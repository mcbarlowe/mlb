#!/usr/bin/env python3
"""Create daily moneyline paper-trade picks from live odds.

Default strategy:
- champion win model vs consensus no-vig h2h market
- edge >= 0.05
- execute at best available book price for the selected side
- quarter-Kelly stake, 5% cap, fixed bankroll
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.ingest import team_abbrev_to_id
from src.betting.paper_trading import (
    PaperOddsLine,
    PaperTradePick,
    select_moneyline_paper_trade,
)
from src.ml.mlflow_utils import DEFAULT_MLFLOW_TRACKING_URI
from src.sim.slate import SlateGame, active_roster_ids, fetch_slate_games
from src.sim.team_strength import (
    DEFAULT_REGISTERED_STRENGTH_MODEL,
    TeamStrengthPredictor,
    build_live_strength_predictor,
)

CURRENT_ODDS_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
STRATEGY_VERSION = "moneyline_champion_best_open_edge05_v1"
NAME_ALIASES = {
    "Cleveland Guardians": "Cleveland Indians",
    "Miami Marlins": "Florida Marlins",
}
CSV_FIELDS = (
    "strategy_version",
    "paper_date",
    "snapshot_time_utc",
    "game_pk",
    "game_time",
    "away_team",
    "home_team",
    "away_team_id",
    "home_team_id",
    "away_probable",
    "home_probable",
    "side",
    "model_prob_home",
    "selected_model_prob",
    "consensus_market_prob",
    "edge",
    "consensus_home_prob",
    "consensus_away_prob",
    "consensus_home_ml",
    "consensus_away_ml",
    "best_books",
    "best_ml",
    "best_decimal",
    "best_fair_prob",
    "staking",
    "stake_fraction",
    "stake_units",
    "status",
    "close_ml",
    "close_fair_prob",
    "clv",
    "result",
    "profit_units",
)


def _parse_date(value: str | None) -> date:
    if value is None:
        return datetime.now(tz=UTC).astimezone().date()
    return date.fromisoformat(value)


def _parse_datetime(value: object | None) -> datetime | None:
    if value is None:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _fetch_current_odds(api_key: str, regions: str) -> tuple[object, dict[str, str]]:
    response = requests.get(
        CURRENT_ODDS_URL,
        params={
            "apiKey": api_key,
            "regions": regions,
            "markets": "h2h",
            "oddsFormat": "american",
        },
        timeout=30,
    )
    response.raise_for_status()
    headers = {
        "x-requests-last": response.headers.get("x-requests-last", ""),
        "x-requests-remaining": response.headers.get("x-requests-remaining", ""),
        "x-requests-used": response.headers.get("x-requests-used", ""),
    }
    return response.json(), headers


def _odds_games(payload: object) -> list[Mapping[str, object]]:
    if isinstance(payload, Mapping):
        data = payload.get("data")
        if isinstance(data, list):
            return [item for item in data if isinstance(item, Mapping)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    return []


def _resolve_team_id(name: object, mapping: Mapping[str, int]) -> int | None:
    if name is None:
        return None
    text = str(name).strip()
    alias = NAME_ALIASES.get(text, text)
    return mapping.get(alias.upper())


def _market_outcomes(book: Mapping[str, object]) -> Mapping[str, object] | None:
    markets = book.get("markets")
    if not isinstance(markets, list):
        return None
    for market in markets:
        if isinstance(market, Mapping) and market.get("key") == "h2h":
            return market
    return None


def _match_slate_game(
    *,
    odds_game: Mapping[str, object],
    slate_by_pair: Mapping[tuple[int, int], Sequence[SlateGame]],
    team_mapping: Mapping[str, int],
    max_hours: float,
) -> SlateGame | None:
    away_id = _resolve_team_id(odds_game.get("away_team"), team_mapping)
    home_id = _resolve_team_id(odds_game.get("home_team"), team_mapping)
    if away_id is None or home_id is None:
        return None
    candidates = slate_by_pair.get((away_id, home_id), ())
    if not candidates:
        return None
    odds_time = _parse_datetime(odds_game.get("commence_time"))
    if odds_time is None:
        return candidates[0]
    scored: list[tuple[float, SlateGame]] = []
    for game in candidates:
        game_time = _parse_datetime(game.game_datetime)
        if game_time is None:
            continue
        diff_hours = abs((game_time - odds_time).total_seconds()) / 3600.0
        scored.append((diff_hours, game))
    if not scored:
        return candidates[0]
    diff, game = min(scored, key=lambda item: item[0])
    return game if diff <= max_hours else None


def _odds_by_game_pk(
    *,
    payload: object,
    slate_games: Sequence[SlateGame],
    max_match_hours: float,
) -> dict[int, list[PaperOddsLine]]:
    team_mapping = team_abbrev_to_id()
    slate_by_pair: dict[tuple[int, int], list[SlateGame]] = defaultdict(list)
    for game in slate_games:
        slate_by_pair[(game.away_team_id, game.home_team_id)].append(game)

    out: dict[int, list[PaperOddsLine]] = defaultdict(list)
    for odds_game in _odds_games(payload):
        slate_game = _match_slate_game(
            odds_game=odds_game,
            slate_by_pair=slate_by_pair,
            team_mapping=team_mapping,
            max_hours=max_match_hours,
        )
        if slate_game is None:
            continue
        home_name = odds_game.get("home_team")
        away_name = odds_game.get("away_team")
        bookmakers = odds_game.get("bookmakers")
        if not isinstance(bookmakers, list):
            continue
        for book in bookmakers:
            if not isinstance(book, Mapping):
                continue
            market = _market_outcomes(book)
            if market is None:
                continue
            outcomes = market.get("outcomes")
            if not isinstance(outcomes, list):
                continue
            prices = {
                str(item.get("name")): item.get("price")
                for item in outcomes
                if isinstance(item, Mapping)
            }
            home_ml = prices.get(str(home_name))
            away_ml = prices.get(str(away_name))
            if home_ml is None or away_ml is None:
                continue
            out[slate_game.game_pk].append(
                PaperOddsLine(
                    bookmaker=str(book.get("key") or book.get("title") or "unknown"),
                    home_ml=float(home_ml),
                    away_ml=float(away_ml),
                    last_update=str(market.get("last_update") or ""),
                )
            )
    return dict(out)


def _active_rosters(game: SlateGame, *, enabled: bool) -> tuple[tuple[int, ...], ...]:
    if not enabled:
        return (), (), (), ()
    away_batters, away_pitchers = active_roster_ids(game.away_team_id, game.slate_date)
    home_batters, home_pitchers = active_roster_ids(game.home_team_id, game.slate_date)
    return away_batters, home_batters, away_pitchers, home_pitchers


def _predict_home_probability(
    *,
    game: SlateGame,
    predictor: TeamStrengthPredictor,
    target_date: date,
    include_active_rosters: bool,
) -> float:
    away_batters, home_batters, away_pitchers, home_pitchers = _active_rosters(
        game,
        enabled=include_active_rosters,
    )
    return predictor.predict_home_probability(
        season=target_date.year,
        away_team_id=game.away_team_id,
        home_team_id=game.home_team_id,
        away_starter_id=game.away_probable.player_id or 0,
        home_starter_id=game.home_probable.player_id or 0,
        prediction_date=target_date,
        away_active_batter_ids=away_batters,
        home_active_batter_ids=home_batters,
        away_reliever_ids=away_pitchers,
        home_reliever_ids=home_pitchers,
    )


def _pick_row(
    *,
    game: SlateGame,
    pick: PaperTradePick,
    model_prob_home: float,
    snapshot_time: str,
    paper_date: str,
    staking: str,
) -> dict[str, str]:
    return {
        "strategy_version": STRATEGY_VERSION,
        "paper_date": paper_date,
        "snapshot_time_utc": snapshot_time,
        "game_pk": str(game.game_pk),
        "game_time": game.game_datetime or "",
        "away_team": game.away_abbrev,
        "home_team": game.home_abbrev,
        "away_team_id": str(game.away_team_id),
        "home_team_id": str(game.home_team_id),
        "away_probable": game.away_probable.display_name,
        "home_probable": game.home_probable.display_name,
        "side": pick.side,
        "model_prob_home": f"{model_prob_home:.6f}",
        "selected_model_prob": f"{pick.model_prob:.6f}",
        "consensus_market_prob": f"{pick.consensus_market_prob:.6f}",
        "edge": f"{pick.edge:.6f}",
        "consensus_home_prob": f"{pick.consensus_home_prob:.6f}",
        "consensus_away_prob": f"{pick.consensus_away_prob:.6f}",
        "consensus_home_ml": f"{pick.consensus_home_ml:.1f}",
        "consensus_away_ml": f"{pick.consensus_away_ml:.1f}",
        "best_books": "|".join(pick.best_books),
        "best_ml": f"{pick.best_ml:.1f}",
        "best_decimal": f"{pick.best_decimal:.6f}",
        "best_fair_prob": f"{pick.best_fair_prob:.6f}",
        "staking": staking,
        "stake_fraction": f"{pick.stake_fraction:.6f}",
        "stake_units": f"{pick.stake_units:.4f}",
        "status": "open",
        "close_ml": "",
        "close_fair_prob": "",
        "clv": "",
        "result": "",
        "profit_units": "",
    }


def _read_existing(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_picks(
    *,
    path: Path,
    rows: Sequence[dict[str, str]],
    paper_date: str,
    replace_date: bool,
) -> tuple[int, int]:
    existing = _read_existing(path)
    if replace_date:
        existing = [
            row
            for row in existing
            if not (
                row.get("paper_date") == paper_date
                and row.get("strategy_version") == STRATEGY_VERSION
            )
        ]
    existing_keys = {
        (row.get("strategy_version", ""), row.get("game_pk", ""))
        for row in existing
    }
    new_rows: list[dict[str, str]] = []
    skipped = 0
    for row in rows:
        key = (row["strategy_version"], row["game_pk"])
        if key in existing_keys:
            skipped += 1
            continue
        existing_keys.add(key)
        new_rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_FIELDS)
        for row in existing + new_rows:
            writer.writerow([row.get(field, "") for field in CSV_FIELDS])
    return len(new_rows), skipped


def _print_rows(rows: Iterable[dict[str, str]]) -> None:
    for row in rows:
        side_label = row["home_team"] if row["side"] == "home" else row["away_team"]
        print(
            f"{row['game_pk']} {row['away_team']} @ {row['home_team']} | "
            f"pick {side_label} ({row['side']}) {row['best_ml']} at {row['best_books']} | "
            f"edge {float(row['edge']):+.3f} stake {row['stake_units']}u"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--edge-threshold", type=float, default=0.05)
    parser.add_argument("--staking", choices=("flat", "kelly"), default="kelly")
    parser.add_argument("--bankroll-units", type=float, default=100.0)
    parser.add_argument("--flat-stake-units", type=float, default=1.0)
    parser.add_argument("--kelly-multiplier", type=float, default=0.25)
    parser.add_argument("--kelly-cap", type=float, default=0.05)
    parser.add_argument("--regions", default="us")
    parser.add_argument("--odds-json", type=str, default=None)
    parser.add_argument("--max-match-hours", type=float, default=12.0)
    parser.add_argument("--all-games", action="store_true")
    parser.add_argument("--skip-active-rosters", action="store_true")
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=DEFAULT_MLFLOW_TRACKING_URI,
        help="Tracking server used for champion artifacts.",
    )
    parser.add_argument(
        "--win-model-name",
        type=str,
        default=DEFAULT_REGISTERED_STRENGTH_MODEL,
    )
    parser.add_argument(
        "--out",
        type=str,
        default="output/paper_trades/moneyline_paper_trades.csv",
    )
    parser.add_argument("--replace-date", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_date = _parse_date(args.date)
    paper_date = target_date.isoformat()
    states = None if args.all_games else {"Preview"}
    slate_games = fetch_slate_games(target_date, abstract_states=states)
    if not slate_games:
        raise SystemExit(f"No slate games found for {paper_date}")

    if args.odds_json:
        odds_payload = json.loads(Path(args.odds_json).read_text())
        headers: dict[str, str] = {}
    else:
        api_key = os.environ.get("ODDS_API_KEY")
        if not api_key:
            raise SystemExit("ODDS_API_KEY is required unless --odds-json is provided")
        odds_payload, headers = _fetch_current_odds(api_key, args.regions)

    odds_by_game = _odds_by_game_pk(
        payload=odds_payload,
        slate_games=slate_games,
        max_match_hours=args.max_match_hours,
    )
    predictor = build_live_strength_predictor(
        target_date,
        tracking_uri=args.mlflow_tracking_uri,
        registered_model_name=args.win_model_name,
    )
    snapshot_time = datetime.now(tz=UTC).isoformat()
    rows: list[dict[str, str]] = []
    skipped_no_odds = 0
    skipped_no_edge = 0
    for game in slate_games:
        odds_lines = odds_by_game.get(game.game_pk, [])
        if not odds_lines:
            skipped_no_odds += 1
            continue
        model_prob_home = _predict_home_probability(
            game=game,
            predictor=predictor,
            target_date=target_date,
            include_active_rosters=not args.skip_active_rosters,
        )
        pick = select_moneyline_paper_trade(
            model_prob_home=model_prob_home,
            odds_lines=odds_lines,
            edge_threshold=args.edge_threshold,
            staking=args.staking,
            bankroll_units=args.bankroll_units,
            flat_stake_units=args.flat_stake_units,
            kelly_multiplier=args.kelly_multiplier,
            kelly_cap=args.kelly_cap,
        )
        if pick is None:
            skipped_no_edge += 1
            continue
        rows.append(
            _pick_row(
                game=game,
                pick=pick,
                model_prob_home=model_prob_home,
                snapshot_time=snapshot_time,
                paper_date=paper_date,
                staking=args.staking,
            )
        )

    print(
        f"Paper-trade moneyline {paper_date}: slate={len(slate_games)} "
        f"odds_matched={len(odds_by_game)} picks={len(rows)} "
        f"no_odds={skipped_no_odds} no_edge={skipped_no_edge}"
    )
    if headers:
        print(
            "Odds API credits: "
            f"last={headers.get('x-requests-last', '')} "
            f"used={headers.get('x-requests-used', '')} "
            f"remaining={headers.get('x-requests-remaining', '')}"
        )
    _print_rows(rows)

    if args.dry_run:
        print("dry-run: no paper-trade rows written")
        return
    written, skipped_existing = _write_picks(
        path=Path(args.out),
        rows=rows,
        paper_date=paper_date,
        replace_date=args.replace_date,
    )
    print(f"wrote {written} rows to {args.out}; skipped_existing={skipped_existing}")


if __name__ == "__main__":
    main()
