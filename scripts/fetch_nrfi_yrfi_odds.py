#!/usr/bin/env python3
"""Fetch and store current MLB NRFI/YRFI odds.

The Odds API exposes NRFI/YRFI as first-inning totals: Over 0.5 = YRFI and
Under 0.5 = NRFI. Stores one idempotent row per game/book/line_type in
``mlb.nrfi_yrfi_odds``.

    uv run python scripts/fetch_nrfi_yrfi_odds.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.market_data.nrfi_yrfi_odds import (
    NRFI_YRFI_MARKET,
    NrfiYrfiOddsApiRow,
    parse_nrfi_yrfi_odds_rows,
)
from src.market_data.nrfi_yrfi_odds_store import upsert_nrfi_yrfi_odds_rows
from src.market_data.team_mapping import team_abbrev_to_id
from src.sim.slate import SlateGame, fetch_slate_games

CURRENT_ODDS_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
NAME_ALIASES = {
    "Cleveland Guardians": "Cleveland Indians",
    "Miami Marlins": "Florida Marlins",
}
CSV_FIELDS = (
    "game_pk",
    "game_date",
    "game_time",
    "away_team",
    "home_team",
    "away_team_id",
    "home_team_id",
    "bookmaker",
    "line_type",
    "snapshot_time",
    "market_key",
    "market_last_update",
    "total_point",
    "yrfi_ml",
    "nrfi_ml",
    "source",
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


def _fetch_current_nrfi_yrfi_odds(
    api_key: str,
    regions: str,
    *,
    max_events: int,
) -> tuple[object, dict[str, str]]:
    events_response = requests.get(
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/events",
        params={"apiKey": api_key, "dateFormat": "iso"},
        timeout=30,
    )
    events_response.raise_for_status()
    events = events_response.json()
    if not isinstance(events, list):
        raise TypeError("Unexpected events response from Odds API")
    if max_events > 0:
        events = events[:max_events]

    payloads: list[object] = []
    total_last = 0
    used = events_response.headers.get("x-requests-used", "")
    remaining = events_response.headers.get("x-requests-remaining", "")
    for event in events:
        if not isinstance(event, Mapping) or event.get("id") is None:
            continue
        response = requests.get(
            f"{CURRENT_ODDS_URL.rsplit('/odds', 1)[0]}/events/{event['id']}/odds",
            params={
                "apiKey": api_key,
                "regions": regions,
                "markets": NRFI_YRFI_MARKET,
                "oddsFormat": "american",
                "dateFormat": "iso",
            },
            timeout=30,
        )
        response.raise_for_status()
        last = response.headers.get("x-requests-last", "")
        total_last += int(last) if last.isdigit() else 0
        used = response.headers.get("x-requests-used", used)
        remaining = response.headers.get("x-requests-remaining", remaining)
        payloads.append(response.json())
    headers = {
        "x-requests-last": str(total_last),
        "x-requests-remaining": remaining,
        "x-requests-used": used,
    }
    return payloads, headers


def _resolve_team_id(name: object, mapping: Mapping[str, int]) -> int | None:
    if name is None:
        return None
    text = str(name).strip()
    alias = NAME_ALIASES.get(text, text)
    return mapping.get(alias.upper())


def _match_slate_game(
    *,
    odds_row: NrfiYrfiOddsApiRow,
    slate_by_pair: Mapping[tuple[int, int], Sequence[SlateGame]],
    team_mapping: Mapping[str, int],
    max_hours: float,
) -> SlateGame | None:
    away_id = _resolve_team_id(odds_row.away_team, team_mapping)
    home_id = _resolve_team_id(odds_row.home_team, team_mapping)
    if away_id is None or home_id is None:
        return None
    candidates = slate_by_pair.get((away_id, home_id), ())
    if not candidates:
        return None
    odds_time = _parse_datetime(odds_row.commence_time)
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


def _db_row(
    *,
    game: SlateGame,
    odds_row: NrfiYrfiOddsApiRow,
    snapshot_time: str,
    line_type: str,
) -> dict[str, object | None]:
    return {
        "game_pk": game.game_pk,
        "game_date": game.slate_date,
        "game_time": game.game_datetime,
        "away_team": game.away_abbrev,
        "home_team": game.home_abbrev,
        "away_team_id": game.away_team_id,
        "home_team_id": game.home_team_id,
        "bookmaker": odds_row.bookmaker,
        "line_type": line_type,
        "snapshot_time": snapshot_time,
        "market_key": NRFI_YRFI_MARKET,
        "market_last_update": odds_row.market_last_update,
        "total_point": odds_row.total_point,
        "yrfi_ml": odds_row.yrfi_ml,
        "nrfi_ml": odds_row.nrfi_ml,
        "source": "the-odds-api",
    }


def nrfi_yrfi_rows_by_game_pk(
    *,
    payload: object,
    slate_games: Sequence[SlateGame],
    max_match_hours: float,
    snapshot_time: str,
    line_type: str,
) -> tuple[list[dict[str, object | None]], int]:
    team_mapping = team_abbrev_to_id()
    slate_by_pair: dict[tuple[int, int], list[SlateGame]] = defaultdict(list)
    for game in slate_games:
        slate_by_pair[(game.away_team_id, game.home_team_id)].append(game)

    rows: list[dict[str, object | None]] = []
    unmatched = 0
    for odds_row in parse_nrfi_yrfi_odds_rows(payload):
        game = _match_slate_game(
            odds_row=odds_row,
            slate_by_pair=slate_by_pair,
            team_mapping=team_mapping,
            max_hours=max_match_hours,
        )
        if game is None:
            unmatched += 1
            continue
        rows.append(
            _db_row(
                game=game,
                odds_row=odds_row,
                snapshot_time=snapshot_time,
                line_type=line_type,
            )
        )
    return rows, unmatched


def _write_rows(path: Path, rows: Sequence[Mapping[str, object | None]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_FIELDS)
        for row in rows:
            writer.writerow([row.get(field, "") or "" for field in CSV_FIELDS])


def _print_rows(rows: Sequence[Mapping[str, object | None]]) -> None:
    for row in rows:
        print(
            f"{row['game_pk']} {row['away_team']} @ {row['home_team']} "
            f"{row['bookmaker']} | YRFI {row.get('yrfi_ml')} "
            f"NRFI {row.get('nrfi_ml')} point {row.get('total_point')}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--regions", default="us")
    parser.add_argument("--line-type", default="current")
    parser.add_argument("--odds-json", type=str, default=None)
    parser.add_argument("--max-match-hours", type=float, default=12.0)
    parser.add_argument("--max-events", type=int, default=0, help="limit API event-odds calls")
    parser.add_argument("--all-games", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-db-log", action="store_true")
    parser.add_argument("--out", default="output/odds/nrfi_yrfi_odds_current.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_date = _parse_date(args.date)
    states = None if args.all_games else {"Preview"}
    slate_games = fetch_slate_games(target_date, abstract_states=states)
    if not slate_games:
        raise SystemExit(f"No slate games found for {target_date.isoformat()}")

    if args.odds_json:
        odds_payload = json.loads(Path(args.odds_json).read_text())
        headers: dict[str, str] = {}
    else:
        api_key = os.environ.get("ODDS_API_KEY")
        if not api_key:
            raise SystemExit("ODDS_API_KEY is required unless --odds-json is provided")
        odds_payload, headers = _fetch_current_nrfi_yrfi_odds(
            api_key,
            args.regions,
            max_events=args.max_events,
        )

    snapshot_time = datetime.now(tz=UTC).isoformat()
    rows, unmatched = nrfi_yrfi_rows_by_game_pk(
        payload=odds_payload,
        slate_games=slate_games,
        max_match_hours=args.max_match_hours,
        snapshot_time=snapshot_time,
        line_type=args.line_type,
    )
    matched_games = {int(str(row["game_pk"])) for row in rows if row.get("game_pk") is not None}
    print(
        f"NRFI/YRFI odds {target_date.isoformat()}: slate={len(slate_games)} "
        f"matched_games={len(matched_games)} rows={len(rows)} unmatched_book_rows={unmatched}"
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
        print("dry-run: no NRFI/YRFI odds rows written to DB or CSV")
        return
    if not args.no_db_log:
        db_written = upsert_nrfi_yrfi_odds_rows(rows)
        print(f"upserted {db_written} rows into nrfi_yrfi_odds")
    _write_rows(Path(args.out), rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
