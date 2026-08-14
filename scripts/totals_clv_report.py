#!/usr/bin/env python3
"""Evaluate totals model picks against opening lines and closing-line value.

The model side and edge are measured against consensus opening totals. Bets are
settled at the consensus opening price/point, then CLV is measured against the
consensus close by point movement and same-side de-vigged probability movement.

    uv run python scripts/totals_clv_report.py --seasons 2024,2025 --games 400 --sims 500
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.totals_clv import (
    TotalsBookLine,
    TotalsClvBet,
    TotalsClvGame,
    TotalsClvSummary,
    consensus_totals_line,
    summarize_totals_clv,
)
from src.database import PostgresConfig
from src.ml.mlflow_utils import DEFAULT_MLFLOW_TRACKING_URI
from src.sim.contact_environment import ContactEnvironment
from src.sim.db_games import GameDataStore
from src.sim.slate import build_day_ahead_simulator

MarketLines = dict[int, dict[str, dict[str, TotalsBookLine]]]


def _parse_ints(value: str) -> tuple[int, ...]:
    try:
        seasons = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integer seasons") from exc
    if not seasons:
        raise argparse.ArgumentTypeError("at least one season is required")
    return seasons


def _parse_floats(value: str) -> tuple[float, ...]:
    try:
        edges = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numeric edges") from exc
    if not edges:
        raise argparse.ArgumentTypeError("at least one edge is required")
    if any(edge < 0.0 for edge in edges):
        raise argparse.ArgumentTypeError("edges must be non-negative")
    return edges


def _staking_plans(value: str) -> tuple[str, ...]:
    if value == "both":
        return "flat", "kelly"
    return (value,)


def load_totals_market_lines(
    seasons: Sequence[int],
) -> tuple[MarketLines, dict[int, int], dict[int, int]]:
    config = PostgresConfig.from_env()
    by_game: MarketLines = defaultdict(lambda: {"open": {}, "close": {}})
    game_seasons: dict[int, int] = {}
    with psycopg.connect(
        dbname=config.dbname,
        user=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
        connect_timeout=15,
    ) as conn, conn.cursor() as cursor:
        query = sql.SQL(
            """
            SELECT g.season::int, o.game_pk, o.bookmaker, o.line_type,
                   o.total_point, o.over_ml, o.under_ml
            FROM {}.odds_totals o JOIN {}.games g USING(game_pk)
            WHERE g.season::int = ANY(%s)
              AND g.game_type = 'R'
              AND o.line_type IN ('open', 'close')
              AND o.total_point IS NOT NULL
              AND o.over_ml IS NOT NULL
              AND o.under_ml IS NOT NULL
            """
        ).format(sql.Identifier(config.schema), sql.Identifier(config.schema))
        cursor.execute(query, (list(seasons),))
        for season, game_pk, bookmaker, line_type, point, over_ml, under_ml in cursor.fetchall():
            pk = int(game_pk)
            game_seasons[pk] = int(season)
            by_game[pk][str(line_type)][str(bookmaker)] = TotalsBookLine(
                bookmaker=str(bookmaker),
                point=float(point),
                over_ml=float(over_ml),
                under_ml=float(under_ml),
            )
    coverage: Counter[int] = Counter()
    eligible: MarketLines = {}
    for game_pk, lines in by_game.items():
        if lines["open"] and lines["close"]:
            eligible[game_pk] = lines
            coverage[game_seasons[game_pk]] += 1
    return eligible, dict(sorted(coverage.items())), game_seasons


class _SeasonRuntime:
    def __init__(
        self,
        *,
        season: int,
        seed: int,
        mlflow_tracking_uri: str,
        pa_calibration_path: str | None,
    ) -> None:
        self.store = GameDataStore.load(season)
        self.simulator, self.outcome_run_dir = build_day_ahead_simulator(
            season=season,
            seed=seed,
            tracking_uri=mlflow_tracking_uri,
            pa_calibration_path=pa_calibration_path,
        )
        self.contact_env = ContactEnvironment.load(season)
        self.envs = self._load_envs(season) if self.contact_env else {}

    @staticmethod
    def _load_envs(season: int) -> dict[int, tuple[Any, Any]]:
        from scripts.sim_totals_eval import game_environments

        return game_environments(season)


def build_clv_games(
    *,
    market_lines: MarketLines,
    game_seasons: dict[int, int],
    games_requested: int,
    sims: int,
    seed: int,
    mlflow_tracking_uri: str,
    pa_calibration_path: str | None,
) -> tuple[list[TotalsClvGame], dict[int, str]]:
    rng = random.Random(seed)
    candidates = [(game_pk, lines) for game_pk, lines in market_lines.items()]
    rng.shuffle(candidates)
    runtimes: dict[int, _SeasonRuntime] = {}
    outcome_dirs: dict[int, str] = {}
    games: list[TotalsClvGame] = []

    for game_pk, lines in candidates:
        if len(games) >= games_requested:
            break
        season = game_seasons[game_pk]
        runtime = runtimes.get(season)
        if runtime is None:
            runtime = _SeasonRuntime(
                season=season,
                seed=seed,
                mlflow_tracking_uri=mlflow_tracking_uri,
                pa_calibration_path=pa_calibration_path,
            )
            runtimes[season] = runtime
            outcome_dirs[season] = str(runtime.outcome_run_dir)
        if game_pk not in runtime.store.final_game_pks(seed, 10_000):
            continue
        try:
            away = runtime.store.lineup(game_pk, "away", individual_bullpen=True)
            home = runtime.store.lineup(game_pk, "home", individual_bullpen=True)
            away_runs, home_runs = runtime.store.final(game_pk)
        except (ValueError, KeyError):
            continue
        environment = None
        if runtime.contact_env:
            venue_id, weather = runtime.envs.get(game_pk, (None, None))
            environment = runtime.contact_env.multipliers(venue_id, weather)
        results = runtime.simulator.simulate_many(
            away,
            home,
            sims,
            environment=environment,
        )
        totals = [result.away_runs + result.home_runs for result in results]
        game = TotalsClvGame(
            game_pk=game_pk,
            season=season,
            open_lines=tuple(lines["open"].values()),
            close_lines=tuple(lines["close"].values()),
            simulated_totals=totals,
            actual_total=away_runs + home_runs,
        )
        games.append(game)
        open_line = consensus_totals_line(game.open_lines)
        close_line = consensus_totals_line(game.close_lines)
        if open_line is not None and close_line is not None:
            print(
                f"{game_pk} open={open_line.point:g} close={close_line.point:g} "
                f"actual={game.actual_total:g}",
                flush=True,
            )
    return games, outcome_dirs




def _summary_line(edge: float, summary: TotalsClvSummary) -> str:
    return (
        f"edge>={edge:.2f}: bets {summary.n_bets:4d} | "
        f"ROI {summary.roi:+.2%} | win {summary.win_rate:.1%} | "
        f"push {summary.pushes:3d} | staked {summary.total_staked:.2f}u | "
        f"profit {summary.net_profit:+.2f}u | edge {summary.avg_edge:+.4f} | "
        f"point CLV {summary.avg_point_clv:+.3f} | prob CLV {summary.avg_prob_clv:+.4f} | "
        f"beat close {summary.beat_close_rate:.1%}"
    )


def _game_row(game: TotalsClvGame) -> dict[str, float | int | str]:
    open_line = consensus_totals_line(game.open_lines)
    close_line = consensus_totals_line(game.close_lines)
    if open_line is None or close_line is None:
        raise ValueError("game missing open or close consensus")
    return {
        "game_pk": game.game_pk,
        "season": game.season,
        "open_point": open_line.point,
        "close_point": close_line.point,
        "open_prob_over": open_line.prob_over,
        "close_prob_over": close_line.prob_over,
        "sim_mean_total": sum(game.simulated_totals) / len(game.simulated_totals),
        "actual_total": game.actual_total,
        "open_books": len(game.open_lines),
        "close_books": len(game.close_lines),
    }


def _bet_row(edge: float, staking: str, bet: TotalsClvBet) -> dict[str, float | int | str | bool]:
    return {"edge_threshold": edge, "staking": staking, **asdict(bet)}


def write_outputs(
    *,
    games: Sequence[TotalsClvGame],
    summaries: Sequence[TotalsClvSummary],
    bets_by_key: dict[tuple[str, float], list[TotalsClvBet]],
    out_json: Path | None,
    out_csv: Path | None,
) -> None:
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(
                {
                    "summaries": [asdict(summary) for summary in summaries],
                    "games": [_game_row(game) for game in games],
                    "bets": [
                        _bet_row(edge, staking, bet)
                        for (staking, edge), bets in bets_by_key.items()
                        for bet in bets
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        print(f"wrote JSON totals CLV report to {out_json}")
    if out_csv is not None:
        rows = [
            _bet_row(edge, staking, bet)
            for (staking, edge), bets in bets_by_key.items()
            for bet in bets
        ]
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as handle:
            fieldnames = list(rows[0]) if rows else ["edge_threshold", "staking"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote CSV totals CLV bets to {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=_parse_ints, default=(2024, 2025))
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--sims", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--edges", type=_parse_floats, default=(0.0, 0.03, 0.05, 0.08))
    parser.add_argument("--staking", choices=("flat", "kelly", "both"), default="both")
    parser.add_argument("--flat-stake", type=float, default=1.0)
    parser.add_argument("--kelly-multiplier", type=float, default=0.25)
    parser.add_argument("--kelly-cap", type=float, default=0.05)
    parser.add_argument("--mlflow-tracking-uri", default=DEFAULT_MLFLOW_TRACKING_URI)
    parser.add_argument("--pa-calibration", default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    market_lines, coverage, game_seasons = load_totals_market_lines(args.seasons)
    print("Totals open-close CLV report")
    print(f"seasons: {args.seasons}")
    print(f"market coverage with open+close totals: {coverage} total={len(market_lines)}")
    print("selection: sim-vs-consensus-open side/edge; settlement: consensus open")
    print("CLV: selected side vs consensus close; point CLV dominates price CLV")
    print("Kelly staking: quarter Kelly, 5% cap, fixed bankroll/no compounding")

    games, outcome_dirs = build_clv_games(
        game_seasons=game_seasons,
        market_lines=market_lines,
        games_requested=args.games,
        sims=args.sims,
        seed=args.seed,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        pa_calibration_path=args.pa_calibration,
    )
    print(f"simulated games: {len(games)} outcome_dirs={outcome_dirs}")
    if not games:
        raise SystemExit("no simulatable games with open+close totals")

    summaries: list[TotalsClvSummary] = []
    bets_by_key: dict[tuple[str, float], list[TotalsClvBet]] = {}
    for staking in _staking_plans(args.staking):
        print(f"\n=== {staking.upper()} staking ===")
        for edge in args.edges:
            summary, bets = summarize_totals_clv(
                games,
                edge_threshold=edge,
                staking=staking,
                flat_stake=args.flat_stake,
                kelly_multiplier=args.kelly_multiplier,
                kelly_cap=args.kelly_cap,
            )
            summaries.append(summary)
            bets_by_key[(staking, edge)] = bets
            print(_summary_line(edge, summary))

    write_outputs(
        games=games,
        summaries=summaries,
        bets_by_key=bets_by_key,
        out_json=args.out_json,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    main()
