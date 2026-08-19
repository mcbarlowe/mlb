#!/usr/bin/env python3
"""Evaluate first-five totals picks against F5 market lines.

Tracks simulated pitch sequences and outcome models to generate
first-five (F5) simulation probabilities. Compares against market
pricing via consensus, best-book, and optionally market-anchor blends.
Logs progress to optional file.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import psycopg
from psycopg import sql

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.sim_totals_eval import game_environments
from src.betting.f5_clv import (
    F5BookLine,
    F5ClvBet,
    F5ClvGame,
    F5ClvSummary,
    F5LineShoppingBet,
    F5LineShoppingSummary,
    F5ModelProbability,
    compare_f5_line_shopping,
    consensus_f5_totals_line,
    summarize_f5_clv,
)
from src.database import PostgresConfig
from src.sim.slate import build_day_ahead_simulator


# Configure logging
def setup_logging(log_file: Path | None = None) -> logging.Logger:
    """Setup logging with optional file output."""
    logger = logging.getLogger("f5_clv_report")
    logger.setLevel(logging.DEBUG)
    
    # Console handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (DEBUG level) if provided
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

from src.ml.mlflow_utils import DEFAULT_MLFLOW_TRACKING_URI
from src.sim.contact_environment import ContactEnvironment
from src.sim.db_games import GameDataStore

MarketLines = dict[int, dict[str, dict[str, F5BookLine]]]


def _parse_ints(value: str) -> tuple[int, ...]:
    try:
        seasons = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not seasons:
        raise argparse.ArgumentTypeError("at least one season is required")
    return seasons


def _parse_floats(value: str) -> tuple[float, ...]:
    try:
        edges = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not edges:
        raise argparse.ArgumentTypeError("at least one edge is required")
    return edges


def _staking_plans(value: str) -> tuple[str, ...]:
    if value == "both":
        return "flat", "kelly"
    return (value,)


def _connect(config: PostgresConfig):
    return psycopg.connect(
        dbname=config.dbname,
        user=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
        connect_timeout=15,
    )


def load_f5_market_lines(
    seasons: Sequence[int],
) -> tuple[MarketLines, dict[str, dict[int, int]], dict[int, int], int]:
    config = PostgresConfig.from_env()
    by_game: MarketLines = defaultdict(lambda: defaultdict(dict))
    game_seasons: dict[int, int] = {}
    total_rows = 0
    with _connect(config) as conn, conn.cursor() as cursor:
        count_query = sql.SQL(
            """
            SELECT COUNT(*)
            FROM {}.f5_odds
            WHERE total_point IS NOT NULL
              AND over_ml IS NOT NULL
              AND under_ml IS NOT NULL
            """
        ).format(sql.Identifier(config.schema))
        cursor.execute(count_query)
        count_row = cursor.fetchone()
        total_rows = int(count_row[0]) if count_row is not None else 0
        if total_rows == 0:
            raise SystemExit(
                f"{config.schema}.f5_odds has no rows with F5 totals lines"
            )

        query = sql.SQL(
            """
            SELECT g.season::int, f.game_pk, f.bookmaker, f.line_type,
                   f.total_point, f.over_ml, f.under_ml
            FROM {}.f5_odds f
            JOIN {}.games g USING(game_pk)
            WHERE g.season::int = ANY(%s)
              AND g.game_type = 'R'
              AND f.line_type IN ('open', 'current', 'close')
              AND f.total_point IS NOT NULL
              AND f.over_ml IS NOT NULL
              AND f.under_ml IS NOT NULL
            """
        ).format(sql.Identifier(config.schema), sql.Identifier(config.schema))
        cursor.execute(query, (list(seasons),))
        for season, game_pk, bookmaker, line_type, point, over_ml, under_ml in cursor.fetchall():
            pk = int(game_pk)
            bucket = str(line_type).lower()
            game_seasons[pk] = int(season)
            by_game[pk][bucket][str(bookmaker)] = F5BookLine(
                bookmaker=str(bookmaker),
                point=float(point),
                over_ml=float(over_ml),
                under_ml=float(under_ml),
            )
    if not by_game:
        raise SystemExit(
            f"{config.schema}.f5_odds has no F5 totals rows for seasons {tuple(seasons)}"
        )

    coverage: dict[str, Counter[int]] = {
        "open": Counter(),
        "current": Counter(),
        "close": Counter(),
        "open_style": Counter(),
        "open_style_with_close": Counter(),
    }
    eligible: MarketLines = {}
    for game_pk, buckets in by_game.items():
        season = game_seasons[game_pk]
        for line_type in ("open", "current", "close"):
            if buckets.get(line_type):
                coverage[line_type][season] += 1
        take_bucket = "open" if buckets.get("open") else "current"
        if not buckets.get(take_bucket):
            continue
        coverage["open_style"][season] += 1
        if buckets.get("close"):
            coverage["open_style_with_close"][season] += 1
        eligible[game_pk] = buckets
    return (
        eligible,
        {key: dict(sorted(counter.items())) for key, counter in coverage.items()},
        game_seasons,
        total_rows,
    )


def load_f5_actual_totals(
    game_pks: Sequence[int], *, prefix_innings: int
) -> dict[int, float]:
    if not game_pks:
        return {}
    config = PostgresConfig.from_env()
    totals: dict[int, float] = {}
    with _connect(config) as conn, conn.cursor() as cursor:
        query = sql.SQL(
            """
            SELECT game_pk,
                   SUM(runs) FILTER (WHERE team_type = 'away')::float AS away_runs,
                   SUM(runs) FILTER (WHERE team_type = 'home')::float AS home_runs,
                   COUNT(*) FILTER (
                       WHERE team_type = 'away' AND runs IS NOT NULL
                   ) AS away_rows,
                   COUNT(*) FILTER (
                       WHERE team_type = 'home' AND runs IS NOT NULL
                   ) AS home_rows
            FROM {}.linescore
            WHERE game_pk = ANY(%s)
              AND inning BETWEEN 1 AND %s
            GROUP BY game_pk
            """
        ).format(sql.Identifier(config.schema))
        cursor.execute(query, (list(game_pks), prefix_innings))
        for game_pk, away_runs, home_runs, away_rows, home_rows in cursor.fetchall():
            if away_rows < prefix_innings or home_rows < prefix_innings:
                continue
            if away_runs is None or home_runs is None:
                continue
            totals[int(game_pk)] = float(away_runs) + float(home_runs)
    return totals


class _SeasonRuntime:
    def __init__(
        self,
        *,
        season: int,
        seed: int,
        mlflow_tracking_uri: str,
        pa_calibration_path: str | None,
        pitch_type_model: str | None,
        pitch_type_model_weight: float,
    ) -> None:
        self.store = GameDataStore.load(season)
        self.final_pks = set(self.store.final_game_pks(seed, 10_000))
        simulator_builder = cast(Any, build_day_ahead_simulator)
        self.simulator, self.outcome_run_dir = simulator_builder(
            season=season,
            seed=seed,
            tracking_uri=mlflow_tracking_uri,
            pa_calibration_path=pa_calibration_path,
            pitch_type_model_dir=pitch_type_model,
            pitch_type_model_weight=pitch_type_model_weight,
        )
        self.contact_env = ContactEnvironment.load(season)
        self.envs = game_environments(season) if self.contact_env else {}


def build_f5_clv_games(
    *,
    market_lines: MarketLines,
    game_seasons: dict[int, int],
    actual_totals: dict[int, float],
    games_requested: int,
    sims: int,
    seed: int,
    prefix_innings: int,
    mlflow_tracking_uri: str,
    pa_calibration_path: str | None,
    pitch_type_model: str | None,
    pitch_type_model_weight: float,
) -> tuple[list[F5ClvGame], dict[int, str]]:
    rng = random.Random(seed)
    candidates = [(game_pk, lines) for game_pk, lines in market_lines.items()]
    rng.shuffle(candidates)
    runtimes: dict[int, _SeasonRuntime] = {}
    outcome_dirs: dict[int, str] = {}
    games: list[F5ClvGame] = []

    for game_pk, lines in candidates:
        if len(games) >= games_requested:
            break
        if game_pk not in actual_totals:
            continue
        season = game_seasons[game_pk]
        runtime = runtimes.get(season)
        if runtime is None:
            runtime = _SeasonRuntime(
                season=season,
                seed=seed,
                mlflow_tracking_uri=mlflow_tracking_uri,
                pa_calibration_path=pa_calibration_path,
                pitch_type_model=pitch_type_model,
                pitch_type_model_weight=pitch_type_model_weight,
            )
            runtimes[season] = runtime
            outcome_dirs[season] = str(runtime.outcome_run_dir)
        if game_pk not in runtime.final_pks:
            continue
        try:
            away = runtime.store.lineup(game_pk, "away", individual_bullpen=True)
            home = runtime.store.lineup(game_pk, "home", individual_bullpen=True)
        except (ValueError, KeyError):
            continue
        environment = None
        if runtime.contact_env:
            venue_id, weather = runtime.envs.get(game_pk, (None, None))
            environment = runtime.contact_env.multipliers(venue_id, weather)
        results = runtime.simulator.simulate_prefix_many(
            away,
            home,
            sims,
            innings=prefix_innings,
            environment=environment,
        )
        simulated_totals = [result.away_runs + result.home_runs for result in results]
        take_type = "open" if lines.get("open") else "current"
        game = F5ClvGame(
            game_pk=game_pk,
            season=season,
            open_lines=tuple(lines[take_type].values()),
            close_lines=tuple(lines.get("close", {}).values()),
            simulated_totals=simulated_totals,
            actual_total=actual_totals[game_pk],
            take_line_type=take_type,
        )
        games.append(game)
        open_line = consensus_f5_totals_line(game.open_lines)
        close_line = consensus_f5_totals_line(game.close_lines)
        close_text = "n/a" if close_line is None else f"{close_line.point:g}"
        if open_line is not None:
            print(
                f"{game_pk} {take_type}={open_line.point:g} close={close_text} "
                f"actual_f5={game.actual_total:g}",
                flush=True,
            )
    return games, outcome_dirs


def _fmt_optional(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def _sim_prob_over(simulated_totals: Sequence[float], point: float) -> float:
    if not simulated_totals:
        raise ValueError("simulated_totals must not be empty")
    wins = sum(1 for total in simulated_totals if total > point)
    pushes = sum(1 for total in simulated_totals if math.isclose(total, point))
    return (wins + 0.5 * pushes) / len(simulated_totals)

def _sim_probability(
    simulated_totals: Sequence[float], *, point: float, side: str
) -> float:
    if not simulated_totals:
        raise ValueError("simulated_totals must not be empty")
    pushes = sum(1 for total in simulated_totals if math.isclose(total, point))
    if side == "over":
        wins = sum(1 for total in simulated_totals if total > point)
    elif side == "under":
        wins = sum(1 for total in simulated_totals if total < point)
    else:
        raise ValueError(f"unknown F5 totals side {side!r}")
    return (wins + 0.5 * pushes) / len(simulated_totals)


def _market_anchor_probability(lambda_value: float) -> F5ModelProbability:
    if not math.isfinite(lambda_value):
        raise ValueError("market anchor lambda must be finite")

    def probability(
        game: F5ClvGame,
        point: float,
        side: str,
        market_prob: float,
    ) -> float:
        sim_prob = _sim_probability(game.simulated_totals, point=point, side=side)
        return min(
            max(market_prob + lambda_value * (sim_prob - market_prob), 1e-6),
            1.0 - 1e-6,
        )

    return probability


def _summary_line(edge: float, summary: F5ClvSummary) -> str:
    return (
        f"edge>={edge:.2f}: bets {summary.n_bets:4d} | "
        f"settled {summary.n_settled_bets:4d} | ROI {summary.roi:+.2%} | "
        f"win {summary.win_rate:.1%} | push {summary.pushes:3d} | "
        f"profit {summary.net_profit:+.2f}u | edge {summary.avg_edge:+.4f} | "
        f"model/market Brier "
        f"{_fmt_optional(summary.model_brier_all, '.4f')}/"
        f"{_fmt_optional(summary.market_brier_all, '.4f')} | "
        f"log loss {_fmt_optional(summary.model_log_loss_all, '.4f')}/"
        f"{_fmt_optional(summary.market_log_loss_all, '.4f')} | "
        f"point CLV {_fmt_optional(summary.avg_point_clv, '+.3f')} | "
        f"beat close {_fmt_optional(summary.beat_close_rate, '.1%')}"
    )


def _line_shopping_summary_line(edge: float, summary: F5LineShoppingSummary) -> str:
    return (
        f"line-shopping edge>={edge:.2f}: "
        f"ROI consensus {summary.consensus_roi:+.2%} -> "
        f"best {summary.best_roi:+.2%} "
        f"lift {summary.roi_lift:+.2%} | "
        f"point lift {summary.avg_point_lift:+.3f} | "
        f"decimal lift {summary.avg_decimal_lift:+.4f} | "
        f"point CLV lift {_fmt_optional(summary.point_clv_lift, '+.3f')} | "
        f"prob CLV lift {_fmt_optional(summary.prob_clv_lift, '+.4f')} | "
        f"close n {summary.close_n}"
    )


def _book_line_row(line: F5BookLine) -> dict[str, float | str]:
    return {
        "bookmaker": line.bookmaker,
        "point": line.point,
        "over_ml": line.over_ml,
        "under_ml": line.under_ml,
    }


def _game_row(game: F5ClvGame) -> dict[str, object]:
    open_line = consensus_f5_totals_line(game.open_lines)
    close_line = consensus_f5_totals_line(game.close_lines)
    if open_line is None:
        raise ValueError("game missing F5 take consensus")
    return {
        "game_pk": game.game_pk,
        "season": game.season,
        "take_line_type": game.take_line_type,
        "take_point": open_line.point,
        "close_point": None if close_line is None else close_line.point,
        "take_prob_over": open_line.prob_over,
        "close_prob_over": None if close_line is None else close_line.prob_over,
        "sim_mean_total": sum(game.simulated_totals) / len(game.simulated_totals),
        "sim_prob_over": _sim_prob_over(game.simulated_totals, open_line.point),
        "simulated_totals": list(game.simulated_totals),
        "open_lines": [_book_line_row(line) for line in game.open_lines],
        "close_lines": [_book_line_row(line) for line in game.close_lines],
        "actual_f5_total": game.actual_total,
        "take_books": len(game.open_lines),
        "close_books": len(game.close_lines),
    }


def _bet_row(edge: float, staking: str, bet: F5ClvBet) -> dict[str, object]:
    return {"edge_threshold": edge, "staking": staking, **asdict(bet)}


def _line_shopping_bet_row(
    edge: float, staking: str, bet: F5LineShoppingBet
) -> dict[str, object]:
    return {"edge_threshold": edge, "staking": staking, **asdict(bet)}


def write_outputs(
    *,
    games: Sequence[F5ClvGame],
    summaries: Sequence[F5ClvSummary],
    bets_by_key: dict[tuple[str, float], list[F5ClvBet]],
    line_shopping_summaries: Sequence[F5LineShoppingSummary],
    line_shopping_bets_by_key: dict[tuple[str, float], list[F5LineShoppingBet]],
    coverage: dict[str, dict[int, int]],
    outcome_dirs: dict[int, str],
    total_f5_rows: int,
    out_json: Path | None,
    out_csv: Path | None,
    anchor_blend_lambda: float | None = None,
    anchor_blend_summaries: Sequence[F5ClvSummary] = (),
    anchor_blend_bets_by_key: dict[tuple[str, float], list[F5ClvBet]] | None = None,
    anchor_blend_line_shopping_summaries: Sequence[F5LineShoppingSummary] = (),
    anchor_blend_line_shopping_bets_by_key: dict[tuple[str, float], list[F5LineShoppingBet]] | None = None,
) -> None:
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "coverage": coverage,
            "total_f5_rows": total_f5_rows,
            "outcome_dirs": outcome_dirs,
            "summaries": [asdict(summary) for summary in summaries],
            "line_shopping_summaries": [
                asdict(summary) for summary in line_shopping_summaries
            ],
            "games": [_game_row(game) for game in games],
            "bets": [
                _bet_row(edge, staking, bet)
                for (staking, edge), bets in bets_by_key.items()
                for bet in bets
            ],
            "line_shopping_bets": [
                _line_shopping_bet_row(edge, staking, bet)
                for (staking, edge), bets in line_shopping_bets_by_key.items()
                for bet in bets
            ],
        }
        if anchor_blend_lambda is not None:
            payload["anchor_blend"] = {
                "lambda": anchor_blend_lambda,
                "summaries": [
                    asdict(summary) for summary in anchor_blend_summaries
                ],
                "line_shopping_summaries": [
                    asdict(summary)
                    for summary in anchor_blend_line_shopping_summaries
                ],
                "bets": [
                    _bet_row(edge, staking, bet)
                    for (staking, edge), bets in (
                        anchor_blend_bets_by_key or {}
                    ).items()
                    for bet in bets
                ],
                "line_shopping_bets": [
                    _line_shopping_bet_row(edge, staking, bet)
                    for (staking, edge), bets in (
                        anchor_blend_line_shopping_bets_by_key or {}
                    ).items()
                    for bet in bets
                ],
            }
        out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"wrote JSON F5 CLV report to {out_json}")
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
        print(f"wrote CSV F5 CLV bets to {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=_parse_ints, default=(2025,))
    parser.add_argument("--games", type=int, default=25)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--prefix-innings", type=int, default=5)
    parser.add_argument("--edges", type=_parse_floats, default=(0.0, 0.03, 0.05))
    parser.add_argument("--staking", choices=("flat", "kelly", "both"), default="flat")
    parser.add_argument("--execution", choices=("best", "consensus"), default="best")
    parser.add_argument(
        "--market-anchor-lambda",
        type=float,
        default=None,
        help="Also evaluate p_market + lambda * (p_sim - p_market) as a non-default diagnostic.",
    )
    parser.add_argument("--flat-stake", type=float, default=1.0)
    parser.add_argument("--kelly-multiplier", type=float, default=0.25)
    parser.add_argument("--kelly-cap", type=float, default=0.05)
    parser.add_argument("--mlflow-tracking-uri", default=DEFAULT_MLFLOW_TRACKING_URI)
    parser.add_argument("--pa-calibration", default=None)
    parser.add_argument("--pitch-type-model", default=None)
    parser.add_argument("--pitch-type-model-weight", type=float, default=1.0)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    market_lines, coverage, game_seasons, total_f5_rows = load_f5_market_lines(
        args.seasons
    )
    actual_totals = load_f5_actual_totals(
        tuple(market_lines),
        prefix_innings=args.prefix_innings,
    )
    print("F5 totals CLV report")
    print(f"seasons: {args.seasons}")
    print(f"prefix innings: {args.prefix_innings}")
    print(f"F5 totals rows in table: {total_f5_rows}")
    print(f"market coverage: {coverage}")
    print(f"final F5 actual coverage: {len(actual_totals)} of {len(market_lines)} games")
    print("selection: sim-vs-consensus open/current side/edge")
    print(f"execution: {args.execution} for the primary CLV summary")
    print("line-shopping: fixed selected side, consensus execution vs best book")
    print("CLV: execution lines vs consensus close when close rows exist")
    if not any(coverage["close"].values()):
        print("close coverage: none; CLV fields will be n/a")

    games, outcome_dirs = build_f5_clv_games(
        game_seasons=game_seasons,
        market_lines=market_lines,
        actual_totals=actual_totals,
        games_requested=args.games,
        sims=args.sims,
        seed=args.seed,
        prefix_innings=args.prefix_innings,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        pa_calibration_path=args.pa_calibration,
        pitch_type_model=args.pitch_type_model,
        pitch_type_model_weight=args.pitch_type_model_weight,
    )
    print(f"simulated games: {len(games)} outcome_dirs={outcome_dirs}")
    if not games:
        raise SystemExit("no simulatable final games with F5 totals lines")

    summaries: list[F5ClvSummary] = []
    bets_by_key: dict[tuple[str, float], list[F5ClvBet]] = {}
    line_shopping_summaries: list[F5LineShoppingSummary] = []
    line_shopping_bets_by_key: dict[tuple[str, float], list[F5LineShoppingBet]] = {}
    anchor_model_probability = (
        _market_anchor_probability(args.market_anchor_lambda)
        if args.market_anchor_lambda is not None
        else None
    )
    anchor_blend_summaries: list[F5ClvSummary] = []
    anchor_blend_bets_by_key: dict[tuple[str, float], list[F5ClvBet]] = {}
    anchor_blend_line_shopping_summaries: list[F5LineShoppingSummary] = []
    anchor_blend_line_shopping_bets_by_key: dict[
        tuple[str, float], list[F5LineShoppingBet]
    ] = {}
    if anchor_model_probability is not None:
        print(f"market-anchor blend diagnostic: lambda={args.market_anchor_lambda:g}")
    for staking in _staking_plans(args.staking):
        print(f"\n=== {staking.upper()} staking ===")
        for edge in args.edges:
            summary, bets = summarize_f5_clv(
                games,
                edge_threshold=edge,
                execution=args.execution,
                staking=staking,
                flat_stake=args.flat_stake,
                kelly_multiplier=args.kelly_multiplier,
                kelly_cap=args.kelly_cap,
            )
            summaries.append(summary)
            bets_by_key[(staking, edge)] = bets
            print(_summary_line(edge, summary))
            line_shop_summary, line_shop_bets = compare_f5_line_shopping(
                games,
                edge_threshold=edge,
                staking=staking,
                flat_stake=args.flat_stake,
                kelly_multiplier=args.kelly_multiplier,
                kelly_cap=args.kelly_cap,
            )
            line_shopping_summaries.append(line_shop_summary)
            line_shopping_bets_by_key[(staking, edge)] = line_shop_bets
            print(_line_shopping_summary_line(edge, line_shop_summary))
            if anchor_model_probability is not None:
                anchor_summary, anchor_bets = summarize_f5_clv(
                    games,
                    edge_threshold=edge,
                    execution=args.execution,
                    staking=staking,
                    flat_stake=args.flat_stake,
                    kelly_multiplier=args.kelly_multiplier,
                    kelly_cap=args.kelly_cap,
                    model_probability=anchor_model_probability,
                )
                anchor_blend_summaries.append(anchor_summary)
                anchor_blend_bets_by_key[(staking, edge)] = anchor_bets
                print(
                    f"anchor lambda={args.market_anchor_lambda:g} "
                    + _summary_line(edge, anchor_summary)
                )
                anchor_line_shop_summary, anchor_line_shop_bets = compare_f5_line_shopping(
                    games,
                    edge_threshold=edge,
                    staking=staking,
                    flat_stake=args.flat_stake,
                    kelly_multiplier=args.kelly_multiplier,
                    kelly_cap=args.kelly_cap,
                    model_probability=anchor_model_probability,
                )
                anchor_blend_line_shopping_summaries.append(anchor_line_shop_summary)
                anchor_blend_line_shopping_bets_by_key[
                    (staking, edge)
                ] = anchor_line_shop_bets
                print(
                    f"anchor lambda={args.market_anchor_lambda:g} "
                    + _line_shopping_summary_line(edge, anchor_line_shop_summary)
                )

    write_outputs(
        games=games,
        summaries=summaries,
        bets_by_key=bets_by_key,
        line_shopping_summaries=line_shopping_summaries,
        line_shopping_bets_by_key=line_shopping_bets_by_key,
        coverage=coverage,
        outcome_dirs=outcome_dirs,
        total_f5_rows=total_f5_rows,
        out_json=args.out_json,
        out_csv=args.out_csv,
        anchor_blend_lambda=args.market_anchor_lambda,
        anchor_blend_summaries=anchor_blend_summaries,
        anchor_blend_bets_by_key=anchor_blend_bets_by_key,
        anchor_blend_line_shopping_summaries=anchor_blend_line_shopping_summaries,
        anchor_blend_line_shopping_bets_by_key=anchor_blend_line_shopping_bets_by_key,
    )


if __name__ == "__main__":
    main()
