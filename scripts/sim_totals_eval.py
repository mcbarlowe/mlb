"""Benchmark the Monte Carlo sim against the totals (O/U) market.

The sim's whole-game score distribution yields P(total > line) directly. We
compare that to the consensus de-vigged market P(over) against realized totals
(Brier + log loss + flat-bet ROI), on a sample of games.

    uv run python scripts/sim_totals_eval.py --season 2025 --games 500 --sims 500

The existing per-game stdout line is preserved for ``totals_blend_eval.py``::

    <game_pk> pt=<line> sim_over=<p> mkt_over=<p> actual=<runs>

By default, runs are inserted into ``mlb.totals_eval_runs`` and per-game rows
into ``mlb.totals_eval_games``. Use ``--out-json`` or ``--out-csv`` for
additional file artifacts; use ``--no-db-log`` for stdout/files only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import psycopg
from psycopg import sql

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.odds import no_vig_two_way
from src.database import PostgresConfig
from src.ml.mlflow_utils import DEFAULT_MLFLOW_TRACKING_URI
from src.sim.contact_environment import ContactEnvironment, parse_weather
from src.sim.db_games import GameDataStore
from src.sim.slate import build_day_ahead_simulator
from src.sim.totals_eval_store import insert_totals_eval_run

EPS = 1e-9
DEFAULT_EDGE_BUCKETS = (0.03, 0.05, 0.08)
WIN_PROFIT = 100.0 / 110.0


@dataclass(frozen=True)
class TotalsEvalRow:
    season: int
    game_pk: int
    point: float
    sim_prob_over: float
    sim_prob_under: float
    sim_prob_push: float
    market_prob_over: float
    sim_mean_total: float
    sim_total_stdev: float
    actual_total: float
    outcome: str

    @property
    def actual_over(self) -> int | None:
        if self.outcome == "over":
            return 1
        if self.outcome == "under":
            return 0
        return None


def market_totals(season: int) -> dict[int, tuple[float, float]]:
    """game_pk -> (consensus_point, consensus_devigged_p_over)."""
    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname,
        user=c.user,
        password=c.password,
        host=c.host,
        port=c.port,
        connect_timeout=15,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """SELECT o.game_pk, o.total_point, o.over_ml, o.under_ml
                    FROM {schema}.odds_totals o JOIN {schema}.games g USING(game_pk)
                    WHERE g.season::int=%s AND o.line_type='close'"""
                ).format(schema=sql.Identifier(c.schema)),
                (season,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    pts: dict[int, list[float]] = {}
    povs: dict[int, list[float]] = {}
    for game_pk, point, over_ml, under_ml in rows:
        p_over, _ = no_vig_two_way(float(over_ml), float(under_ml))
        pts.setdefault(int(game_pk), []).append(float(point))
        povs.setdefault(int(game_pk), []).append(p_over)
    return {
        pk: (statistics.median(pts[pk]), sum(povs[pk]) / len(povs[pk]))
        for pk in pts
    }


def game_environments(season: int) -> dict[int, tuple]:
    """game_pk -> (venue_id, GameWeather) for the contact environment."""
    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname,
        user=c.user,
        password=c.password,
        host=c.host,
        port=c.port,
        connect_timeout=15,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """SELECT game_pk, venue_id, weather_temp, weather_wind,
                           weather_condition
                    FROM {schema}.games WHERE season::int=%s"""
                ).format(schema=sql.Identifier(c.schema)),
                (season,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return {
        int(pk): (int(v) if v is not None else None, parse_weather(t, w, cond))
        for pk, v, t, w, cond in rows
    }


def brier(p: float, o: float) -> float:
    return (p - o) ** 2


def logloss(p: float, o: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return -(o * math.log(p) + (1 - o) * math.log(1 - p))


def mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def make_eval_row(
    *,
    season: int,
    game_pk: int,
    point: float,
    market_prob_over: float,
    simulated_totals: Sequence[int],
    actual_total: float,
) -> TotalsEvalRow:
    n = len(simulated_totals)
    if n == 0:
        raise ValueError("simulated_totals must not be empty")
    over = sum(1 for total in simulated_totals if total > point)
    push = sum(1 for total in simulated_totals if total == point)
    under = n - over - push
    sim_mean_total = sum(simulated_totals) / n
    sim_total_stdev = statistics.pstdev(simulated_totals) if n > 1 else 0.0
    if actual_total > point:
        outcome = "over"
    elif actual_total < point:
        outcome = "under"
    else:
        outcome = "push"
    return TotalsEvalRow(
        season=season,
        game_pk=game_pk,
        point=point,
        sim_prob_over=(over + 0.5 * push) / n,
        sim_prob_under=(under + 0.5 * push) / n,
        sim_prob_push=push / n,
        market_prob_over=market_prob_over,
        sim_mean_total=sim_mean_total,
        sim_total_stdev=sim_total_stdev,
        actual_total=actual_total,
        outcome=outcome,
    )


def scored_rows(rows: Iterable[TotalsEvalRow]) -> list[TotalsEvalRow]:
    return [row for row in rows if row.actual_over is not None]


def bootstrap_brier_gap(rows: Sequence[TotalsEvalRow]) -> tuple[float, float, float] | None:
    diffs: list[float] = []
    for row in rows:
        actual_over = row.actual_over
        if actual_over is None:
            continue
        diffs.append(
            brier(row.sim_prob_over, actual_over)
            - brier(row.market_prob_over, actual_over)
        )
    if not diffs:
        return None
    mean_diff = sum(diffs) / len(diffs)
    rng = random.Random(0)
    boots = sorted(
        sum(diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))) / len(diffs)
        for _ in range(4000)
    )
    return mean_diff, boots[100], boots[3899]


def calibration_buckets(
    rows: Sequence[TotalsEvalRow],
    *,
    model: str,
    n_bins: int = 5,
) -> list[dict[str, float | int | None]]:
    scored = scored_rows(rows)
    buckets: list[dict[str, float | int | None]] = []
    for idx in range(n_bins):
        lo = idx / n_bins
        hi = (idx + 1) / n_bins
        if idx == n_bins - 1:
            bucket_rows = [
                row for row in scored if lo <= _model_probability(row, model) <= hi
            ]
        else:
            bucket_rows = [
                row for row in scored if lo <= _model_probability(row, model) < hi
            ]
        actuals = [row.actual_over for row in bucket_rows if row.actual_over is not None]
        probs = [_model_probability(row, model) for row in bucket_rows]
        buckets.append(
            {
                "lo": lo,
                "hi": hi,
                "n": len(bucket_rows),
                "mean_pred": mean_or_none(probs),
                "actual_over_rate": mean_or_none([float(value) for value in actuals]),
            }
        )
    return buckets


def _model_probability(row: TotalsEvalRow, model: str) -> float:
    if model == "sim":
        return row.sim_prob_over
    if model == "market":
        return row.market_prob_over
    raise ValueError(f"unknown model {model!r}")


def roi_by_edge(
    rows: Sequence[TotalsEvalRow],
    edges: Sequence[float],
) -> list[dict[str, float | int]]:
    scored = scored_rows(rows)
    out: list[dict[str, float | int]] = []
    for edge in edges:
        bets = wins = 0
        for row in scored:
            actual_over = row.actual_over
            if actual_over is None:
                continue
            if row.sim_prob_over - row.market_prob_over > edge:
                bets += 1
                wins += actual_over
            elif row.market_prob_over - row.sim_prob_over > edge:
                bets += 1
                wins += 1 - actual_over
        losses = bets - wins
        profit = wins * WIN_PROFIT - losses
        out.append(
            {
                "edge": edge,
                "bets": bets,
                "wins": wins,
                "roi": profit / bets if bets else 0.0,
            }
        )
    return out


def summarize_rows(
    rows: Sequence[TotalsEvalRow],
    *,
    run_id: str,
    season: int,
    games_requested: int,
    sims: int,
    seed: int,
    edge: float,
    edge_buckets: Sequence[float],
    pa_calibration_path: str | None,
    contact_environment_enabled: bool,
    mlflow_tracking_uri: str,
    outcome_run_dir: str,
) -> dict[str, Any]:
    scored = scored_rows(rows)
    actuals = [float(row.actual_over) for row in scored if row.actual_over is not None]
    sim_probs = [row.sim_prob_over for row in scored]
    market_probs = [row.market_prob_over for row in scored]
    gap = bootstrap_brier_gap(scored)
    roi = roi_by_edge(scored, edge_buckets)
    return {
        "run_id": run_id,
        "season": season,
        "games_requested": games_requested,
        "games_evaluated": len(rows),
        "non_push_games": len(scored),
        "push_games": len(rows) - len(scored),
        "sims_per_game": sims,
        "seed": seed,
        "edge_threshold": edge,
        "edge_buckets": list(edge_buckets),
        "pa_calibration_path": pa_calibration_path,
        "contact_environment": contact_environment_enabled,
        "mlflow_tracking_uri": mlflow_tracking_uri,
        "outcome_run_dir": outcome_run_dir,
        "metrics": {
            "sim": {
                "brier": mean_or_none(
                    [brier(prob, actual) for prob, actual in zip(sim_probs, actuals)]
                ),
                "log_loss": mean_or_none(
                    [logloss(prob, actual) for prob, actual in zip(sim_probs, actuals)]
                ),
            },
            "market": {
                "brier": mean_or_none(
                    [brier(prob, actual) for prob, actual in zip(market_probs, actuals)]
                ),
                "log_loss": mean_or_none(
                    [logloss(prob, actual) for prob, actual in zip(market_probs, actuals)]
                ),
            },
            "brier_gap_sim_minus_market": None if gap is None else gap[0],
            "brier_gap_ci95": None if gap is None else {"lo": gap[1], "hi": gap[2]},
        },
        "totals": {
            "mean_market_total": mean_or_none([row.point for row in rows]),
            "mean_sim_total": mean_or_none([row.sim_mean_total for row in rows]),
            "mean_actual_total": mean_or_none([row.actual_total for row in rows]),
            "mean_sim_minus_market_total": mean_or_none(
                [row.sim_mean_total - row.point for row in rows]
            ),
            "mean_sim_minus_actual_total": mean_or_none(
                [row.sim_mean_total - row.actual_total for row in rows]
            ),
            "mean_abs_sim_minus_actual_total": mean_or_none(
                [abs(row.sim_mean_total - row.actual_total) for row in rows]
            ),
            "mean_sim_total_stdev": mean_or_none([row.sim_total_stdev for row in rows]),
        },
        "calibration": {
            "sim": calibration_buckets(scored, model="sim"),
            "market": calibration_buckets(scored, model="market"),
        },
        "roi_by_edge": roi,
    }


def selected_side(row: TotalsEvalRow, edge: float) -> str | None:
    if row.sim_prob_over - row.market_prob_over > edge:
        return "over"
    if row.market_prob_over - row.sim_prob_over > edge:
        return "under"
    return None


def row_output(
    row: TotalsEvalRow,
    edge: float,
    *,
    run_id: str | None = None,
) -> dict[str, float | int | str | None]:
    actual_over = row.actual_over
    side = selected_side(row, edge)
    if side is None:
        bet_result = None
    elif row.outcome == "push":
        bet_result = "push"
    elif side == row.outcome:
        bet_result = "win"
    else:
        bet_result = "loss"
    output: dict[str, float | int | str | None] = {
        **asdict(row),
        "actual_over": actual_over,
        "sim_brier": None if actual_over is None else brier(row.sim_prob_over, actual_over),
        "market_brier": None
        if actual_over is None
        else brier(row.market_prob_over, actual_over),
        "sim_log_loss": None
        if actual_over is None
        else logloss(row.sim_prob_over, actual_over),
        "market_log_loss": None
        if actual_over is None
        else logloss(row.market_prob_over, actual_over),
        "sim_edge_over": row.sim_prob_over - row.market_prob_over,
        "sim_edge_under": row.market_prob_over - row.sim_prob_over,
        "bet_side_at_threshold": side,
        "bet_result_at_threshold": bet_result,
    }
    if run_id is not None:
        output = {"run_id": run_id, **output}
    return output


def write_outputs(
    *,
    rows: Sequence[TotalsEvalRow],
    summary: dict[str, Any],
    edge: float,
    out_json: Path | None,
    out_csv: Path | None,
) -> None:
    run_id = str(summary["run_id"])
    row_dicts = [row_output(row, edge, run_id=run_id) for row in rows]
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps({"summary": summary, "rows": row_dicts}, indent=2, sort_keys=True)
        )
        print(f"wrote JSON totals eval to {out_json}")
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(row_dicts[0]) if row_dicts else list(row_output(_empty_row(), edge))
        with out_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(row_dicts)
        print(f"wrote CSV totals eval rows to {out_csv}")


def _empty_row() -> TotalsEvalRow:
    return TotalsEvalRow(
        season=0,
        game_pk=0,
        point=0.0,
        sim_prob_over=0.0,
        sim_prob_under=0.0,
        sim_prob_push=0.0,
        market_prob_over=0.0,
        sim_mean_total=0.0,
        sim_total_stdev=0.0,
        actual_total=0.0,
        outcome="push",
    )


def parse_edges(value: str) -> tuple[float, ...]:
    try:
        edges = tuple(float(part) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--edge-buckets must be comma-separated numbers"
        ) from exc
    if not edges:
        raise argparse.ArgumentTypeError("--edge-buckets must not be empty")
    if any(edge < 0.0 for edge in edges):
        raise argparse.ArgumentTypeError("--edge-buckets must be non-negative")
    return edges


def _edge_buckets_with_active(edges: Sequence[float], edge: float) -> tuple[float, ...]:
    return tuple(dict.fromkeys((*edges, edge)))


def default_run_id(season: int) -> str:
    created = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"totals_eval_{season}_{created}_{uuid4().hex[:8]}"


def print_summary(summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    sim = metrics["sim"]
    market = metrics["market"]
    gap = metrics["brier_gap_sim_minus_market"]
    ci = metrics["brier_gap_ci95"]
    print(
        f"\nSeason {summary['season']}: {summary['games_evaluated']} games, "
        f"{summary['sims_per_game']} sims each "
        f"({summary['non_push_games']} non-push, {summary['push_games']} pushes)"
    )
    print(f"run_id: {summary['run_id']}")
    print(f"{'':8}{'Brier':>10}{'log loss':>10}")
    print(f"{'sim':8}{_fmt_optional(sim['brier']):>10}{_fmt_optional(sim['log_loss']):>10}")
    print(
        f"{'market':8}{_fmt_optional(market['brier']):>10}"
        f"{_fmt_optional(market['log_loss']):>10}"
    )
    if gap is None or ci is None:
        print("Brier gap (sim - market): n/a")
    else:
        print(
            "Brier gap (sim - market): "
            f"{gap:+.4f}  95% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]  "
            "(neg = sim better)"
        )
    active_edge = summary["edge_threshold"]
    active_roi = next(
        item for item in summary["roi_by_edge"] if item["edge"] == active_edge
    )
    if active_roi["bets"]:
        print(
            f"flat-bet @ edge>{active_edge}: {active_roi['bets']} bets, "
            f"{active_roi['wins']:.1f} wins "
            f"({active_roi['wins'] / active_roi['bets']:.1%}), "
            f"ROI {active_roi['roi']:+.1%}"
        )
    else:
        print("no bets")
    totals = summary["totals"]
    print(
        "totals diagnostics: "
        f"mean sim {totals['mean_sim_total']:.2f}, "
        f"mean market {totals['mean_market_total']:.2f}, "
        f"mean actual {totals['mean_actual_total']:.2f}, "
        f"sim-actual bias {totals['mean_sim_minus_actual_total']:+.2f}, "
        f"mean sim stdev {totals['mean_sim_total_stdev']:.2f}"
    )


def _fmt_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--sims", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--edge", type=float, default=0.03, help="flat-bet edge threshold")
    ap.add_argument(
        "--edge-buckets",
        type=parse_edges,
        default=DEFAULT_EDGE_BUCKETS,
        help="comma-separated edge thresholds for structured ROI reporting",
    )
    ap.add_argument(
        "--pa-calibration",
        default=None,
        help="off-season PA calibration path (OOS totals test)",
    )
    ap.add_argument("--mlflow-tracking-uri", default=DEFAULT_MLFLOW_TRACKING_URI)
    ap.add_argument("--run-id", default=None, help="DB run identifier; generated by default")
    ap.add_argument("--no-db-log", action="store_true", help="skip totals eval DB insert")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    market = market_totals(args.season)
    store = GameDataStore.load(args.season)
    simulator, outcome_run_dir = build_day_ahead_simulator(
        season=args.season,
        seed=args.seed,
        tracking_uri=args.mlflow_tracking_uri,
        pa_calibration_path=args.pa_calibration,
    )
    contact_env = ContactEnvironment.load(args.season)
    envs = game_environments(args.season) if contact_env else {}
    print(f"contact environment: {'ON' if contact_env else 'OFF (no park factors)'}")

    candidates = [pk for pk in store.final_game_pks(args.seed, 10_000) if pk in market]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    rows: list[TotalsEvalRow] = []
    for pk in candidates:
        if len(rows) >= args.games:
            break
        point, mkt_over = market[pk]
        try:
            away = store.lineup(pk, "away", individual_bullpen=True)
            home = store.lineup(pk, "home", individual_bullpen=True)
        except (ValueError, KeyError):
            continue
        environment = None
        if contact_env:
            venue_id, weather = envs.get(pk, (None, None))
            environment = contact_env.multipliers(venue_id, weather)
        results = simulator.simulate_many(
            away,
            home,
            args.sims,
            environment=environment,
        )
        totals = [result.away_runs + result.home_runs for result in results]
        away_runs, home_runs = store.final(pk)
        actual = away_runs + home_runs
        row = make_eval_row(
            season=args.season,
            game_pk=pk,
            point=point,
            market_prob_over=mkt_over,
            simulated_totals=totals,
            actual_total=actual,
        )
        rows.append(row)
        print(
            f"{pk} pt={point} sim_over={row.sim_prob_over:.2f} "
            f"mkt_over={mkt_over:.2f} actual={actual}",
            flush=True,
        )

    if not rows:
        raise SystemExit("no games")
    edge_buckets = _edge_buckets_with_active(args.edge_buckets, args.edge)
    run_id = args.run_id or default_run_id(args.season)
    summary = summarize_rows(
        rows,
        run_id=run_id,
        season=args.season,
        games_requested=args.games,
        sims=args.sims,
        seed=args.seed,
        edge=args.edge,
        edge_buckets=edge_buckets,
        pa_calibration_path=args.pa_calibration,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        outcome_run_dir=str(outcome_run_dir),
        contact_environment_enabled=bool(contact_env),
    )
    print_summary(summary)
    write_outputs(
        rows=rows,
        summary=summary,
        edge=args.edge,
        out_json=args.out_json,
        out_csv=args.out_csv,
    )
    if args.no_db_log:
        print("totals eval DB log skipped by --no-db-log")
        return
    db_rows = [row_output(row, args.edge, run_id=run_id) for row in rows]
    run_inserted, game_inserted = insert_totals_eval_run(run=summary, rows=db_rows)
    print(
        "inserted totals eval DB rows: "
        f"runs={run_inserted} games={game_inserted} run_id={run_id}"
    )


if __name__ == "__main__":
    main()
