"""Fit the PA-outcome calibration so the sim's PA mix matches league rates.

The base-out engine is faithful when fed the league PA-outcome distribution
(see scripts/diagnose_sim_run_inflation.py), so pinning the sim's emergent
9-class PA-outcome mix to league fixes the run environment. This computes each
matchup's closed-form PA-outcome distribution (from the per-pitch-calibrated
providers), aggregates a frequency-weighted league mix per side, and fits
multipliers by iterative proportional fitting toward the actual league mix.

Writes models/sim/pa_outcome_calibration.json. Run with the shared server:
    MLFLOW_TRACKING_URI=http://10.0.0.171:5001 \
        uv run python scripts/fit_pa_outcome_calibration.py --season 2024
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mlb.sim.base_out import AB_EVENT_TO_PA_OUTCOME, BaseOutEngine
from mlb.sim.calibration import PAOutcomeCalibration, apply_multipliers
from mlb.sim.count_machine import PA_OUTCOMES
from mlb.sim.game import Batter, Pitcher
from mlb.sim.matchup import MatchupProviderFactory
from mlb.sim.pa import pa_outcome_distribution
from mlb.sim.slate import build_day_ahead_simulator

SIDES = {True: "top", False: "bottom"}


def league_pa_by_side(
    seasons: Sequence[int],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """League PA-outcome distribution split by batting side, plus runs/team-game."""
    from mlb.database.postgres_handler import PostgresHandler

    handler = PostgresHandler()
    dists: dict[str, dict[str, float]] = {}
    with handler.connection.cursor() as cur:
        for is_top, side in SIDES.items():
            half = "top" if is_top else "bottom"
            cur.execute(
                """
                SELECT event_type, COUNT(*) FROM (
                    SELECT game_pk, at_bat_index, MAX(event_type) AS event_type
                    FROM mlb.pitches
                    WHERE season = ANY(%s) AND game_type = 'R' AND half_inning = %s
                    GROUP BY game_pk, at_bat_index
                ) t GROUP BY event_type
                """,
                (list(seasons), half),
            )
            counts: Counter = Counter()
            for event_type, n in cur.fetchall():
                cls = AB_EVENT_TO_PA_OUTCOME.get(event_type)
                if cls is not None:
                    counts[cls] += n
            total = sum(counts.values())
            dists[side] = {cls: counts.get(cls, 0) / total for cls in PA_OUTCOMES}
        cur.execute(
            """
            SELECT SUM(runs), COUNT(DISTINCT game_pk) FROM linescore
            WHERE game_pk IN (
                SELECT game_pk FROM mlb.pitches WHERE season = ANY(%s) AND game_type = 'R'
            )
            """,
            (list(seasons),),
        )
        total_runs, n_games = cur.fetchone()
    runs = {"per_team": total_runs / (2 * n_games)}
    return dists, runs


def sample_matchups(seasons: Sequence[int], n: int, rng: random.Random) -> list[tuple]:
    from mlb.database.postgres_handler import PostgresHandler

    handler = PostgresHandler()
    with handler.connection.cursor() as cur:
        cur.execute(
            """
            SELECT pitcher_id, batter_id, throw_side, bat_side, half_inning, COUNT(*)
            FROM (
                SELECT game_pk, at_bat_index, MAX(pitcher_id) AS pitcher_id,
                       MAX(batter_id) AS batter_id, MAX(throw_side) AS throw_side,
                       MAX(bat_side) AS bat_side, MAX(half_inning) AS half_inning
                FROM mlb.pitches WHERE season = ANY(%s) AND game_type = 'R'
                GROUP BY game_pk, at_bat_index
            ) t
            WHERE throw_side IS NOT NULL AND bat_side IS NOT NULL
            GROUP BY pitcher_id, batter_id, throw_side, bat_side, half_inning
            """,
            (list(seasons),),
        )
        rows = cur.fetchall()
    population = [
        (int(p), int(b), t, s, (h == "top")) for p, b, t, s, h, _ in rows
    ]
    weights = [int(c) for *_, c in rows]
    return rng.choices(population, weights=weights, k=n)


def base_distributions(
    factory: MatchupProviderFactory, matchups: list[tuple], rng: random.Random
) -> dict[str, list[dict[str, float]]]:
    """Per-side list of each matchup's closed-form PA-outcome distribution."""
    by_side: dict[str, list[dict[str, float]]] = {"top": [], "bottom": []}
    for pitcher_id, batter_id, throw, bat, is_top in matchups:
        stretch = rng.random() < 0.43
        provider = factory(
            Pitcher(player_id=pitcher_id, throw_side=throw or "R"),
            Batter(player_id=batter_id, bat_side=bat or "R"),
            is_top,
            stretch,
        )
        dist = pa_outcome_distribution(provider._result, provider._event)
        by_side[SIDES[is_top]].append(dist)
    return by_side


def _aggregate(dists: list[dict[str, float]], mult: dict[str, float]) -> dict[str, float]:
    agg: dict[str, float] = dict.fromkeys(PA_OUTCOMES, 0.0)
    for d in dists:
        adjusted = apply_multipliers(d, mult) if mult else d
        for cls in PA_OUTCOMES:
            agg[cls] += adjusted[cls]
    total = sum(agg.values())
    return {cls: agg[cls] / total for cls in PA_OUTCOMES}


def fit_side(
    base: list[dict[str, float]], league: dict[str, float], iterations: int
) -> dict[str, float]:
    mult: dict[str, float] = dict.fromkeys(PA_OUTCOMES, 1.0)
    for _ in range(iterations):
        agg = _aggregate(base, mult)
        for cls in PA_OUTCOMES:
            if agg[cls] > 0 and league[cls] > 0:
                mult[cls] *= league[cls] / agg[cls]
    return mult


def engine_runs(engine: BaseOutEngine, dist: dict[str, float], rng, n_games: int) -> float:
    classes = list(dist.keys())
    weights = [dist[c] for c in classes]
    total = 0
    for _ in range(n_games):
        for _inning in range(9):
            outs = runners = 0
            while outs < 3:
                t = engine.sample(rng.choices(classes, weights=weights)[0], runners, outs)
                runners, outs = t.runners_after, t.outs_after
                total += t.runs
    return total / n_games


def match_runs_scalar(
    base: list[dict[str, float]],
    mult: dict[str, float],
    engine: BaseOutEngine,
    target: float,
    n_games: int = 6000,
) -> dict[str, float]:
    """Boost the 'out' multiplier so calibrated engine runs match the actual
    league runs/team-game. The base-out engine over-produces runs on the pure
    league PA mix, so matching runs (not just the mix) removes that level bias.
    """
    lo, hi = 0.5, 4.0
    result = dict(mult)
    for _ in range(14):
        s = (lo + hi) / 2.0
        trial = dict(mult)
        trial["out"] = mult["out"] * s
        r = engine_runs(engine, _aggregate(base, trial), random.Random(999), n_games)
        result = trial
        if abs(r - target) < 0.02:
            break
        if r > target:  # too many runs -> need more outs
            lo = s
        else:
            hi = s
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--seasons", default=None,
                        help="comma prior seasons (trailing window); overrides --season")
    parser.add_argument("--matchups", type=int, default=1500)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--engine-games", type=int, default=20000)
    parser.add_argument("--tracking-uri", type=str, default="http://10.0.0.171:5001")
    parser.add_argument("--out", default="models/sim/pa_outcome_calibration.json",
                        help="output calibration path")
    parser.add_argument("--match-runs", action="store_true",
                        help="also match actual runs/team-game (fix base-out level bias)")
    args = parser.parse_args()
    seasons = ([int(s) for s in args.seasons.split(",")]
               if args.seasons else [args.season])
    model_season = max(seasons)

    started = time.time()
    rng = random.Random(23)
    league, runs = league_pa_by_side(seasons)
    print(f"Actual runs/team-game {seasons}: {runs['per_team']:.3f}")

    sim, _ = build_day_ahead_simulator(
        season=model_season, tracking_uri=args.tracking_uri
    )
    # Fit against per-pitch-calibrated providers only (no existing PA layer).
    factory = MatchupProviderFactory(
        sim._factory._predictor,
        sim._factory._mix,
        season=model_season,
        seed=23,
        calibration=sim._factory._calibration,
        pa_outcome_calibration=None,
    )
    engine = sim._engine

    matchups = sample_matchups(seasons, args.matchups, rng)
    print(f"Sampled {len(matchups):,} matchups; computing closed-form PA distributions")
    base = base_distributions(factory, matchups, rng)

    multipliers: dict[str, dict[str, float]] = {}
    for side in ("top", "bottom"):
        multipliers[side] = fit_side(base[side], league[side], args.iterations)
        if args.match_runs:
            multipliers[side] = match_runs_scalar(
                base[side], multipliers[side], engine, runs["per_team"]
            )

    calibration = PAOutcomeCalibration(multipliers=multipliers)
    calibration.save(
        path=Path(args.out),
        meta={
            "seasons": seasons,
            "matchups": len(matchups),
            "iterations": args.iterations,
            "match_runs": args.match_runs,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    print("\nFitted multipliers:")
    for side in ("top", "bottom"):
        print(f"  {side}: " + ", ".join(
            f"{c}={multipliers[side][c]:.3f}" for c in PA_OUTCOMES
        ))

    print("\nValidation — runs/team-game through the real base-out engine:")
    for side in ("top", "bottom"):
        raw_agg = _aggregate(base[side], {})
        cal_agg = _aggregate(base[side], multipliers[side])
        print(f"  [{side}] league={league[side]['walk']:.4f}w "
              f"sim_raw_walk={raw_agg['walk']:.4f} sim_cal_walk={cal_agg['walk']:.4f}")
        r_raw = engine_runs(engine, raw_agg, rng, args.engine_games)
        r_cal = engine_runs(engine, cal_agg, rng, args.engine_games)
        r_league = engine_runs(engine, league[side], rng, args.engine_games)
        print(f"        runs/team-game  raw={r_raw:.3f}  calibrated={r_cal:.3f}  "
              f"league={r_league:.3f}  (actual {runs['per_team']:.3f})")
    print(f"\nSaved {args.out} in {time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
