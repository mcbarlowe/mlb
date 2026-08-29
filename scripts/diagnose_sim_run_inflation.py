"""Diagnose the simulator's run-total inflation.

Partitions the +runs error between (a) the empirical base-out engine and
(b) the sim's PA-outcome distribution:

  1. Measures the sim's per-PA outcome distribution on a frequency-weighted
     sample of real matchups, both calibrated (as served) and raw
     (calibration off).
  2. Compares each to the actual league PA distribution.
  3. Feeds each distribution through the real base-out engine to attribute
     runs/team-game.

Run with the shared HTTP MLflow server:
    MLFLOW_TRACKING_URI=http://10.0.0.171:5001 \
        uv run python scripts/diagnose_sim_run_inflation.py --season 2024
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mlb.sim.base_out import AB_EVENT_TO_PA_OUTCOME, BaseOutEngine
from mlb.sim.game import Batter, Pitcher
from mlb.sim.matchup import MatchupProviderFactory
from mlb.sim.pa import simulate_plate_appearance
from mlb.sim.slate import build_day_ahead_simulator

PA_CLASSES = (
    "out", "strikeout", "walk", "hit_by_pitch",
    "single", "double", "triple", "home_run", "reached_on_error",
)


def league_pa_distribution(season: int) -> tuple[dict[str, float], float]:
    """League per-PA outcome frequencies and actual runs/team-game."""
    from mlb.database.postgres_handler import PostgresHandler

    handler = PostgresHandler()
    with handler.connection.cursor() as cur:
        cur.execute(
            """
            SELECT event_type, COUNT(*) FROM (
                SELECT game_pk, at_bat_index, MAX(event_type) AS event_type
                FROM mlb.pitches WHERE season = %s AND game_type = 'R'
                GROUP BY game_pk, at_bat_index
            ) t GROUP BY event_type
            """,
            (season,),
        )
        counts = Counter()
        for event_type, n in cur.fetchall():
            cls = AB_EVENT_TO_PA_OUTCOME.get(event_type)
            if cls is not None:
                counts[cls] += n
        cur.execute(
            """
            SELECT SUM(runs), COUNT(DISTINCT game_pk) FROM linescore
            WHERE game_pk IN (
                SELECT game_pk FROM mlb.pitches
                WHERE season = %s AND game_type = 'R'
            )
            """,
            (season,),
        )
        total_runs, n_games = cur.fetchone()
    total = sum(counts.values())
    dist = {cls: counts.get(cls, 0) / total for cls in PA_CLASSES}
    return dist, total_runs / (2 * n_games)


def sample_matchups(season: int, n: int, rng: random.Random) -> list[tuple]:
    """Frequency-weighted sample of (pitcher_id, batter_id, throw, bat)."""
    from mlb.database.postgres_handler import PostgresHandler

    handler = PostgresHandler()
    with handler.connection.cursor() as cur:
        cur.execute(
            """
            SELECT pitcher_id, batter_id, throw_side, bat_side, COUNT(*) AS pa
            FROM (
                SELECT game_pk, at_bat_index, MAX(pitcher_id) AS pitcher_id,
                       MAX(batter_id) AS batter_id, MAX(throw_side) AS throw_side,
                       MAX(bat_side) AS bat_side
                FROM mlb.pitches WHERE season = %s AND game_type = 'R'
                GROUP BY game_pk, at_bat_index
            ) t
            WHERE throw_side IS NOT NULL AND bat_side IS NOT NULL
            GROUP BY pitcher_id, batter_id, throw_side, bat_side
            """,
            (season,),
        )
        rows = cur.fetchall()
    population = [(int(p), int(b), t, s) for p, b, t, s, _ in rows]
    weights = [int(pa) for *_, pa in rows]
    return rng.choices(population, weights=weights, k=n)


def measure_sim_distribution(
    factory: MatchupProviderFactory,
    matchups: list[tuple],
    pas_per_matchup: int,
    rng: random.Random,
) -> dict[str, float]:
    counts: Counter = Counter()
    for pitcher_id, batter_id, throw, bat in matchups:
        pitcher = Pitcher(player_id=pitcher_id, throw_side=throw or "R")
        batter = Batter(player_id=batter_id, bat_side=bat or "R")
        for _ in range(pas_per_matchup):
            stretch = rng.random() < 0.43  # league on-base occupancy
            is_top = rng.random() < 0.5
            try:
                provider = factory(pitcher, batter, is_top, stretch)
                result = simulate_plate_appearance(provider, rng)
            except (ValueError, KeyError, RuntimeError):
                continue
            counts[result.outcome] += 1
    total = sum(counts.values())
    return {cls: counts.get(cls, 0) / total for cls in PA_CLASSES}


def engine_runs_per_team_game(
    engine: BaseOutEngine, dist: dict[str, float], rng: random.Random, n_games: int
) -> float:
    classes = list(dist.keys())
    weights = [dist[c] for c in classes]
    total = 0
    for _ in range(n_games):
        for _inning in range(9):
            outs = 0
            runners = 0
            while outs < 3:
                outcome = rng.choices(classes, weights=weights)[0]
                t = engine.sample(outcome, runners, outs)
                runners, outs = t.runners_after, t.outs_after
                total += t.runs
    return total / n_games


def _print_dist(label: str, dist: dict[str, float], league: dict[str, float]) -> None:
    print(f"\n{label:<28} (Δ vs league)")
    for cls in PA_CLASSES:
        d = dist.get(cls, 0.0)
        print(f"  {cls:<16} {d:.4f}  ({d - league[cls]:+.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--matchups", type=int, default=600)
    parser.add_argument("--pas", type=int, default=40)
    parser.add_argument("--engine-games", type=int, default=10000)
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--tracking-uri", type=str, default="http://10.0.0.171:5001")
    args = parser.parse_args()

    rng = random.Random(17)
    league, actual_rpg = league_pa_distribution(args.season)
    print(f"Actual runs/team-game {args.season}: {actual_rpg:.3f}")

    sim, _ = build_day_ahead_simulator(
        season=args.season, tracking_uri=args.tracking_uri
    )
    calibrated_factory = sim._factory
    raw_factory = MatchupProviderFactory(
        calibrated_factory._predictor,
        calibrated_factory._mix,
        season=args.season,
        seed=17,
        calibration=None,
    )
    engine = sim._engine

    matchups = sample_matchups(args.season, args.matchups, rng)
    print(f"Sampled {len(matchups):,} frequency-weighted matchups; "
          f"{args.pas} PAs each")

    cal_dist = measure_sim_distribution(calibrated_factory, matchups, args.pas, rng)
    _print_dist("LEAGUE", league, league)
    _print_dist("SIM calibrated (served)", cal_dist, league)
    if not args.skip_raw:
        raw_dist = measure_sim_distribution(raw_factory, matchups, args.pas, rng)
        _print_dist("SIM raw (no calibration)", raw_dist, league)

    print("\nRuns/team-game through the real base-out engine:")
    print(f"  league dist:     {engine_runs_per_team_game(engine, league, rng, args.engine_games):.3f}"
          f"   (actual {actual_rpg:.3f})")
    print(f"  sim calibrated:  {engine_runs_per_team_game(engine, cal_dist, rng, args.engine_games):.3f}")
    if not args.skip_raw:
        print(f"  sim raw:         {engine_runs_per_team_game(engine, raw_dist, rng, args.engine_games):.3f}")


if __name__ == "__main__":
    main()
