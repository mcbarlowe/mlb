"""Simulate MLB games with the trained model chain and report win probability.

Single game (lineups pulled from the archived live feed):

    uv run python scripts/simulate_game.py --game-pk 777284 --sims 2000

Aggregate calibration over a sample of a season's games:

    uv run python scripts/simulate_game.py --validate --season 2025 --games 20

Uses the latest local outcome run (``models/outcome/latest_run.txt``), the
pitch mix profiles (``scripts/export_pitch_mix.py``), and the base-out
tables (``scripts/build_base_out_tables.py``).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.outcome.inference import PitchOutcomePredictor
from src.sim.base_out import BaseOutEngine
from src.sim.game import GameSimulator, summarize
from src.sim.lineups import actual_final, describe_game, lineup_from_feed
from src.sim.matchup import MatchupProviderFactory
from src.sim.pitch_mix import PitchMixProfiles

LIVEFEED_ROOT = Path("data/raw/livefeeds")


def load_feed(game_pk: int, season: int | None) -> dict:
    seasons = [season] if season else [p.name for p in sorted(LIVEFEED_ROOT.iterdir())]
    for candidate in seasons:
        path = LIVEFEED_ROOT / str(candidate) / f"{game_pk}.json"
        if path.exists():
            return json.loads(path.read_text())
    raise SystemExit(f"No archived feed for game {game_pk}; pass --season or backfill")


def build_simulator(args: argparse.Namespace, season: int) -> GameSimulator:
    pointer = Path("models/outcome/latest_run.txt").read_text().strip()
    predictor = PitchOutcomePredictor(Path("models/outcome") / pointer)
    mix = PitchMixProfiles.load(seed=args.seed)
    factory = MatchupProviderFactory(predictor, mix, season=season, seed=args.seed)
    engine = BaseOutEngine.load(seed=args.seed)
    return GameSimulator(factory, engine, rng=random.Random(args.seed))


def run_single(args: argparse.Namespace) -> None:
    feed = load_feed(args.game_pk, args.season)
    season = int(feed["gameData"]["game"]["season"])
    away = lineup_from_feed(feed, "away")
    home = lineup_from_feed(feed, "home")
    simulator = build_simulator(args, season)

    print(f"Simulating {describe_game(feed)} x{args.sims}...")
    results = simulator.simulate_many(away, home, args.sims)
    stats = summarize(results)

    away_actual, home_actual = actual_final(feed)
    print(f"\nHome win probability: {stats['home_win_probability']:.1%}")
    print(
        f"Mean score: away {stats['mean_away_runs']:.2f} - home {stats['mean_home_runs']:.2f}"
        f" (total {stats['mean_total_runs']:.2f})"
    )
    print(f"Mean innings: {stats['mean_innings']:.2f}; tie rate {stats['tie_rate']:.2%}")
    print(f"Actual final: away {away_actual} - home {home_actual}")


def run_validation(args: argparse.Namespace) -> None:
    season_dir = LIVEFEED_ROOT / str(args.season)
    files = sorted(season_dir.glob("*.json"))
    rng = random.Random(args.seed)
    rng.shuffle(files)

    simulator = build_simulator(args, args.season)
    rows = []
    for path in files:
        if len(rows) >= args.games:
            break
        feed = json.loads(path.read_text())
        game = feed.get("gameData", {}).get("game", {})
        status = feed.get("gameData", {}).get("status", {}).get("abstractGameState")
        if game.get("type") != "R" or status != "Final":
            continue
        try:
            away = lineup_from_feed(feed, "away")
            home = lineup_from_feed(feed, "home")
        except (ValueError, KeyError):
            continue
        results = simulator.simulate_many(away, home, args.sims)
        stats = summarize(results)
        away_actual, home_actual = actual_final(feed)
        rows.append(
            {
                "game": describe_game(feed),
                "p_home": stats["home_win_probability"],
                "sim_total": stats["mean_total_runs"],
                "actual_total": away_actual + home_actual,
                "home_won": home_actual > away_actual,
            }
        )
        print(
            f"{rows[-1]['game']:12s} p(home)={stats['home_win_probability']:.2f}"
            f" sim total={stats['mean_total_runs']:.1f}"
            f" actual {away_actual}-{home_actual}"
        )

    n = len(rows)
    if n == 0:
        raise SystemExit("No usable games found")
    brier = sum((r["p_home"] - (1.0 if r["home_won"] else 0.0)) ** 2 for r in rows) / n
    mean_p_home = sum(r["p_home"] for r in rows) / n
    print(f"\nGames: {n}")
    print(f"Mean simulated total runs: {sum(r['sim_total'] for r in rows) / n:.2f}")
    print(f"Mean actual total runs:    {sum(r['actual_total'] for r in rows) / n:.2f}")
    print(f"Mean p(home): {mean_p_home:.3f}; actual home win rate: "
          f"{sum(1 for r in rows if r['home_won']) / n:.3f}")
    print(f"Brier score (home win): {brier:.4f} (0.25 = coin flip)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo game simulation.")
    parser.add_argument("--game-pk", type=int)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--sims", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--games", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate:
        if args.season is None:
            raise SystemExit("--validate requires --season")
        run_validation(args)
    elif args.game_pk:
        run_single(args)
    else:
        raise SystemExit("Pass --game-pk or --validate")


if __name__ == "__main__":
    main()
