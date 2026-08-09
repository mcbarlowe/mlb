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

    if not args.no_card:
        card_path = render_card(args, feed, results)
        print(f"Card: {card_path}")


def render_card(args: argparse.Namespace, feed: dict, results) -> Path:
    from src.live.game_sim_card import card_data_from_results, render_game_sim_card

    game_data = feed["gameData"]
    teams = game_data["teams"]
    players = game_data["players"]
    box = feed["liveData"]["boxscore"]["teams"]

    def starter_name(side: str) -> str:
        pid = box[side]["pitchers"][0]
        return players[f"ID{pid}"]["fullName"]

    data = card_data_from_results(
        results,
        away_abbrev=teams["away"]["abbreviation"],
        home_abbrev=teams["home"]["abbreviation"],
        away_team_id=teams["away"].get("id"),
        home_team_id=teams["home"].get("id"),
        away_starter=starter_name("away"),
        home_starter=starter_name("home"),
        game_date=game_data.get("datetime", {}).get("officialDate", ""),
        venue=game_data.get("venue", {}).get("name"),
    )
    out_path = Path(args.card_out) if args.card_out else Path(
        f"output/sim_cards/sim_{args.game_pk}.jpg"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return render_game_sim_card(data, out_path)


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
    outcomes = [1.0 if r["home_won"] else 0.0 for r in rows]
    home_rate = sum(outcomes) / n
    brier = sum((r["p_home"] - y) ** 2 for r, y in zip(rows, outcomes)) / n
    # League home-field advantage benchmark (long-run MLB home win rate).
    league_home_rate = 0.543
    brier_league = sum((league_home_rate - y) ** 2 for y in outcomes) / n
    brier_always_home = sum((1.0 - y) ** 2 for y in outcomes) / n
    picks = sum(1 for r, y in zip(rows, outcomes) if (r["p_home"] > 0.5) == bool(y)) / n
    mean_p_home = sum(r["p_home"] for r in rows) / n
    print(f"\nGames: {n}")
    print(f"Mean simulated total runs: {sum(r['sim_total'] for r in rows) / n:.2f}")
    print(f"Mean actual total runs:    {sum(r['actual_total'] for r in rows) / n:.2f}")
    print(f"Mean p(home): {mean_p_home:.3f}; actual home win rate: {home_rate:.3f}")
    print("Brier score (home win), lower is better:")
    print(f"  model:                    {brier:.4f}")
    print("  coin flip (p=0.5):        0.2500")
    print(f"  league home rate (p={league_home_rate}): {brier_league:.4f}")
    print(f"  always-home hard pick:    {brier_always_home:.4f}")
    print(f"Pick accuracy: model {picks:.1%} vs always-home {home_rate:.1%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo game simulation.")
    parser.add_argument("--game-pk", type=int)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--sims", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--no-card", action="store_true")
    parser.add_argument("--card-out", type=str, default=None)
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
