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
import math
import random
import sys
import time
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


def _is_final(feed: dict) -> bool:
    return (
        feed.get("gameData", {}).get("status", {}).get("abstractGameState")
        == "Final"
    )


def load_feed(game_pk: int, season: int | None) -> dict:
    """Archived feed for a game; refreshed from the API when the local copy
    is a stale pre-game stub (season backfills download future games too)."""
    seasons = [season] if season else [p.name for p in sorted(LIVEFEED_ROOT.iterdir())]
    local: dict | None = None
    for candidate in seasons:
        path = LIVEFEED_ROOT / str(candidate) / f"{game_pk}.json"
        if path.exists():
            local = json.loads(path.read_text())
            break
    if local is not None and _is_final(local):
        return local
    try:
        return _fetch_json(
            f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
        )
    except Exception:
        if local is not None:
            return local
        raise SystemExit(
            f"No archived feed for game {game_pk}; pass --season or backfill"
        )


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

    def brier_of(probs) -> float:
        return sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / n

    def log_loss_of(probs) -> float:
        eps = 1e-6
        total = 0.0
        for p, y in zip(probs, outcomes):
            p = min(max(p, eps), 1 - eps)
            total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        return total / n

    model_p = [r["p_home"] for r in rows]
    # League home-field advantage benchmark (long-run MLB home win rate).
    league_home_rate = 0.543
    league_p = [league_home_rate] * n
    coin_p = [0.5] * n
    picks = sum(1 for r, y in zip(rows, outcomes) if (r["p_home"] > 0.5) == bool(y)) / n
    mean_p_home = sum(model_p) / n
    metrics = {
        "win_brier": brier_of(model_p),
        "win_log_loss": log_loss_of(model_p),
        "win_brier_coin": brier_of(coin_p),
        "win_log_loss_coin": log_loss_of(coin_p),
        "win_brier_league_home": brier_of(league_p),
        "win_log_loss_league_home": log_loss_of(league_p),
        "win_brier_always_home": brier_of([1.0] * n),
        "pick_accuracy": picks,
        "pick_accuracy_always_home": home_rate,
        "mean_p_home": mean_p_home,
        "actual_home_rate": home_rate,
        "sim_mean_total_runs": sum(r["sim_total"] for r in rows) / n,
        "actual_mean_total_runs": sum(r["actual_total"] for r in rows) / n,
    }

    print(f"\nGames: {n}")
    print(f"Mean simulated total runs: {metrics['sim_mean_total_runs']:.2f}")
    print(f"Mean actual total runs:    {metrics['actual_mean_total_runs']:.2f}")
    print(f"Mean p(home): {mean_p_home:.3f}; actual home win rate: {home_rate:.3f}")
    print("Home-win Brier / log loss, lower is better:")
    print(f"  model:                    {metrics['win_brier']:.4f} / {metrics['win_log_loss']:.4f}")
    print(f"  coin flip (p=0.5):        {metrics['win_brier_coin']:.4f} / {metrics['win_log_loss_coin']:.4f}")
    print(f"  league home rate (p={league_home_rate}): {metrics['win_brier_league_home']:.4f} / {metrics['win_log_loss_league_home']:.4f}")
    print(f"  always-home hard pick:    {metrics['win_brier_always_home']:.4f} / -")
    print(f"Pick accuracy: model {picks:.1%} vs always-home {home_rate:.1%}")

    _log_validation_to_mlflow(args, metrics, n)


def _log_validation_to_mlflow(args: argparse.Namespace, metrics: dict, n: int) -> None:
    """Track game-level validation metrics in shared MLflow when configured."""
    import os

    tracking_uri = args.mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("MLFLOW_TRACKING_URI not set; validation metrics not tracked")
        return
    import mlflow

    from src.ml.mlflow_utils import configure_mlflow

    configure_mlflow(
        args.mlflow_experiment, tracking_uri, require_tracking_uri=True
    )
    pointer = Path("models/outcome/latest_run.txt").read_text().strip()
    with mlflow.start_run(run_name=f"sim-validation-{time.strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params(
            {
                "games": n,
                "sims": args.sims,
                "season": args.season,
                "seed": args.seed,
                "outcome_run": pointer,
            }
        )
        mlflow.log_metrics(metrics)
    print("Validation metrics logged to MLflow")


STATS_API = "https://statsapi.mlb.com/api/v1"


def _fetch_json(url: str, params: dict | None = None) -> dict:
    import requests

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _feed_for_game(game_pk: int, season: int) -> dict:
    path = LIVEFEED_ROOT / str(season) / f"{game_pk}.json"
    if path.exists():
        feed = json.loads(path.read_text())
        if _is_final(feed):
            return feed
    return _fetch_json(f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live")


def _pitch_hand(player_id: int) -> str:
    data = _fetch_json(f"{STATS_API}/people/{player_id}")
    people = data.get("people", [])
    return people[0].get("pitchHand", {}).get("code", "R") if people else "R"


def _projected_batters(team_id: int, date: str, season: int) -> list:
    """Batting order from the team's most recent completed game."""
    from datetime import date as date_cls
    from datetime import timedelta

    end = date_cls.fromisoformat(date)
    schedule = _fetch_json(
        f"{STATS_API}/schedule",
        {
            "sportId": 1,
            "teamId": team_id,
            "startDate": (end - timedelta(days=10)).isoformat(),
            "endDate": (end - timedelta(days=1)).isoformat(),
        },
    )
    finals: list[tuple[str, int, str]] = []
    for day in schedule.get("dates", []):
        for game in day.get("games", []):
            if game.get("status", {}).get("abstractGameState") != "Final":
                continue
            side = "home" if game["teams"]["home"]["team"]["id"] == team_id else "away"
            finals.append((game["gameDate"], game["gamePk"], side))
    for _, game_pk, side in sorted(finals, reverse=True):
        try:
            return lineup_from_feed(_feed_for_game(game_pk, season), side).batters
        except (ValueError, KeyError):
            continue
    raise ValueError(f"No recent lineup found for team {team_id}")


def run_slate(args: argparse.Namespace) -> None:
    from datetime import UTC, datetime

    from src.sim.game import BULLPEN_ARM, Lineup, Pitcher

    slate_date = args.date or datetime.now(UTC).astimezone().date().isoformat()
    season = int(slate_date[:4])
    schedule = _fetch_json(
        f"{STATS_API}/schedule",
        {"sportId": 1, "date": slate_date, "hydrate": "probablePitcher,team"},
    )
    games = [
        g
        for day in schedule.get("dates", [])
        for g in day.get("games", [])
        if g.get("status", {}).get("abstractGameState") == "Preview"
    ]
    if not games:
        raise SystemExit(f"No unstarted games on {slate_date}")
    print(f"{len(games)} unstarted games on {slate_date}\n")

    simulator = build_simulator(args, season)
    for game in games:
        teams = game["teams"]
        away_abbrev = teams["away"]["team"].get("abbreviation") or teams["away"]["team"]["name"]
        home_abbrev = teams["home"]["team"].get("abbreviation") or teams["home"]["team"]["name"]
        label = f"{away_abbrev} @ {home_abbrev}"
        try:
            lineups = {}
            starters = {}
            for side in ("away", "home"):
                team = teams[side]["team"]
                probable = teams[side].get("probablePitcher")
                if probable:
                    starter = Pitcher(probable["id"], _pitch_hand(probable["id"]))
                    starters[side] = probable.get("fullName", str(probable["id"]))
                else:
                    starter = BULLPEN_ARM
                    starters[side] = "TBD (league-average arm)"
                batters = _projected_batters(team["id"], slate_date, season)
                lineups[side] = Lineup(batters=batters, starter=starter)
        except (ValueError, KeyError) as exc:
            print(f"{label:12s} SKIPPED ({exc})")
            continue

        results = simulator.simulate_many(lineups["away"], lineups["home"], args.sims)
        stats = summarize(results)
        print(
            f"{label:12s} p(home)={stats['home_win_probability']:.2f}  "
            f"proj {stats['mean_away_runs']:.1f}-{stats['mean_home_runs']:.1f}  "
            f"{starters['away']} vs {starters['home']}"
        )

        if not args.no_card:
            from src.live.game_sim_card import (
                card_data_from_results,
                render_game_sim_card,
            )

            data = card_data_from_results(
                results,
                away_abbrev=away_abbrev,
                home_abbrev=home_abbrev,
                away_team_id=teams["away"]["team"].get("id"),
                home_team_id=teams["home"]["team"].get("id"),
                away_starter=starters["away"],
                home_starter=starters["home"],
                game_date=slate_date,
                venue=game.get("venue", {}).get("name"),
            )
            out = Path(f"output/sim_cards/slate_{slate_date}_{game['gamePk']}.jpg")
            out.parent.mkdir(parents=True, exist_ok=True)
            card_path = render_game_sim_card(data, out)
            print(f"             card: {card_path}")

            if args.post:
                from src.live.publisher import BlueskyPublisher, PredictionPost

                p_home = stats["home_win_probability"]
                favorite, p_fav = (
                    (home_abbrev, p_home)
                    if p_home >= 0.5
                    else (away_abbrev, 1 - p_home)
                )
                # Production caption: plain matchup text, no model internals.
                caption = (
                    f"{label}: {favorite} {p_fav:.0%} to win. "
                    f"Projected score {away_abbrev} {stats['mean_away_runs']:.1f} - "
                    f"{home_abbrev} {stats['mean_home_runs']:.1f}. "
                    f"Probables: {starters['away']} vs {starters['home']}."
                )
                post_id = BlueskyPublisher().publish(
                    PredictionPost(text=caption, image_path=card_path)
                )
                print(f"             posted: {post_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo game simulation.")
    parser.add_argument("--game-pk", type=int)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--sims", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--post", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--slate", action="store_true")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--no-card", action="store_true")
    parser.add_argument("--card-out", type=str, default=None)
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument(
        "--mlflow-experiment", type=str, default="mlb-model-training"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate:
        if args.season is None:
            raise SystemExit("--validate requires --season")
        run_validation(args)
    elif args.slate:
        run_slate(args)
    elif args.game_pk:
        run_single(args)
    else:
        raise SystemExit("Pass --game-pk, --slate, or --validate")


if __name__ == "__main__":
    main()
