"""Simulate MLB games with the trained model chain and report win probability.

Single game (lineups pulled from the archived live feed):

    uv run python scripts/simulate_game.py --game-pk 777284 --sims 2000

Aggregate calibration over a sample of a season's games:

    uv run python scripts/simulate_game.py --validate --season 2025 --games 20

Uses shared MLflow production outcome artifacts when available (or the latest
local outcome run), plus the pitch mix profiles (``scripts/export_pitch_mix.py``)
and base-out tables (``scripts/build_base_out_tables.py``).
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

from src.sim.game import GameSimulator, summarize
from src.sim.lineups import actual_final, describe_game, lineup_from_feed
from src.sim.slate import (
    build_day_ahead_simulator,
    fetch_slate_games,
    render_prediction_card,
    simulate_slate_game,
)

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


def build_simulator(args: argparse.Namespace, season: int) -> tuple[GameSimulator, Path]:
    simulator, run_dir = build_day_ahead_simulator(
        season=season,
        seed=args.seed,
        outcome_run_dir=args.outcome_run_dir,
        tracking_uri=args.mlflow_tracking_uri,
    )
    print(f"Loaded outcome models from {run_dir}")
    return simulator, run_dir


def run_single(args: argparse.Namespace) -> None:
    from src.sim.calibration import load_win_calibration

    feed = load_feed(args.game_pk, args.season)
    season = int(feed["gameData"]["game"]["season"])
    away = lineup_from_feed(feed, "away")
    home = lineup_from_feed(feed, "home")
    simulator, _ = build_simulator(args, season)
    print(f"Simulating {describe_game(feed)} x{args.sims}...")
    results = simulator.simulate_many(away, home, args.sims)
    stats = summarize(results)
    calibration = load_win_calibration()
    raw_p = stats["home_win_probability"]
    if calibration is not None:
        stats["home_win_probability"] = calibration.apply(raw_p)

    away_actual, home_actual = actual_final(feed)
    print(f"\nHome win probability: {stats['home_win_probability']:.1%}"
          + (f" (raw {raw_p:.1%})" if calibration is not None else ""))
    print(
        f"Mean score: away {stats['mean_away_runs']:.2f} - home {stats['mean_home_runs']:.2f}"
        f" (total {stats['mean_total_runs']:.2f})"
    )
    print(f"Mean innings: {stats['mean_innings']:.2f}; tie rate {stats['tie_rate']:.2%}")
    print(f"Actual final: away {away_actual} - home {home_actual}")

    if not args.no_card:
        card_path = render_card(args, feed, results, stats["home_win_probability"])
        print(f"Card: {card_path}")


def render_card(
    args: argparse.Namespace, feed: dict, results, home_win_probability: float
) -> Path:
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
        home_win_probability=home_win_probability,
    )
    out_path = Path(args.card_out) if args.card_out else Path(
        f"output/sim_cards/sim_{args.game_pk}.jpg"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return render_game_sim_card(data, out_path)


def collect_validation_rows(args: argparse.Namespace, simulator) -> list[dict]:
    """Simulate a sample of archived finals; returns per-game rows with raw p."""
    season_dir = LIVEFEED_ROOT / str(args.season)
    files = sorted(season_dir.glob("*.json"))
    rng = random.Random(args.seed)
    rng.shuffle(files)

    rows: list[dict] = []
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
    return rows


def run_fit_win_calibration(args: argparse.Namespace) -> None:
    """Fit Platt scaling of p(home) on simulated val-season games."""
    from src.sim.calibration import fit_win_calibration

    if args.season == 2025:
        raise SystemExit(
            "Never fit win calibration on the 2025 test season; use --season 2024"
        )
    simulator, run_dir = build_simulator(args, args.season)
    rows = collect_validation_rows(args, simulator)
    if len(rows) < 30:
        raise SystemExit(f"Only {len(rows)} usable games; need at least 30")
    probabilities = [r["p_home"] for r in rows]
    outcomes = [1.0 if r["home_won"] else 0.0 for r in rows]
    calibration = fit_win_calibration(probabilities, outcomes)
    calibration.save(
        meta={
            "fit_season": args.season,
            "games": len(rows),
            "sims": args.sims,
            "outcome_run": run_dir.name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    n = len(rows)
    raw_brier = sum((p - y) ** 2 for p, y in zip(probabilities, outcomes)) / n
    cal_brier = sum(
        (calibration.apply(p) - y) ** 2 for p, y in zip(probabilities, outcomes)
    ) / n
    print(
        f"\nFitted on {n} games: intercept={calibration.intercept:.3f} "
        f"slope={calibration.slope:.3f}"
    )
    print(f"Fit-sample Brier raw {raw_brier:.4f} -> calibrated {cal_brier:.4f}")


def run_validation(args: argparse.Namespace) -> None:
    from src.sim.calibration import load_win_calibration

    simulator, run_dir = build_simulator(args, args.season)
    rows = collect_validation_rows(args, simulator)

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
    calibration = load_win_calibration()
    if calibration is not None:
        calibrated_p = [calibration.apply(p) for p in model_p]
        metrics["win_brier_calibrated"] = brier_of(calibrated_p)
        metrics["win_log_loss_calibrated"] = log_loss_of(calibrated_p)
        metrics["mean_p_home_calibrated"] = sum(calibrated_p) / n
        metrics["pick_accuracy_calibrated"] = (
            sum(1 for p, y in zip(calibrated_p, outcomes) if (p > 0.5) == bool(y)) / n
        )

    print(f"\nGames: {n}")
    print(f"Mean simulated total runs: {metrics['sim_mean_total_runs']:.2f}")
    print(f"Mean actual total runs:    {metrics['actual_mean_total_runs']:.2f}")
    print(f"Mean p(home): {mean_p_home:.3f}; actual home win rate: {home_rate:.3f}")
    print("Home-win Brier / log loss, lower is better:")
    print(f"  model (raw):              {metrics['win_brier']:.4f} / {metrics['win_log_loss']:.4f}")
    if calibration is not None:
        print(
            f"  model (calibrated):       {metrics['win_brier_calibrated']:.4f}"
            f" / {metrics['win_log_loss_calibrated']:.4f}"
        )
    print(f"  coin flip (p=0.5):        {metrics['win_brier_coin']:.4f} / {metrics['win_log_loss_coin']:.4f}")
    print(f"  league home rate (p={league_home_rate}): {metrics['win_brier_league_home']:.4f} / {metrics['win_log_loss_league_home']:.4f}")
    print(f"  always-home hard pick:    {metrics['win_brier_always_home']:.4f} / -")
    print(f"Pick accuracy: model {picks:.1%} vs always-home {home_rate:.1%}")

    _log_validation_to_mlflow(args, metrics, n, run_dir.name)


def _log_validation_to_mlflow(
    args: argparse.Namespace,
    metrics: dict,
    n: int,
    outcome_run: str,
) -> None:
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
    with mlflow.start_run(run_name=f"sim-validation-{time.strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params(
            {
                "games": n,
                "sims": args.sims,
                "season": args.season,
                "seed": args.seed,
                "outcome_run": outcome_run,
            }
        )
        mlflow.log_metrics(metrics)
    print("Validation metrics logged to MLflow")


def _fetch_json(url: str, params: dict | None = None) -> dict:
    import requests

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def run_slate(args: argparse.Namespace) -> None:
    from datetime import UTC, datetime

    slate_date = args.date or datetime.now(UTC).astimezone().date().isoformat()
    season = int(slate_date[:4])
    games = fetch_slate_games(datetime.fromisoformat(slate_date).date(), abstract_states={"Preview"})
    if not games:
        raise SystemExit(f"No unstarted games on {slate_date}")
    print(f"{len(games)} unstarted games on {slate_date}\n")

    simulator, _ = build_simulator(args, season)
    for game in games:
        try:
            prediction = simulate_slate_game(
                game,
                simulator,
                season=season,
                n_sims=args.sims,
            )
        except (ValueError, KeyError) as exc:
            print(f"{game.label:12s} SKIPPED ({exc})")
            continue

        stats = prediction.stats
        print(
            f"{prediction.game.label:12s} p(home)={stats['home_win_probability']:.2f}  "
            f"proj {stats['mean_away_runs']:.1f}-{stats['mean_home_runs']:.1f}  "
            f"{prediction.away_starter} vs {prediction.home_starter}"
        )

        if not args.no_card:
            out = Path(f"output/sim_cards/slate_{slate_date}_{prediction.game.game_pk}.jpg")
            card_path = render_prediction_card(prediction, out)
            print(f"             card: {card_path}")

            if args.post:
                from src.live.publisher import PredictionPost, build_publisher

                p_home = stats["home_win_probability"]
                favorite, p_fav = (
                    (prediction.game.home_abbrev, p_home)
                    if p_home >= 0.5
                    else (prediction.game.away_abbrev, 1 - p_home)
                )
                caption = (
                    f"{prediction.game.label}: {favorite} {p_fav:.0%} to win. "
                    f"Projected score {prediction.game.away_abbrev} {stats['mean_away_runs']:.1f} - "
                    f"{prediction.game.home_abbrev} {stats['mean_home_runs']:.1f}. "
                    f"Probables: {prediction.away_starter} vs {prediction.home_starter}."
                )
                post_id = build_publisher(
                    post=True,
                    provider=args.post_provider,
                ).publish(PredictionPost(text=caption, image_path=card_path))
                print(f"             posted: {post_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo game simulation.")
    parser.add_argument("--game-pk", type=int)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--sims", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--post", action="store_true")
    parser.add_argument(
        "--post-provider",
        type=str,
        choices=("bluesky", "x", "both"),
        default="bluesky",
        help="Posting backend to use when --post is set (default: bluesky)",
    )
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
    parser.add_argument(
        "--outcome-run-dir",
        type=str,
        default="auto",
        help=(
            "Outcome model run directory. Default 'auto' prefers shared MLflow "
            "production runs when MLFLOW_TRACKING_URI is set, then falls back to "
            "models/outcome/latest_run.txt or the newest local run_* directory."
        ),
    )
    parser.add_argument("--fit-win-calibration", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fit_win_calibration:
        if args.season is None:
            raise SystemExit("--fit-win-calibration requires --season (use 2024)")
        run_fit_win_calibration(args)
    elif args.validate:
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
