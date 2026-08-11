"""Run the live next-pitch prediction pipeline.

Waits for game start using the MLB schedule, polls live feeds while
games are in progress, predicts the next pitch before it happens,
renders a pitch card, and posts it to the configured social platform
(or saves locally in dry-run mode).

Usage:
    # Monitor all of today's games (dry run, no posts)
    uv run python scripts/run_live_pipeline.py

    # Monitor a specific date and post to X
    uv run python scripts/run_live_pipeline.py --date 2026-08-09 --post --post-provider x

    # Follow a single game
    uv run python scripts/run_live_pipeline.py --game-pk 823490
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import argparse
import asyncio
import sys
from datetime import UTC, date, datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.live.pipeline import (
    LiveGamePredictionService,
    run_live_day,
    run_live_game,
    run_random_live_game,
)
from src.live.predictor import LiveNextPitchPredictor
from src.live.publisher import POST_PROVIDER_CHOICES, build_publisher
from src.ml.mlflow_utils import resolve_mlflow_tracking_uri
from src.ml.run_dirs import resolve_location_run_dir, resolve_pitch_type_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict the next pitch for live MLB games and publish cards.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to monitor (YYYY-MM-DD, default today)",
    )
    parser.add_argument(
        "--game-pk",
        type=int,
        default=None,
        help="Follow one specific game instead of the whole schedule",
    )
    parser.add_argument(
        "--random-game",
        action="store_true",
        help="Pick one game at random from the schedule and follow only it",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for --random-game selection",
    )
    parser.add_argument(
        "--pitch-type-model",
        type=str,
        default="auto",
        help=(
            "Directory of the trained pitch-type model (default 'auto': "
            "newest complete non-quick models/pitch_type/run_*)"
        ),
    )
    parser.add_argument(
        "--location-model",
        type=str,
        default="auto",
        help=(
            "Directory of the conditioned location model (default 'auto': "
            "newest complete models/pitch_type_location/pitch_type_location_*; "
            "'' disables)"
        ),
    )
    parser.add_argument(
        "--outcome-run-dir",
        type=str,
        default="auto",
        help=(
            "Directory of trained outcome models (Stage A/B). "
            "Default 'auto' uses models/outcome/latest_run.txt or the newest "
            "models/outcome/run_* directory when present; use 'none' to disable "
            "the outcome-odds strip."
        ),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument(
        "--lead-minutes",
        type=float,
        default=15.0,
        help="Start monitoring this many minutes before first pitch",
    )
    parser.add_argument("--output-dir", type=str, default="output/live_cards")
    parser.add_argument(
        "--post",
        action="store_true",
        help="Actually post to the selected provider (default: dry run that saves cards)",
    )
    parser.add_argument(
        "--post-provider",
        type=str,
        choices=POST_PROVIDER_CHOICES,
        default="bluesky",
        help="Posting backend to use when --post is set (default: bluesky)",
    )
    parser.add_argument(
        "--post-cadence",
        type=str,
        choices=["at_bat", "pitch", "random_pitch", "half_inning"],
        default="half_inning",
        help=(
            "Post once per half-inning (default), once per at-bat, on every "
            "pitch, or on one randomly chosen pitch per at-bat"
        ),
    )
    parser.add_argument(
        "--random-pitch-ceiling",
        type=int,
        default=4,
        help="Highest pitch number the random_pitch cadence can target",
    )
    parser.add_argument(
        "--max-posts-per-game",
        type=int,
        default=40,
        help="Hard cap on posts per game",
    )
    parser.add_argument(
        "--card-style",
        type=str,
        choices=["html", "matplotlib"],
        default="html",
        help=(
            "Card renderer: 'html' (Chromium, broadcast style, default) "
            "or 'matplotlib' (legacy)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    target_date = (
        date.fromisoformat(args.date)
        if args.date
        else datetime.now(tz=UTC).astimezone().date()
    )

    print("Loading models...")
    pitch_type_model_dir = resolve_pitch_type_run_dir(args.pitch_type_model)
    location_model_dir = (
        resolve_location_run_dir(args.location_model)
        if args.location_model
        else None
    )
    print(f"  Pitch type model: {pitch_type_model_dir}")
    print(f"  Location model: {location_model_dir or 'disabled'}")
    predictor = LiveNextPitchPredictor(
        pitch_type_model_dir=pitch_type_model_dir,
        location_model_dir=location_model_dir,
        device=args.device,
    )
    outcome_predictor = None
    if args.outcome_run_dir.lower() != "none":
        from src.outcome.inference import PitchOutcomePredictor
        from src.outcome.mlflow_artifacts import resolve_outcome_artifact_dirs

        resolved = resolve_outcome_artifact_dirs(
            args.outcome_run_dir,
            tracking_uri=resolve_mlflow_tracking_uri(),
        )
        if resolved is not None:
            outcome_run_dir, profiles_dir = resolved
            outcome_predictor = PitchOutcomePredictor(
                outcome_run_dir, profiles_dir=profiles_dir
            )
        else:
            print(
                "Outcome models not found locally or in shared MLflow; rendering without outcome-odds strip"
            )

    publisher = build_publisher(post=args.post, provider=args.post_provider)
    service = LiveGamePredictionService(
        predictor=predictor,
        publisher=publisher,
        output_dir=args.output_dir,
        post_cadence=args.post_cadence,
        max_posts_per_game=args.max_posts_per_game,
        random_pitch_ceiling=args.random_pitch_ceiling,
        seed=args.seed,
        card_style=args.card_style,
        outcome_predictor=outcome_predictor,
    )

    if args.game_pk is not None:
        result = asyncio.run(
            run_live_game(
                args.game_pk,
                service,
                poll_interval=args.poll_interval,
            )
        )
    elif args.random_game:
        result = asyncio.run(
            run_random_live_game(
                target_date,
                service,
                poll_interval=args.poll_interval,
                lead_minutes=args.lead_minutes,
                seed=args.seed,
            )
        )
    else:
        result = asyncio.run(
            run_live_day(
                target_date,
                service,
                poll_interval=args.poll_interval,
                lead_minutes=args.lead_minutes,
            )
        )

    print(f"\nLive pipeline finished: {result}")


if __name__ == "__main__":
    main()
