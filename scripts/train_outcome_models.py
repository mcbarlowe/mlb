"""Train the pitch outcome models (Stage A and Stage B) with MLflow tracking.

Usage:
    uv run python scripts/train_outcome_models.py                # full training
    uv run python scripts/train_outcome_models.py --quick        # 1-season smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlflow

from src.ml.mlflow_utils import configure_mlflow
from src.outcome.dataset import (
    build_training_frame,
    load_pitches,
    stage_a_frame,
    stage_b_frame,
)
from src.outcome.models import (
    conditional_baseline_log_loss,
    evaluate_model,
    save_model,
    train_outcome_model,
)

DEFAULT_TRAIN_SEASONS = list(range(2015, 2024))
DEFAULT_VAL_SEASON = 2024
DEFAULT_TEST_SEASON = 2025


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train pitch outcome models (Stage A: result, Stage B: in-play event).",
    )
    parser.add_argument(
        "--train-seasons",
        nargs="+",
        type=int,
        default=DEFAULT_TRAIN_SEASONS,
        help="Seasons used for training (default: 2015-2023)",
    )
    parser.add_argument("--val-season", type=int, default=DEFAULT_VAL_SEASON)
    parser.add_argument("--test-season", type=int, default=DEFAULT_TEST_SEASON)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="models/outcome")
    parser.add_argument("--mlflow-experiment", type=str, default="mlb-model-training")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument(
        "--stage",
        choices=["a", "b", "both"],
        default="both",
        help="Which model(s) to train",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke mode: train on 2023 only with 200 iterations",
    )
    return parser.parse_args()


def run_stage(
    name: str,
    label_column: str,
    baseline_conditions: list[str],
    train,
    val,
    test,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict:
    print(f"\n=== Stage {name.upper()} ({label_column}) ===")
    print(f"train={train.height:,} val={val.height:,} test={test.height:,}")

    start = time.perf_counter()
    model = train_outcome_model(
        train,
        val,
        label_column,
        iterations=args.iterations,
        depth=args.depth,
    )
    train_seconds = time.perf_counter() - start

    metrics = {
        "train_seconds": train_seconds,
        "best_iteration": int(model.get_best_iteration() or 0),
        "val": evaluate_model(model, val, label_column),
        "test": evaluate_model(model, test, label_column),
        "baseline_val_log_loss": conditional_baseline_log_loss(
            train, val, label_column, baseline_conditions
        ),
        "baseline_test_log_loss": conditional_baseline_log_loss(
            train, test, label_column, baseline_conditions
        ),
    }
    save_model(model, output_dir, f"stage_{name}", metrics)

    print(f"val   log_loss={metrics['val']['log_loss']:.4f} "
          f"(baseline {metrics['baseline_val_log_loss']:.4f})")
    print(f"test  log_loss={metrics['test']['log_loss']:.4f} "
          f"(baseline {metrics['baseline_test_log_loss']:.4f})")

    mlflow.log_metrics(
        {
            f"stage_{name}_val_log_loss": metrics["val"]["log_loss"],
            f"stage_{name}_val_accuracy": metrics["val"]["accuracy"],
            f"stage_{name}_test_log_loss": metrics["test"]["log_loss"],
            f"stage_{name}_test_accuracy": metrics["test"]["accuracy"],
            f"stage_{name}_baseline_val_log_loss": metrics["baseline_val_log_loss"],
            f"stage_{name}_baseline_test_log_loss": metrics["baseline_test_log_loss"],
            f"stage_{name}_train_seconds": train_seconds,
        }
    )
    return metrics


def main() -> None:
    args = parse_args()
    if args.quick:
        args.train_seasons = [2023]
        args.iterations = 200

    output_dir = Path(args.output_dir) / time.strftime("run_%Y%m%d_%H%M%S")
    seasons = sorted(set(args.train_seasons + [args.val_season, args.test_season]))

    tracking_uri = configure_mlflow(
        args.mlflow_experiment,
        args.mlflow_tracking_uri,
        require_tracking_uri=True,
    )
    print(f"MLflow tracking URI: {tracking_uri}")

    print(f"Loading pitches for seasons {seasons}...")
    start = time.perf_counter()
    raw = load_pitches(seasons)
    print(f"Loaded {raw.height:,} rows in {time.perf_counter() - start:.1f}s")

    frame = build_training_frame(raw)
    train_mask = frame["season"].is_in(args.train_seasons)
    splits = {
        "train": frame.filter(train_mask),
        "val": frame.filter(frame["season"] == args.val_season),
        "test": frame.filter(frame["season"] == args.test_season),
    }

    with mlflow.start_run(run_name=output_dir.name):
        mlflow.log_params(
            {
                "train_seasons": ",".join(map(str, args.train_seasons)),
                "val_season": args.val_season,
                "test_season": args.test_season,
                "iterations": args.iterations,
                "depth": args.depth,
                "quick": args.quick,
            }
        )

        if args.stage in ("a", "both"):
            run_stage(
                "a",
                "label_result",
                ["balls_before", "strikes_before"],
                *(stage_a_frame(splits[key]) for key in ("train", "val", "test")),
                args=args,
                output_dir=output_dir,
            )
        if args.stage in ("b", "both"):
            run_stage(
                "b",
                "label_event",
                ["pitch_type", "throw_side", "bat_side"],
                *(stage_b_frame(splits[key]) for key in ("train", "val", "test")),
                args=args,
                output_dir=output_dir,
            )

    print(f"\nDone. Models under {output_dir}")


if __name__ == "__main__":
    main()
