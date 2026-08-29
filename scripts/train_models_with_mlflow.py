from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlflow

from scripts import run_full_training, train_pitch_type_location_model
from mlb.ml.mlflow_utils import (
    DEFAULT_MLFLOW_EXPERIMENT,
    build_metric_dict,
    build_param_dict,
    configure_mlflow,
    log_path_if_exists,
)
from mlb.ml.season_splits import (
    DEFAULT_TEST_SEASON,
    DEFAULT_VAL_SEASON,
    default_data_source_train_seasons,
    discover_available_seasons,
)

VALIDATION_SEASON = DEFAULT_VAL_SEASON
TEST_SEASON = DEFAULT_TEST_SEASON



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the documented pitch models with MLflow tracking.",
    )
    parser.add_argument("--data-path", type=str, default="postgres")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--mlflow-experiment", type=str, default=DEFAULT_MLFLOW_EXPERIMENT)
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()



def build_pitch_type_args(
    base_output_dir: Path,
    data_path: str,
    train_seasons: list[str],
    quick: bool,
    device: str,
    low_memory: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        data_path=data_path,
        train_seasons=["2021"] if quick else train_seasons,
        val_season=VALIDATION_SEASON,
        test_season=TEST_SEASON,
        exclude_2020=True,
        include_2020=False,
        model_type="lstm_attention",
        hidden_dim=128 if quick else 256,
        n_layers=2,
        dropout=0.3,
        n_components=3,
        embedding_dim=32,
        n_attention_heads=4 if quick else 8,
        n_attention_layers=1 if quick else 2,
        batch_size=64,
        n_epochs=3 if quick else 50,
        patience=2 if quick else 7,
        learning_rate=1e-3,
        type_weight=1.0,
        location_weight=0.5,
        use_class_weights=True,
        no_class_weights=False,
        class_weight_smoothing=0.5,
        output_dir=str(base_output_dir / "pitch_type"),
        plots=not quick,
        no_plots=quick,
        show_batch_progress=False,
        device=device,
        seed=42,
        quick=quick,
        player_dropout=0.0 if quick else 0.02,
        low_memory=low_memory,

    )



def build_location_args(
    base_output_dir: Path,
    data_path: str,
    train_seasons: list[str],
    quick: bool,
    device: str,
    low_memory: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        data_path=data_path,
        train_seasons=["2021"] if quick else train_seasons,
        val_season=VALIDATION_SEASON,
        test_season=TEST_SEASON,
        hidden_dims=[128, 64] if quick else [256, 128],
        n_components=3,
        dropout=0.2,
        learning_rate=1e-3,
        batch_size=128 if quick else 256,
        n_epochs=2 if quick else 100,
        seed=42,
        device=device,
        output_dir=str(base_output_dir / "pitch_type_location"),
        # Diagnostic-only baseline arm; measured twice (~4.4% conditioning
        # gain) and never served — enable deliberately when re-validating.
        compare_baseline=False,
        quick=quick,
        low_memory=low_memory,
    )



def new_run_dir(base_dir: Path, prefix: str, before: set[Path]) -> Path:
    after = set(base_dir.glob(f"{prefix}_*"))
    new_dirs = sorted(after - before)
    if not new_dirs:
        raise RuntimeError(f"No new run directory created under {base_dir} for prefix {prefix}")
    return new_dirs[-1]



def train_pitch_type_model(args: SimpleNamespace) -> dict:
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    before = set(output_root.glob("run_*"))
    results = run_full_training.run_training(args)
    results["output_dir"] = str(new_run_dir(output_root, "run", before))
    return results



def train_location_model(args: SimpleNamespace) -> dict:
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    before = set(output_root.glob("pitch_type_location_*"))
    _, test_metrics = train_pitch_type_location_model.run_training(args)
    output_dir = new_run_dir(output_root, "pitch_type_location", before)
    return {
        "output_dir": str(output_dir),
        "test_metrics": test_metrics,
    }



def main() -> None:
    args = parse_args()
    data_path = args.data_path
    output_dir = Path(args.output_dir)

    tracking_uri = configure_mlflow(
        args.mlflow_experiment,
        args.mlflow_tracking_uri,
        require_tracking_uri=True,
    )

    train_seasons = default_data_source_train_seasons(
        data_path,
        val_season=VALIDATION_SEASON,
        test_season=TEST_SEASON,
        exclude_2020=True,
    )

    if data_path == "postgres":
        available_seasons = set(discover_available_seasons(data_path))
        required_seasons = {*train_seasons, VALIDATION_SEASON, TEST_SEASON}
        missing_seasons = sorted(required_seasons - available_seasons)
        if missing_seasons:
            raise ValueError(
                f"PostgreSQL training source is missing required seasons: {missing_seasons}. "
                f"Available seasons: {sorted(available_seasons)}"
            )

    pitch_args = build_pitch_type_args(
        output_dir, data_path, train_seasons, args.quick, args.device, args.low_memory
    )
    location_args = build_location_args(
        output_dir, data_path, train_seasons, args.quick, args.device, args.low_memory
    )

    common_tags = {
        "data_path": data_path,
        "train_seasons": ",".join(location_args.train_seasons),
        "val_season": location_args.val_season,
        "test_season": location_args.test_season,
        "quick": str(args.quick),
        "low_memory": str(args.low_memory),
    }

    with mlflow.start_run(run_name=f"pitch-type-{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}"):
        mlflow.set_tags({**common_tags, "model_family": "pitch_type_lstm_attention"})
        mlflow.log_params(build_param_dict(vars(pitch_args), prefix="arg"))
        pitch_results = train_pitch_type_model(pitch_args)
        mlflow.log_dict(pitch_results, "results.json")
        mlflow.log_metrics(build_metric_dict(pitch_results.get("training", {}), prefix="training"))
        mlflow.log_metrics(build_metric_dict(pitch_results.get("test_results", {}), prefix="test"))
        log_path_if_exists(Path(pitch_results["output_dir"]), "artifacts")

    with mlflow.start_run(run_name=f"pitch-location-{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}"):
        mlflow.set_tags({**common_tags, "model_family": "pitch_type_conditioned_location"})
        mlflow.log_params(build_param_dict(vars(location_args), prefix="arg"))
        location_results = train_location_model(location_args)
        mlflow.log_dict(location_results["test_metrics"], "test_metrics.json")
        mlflow.log_metrics(build_metric_dict(location_results["test_metrics"].get("overall", {}), prefix="test"))
        baseline = location_results["test_metrics"].get("baseline_comparison")
        if isinstance(baseline, dict):
            mlflow.log_metrics(build_metric_dict(baseline, prefix="baseline"))
        log_path_if_exists(Path(location_results["output_dir"]), "artifacts")

    print("\nTraining runs finished")
    print(f"- MLflow experiment: {args.mlflow_experiment}")
    print(f"- Tracking URI: {tracking_uri}")
    print(f"- Pitch type output: {pitch_results['output_dir']}")
    print(f"- Location output: {location_results['output_dir']}")


if __name__ == "__main__":
    main()
