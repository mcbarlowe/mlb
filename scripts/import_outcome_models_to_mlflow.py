"""Import locally trained outcome model artifacts into the shared MLflow backend.

This is the production-import path for the current Stage A / Stage B CatBoost
artifacts. It exists because the full training run happened on a laptop before
we hardened the shared MLflow setup; importing here brings the run metadata and
artifacts into the shared Postgres-backed MLflow system used by the rest of the
repo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlflow

from src.ml.mlflow_utils import (
    build_metric_dict,
    build_param_dict,
    configure_mlflow,
    log_path_if_exists,
)

DEFAULT_EXPERIMENT = "mlb-model-training"
DEFAULT_ARTIFACT_ROOT = "file:///Users/matthewbarlowe/mlflow-artifacts/mlb-model-training"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import outcome model artifacts into shared MLflow.")
    parser.add_argument("--run-dir", type=str, default="auto", help="Outcome run directory (default: models/outcome/latest_run.txt)")
    parser.add_argument("--profiles-dir", type=str, default="models/outcome", help="Directory containing pitcher_profiles.parquet and batter_priors.parquet")
    parser.add_argument("--mlflow-experiment", type=str, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--mlflow-artifact-root", type=str, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--production-model", action="store_true", help="Tag imported runs as production_model=true")
    return parser.parse_args()


def resolve_run_dir(raw: str) -> Path:
    if raw != "auto":
        return Path(raw)
    pointer = Path("models/outcome/latest_run.txt")
    if pointer.exists():
        return Path("models/outcome") / pointer.read_text().strip()
    runs = sorted(Path("models/outcome").glob("run_*"), reverse=True)
    if not runs:
        raise FileNotFoundError("No models/outcome/run_* directory found")
    return runs[0]


def ensure_experiment(name: str, artifact_root: str) -> str:
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(name, artifact_location=artifact_root)


def import_stage(
    *,
    experiment_id: str,
    run_dir: Path,
    profiles_dir: Path,
    stage: str,
    model_family: str,
    production_model: bool,
) -> None:
    features = json.loads((run_dir / f"stage_{stage}_features.json").read_text())
    metrics = json.loads((run_dir / f"stage_{stage}_metrics.json").read_text())
    run_name = f"import-production-outcome-stage-{stage}-{run_dir.name.removeprefix('run_')}"
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        mlflow.set_tags(
            {
                "source": "production-import",
                "model_family": model_family,
                "imported_from_path": str(run_dir),
                "profiles_path": str(profiles_dir),
                **({"production_model": "true"} if production_model else {}),
            }
        )
        mlflow.log_params(
            build_param_dict(
                {
                    "stage": stage,
                    "run_dir": str(run_dir),
                    "feature_count": len(features["feature_columns"]),
                    "categorical_count": len(features["categorical_features"]),
                    "classes": features["classes"],
                }
            )
        )
        mlflow.log_metrics(build_metric_dict(metrics))
        log_path_if_exists(run_dir / f"stage_{stage}.cbm", artifact_path="model")
        log_path_if_exists(
            run_dir / f"stage_{stage}_features.json", artifact_path="model"
        )
        log_path_if_exists(
            run_dir / f"stage_{stage}_metrics.json", artifact_path="model"
        )
        log_path_if_exists(
            profiles_dir / "pitcher_profiles.parquet", artifact_path="profiles"
        )
        log_path_if_exists(
            profiles_dir / "batter_priors.parquet", artifact_path="profiles"
        )
        active = mlflow.active_run()
        assert active is not None
        print(f"Imported stage {stage} -> run {active.info.run_id}")


def main() -> None:
    args = parse_args()
    tracking_uri = configure_mlflow(
        args.mlflow_experiment,
        args.mlflow_tracking_uri,
        require_tracking_uri=True,
    )
    print(f"MLflow tracking URI: {tracking_uri}")
    experiment_id = ensure_experiment(args.mlflow_experiment, args.mlflow_artifact_root)
    print(f"Experiment id: {experiment_id}")

    run_dir = resolve_run_dir(args.run_dir)
    profiles_dir = Path(args.profiles_dir)
    if not (profiles_dir / "pitcher_profiles.parquet").exists() or not (profiles_dir / "batter_priors.parquet").exists():
        raise FileNotFoundError("Profile stores missing; run scripts/export_outcome_profiles.py first")

    import_stage(
        experiment_id=experiment_id,
        run_dir=run_dir,
        profiles_dir=profiles_dir,
        stage="a",
        model_family="pitch_result_stage_a",
        production_model=args.production_model,
    )
    import_stage(
        experiment_id=experiment_id,
        run_dir=run_dir,
        profiles_dir=profiles_dir,
        stage="b",
        model_family="in_play_event_stage_b",
        production_model=args.production_model,
    )


if __name__ == "__main__":
    main()
