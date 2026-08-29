"""Register a paired Stage A/Stage B outcome release in shared MLflow.

The import path converts existing CatBoost files into native MLflow models,
creates immutable registered-model versions, and optionally advances both
``champion`` aliases after the paired release passes its promotion gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlflow

from mlb.ml.mlflow_utils import DEFAULT_MLFLOW_EXPERIMENT, configure_mlflow
from mlb.outcome.mlflow_registry import (
    log_outcome_release_models,
    resolve_sim_inputs_run_id,
)

DEFAULT_EXPERIMENT = DEFAULT_MLFLOW_EXPERIMENT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register outcome model artifacts in shared MLflow."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="auto",
        help="Outcome run directory (default: models/outcome/latest_run.txt).",
    )
    parser.add_argument(
        "--profiles-dir",
        type=str,
        default="models/outcome",
        help="Directory containing pitcher and batter profile stores.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=DEFAULT_EXPERIMENT,
    )
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument(
        "--sim-inputs-run-id",
        type=str,
        default="auto",
        help="Immutable sim-inputs run ID; default resolves the latest finished release.",
    )
    parser.add_argument(
        "--set-champion",
        action="store_true",
        help="Advance both champion aliases only when the paired release passes.",
    )
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


def source_provenance_tags(
    run_dir: Path,
    tracking_uri: str,
) -> dict[str, str]:
    manifest_path = run_dir.parent / "manifest.json"
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise TypeError(f"Expected a JSON object in {manifest_path}")

    source_run_ids: dict[str, str] = {}
    for stage in ("a", "b"):
        run_id = manifest.get(f"stage_{stage}_run_id")
        if not isinstance(run_id, str):
            raise TypeError(f"Cached outcome manifest is missing stage_{stage}_run_id")
        source_run_ids[stage] = run_id

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    stage_a_commit = client.get_run(source_run_ids["a"]).data.tags.get(
        "mlflow.source.git.commit"
    )
    stage_b_commit = client.get_run(source_run_ids["b"]).data.tags.get(
        "mlflow.source.git.commit"
    )
    tags = {
        "source_stage_a_run_id": source_run_ids["a"],
        "source_stage_b_run_id": source_run_ids["b"],
    }
    if isinstance(stage_a_commit, str) and stage_a_commit == stage_b_commit:
        tags["source_model_git_commit"] = stage_a_commit
    return tags


def main() -> None:
    args = parse_args()
    tracking_uri = configure_mlflow(
        args.mlflow_experiment,
        args.mlflow_tracking_uri,
        require_tracking_uri=True,
    )
    print(f"MLflow tracking URI: {tracking_uri}")

    run_dir = resolve_run_dir(args.run_dir)
    profiles_dir = Path(args.profiles_dir)
    for path in (
        run_dir / "stage_a.cbm",
        run_dir / "stage_a_features.json",
        run_dir / "stage_a_metrics.json",
        run_dir / "stage_b.cbm",
        run_dir / "stage_b_features.json",
        run_dir / "stage_b_metrics.json",
        profiles_dir / "pitcher_profiles.parquet",
        profiles_dir / "batter_priors.parquet",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    sim_inputs_run_id = resolve_sim_inputs_run_id(
        tracking_uri,
        args.sim_inputs_run_id,
    )
    with mlflow.start_run(
        run_name=f"outcome-model-release-{run_dir.name.removeprefix('run_')}"
    ):
        mlflow.set_tags(
            {
                "source": "production-import",
                "imported_from_path": str(run_dir),
                "profiles_path": str(profiles_dir),
                **source_provenance_tags(run_dir, tracking_uri),
            }
        )
        registration = log_outcome_release_models(
            run_dir=run_dir,
            profiles_dir=profiles_dir,
            sim_inputs_run_id=sim_inputs_run_id,
            set_champion=args.set_champion,
        )

    print(
        "Registered outcome release "
        f"{registration.release_id} from run {registration.run_id}"
    )
    print(
        f"Stage A: {registration.stage_a.registered_model_name} "
        f"v{registration.stage_a.version}"
    )
    print(
        f"Stage B: {registration.stage_b.registered_model_name} "
        f"v{registration.stage_b.version}"
    )
    if args.set_champion and not registration.promotion_gate_passed:
        raise SystemExit("Outcome release failed promotion gate; champions unchanged")


if __name__ == "__main__":
    main()
