"""Resolve and cache production outcome model artifacts from shared MLflow.

The live pipeline still loads CatBoost models from local disk. This module
bridges shared MLflow metadata to that local-disk contract by resolving the
latest `production_model=true` Stage A / Stage B runs and downloading their
artifacts into a deterministic cache directory.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient

PRODUCTION_EXPERIMENT = "mlb-model-training"
CACHE_ROOT = Path("models/outcome/mlflow_cache")
REQUIRED_MODEL_FAMILIES = {
    "stage_a": "pitch_result_stage_a",
    "stage_b": "in_play_event_stage_b",
}


@dataclass(frozen=True)
class OutcomeProductionSelection:
    experiment_id: str
    experiment_name: str
    stage_a_run_id: str
    stage_b_run_id: str

    @property
    def cache_key(self) -> str:
        return f"{self.stage_a_run_id}_{self.stage_b_run_id}"


@dataclass(frozen=True)
class CachedOutcomeArtifacts:
    cache_dir: Path
    selection: OutcomeProductionSelection

    @property
    def run_dir(self) -> Path:
        return self.cache_dir / "run"

    @property
    def profiles_dir(self) -> Path:
        return self.cache_dir / "profiles"


def resolve_production_outcome_selection(
    tracking_uri: str,
    experiment_name: str = PRODUCTION_EXPERIMENT,
) -> OutcomeProductionSelection:
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise FileNotFoundError(
            f"MLflow experiment {experiment_name!r} not found at {tracking_uri}"
        )

    def latest_run(model_family: str) -> str:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=(
                "attributes.status = 'FINISHED' "
                f"and tags.production_model = 'true' "
                f"and tags.model_family = '{model_family}'"
            ),
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise FileNotFoundError(
                f"No production MLflow run found for model_family={model_family!r}"
            )
        return runs[0].info.run_id

    return OutcomeProductionSelection(
        experiment_id=experiment.experiment_id,
        experiment_name=experiment.name,
        stage_a_run_id=latest_run(REQUIRED_MODEL_FAMILIES["stage_a"]),
        stage_b_run_id=latest_run(REQUIRED_MODEL_FAMILIES["stage_b"]),
    )


def cache_production_outcome_artifacts(
    tracking_uri: str,
    *,
    cache_root: Path = CACHE_ROOT,
    refresh: bool = False,
) -> CachedOutcomeArtifacts:
    selection = resolve_production_outcome_selection(tracking_uri)
    cache_dir = cache_root / selection.cache_key
    run_dir = cache_dir / "run"
    profiles_dir = cache_dir / "profiles"
    manifest = cache_dir / "manifest.json"

    expected = [
        run_dir / "stage_a.cbm",
        run_dir / "stage_a_features.json",
        run_dir / "stage_a_metrics.json",
        run_dir / "stage_b.cbm",
        run_dir / "stage_b_features.json",
        run_dir / "stage_b_metrics.json",
        profiles_dir / "pitcher_profiles.parquet",
        profiles_dir / "batter_priors.parquet",
    ]
    if not refresh and manifest.exists() and all(path.exists() for path in expected):
        return CachedOutcomeArtifacts(cache_dir=cache_dir, selection=selection)

    cache_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    # Each imported production run contains only its own stage files plus a copy
    # of the shared profile stores. Download the directories and then flatten into
    # our local contract: one run dir with both stage files + a sibling profiles dir.
    download_artifacts(
        run_id=selection.stage_a_run_id,
        artifact_path="model",
        dst_path=str(cache_dir),
        tracking_uri=tracking_uri,
    )
    for name in ["stage_a.cbm", "stage_a_features.json", "stage_a_metrics.json"]:
        src = cache_dir / "model" / name
        src.replace(run_dir / name)

    download_artifacts(
        run_id=selection.stage_b_run_id,
        artifact_path="model",
        dst_path=str(cache_dir),
        tracking_uri=tracking_uri,
    )
    for name in ["stage_b.cbm", "stage_b_features.json", "stage_b_metrics.json"]:
        src = cache_dir / "model" / name
        src.replace(run_dir / name)

    download_artifacts(
        run_id=selection.stage_a_run_id,
        artifact_path="profiles",
        dst_path=str(cache_dir),
        tracking_uri=tracking_uri,
    )
    for name in ["pitcher_profiles.parquet", "batter_priors.parquet"]:
        src = cache_dir / "profiles" / name
        src.replace(profiles_dir / name)

    model_dir = cache_dir / "model"
    if model_dir.exists() and not any(model_dir.iterdir()):
        model_dir.rmdir()

    manifest.write_text(json.dumps(asdict(selection), indent=2, sort_keys=True))
    return CachedOutcomeArtifacts(cache_dir=cache_dir, selection=selection)


def resolve_outcome_artifact_dirs(
    run_dir_arg: str,
    *,
    tracking_uri: str | None = None,
    cache_root: Path = CACHE_ROOT,
) -> tuple[Path, Path] | None:
    """Resolve live outcome artifacts from MLflow or local disk.

    Policy:
    - ``none`` -> disabled
    - explicit path -> that run dir + sibling profile store directory
    - ``auto`` -> prefer shared MLflow production runs when a tracking URI is
      available; otherwise fall back to `models/outcome/latest_run.txt`, then the
      newest local `models/outcome/run_*` directory.
    """
    if run_dir_arg.lower() == "none":
        return None

    if run_dir_arg != "auto":
        run_dir = Path(run_dir_arg)
        profiles_dir = run_dir.parent
        return (run_dir, profiles_dir) if run_dir.exists() else None

    if tracking_uri:
        try:
            cached = cache_production_outcome_artifacts(
                tracking_uri, cache_root=cache_root
            )
            return cached.run_dir, cached.profiles_dir
        except Exception as exc:
            print(
                f"Shared MLflow outcome artifact resolution failed ({exc}); "
                "falling back to local models/outcome"
            )

    base = Path("models/outcome").resolve()
    pointer = base / "latest_run.txt"
    if pointer.exists():
        run_dir = base / pointer.read_text().strip()
        if run_dir.exists():
            return run_dir, base
    runs = sorted(base.glob("run_*"), reverse=True)
    if runs:
        return runs[0], base
    return None
