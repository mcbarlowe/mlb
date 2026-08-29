"""Resolve and cache the paired champion outcome release from MLflow.

Production resolves immutable registered-model versions through their
``champion`` aliases. Both versions must carry the same release and simulator
input IDs; a partial promotion therefore fails closed instead of mixing stages.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory
from typing import cast

from mlflow.artifacts import download_artifacts
from mlflow.models import Model
from mlflow.tracking import MlflowClient

from mlb.outcome.mlflow_registry import (
    OUTCOME_CHAMPION_ALIAS,
    OUTCOME_CONTRACT_VERSION,
    OUTCOME_MODEL_COLLECTION,
    OUTCOME_RELEASE_TAG,
    OUTCOME_STAGE_SPECS,
    SIM_INPUTS_RUN_TAG,
)

CACHE_ROOT = Path("models/outcome/mlflow_cache")


@dataclass(frozen=True)
class OutcomeProductionSelection:
    release_id: str
    sim_inputs_run_id: str
    stage_a_run_id: str
    stage_a_version: str
    stage_b_run_id: str
    stage_b_version: str

    @property
    def cache_key(self) -> str:
        return f"{self.release_id[:16]}_a{self.stage_a_version}_b{self.stage_b_version}"


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
) -> OutcomeProductionSelection:
    client = MlflowClient(tracking_uri=tracking_uri)
    versions = {
        stage: client.get_model_version_by_alias(
            spec.registered_model_name,
            OUTCOME_CHAMPION_ALIAS,
        )
        for stage, spec in OUTCOME_STAGE_SPECS.items()
    }

    for stage, version in versions.items():
        spec = OUTCOME_STAGE_SPECS[stage]
        expected_tags = {
            "contract_version": str(OUTCOME_CONTRACT_VERSION),
            "model_collection": OUTCOME_MODEL_COLLECTION,
            "model_family": spec.model_family,
            "outcome_stage": stage,
            "promotion_gate": "passed",
        }
        if version.status != "READY":
            raise RuntimeError(
                f"{spec.registered_model_name} v{version.version} is not ready"
            )
        for key, expected in expected_tags.items():
            if version.tags.get(key) != expected:
                raise ValueError(
                    f"{spec.registered_model_name} v{version.version} has "
                    f"incompatible {key}={version.tags.get(key)!r}"
                )
        if not version.run_id:
            raise ValueError(
                f"{spec.registered_model_name} v{version.version} has no source run"
            )

    release_ids = {
        version.tags.get(OUTCOME_RELEASE_TAG) for version in versions.values()
    }
    sim_inputs_run_ids = {
        version.tags.get(SIM_INPUTS_RUN_TAG) for version in versions.values()
    }
    run_ids = {version.run_id for version in versions.values()}
    if None in release_ids or len(release_ids) != 1:
        raise ValueError("Outcome champion aliases do not select one paired release")
    if None in sim_inputs_run_ids or len(sim_inputs_run_ids) != 1:
        raise ValueError("Outcome champion aliases do not pin one sim-inputs run")
    if len(run_ids) != 1:
        raise ValueError("Outcome champion aliases do not share one source run")

    stage_a = versions["a"]
    stage_b = versions["b"]
    return OutcomeProductionSelection(
        release_id=cast(str, next(iter(release_ids))),
        sim_inputs_run_id=cast(str, next(iter(sim_inputs_run_ids))),
        stage_a_run_id=cast(str, stage_a.run_id),
        stage_a_version=str(stage_a.version),
        stage_b_run_id=cast(str, stage_b.run_id),
        stage_b_version=str(stage_b.version),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _materialize_registered_stage(
    *,
    tracking_uri: str,
    cache_dir: Path,
    run_dir: Path,
    profiles_dir: Path,
    stage: str,
    version: str,
) -> None:
    spec = OUTCOME_STAGE_SPECS[stage]
    with TemporaryDirectory(prefix=f".stage-{stage}-", dir=cache_dir) as temp_dir:
        downloaded = Path(
            download_artifacts(
                artifact_uri=(f"models:/{spec.registered_model_name}/{version}"),
                dst_path=temp_dir,
                tracking_uri=tracking_uri,
            )
        )
        metadata = Model.load(downloaded)
        catboost_flavor = metadata.flavors.get("catboost")
        if not isinstance(catboost_flavor, dict):
            raise TypeError(
                f"{spec.registered_model_name} v{version} is not a CatBoost model"
            )
        model_data = catboost_flavor.get("data")
        if not isinstance(model_data, str):
            raise TypeError(
                f"{spec.registered_model_name} v{version} has no model payload"
            )
        extra_files = downloaded / "extra_files"
        stage_name = f"stage_{stage}"
        copy2(downloaded / model_data, run_dir / f"{stage_name}.cbm")
        for suffix in ("features.json", "metrics.json"):
            copy2(
                extra_files / f"{stage_name}_{suffix}",
                run_dir / f"{stage_name}_{suffix}",
            )

        for profile_name in ("pitcher_profiles.parquet", "batter_priors.parquet"):
            source_profile = extra_files / profile_name
            destination_profile = profiles_dir / profile_name
            if stage == "a":
                copy2(source_profile, destination_profile)
            elif _sha256(source_profile) != _sha256(destination_profile):
                raise ValueError(
                    f"Outcome release has inconsistent {profile_name} artifacts"
                )


def cache_production_outcome_artifacts(
    tracking_uri: str,
    *,
    cache_root: Path = CACHE_ROOT,
    refresh: bool = False,
    selection: OutcomeProductionSelection | None = None,
) -> CachedOutcomeArtifacts:
    selection = selection or resolve_production_outcome_selection(tracking_uri)
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
    _materialize_registered_stage(
        tracking_uri=tracking_uri,
        cache_dir=cache_dir,
        run_dir=run_dir,
        profiles_dir=profiles_dir,
        stage="a",
        version=selection.stage_a_version,
    )
    _materialize_registered_stage(
        tracking_uri=tracking_uri,
        cache_dir=cache_dir,
        run_dir=run_dir,
        profiles_dir=profiles_dir,
        stage="b",
        version=selection.stage_b_version,
    )

    manifest.write_text(json.dumps(asdict(selection), indent=2, sort_keys=True))
    return CachedOutcomeArtifacts(cache_dir=cache_dir, selection=selection)


def resolve_outcome_artifact_dirs(
    run_dir_arg: str,
    *,
    tracking_uri: str | None = None,
    cache_root: Path = CACHE_ROOT,
    selection: OutcomeProductionSelection | None = None,
) -> tuple[Path, Path] | None:
    """Resolve live outcome artifacts from MLflow or local disk.

    ``none`` disables outcome models. An explicit path uses that local run.
    ``auto`` uses the paired registry champions when a tracking URI is present;
    without one, it falls back to the local pointer and then newest local run.
    """
    if run_dir_arg.lower() == "none":
        return None

    if run_dir_arg != "auto":
        run_dir = Path(run_dir_arg)
        profiles_dir = run_dir.parent
        return (run_dir, profiles_dir) if run_dir.exists() else None

    if tracking_uri:
        cached = cache_production_outcome_artifacts(
            tracking_uri,
            cache_root=cache_root,
            selection=selection,
        )
        return cached.run_dir, cached.profiles_dir

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
