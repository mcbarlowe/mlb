"""Register and promote the paired Stage A/Stage B outcome model release."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

OUTCOME_CHAMPION_ALIAS = "champion"
OUTCOME_CONTRACT_VERSION = 1
OUTCOME_MODEL_COLLECTION = "game_simulation_models"
OUTCOME_RELEASE_TAG = "outcome_release_id"
SIM_INPUTS_RUN_TAG = "sim_inputs_run_id"


@dataclass(frozen=True)
class OutcomeStageSpec:
    stage: str
    model_family: str
    registered_model_name: str
    logged_model_name: str
    description: str


OUTCOME_STAGE_SPECS = {
    "a": OutcomeStageSpec(
        stage="a",
        model_family="pitch_result_stage_a",
        registered_model_name="mlb-pitch-result-stage-a",
        logged_model_name="pitch_result_model",
        description=(
            "Stage A CatBoost model for pitch-result probabilities used by the "
            "MLB game simulator."
        ),
    ),
    "b": OutcomeStageSpec(
        stage="b",
        model_family="in_play_event_stage_b",
        registered_model_name="mlb-in-play-event-stage-b",
        logged_model_name="in_play_event_model",
        description=(
            "Stage B CatBoost model for conditional in-play event probabilities "
            "used by the MLB game simulator."
        ),
    ),
}


@dataclass(frozen=True)
class RegisteredOutcomeStage:
    stage: str
    registered_model_name: str
    version: str
    run_id: str


@dataclass(frozen=True)
class OutcomeReleaseRegistration:
    release_id: str
    sim_inputs_run_id: str
    run_id: str
    promotion_gate_passed: bool
    stage_a: RegisteredOutcomeStage
    stage_b: RegisteredOutcomeStage


def resolve_sim_inputs_run_id(
    tracking_uri: str,
    requested_run_id: str = "auto",
) -> str:
    """Resolve and validate the immutable simulator-input artifact run."""
    from mlflow.tracking import MlflowClient

    from src.ml.mlflow_utils import DEFAULT_MLFLOW_EXPERIMENT

    client = MlflowClient(tracking_uri=tracking_uri)
    if requested_run_id != "auto":
        run = client.get_run(requested_run_id)
        if run.info.status != "FINISHED":
            raise ValueError(
                f"Simulator-input run {requested_run_id!r} is not finished"
            )
        if run.data.tags.get("artifact_kind") != "sim-inputs":
            raise ValueError(
                f"MLflow run {requested_run_id!r} is not a simulator-input release"
            )
        return requested_run_id

    experiment = client.get_experiment_by_name(DEFAULT_MLFLOW_EXPERIMENT)
    if experiment is None:
        raise FileNotFoundError(
            f"MLflow experiment {DEFAULT_MLFLOW_EXPERIMENT!r} was not found"
        )
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED' and tags.artifact_kind = 'sim-inputs'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise FileNotFoundError("No finished simulator-input release was found")
    return runs[0].info.run_id


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _release_id(
    run_dir: Path,
    profiles_dir: Path,
    sim_inputs_run_id: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(sim_inputs_run_id.encode())
    for path in (
        run_dir / "stage_a.cbm",
        run_dir / "stage_a_features.json",
        run_dir / "stage_b.cbm",
        run_dir / "stage_b_features.json",
        profiles_dir / "pitcher_profiles.parquet",
        profiles_dir / "batter_priors.parquet",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.name.encode())
        with path.open("rb") as file_handle:
            while chunk := file_handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _metric(metrics: Mapping[str, object], group: str, name: str) -> float:
    nested = metrics.get(group)
    if not isinstance(nested, dict):
        raise TypeError(f"Outcome metrics are missing {group!r}")
    value = nested.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Outcome metrics are missing {group}.{name}")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"Outcome metric {group}.{name} is not finite")
    return resolved


def _scalar_metric(metrics: Mapping[str, object], name: str) -> float:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Outcome metrics are missing {name!r}")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"Outcome metric {name!r} is not finite")
    return resolved


def _promotion_gate(metrics_by_stage: Mapping[str, Mapping[str, object]]) -> bool:
    for metrics in metrics_by_stage.values():
        for split in ("val", "test"):
            _metric(metrics, split, "brier")
            _metric(metrics, split, "accuracy")
        if _metric(metrics, "val", "log_loss") >= _scalar_metric(
            metrics, "baseline_val_log_loss"
        ):
            return False
        if _metric(metrics, "test", "log_loss") >= _scalar_metric(
            metrics, "baseline_test_log_loss"
        ):
            return False
    return True


def _input_example(feature_contract: Mapping[str, object]):
    import pandas as pd

    columns = feature_contract.get("feature_columns")
    categorical = feature_contract.get("categorical_features")
    if not isinstance(columns, list) or not all(
        isinstance(column, str) for column in columns
    ):
        raise ValueError("Outcome feature contract has invalid feature columns")
    if not isinstance(categorical, list) or not all(
        isinstance(column, str) for column in categorical
    ):
        raise ValueError("Outcome feature contract has invalid categorical features")

    categorical_values = {
        "pitch_type": "FF",
        "throw_side": "R",
        "bat_side": "R",
    }
    numeric_values = {
        "outs": 1.0,
        "inning": 5.0,
        "is_top_half": 1.0,
        "season": 2025.0,
        "times_through_order": 2.0,
        "pz": 2.5,
        "zone_norm_height": 0.5,
        "in_zone": 1.0,
    }
    row = {
        column: (
            categorical_values.get(column, "unknown")
            if column in categorical
            else numeric_values.get(column, 0.0)
        )
        for column in columns
    }
    return pd.DataFrame([row], columns=columns)


def _version_metric_tags(metrics: Mapping[str, object]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for split in ("val", "test"):
        for name in ("log_loss", "brier", "accuracy"):
            tags[f"{split}_{name}"] = f"{_metric(metrics, split, name):.12f}"
    tags["baseline_val_log_loss"] = (
        f"{_scalar_metric(metrics, 'baseline_val_log_loss'):.12f}"
    )
    tags["baseline_test_log_loss"] = (
        f"{_scalar_metric(metrics, 'baseline_test_log_loss'):.12f}"
    )
    return tags


def _log_registered_stage(
    *,
    spec: OutcomeStageSpec,
    run_dir: Path,
    profiles_dir: Path,
    release_id: str,
    sim_inputs_run_id: str,
    promotion_gate_passed: bool,
) -> RegisteredOutcomeStage:
    import catboost
    import mlflow
    import numpy as np
    import pandas as pd
    from catboost import CatBoostClassifier
    from mlflow.catboost import log_model
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    model_path = run_dir / f"stage_{spec.stage}.cbm"
    feature_path = run_dir / f"stage_{spec.stage}_features.json"
    metrics_path = run_dir / f"stage_{spec.stage}_metrics.json"
    feature_contract = _read_json(feature_path)
    metrics = _read_json(metrics_path)

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    example = _input_example(feature_contract)
    predictions = model.predict_proba(example)
    classes = feature_contract.get("classes")
    model_classes = model.classes_
    if (
        not isinstance(classes, list)
        or model_classes is None
        or [str(value) for value in model_classes] != [str(value) for value in classes]
    ):
        raise ValueError(f"Stage {spec.stage.upper()} class contract is incompatible")

    env_key = "MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING"
    previous_env_setting = os.environ.get(env_key)
    os.environ[env_key] = "false"
    try:
        model_info = log_model(
            model,
            name=spec.logged_model_name,
            registered_model_name=spec.registered_model_name,
            signature=infer_signature(example, predictions),
            input_example=example,
            metadata={
                "contract_version": OUTCOME_CONTRACT_VERSION,
                "model_collection": OUTCOME_MODEL_COLLECTION,
                "model_family": spec.model_family,
                "outcome_stage": spec.stage,
                OUTCOME_RELEASE_TAG: release_id,
                SIM_INPUTS_RUN_TAG: sim_inputs_run_id,
                "feature_columns": feature_contract["feature_columns"],
                "categorical_features": feature_contract["categorical_features"],
                "classes": classes,
            },
            extra_files=[
                str(feature_path),
                str(metrics_path),
                str(profiles_dir / "pitcher_profiles.parquet"),
                str(profiles_dir / "batter_priors.parquet"),
            ],
            pip_requirements=[
                f"mlflow=={mlflow.__version__}",
                f"catboost=={catboost.__version__}",
                f"numpy=={np.__version__}",
                f"pandas=={pd.__version__}",
            ],
            await_registration_for=120,
        )
    finally:
        if previous_env_setting is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous_env_setting

    if model_info.registered_model_version is None:
        raise RuntimeError(
            f"MLflow did not register Stage {spec.stage.upper()} model version"
        )
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("Outcome model registration requires an active MLflow run")
    run_id = active_run.info.run_id
    version = str(model_info.registered_model_version)
    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
    client.update_registered_model(
        name=spec.registered_model_name,
        description=spec.description,
    )
    for key, value in {
        "model_collection": OUTCOME_MODEL_COLLECTION,
        "model_family": spec.model_family,
        "model_type": spec.logged_model_name,
    }.items():
        client.set_registered_model_tag(spec.registered_model_name, key, value)

    version_tags = {
        "contract_version": str(OUTCOME_CONTRACT_VERSION),
        "model_collection": OUTCOME_MODEL_COLLECTION,
        "model_family": spec.model_family,
        "model_type": spec.logged_model_name,
        "outcome_stage": spec.stage,
        OUTCOME_RELEASE_TAG: release_id,
        SIM_INPUTS_RUN_TAG: sim_inputs_run_id,
        "promotion_gate": "passed" if promotion_gate_passed else "failed",
        **_version_metric_tags(metrics),
    }
    source_run = client.get_run(run_id)
    source_git_commit = source_run.data.tags.get(
        "source_model_git_commit"
    ) or source_run.data.tags.get("mlflow.source.git.commit")
    registration_git_commit = source_run.data.tags.get("mlflow.source.git.commit")
    if source_git_commit:
        version_tags["source_git_commit"] = source_git_commit
    if registration_git_commit:
        version_tags["registration_git_commit"] = registration_git_commit
    for key in (
        "source_stage_a_run_id",
        "source_stage_b_run_id",
    ):
        if value := source_run.data.tags.get(key):
            version_tags[key] = value
    for key in ("train_seasons", "val_season", "test_season"):
        if value := source_run.data.params.get(key):
            version_tags[key] = value
    for key, value in version_tags.items():
        client.set_model_version_tag(
            name=spec.registered_model_name,
            version=version,
            key=key,
            value=value,
        )
    return RegisteredOutcomeStage(
        stage=spec.stage,
        registered_model_name=spec.registered_model_name,
        version=version,
        run_id=run_id,
    )


def _promote_release(registration: OutcomeReleaseRegistration) -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
    # The resolver refuses mismatched release IDs, so the brief two-alias update
    # window fails closed instead of ever mixing incompatible stages.
    for stage in (registration.stage_b, registration.stage_a):
        client.set_registered_model_alias(
            name=stage.registered_model_name,
            alias=OUTCOME_CHAMPION_ALIAS,
            version=stage.version,
        )
        promoted_version = client.get_model_version(
            stage.registered_model_name,
            stage.version,
        )
        registered_model_tags = {
            "champion_version": stage.version,
            "champion_run_id": stage.run_id,
            "champion_release_id": registration.release_id,
            "champion_sim_inputs_run_id": registration.sim_inputs_run_id,
        }
        if source_git_commit := promoted_version.tags.get("source_git_commit"):
            registered_model_tags["champion_source_git_commit"] = source_git_commit
        for key, value in registered_model_tags.items():
            client.set_registered_model_tag(
                stage.registered_model_name,
                key,
                value,
            )


def log_outcome_release_models(
    *,
    run_dir: Path,
    profiles_dir: Path,
    sim_inputs_run_id: str,
    set_champion: bool = False,
) -> OutcomeReleaseRegistration:
    """Log, register, tag, and optionally promote one paired outcome release."""
    import mlflow

    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("Outcome model registration requires an active MLflow run")
    release_id = _release_id(run_dir, profiles_dir, sim_inputs_run_id)
    metrics_by_stage = {
        stage: _read_json(run_dir / f"stage_{stage}_metrics.json")
        for stage in OUTCOME_STAGE_SPECS
    }
    promotion_gate_passed = _promotion_gate(metrics_by_stage)
    mlflow.set_tags(
        {
            "model_collection": OUTCOME_MODEL_COLLECTION,
            OUTCOME_RELEASE_TAG: release_id,
            SIM_INPUTS_RUN_TAG: sim_inputs_run_id,
            "promotion_gate": "passed" if promotion_gate_passed else "failed",
            "production_model": str(set_champion and promotion_gate_passed).lower(),
        }
    )
    mlflow.log_params(
        {
            "outcome_contract_version": OUTCOME_CONTRACT_VERSION,
            OUTCOME_RELEASE_TAG: release_id,
            SIM_INPUTS_RUN_TAG: sim_inputs_run_id,
        }
    )
    for stage in OUTCOME_STAGE_SPECS:
        mlflow.log_artifact(
            str(run_dir / f"stage_{stage}_features.json"),
            artifact_path="contracts",
        )
        mlflow.log_artifact(
            str(run_dir / f"stage_{stage}_metrics.json"),
            artifact_path="metrics",
        )
    mlflow.log_artifact(
        str(profiles_dir / "pitcher_profiles.parquet"),
        artifact_path="profiles",
    )
    mlflow.log_artifact(
        str(profiles_dir / "batter_priors.parquet"),
        artifact_path="profiles",
    )

    stages = {
        stage: _log_registered_stage(
            spec=spec,
            run_dir=run_dir,
            profiles_dir=profiles_dir,
            release_id=release_id,
            sim_inputs_run_id=sim_inputs_run_id,
            promotion_gate_passed=promotion_gate_passed,
        )
        for stage, spec in OUTCOME_STAGE_SPECS.items()
    }
    registration = OutcomeReleaseRegistration(
        release_id=release_id,
        sim_inputs_run_id=sim_inputs_run_id,
        run_id=active_run.info.run_id,
        promotion_gate_passed=promotion_gate_passed,
        stage_a=stages["a"],
        stage_b=stages["b"],
    )
    if set_champion and promotion_gate_passed:
        _promote_release(registration)
    return registration
