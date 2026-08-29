"""Register trained pitch prediction models in the shared MLflow registry.

Mirrors the ``src/outcome/mlflow_registry.py`` conventions: each release is
logged as a native MLflow model inside an experiment run, becomes an immutable
registered-model version tagged with its data split and test metrics, and can
optionally advance the ``champion`` alias.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow

from mlb.ml.mlflow_utils import build_metric_dict, build_param_dict

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import torch

PITCH_CHAMPION_ALIAS = "champion"
# Predicts the thrown pitch (type/location); pitch OUTCOME models
# (mlb-pitch-result-stage-a / mlb-in-play-event-stage-b) live in src/outcome.
PITCH_MODEL_COLLECTION = "pitch_type_prediction_models"


@dataclass(frozen=True)
class PitchModelSpec:
    model_family: str
    registered_model_name: str
    logged_model_name: str
    description: str


PITCH_TYPE_SPEC = PitchModelSpec(
    model_family="pitch_type_lstm_attention",
    registered_model_name="mlb-pitch-type-lstm-attention",
    logged_model_name="pitch_type_model",
    description=(
        "Pitch type prediction model (LSTM+attention over at-bat sequences): "
        "per-pitch type logits plus an MDN location head. Predicts what the "
        "pitcher throws, not the pitch outcome (outcome models: "
        "mlb-pitch-result-stage-a / mlb-in-play-event-stage-b). Load with "
        "mlflow.pytorch.load_model from this repo (requires src.ml on "
        "sys.path and MLFLOW_ALLOW_PICKLE_DESERIALIZATION=true) and "
        "featurize inputs with the PitchFeatureEngine state in "
        "extra_files/feature_engine.json."
    ),
)

PITCH_LOCATION_SPEC = PitchModelSpec(
    model_family="pitch_type_conditioned_location",
    registered_model_name="mlb-pitch-type-conditioned-location",
    logged_model_name="pitch_location_model",
    description=(
        "Pitch location prediction model (MDN conditioned on pitch type): "
        "bivariate Gaussian mixture over plate coordinates with one head per "
        "pitch type. Predicts where the pitcher throws, not the pitch "
        "outcome (outcome models: mlb-pitch-result-stage-a / "
        "mlb-in-play-event-stage-b). Load with mlflow.pytorch.load_model "
        "from this repo (requires src.ml on sys.path and "
        "MLFLOW_ALLOW_PICKLE_DESERIALIZATION=true); the feature contract is "
        "feature_columns in extra_files/config.json."
    ),
)


@dataclass(frozen=True)
class LoadedPitchModel:
    model: torch.nn.Module
    params: dict[str, str | bool | int | float]
    metrics: dict[str, float]
    metadata: dict[str, Any]
    run_tags: dict[str, str]
    version_tags: dict[str, str]
    extra_files: list[Path]


@dataclass(frozen=True)
class RegisteredPitchModel:
    registered_model_name: str
    version: str
    run_id: str


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _mapping(data: Mapping[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Expected object {key!r} in {path}")
    return value


def _coerce_numeric(data: Mapping[str, Any]) -> dict[str, float]:
    """Recover metrics serialized through ``json.dump(..., default=str)``."""
    coerced: dict[str, float] = {}
    for key, value in data.items():
        if isinstance(value, bool):
            continue
        try:
            coerced[key] = float(value)
        except (TypeError, ValueError):
            continue
    return coerced


def _require_files(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)


def _season_tags(
    train_seasons: Iterable[Any],
    val_season: Any,
    test_season: Any,
) -> dict[str, str]:
    return {
        "train_seasons": ",".join(str(season) for season in train_seasons),
        "val_season": str(val_season),
        "test_season": str(test_season),
    }


def _metric_tags(metrics: Mapping[str, float], keys: Iterable[str]) -> dict[str, str]:
    return {key: f"{metrics[key]:.12f}" for key in keys if key in metrics}


def load_pitch_type_release(run_dir: Path) -> LoadedPitchModel:
    """Reconstruct a trained pitch type model from a run_full_training dir."""
    import torch

    from mlb.ml.features import PitchFeatureEngine
    from mlb.ml.model import create_model

    run_dir = Path(run_dir)
    model_path = run_dir / "final_model.pt"
    engine_path = run_dir / "feature_engine.json"
    results_path = run_dir / "results.json"
    _require_files((model_path, engine_path, results_path))

    results = _read_json(results_path)
    config = _mapping(results, "config", results_path)
    train_args = _mapping(results, "args", results_path)
    test_results = _mapping(results, "test_results", results_path)

    engine = PitchFeatureEngine.load(engine_path)
    feature_columns = engine.get_feature_columns()
    model_type = str(config["model_type"])

    model_kwargs: dict[str, Any] = {
        "n_pitch_types": engine.n_pitch_types,
        "n_pitchers": engine.n_pitchers,
        "n_batters": engine.n_batters,
        "n_features": len(feature_columns),
        "model_type": model_type,
        "feature_indices": engine.get_feature_indices(),
        "hidden_dim": config["hidden_dim"],
        "n_layers": config["n_layers"],
        "dropout": config["dropout"],
        "embedding_dim": config["embedding_dim"],
        "n_location_components": config["n_location_components"],
    }
    if model_type in ("lstm_attention", "enhanced_attention"):
        model_kwargs["n_attention_heads"] = config["n_attention_heads"]
        model_kwargs["n_attention_layers"] = config["n_attention_layers"]

    model = create_model(**model_kwargs)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    n_parameters = sum(p.numel() for p in model.parameters())
    training = results.get("training")
    expected = training.get("n_parameters") if isinstance(training, dict) else None
    if expected is not None and int(expected) != n_parameters:
        raise ValueError(
            f"Reconstructed pitch type model has {n_parameters} parameters "
            f"but {results_path} recorded {expected}"
        )

    metrics = build_metric_dict(_coerce_numeric(test_results), prefix="test")
    season_tags = _season_tags(
        train_args.get("train_seasons") or [],
        train_args.get("val_season", ""),
        train_args.get("test_season", ""),
    )
    run_tags = {
        **season_tags,
        "quick": str(bool(train_args.get("quick"))),
        "data_path": str(train_args.get("data_path", "")),
    }
    return LoadedPitchModel(
        model=model,
        params=build_param_dict(train_args, prefix="arg"),
        metrics=metrics,
        metadata={
            "n_pitch_types": engine.n_pitch_types,
            "n_pitchers": engine.n_pitchers,
            "n_batters": engine.n_batters,
            "n_features": len(feature_columns),
            "feature_columns": feature_columns,
            "architecture": dict(config),
            **season_tags,
        },
        run_tags=run_tags,
        version_tags={
            **run_tags,
            "n_parameters": str(n_parameters),
            "source_run": run_dir.name,
            **_metric_tags(
                metrics,
                (
                    "test.accuracy",
                    "test.top3_accuracy",
                    "test.f1_macro",
                    "test.nll",
                    "test.euclidean_error",
                ),
            ),
        },
        extra_files=[engine_path, results_path],
    )


def load_pitch_location_release(run_dir: Path) -> LoadedPitchModel:
    """Reconstruct a trained location model from a training output dir."""
    import torch

    from mlb.ml.features import PITCH_TYPE_CODES
    from mlb.ml.pitch_type_location_model import PitchTypeConditionedMDN

    run_dir = Path(run_dir)
    model_path = run_dir / "pitch_type_location_model.pt"
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "test_metrics.json"
    engine_path = run_dir / "feature_engine.pt"
    _require_files((model_path, config_path, metrics_path, engine_path))

    config = _read_json(config_path)
    test_metrics = _read_json(metrics_path)
    overall = _mapping(test_metrics, "overall", metrics_path)
    feature_columns = config["feature_columns"]
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError(f"Expected non-empty feature_columns in {config_path}")

    model = PitchTypeConditionedMDN(
        n_features=len(feature_columns),
        n_pitch_types=len(PITCH_TYPE_CODES),
        hidden_dims=config["hidden_dims"],
        n_components=config["n_components"],
        dropout=config["dropout"],
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    metrics = build_metric_dict(overall, prefix="test")
    season_tags = _season_tags(
        config.get("train_seasons") or [],
        config.get("val_season", ""),
        config.get("test_season", ""),
    )
    params = {
        key: value for key, value in config.items() if key != "feature_columns"
    }
    return LoadedPitchModel(
        model=model,
        params=build_param_dict(params, prefix="arg"),
        metrics=metrics,
        metadata={
            "n_pitch_types": len(PITCH_TYPE_CODES),
            "n_features": len(feature_columns),
            "feature_columns": feature_columns,
            "hidden_dims": config["hidden_dims"],
            "n_components": config["n_components"],
            "dropout": config["dropout"],
            **season_tags,
        },
        run_tags=dict(season_tags),
        version_tags={
            **season_tags,
            "n_parameters": str(sum(p.numel() for p in model.parameters())),
            "source_run": run_dir.name,
            **_metric_tags(
                metrics,
                (
                    "test.nll",
                    "test.mae_px",
                    "test.mae_pz",
                    "test.euclidean",
                    "test.coverage_90",
                ),
            ),
        },
        extra_files=[config_path, metrics_path, engine_path],
    )


def log_registered_pitch_model(
    *,
    spec: PitchModelSpec,
    loaded: LoadedPitchModel,
    set_champion: bool = False,
) -> RegisteredPitchModel:
    """Log one pitch model release and register an immutable version."""
    import numpy as np
    import torch
    from mlflow.pytorch import log_model
    from mlflow.tracking import MlflowClient

    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("Pitch model registration requires an active MLflow run")

    mlflow.set_tags({**loaded.run_tags, "model_family": spec.model_family})
    mlflow.log_params(loaded.params)
    if loaded.metrics:
        mlflow.log_metrics(loaded.metrics)

    env_key = "MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING"
    previous_env_setting = os.environ.get(env_key)
    os.environ[env_key] = "false"
    try:
        model_info = log_model(
            loaded.model,
            name=spec.logged_model_name,
            registered_model_name=spec.registered_model_name,
            metadata={
                "model_collection": PITCH_MODEL_COLLECTION,
                "model_family": spec.model_family,
                **loaded.metadata,
            },
            extra_files=[str(path) for path in loaded.extra_files],
            # Dynamic-length multi-input forward; pt2 traced-graph export
            # does not apply (and requires torch>=2.4).
            serialization_format="pickle",
            pip_requirements=[
                f"mlflow=={mlflow.__version__}",
                f"torch=={torch.__version__}",
                f"numpy=={np.__version__}",
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
            f"MLflow did not register a {spec.registered_model_name} version"
        )
    version = str(model_info.registered_model_version)
    run_id = active_run.info.run_id

    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
    client.update_registered_model(
        name=spec.registered_model_name,
        description=spec.description,
    )
    for key, value in {
        "model_collection": PITCH_MODEL_COLLECTION,
        "model_family": spec.model_family,
        "model_type": spec.logged_model_name,
    }.items():
        client.set_registered_model_tag(spec.registered_model_name, key, value)

    version_tags = {
        "model_collection": PITCH_MODEL_COLLECTION,
        "model_family": spec.model_family,
        "model_type": spec.logged_model_name,
        **loaded.version_tags,
    }
    for key, value in version_tags.items():
        client.set_model_version_tag(
            name=spec.registered_model_name,
            version=version,
            key=key,
            value=value,
        )
    if set_champion:
        client.set_registered_model_alias(
            name=spec.registered_model_name,
            alias=PITCH_CHAMPION_ALIAS,
            version=version,
        )
    return RegisteredPitchModel(
        registered_model_name=spec.registered_model_name,
        version=version,
        run_id=run_id,
    )
