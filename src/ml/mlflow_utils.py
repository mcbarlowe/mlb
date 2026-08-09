from __future__ import annotations

import json
import os
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

import mlflow


def configure_mlflow(
    experiment_name: str,
    tracking_uri: str | None = None,
    *,
    require_tracking_uri: bool = False,
) -> str:
    resolved_tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if require_tracking_uri and not resolved_tracking_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is required for this training script. "
            "Export the shared Postgres-backed URI or pass "
            "--mlflow-tracking-uri explicitly. If you really want a local "
            "SQLite run, pass --mlflow-tracking-uri sqlite:///..."
        )
    if not resolved_tracking_uri:
        resolved_tracking_uri = f"sqlite:///{Path('mlflow.db').resolve()}"
    mlflow.set_tracking_uri(resolved_tracking_uri)
    mlflow.set_experiment(experiment_name)
    return resolved_tracking_uri



def _flatten_dict(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(_flatten_dict(value, full_key))
        else:
            flattened[full_key] = value
    return flattened



def build_param_dict(data: Mapping[str, Any], prefix: str = "") -> dict[str, str | bool | int | float]:
    params: dict[str, str | bool | int | float] = {}
    for key, value in _flatten_dict(data, prefix).items():
        if value is None:
            continue
        if isinstance(value, (bool, int, float)):
            params[key] = value
        elif isinstance(value, (str, Path)):
            params[key] = str(value)
        else:
            params[key] = json.dumps(value, sort_keys=True)
    return params



def build_metric_dict(data: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in _flatten_dict(data, prefix).items():
        if isinstance(value, bool):
            continue
        if isinstance(value, Real):
            metrics[key] = float(value)
    return metrics



def log_path_if_exists(path: Path, artifact_path: str | None = None) -> None:
    if not path.exists():
        return
    if path.is_file():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
        return
    if any(path.iterdir()):
        mlflow.log_artifacts(str(path), artifact_path=artifact_path)
