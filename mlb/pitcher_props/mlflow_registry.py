"""MLflow pyfunc registration for every pitcher-prop posterior component."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import ModelSignature
from mlflow.pyfunc.model import PythonModel, PythonModelContext
from mlflow.types import ColSpec, Schema

import mlb
from mlb.ml.mlflow_utils import build_metric_dict, build_param_dict, configure_mlflow
from mlb.pitcher_props.evaluation import EvaluationReport
from mlb.pitcher_props.model import (
    BATTERS_FACED_MARKET,
    HITS_ALLOWED_MARKET,
    MODEL_COLLECTION,
    MODEL_CONTRACT_VERSION,
    MODEL_FAMILY,
    OUTS_MARKET,
    STRIKEOUT_MARKET,
    WALKS_MARKET,
    PitcherPropModel,
)

DEFAULT_EXPERIMENT = "mlb-pitcher-prop-models"
REGISTERED_MODEL_NAMES = {
    OUTS_MARKET: "mlb-pitcher-outs-bayes",
    BATTERS_FACED_MARKET: "mlb-pitcher-batters-faced-bayes",
    STRIKEOUT_MARKET: "mlb-pitcher-strikeouts-bayes",
    HITS_ALLOWED_MARKET: "mlb-pitcher-hits-allowed-bayes",
    WALKS_MARKET: "mlb-pitcher-walks-bayes",
}
DEFAULT_POINTS = {
    OUTS_MARKET: 17.5,
    BATTERS_FACED_MARKET: 22.5,
    STRIKEOUT_MARKET: 5.5,
    HITS_ALLOWED_MARKET: 5.5,
    WALKS_MARKET: 1.5,
}
PYFUNC_SIGNATURE = ModelSignature(
    inputs=Schema(
        [
            ColSpec("long", "pitcher_id"),
            ColSpec("long", "opponent_team_id", required=False),
            ColSpec("double", "point"),
            ColSpec("string", "prediction_time"),
            ColSpec("boolean", "is_home", required=False),
            ColSpec("double", "rest_days", required=False),
            ColSpec("long", "seed", required=False),
        ]
    ),
    outputs=Schema(
        [
            ColSpec("string", "market"),
            ColSpec("double", "point"),
            ColSpec("double", "mean"),
            ColSpec("double", "standard_deviation"),
            ColSpec("double", "q05"),
            ColSpec("double", "q50"),
            ColSpec("double", "q95"),
            ColSpec("double", "probability_over"),
            ColSpec("double", "probability_push"),
            ColSpec("double", "probability_under"),
            ColSpec("long", "samples"),
            ColSpec("double", "mc_standard_error"),
            ColSpec("string", "model_contract_version"),
        ]
    ),
)


class PitcherPropPyfuncModel(PythonModel):
    """One registered component backed by the complete opportunity model."""

    def __init__(self, model: PitcherPropModel, market: str) -> None:
        self.model = model
        self.market = market

    def predict(
        self,
        context: PythonModelContext | None,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        del context, params
        required = {"pitcher_id", "point", "prediction_time"}
        missing = required - set(model_input.columns)
        if missing:
            raise ValueError(f"pitcher prop model input is missing {sorted(missing)!r}")
        predictions: list[dict[str, object]] = []
        for row_number, row in model_input.iterrows():
            timestamp = pd.Timestamp(row["prediction_time"])
            if timestamp.tzinfo is None:
                raise ValueError(
                    f"row {row_number} prediction_time must include a timezone"
                )
            opponent_value = row.get("opponent_team_id")
            rest_value = row.get("rest_days")
            seed_value = row.get("seed")
            prediction = self.model.predict(
                pitcher_id=int(row["pitcher_id"]),
                opponent_team_id=(
                    None if pd.isna(opponent_value) else int(opponent_value)
                ),
                market=self.market,
                point=float(row["point"]),
                as_of=timestamp.to_pydatetime(),
                is_home=bool(row.get("is_home", False)),
                rest_days=None if pd.isna(rest_value) else float(rest_value),
                seed=None if pd.isna(seed_value) else int(seed_value),
            )
            values = prediction.to_dict()
            values["model_contract_version"] = MODEL_CONTRACT_VERSION
            predictions.append(values)
        return pd.DataFrame(predictions)


@dataclass(frozen=True)
class RegisteredPitcherModel:
    market: str
    registered_model_name: str
    version: str
    run_id: str
    skipped_existing: bool
    promotion_gate: str



def _ensure_registered_model(
    client: MlflowClient,
    *,
    name: str,
    market: str,
) -> None:
    try:
        client.get_registered_model(name)
    except Exception:
        client.create_registered_model(
            name,
            tags={
                "model_collection": MODEL_COLLECTION,
                "model_family": MODEL_FAMILY,
                "model_type": market,
            },
            description=(
                f"Hierarchical empirical-Bayes {market} posterior predictive model. "
                "Opportunity is shared through pitcher outs and batters faced."
            ),
        )
    for key, value in {
        "model_collection": MODEL_COLLECTION,
        "model_family": MODEL_FAMILY,
        "model_type": market,
    }.items():
        client.set_registered_model_tag(name, key, value)



def _existing_version(
    client: MlflowClient,
    *,
    name: str,
    fingerprint: str,
) -> Any | None:
    for version in client.search_model_versions(f"name = '{name}'"):
        if (
            version.tags.get("model_contract_version") == MODEL_CONTRACT_VERSION
            and version.tags.get("model_fingerprint") == fingerprint
        ):
            return version
    return None



def register_pitcher_prop_models(
    model: PitcherPropModel,
    report: EvaluationReport,
    *,
    tracking_uri: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    set_champion: bool = False,
    await_registration_for: int = 120,
) -> list[RegisteredPitcherModel]:
    """Register five immutable component models; promotion remains gate-controlled."""

    if model.trained_through is None:
        raise ValueError("fitted model is missing its training cutoff")
    resolved_uri = configure_mlflow(
        experiment_name,
        tracking_uri,
        require_tracking_uri=True,
    )
    client = MlflowClient(tracking_uri=resolved_uri)
    fingerprint = model.fingerprint()
    # The logged model bundles this package as code. Locate it from the package
    # itself rather than by counting parent directories: the old
    # ``__file__.parents[2] / "mlb"`` form only happened to be right because this
    # module sits exactly two levels below the container of ``mlb/``, so it would
    # silently point at a nonexistent ``mlb/mlb`` if the module ever moved.
    package_root = Path(mlb.__file__).resolve().parent
    results: list[RegisteredPitcherModel] = []

    for market, registered_name in REGISTERED_MODEL_NAMES.items():
        metrics = report.components[market]
        gate_passed = metrics.mae_improvement > 0.0 and metrics.log_loss_improvement > 0.0
        gate = "passed" if gate_passed else "failed"
        _ensure_registered_model(client, name=registered_name, market=market)
        existing = _existing_version(
            client,
            name=registered_name,
            fingerprint=fingerprint,
        )
        if existing is not None:
            results.append(
                RegisteredPitcherModel(
                    market=market,
                    registered_model_name=registered_name,
                    version=str(existing.version),
                    run_id=str(existing.run_id or ""),
                    skipped_existing=True,
                    promotion_gate=existing.tags.get("promotion_gate", gate),
                )
            )
            continue

        input_example = pd.DataFrame(
            {
                "pitcher_id": [next(iter(model.pitchers), 1)],
                "opponent_team_id": [next(iter(model.opponents), 0)],
                "point": [DEFAULT_POINTS[market]],
                "prediction_time": [
                    (model.trained_through + timedelta(days=1)).isoformat()
                ],
                "is_home": [False],
                "rest_days": [5.0],
                "seed": [model.config.seed],
            }
        )
        pyfunc_model = PitcherPropPyfuncModel(model, market)
        pyfunc_model.predict(None, input_example)
        signature = PYFUNC_SIGNATURE
        contract = {
            "model_contract_version": MODEL_CONTRACT_VERSION,
            "model_family": MODEL_FAMILY,
            "model_collection": MODEL_COLLECTION,
            "model_type": market,
            "model_fingerprint": fingerprint,
            "trained_from": (
                model.trained_from.isoformat() if model.trained_from else None
            ),
            "trained_through": model.trained_through.isoformat(),
            "training_starts": model.training_starts,
            "validation_season": report.validation_season,
            "opportunity_dependencies": (
                []
                if market == OUTS_MARKET
                else [OUTS_MARKET]
                if market == BATTERS_FACED_MARKET
                else [OUTS_MARKET, BATTERS_FACED_MARKET]
            ),
            "config": asdict(model.config),
        }
        run_tags = {
            "model_contract_version": MODEL_CONTRACT_VERSION,
            "model_family": MODEL_FAMILY,
            "model_collection": MODEL_COLLECTION,
            "model_type": market,
            "model_fingerprint": fingerprint,
            "promotion_gate": gate,
            "production_model": "true" if set_champion and gate_passed else "false",
        }
        with mlflow.start_run(run_name=f"{market}-{MODEL_CONTRACT_VERSION}") as run:
            mlflow.set_tags(run_tags)
            mlflow.log_params(build_param_dict(contract["config"]))
            mlflow.log_metrics(build_metric_dict(metrics.to_dict()))
            mlflow.log_dict(contract, "model_contract.json")
            mlflow.log_dict(report.to_dict(), "walk_forward_evaluation.json")
            previous_env = os.environ.get("MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING")
            os.environ["MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING"] = "false"
            try:
                model_info = mlflow.pyfunc.log_model(
                    name=market,
                    python_model=pyfunc_model,
                    registered_model_name=registered_name,
                    signature=signature,
                    input_example=input_example,
                    metadata=contract,
                    code_paths=[str(package_root)],
                    pip_requirements=[
                        f"mlflow=={mlflow.__version__}",
                        f"numpy=={np.__version__}",
                        f"pandas=={pd.__version__}",
                    ],
                    await_registration_for=await_registration_for,
                )
            finally:
                if previous_env is None:
                    os.environ.pop("MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING", None)
                else:
                    os.environ["MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING"] = previous_env
            run_id = run.info.run_id
        if model_info.registered_model_version is None:
            raise RuntimeError(f"MLflow did not register {registered_name}")
        version = str(model_info.registered_model_version)
        version_tags = {
            **run_tags,
            "trained_through": model.trained_through.isoformat(),
            "validation_season": str(report.validation_season),
            "validation_mae": f"{metrics.mae:.12f}",
            "validation_count_log_loss": f"{metrics.count_log_loss:.12f}",
        }
        for key, value in version_tags.items():
            client.set_model_version_tag(
                name=registered_name,
                version=version,
                key=key,
                value=value,
            )
        client.set_registered_model_tag(registered_name, "latest_logged_version", version)
        client.set_registered_model_tag(registered_name, "latest_run_id", run_id)
        if set_champion and gate_passed:
            client.set_registered_model_alias(registered_name, "champion", version)
            client.set_registered_model_tag(registered_name, "champion_version", version)
            client.set_registered_model_tag(registered_name, "champion_run_id", run_id)
        results.append(
            RegisteredPitcherModel(
                market=market,
                registered_model_name=registered_name,
                version=version,
                run_id=run_id,
                skipped_existing=False,
                promotion_gate=gate,
            )
        )
    return results


__all__ = [
    "DEFAULT_EXPERIMENT",
    "REGISTERED_MODEL_NAMES",
    "PitcherPropPyfuncModel",
    "RegisteredPitcherModel",
    "register_pitcher_prop_models",
]
