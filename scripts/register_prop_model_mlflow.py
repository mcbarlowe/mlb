"""Register the batter prop rate estimator lineage in MLflow.

Creates registered model ``mlb-prop-rate-estimator`` (collection
``prop_models``) with one version per estimator generation, each backed by a
run in experiment ``mlb-prop-models`` carrying:

  - params: estimator configuration
  - metrics: leak-free 2024+2025 predictive calibration (realized versus
    estimated rates, Brier score, and log loss)
  - artifacts: estimator contract JSON plus the aging-curve and park-factor
    artifacts the version depends on

Version tags carry an immutable ``model_contract_version``. Betting strategy
identity is deliberately separate and is persisted by the betting service
alongside this model provenance. ``@champion`` points at the deployed model
generation. Reruns skip already-registered model contract versions.

Usage:
  uv run python scripts/register_prop_model_mlflow.py [--set-champion]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from calibrate_prop_estimator import VERSIONS, lineage_metrics, load_lines

MODEL_NAME = "mlb-prop-rate-estimator"
EXPERIMENT = "mlb-prop-models"
DEFAULT_URI = "http://10.0.0.171:5001"
MODEL_VERSION_BY_ESTIMATOR = {
    "props-raw-shrink-v1": "prop-rate-raw-shrink-v1",
    "props-decay400-age-k50-mkt": "prop-rate-decay-age-v2",
    "props-cond-v3": "prop-rate-cond-v3",
}
CHAMPION_MODEL_VERSION = "prop-rate-cond-v3"

SPECS: dict[str, dict] = {
    "props-raw-shrink-v1": {
        "description": (
            "Pooled trailing 3-season per-game rate with empirical-Bayes "
            "shrinkage toward the league rate; uniform window, no aging, "
            "and no role conditioning."
        ),
        "params": {
            "window_seasons": 3,
            "shrink_k": 50,
            "recency_half_life": "inf",
            "aging_curves": False,
            "conditioning": False,
        },
        "artifacts": [],
    },
    "props-decay400-age-k50-mkt": {
        "description": (
            "Adds exponential recency decay (400-game half-life) and "
            "delta-method aging curves on log-odds."
        ),
        "params": {
            "window_seasons": 3,
            "shrink_k": 50,
            "recency_half_life": 400,
            "aging_curves": True,
            "conditioning": False,
        },
        "artifacts": ["models/props/aging_curves.json"],
    },
    "props-cond-v3": {
        "description": "Adds conditioning for all markets except HR: starts-only "
                       "stream (PA>=3), expected-PA rescale on 0.5 lines "
                       "(trailing-30-starts mean PA), tonight's-park shrunk logit "
                       "offset (k_park=2000). HR stays on the v2 estimator "
                       "(calibration shows conditioning does not help HR).",
        "params": {
            "window_seasons": 3,
            "shrink_k": 50,
            "recency_half_life": 400,
            "aging_curves": True,
            "conditioning": True,
            "start_pa": 3,
            "exp_pa_window": 30,
            "k_park": 2000,
            "conditioned_markets": "all except batter_home_runs",
        },
        "artifacts": ["models/props/aging_curves.json",
                      "models/props/park_factors.json"],
    },
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mlflow-tracking-uri", default=DEFAULT_URI)
    ap.add_argument("--set-champion", action="store_true",
                    help=f"point @champion at the {CHAMPION_MODEL_VERSION} version")
    args = ap.parse_args()

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient(tracking_uri=args.mlflow_tracking_uri)

    try:
        client.create_registered_model(
            MODEL_NAME,
            tags={"collection": "prop_models"},
            description=(
                "Batter prop per-game probability estimator lineage: trailing "
                "rates, decay, aging, empirical-Bayes shrinkage, and conditioning."
            ),
        )
        print(f"created registered model {MODEL_NAME}")
    except Exception:
        print(f"registered model {MODEL_NAME} exists")

    registered_versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
    existing: dict[str, object] = {}
    for version in registered_versions:
        model_version = version.tags.get("model_contract_version")
        if model_version is None:
            model_version = MODEL_VERSION_BY_ESTIMATOR.get(
                version.tags.get("strategy_version", "")
            )
        if model_version is not None:
            existing[model_version] = version

    repo = Path(__file__).parent.parent
    frame = load_lines()
    metrics = lineage_metrics(frame)
    champion_version = None
    for estimator_key in VERSIONS:
        model_version = MODEL_VERSION_BY_ESTIMATOR[estimator_key]
        if model_version in existing:
            print(f"skip {model_version}: already registered")
            if model_version == CHAMPION_MODEL_VERSION:
                champion_version = existing[model_version].version
            continue
        spec = SPECS[estimator_key]
        with mlflow.start_run(run_name=model_version) as run:
            mlflow.log_params(spec["params"])
            mlflow.log_metrics(metrics[estimator_key])
            mlflow.set_tags(
                {
                    "model_contract_version": model_version,
                    "collection": "prop_models",
                }
            )
            contract = {
                "model_version": model_version,
                "description": spec["description"],
                "params": spec["params"],
                "artifacts": spec["artifacts"],
                "implementation": "src/data_contracts/prop_predictions.py",
                "calibration": (
                    "scripts/calibrate_prop_estimator.py (held-out 2024+2025 starts)"
                ),
            }
            with tempfile.TemporaryDirectory() as temporary_directory:
                contract_path = Path(temporary_directory) / "estimator_contract.json"
                contract_path.write_text(json.dumps(contract, indent=1))
                mlflow.log_artifact(str(contract_path), artifact_path="estimator")
            for relative_path in spec["artifacts"]:
                mlflow.log_artifact(
                    str(repo / relative_path),
                    artifact_path="estimator",
                )
            version = client.create_model_version(
                name=MODEL_NAME,
                source=f"{run.info.artifact_uri}/estimator",
                run_id=run.info.run_id,
                tags={
                    "model_contract_version": model_version,
                    "collection": "prop_models",
                },
                description=spec["description"],
            )
            client.update_model_version(
                MODEL_NAME,
                version.version,
                description=spec["description"],
            )
            print(f"registered {MODEL_NAME} v{version.version} = {model_version}")
            if model_version == CHAMPION_MODEL_VERSION:
                champion_version = version.version

    if args.set_champion and champion_version is not None:
        client.set_registered_model_alias(MODEL_NAME, "champion", champion_version)
        print(f"@champion -> v{champion_version} ({CHAMPION_MODEL_VERSION})")


if __name__ == "__main__":
    main()
