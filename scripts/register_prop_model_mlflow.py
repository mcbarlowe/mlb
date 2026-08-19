"""Register the batter prop rate estimator lineage in MLflow.

Creates registered model ``mlb-prop-rate-estimator`` (collection
``prop_models``) with one version per estimator generation, each backed by a
run in experiment ``mlb-prop-models`` carrying:

  - params: full estimator + alert-gate configuration
  - metrics: leak-free 2024+2025 calibration (max-pick / top-decile /
    under-decile realized-vs-estimated, Brier, log loss) and flat-1u ROI at
    the production gate price, from scripts/calibrate_prop_estimator.py
  - artifacts: estimator contract JSON plus the aging-curve and park-factor
    artifacts the version depends on

Version tags carry ``strategy_version`` matching ``mlb.prop_paper_bets``, so
live paper-ledger P&L joins to registry versions. ``@champion`` points at the
deployed generation. Idempotent: already-registered strategy_versions are
skipped, so re-running after adding a new generation registers only the new
one.

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
CHAMPION = "props-cond-v3"

SPECS: dict[str, dict] = {
    "props-raw-shrink-v1": {
        "description": "Pooled trailing 3-season per-game rate, EB shrink k=50 "
                       "toward board-pool league rate. Uniform window, no aging, "
                       "no conditioning. Alert gate: EV>=5%, >=150 GP.",
        "params": {
            "window_seasons": 3, "shrink_k": 50, "recency_half_life": "inf",
            "aging_curves": False, "conditioning": False,
            "gate_min_ev": 0.05, "gate_min_ev_hr": 0.10, "gate_min_gp": 150,
            "market_anchor": False,
        },
        "artifacts": [],
    },
    "props-decay400-age-k50-mkt": {
        "description": "Adds exponential recency decay (H=400 games) and "
                       "delta-method aging curves on log-odds; market-anchor "
                       "alert gates (two-sided fair required, <=1.5x/+15pp vs "
                       "fair, price <= +1400, EV <= 50%).",
        "params": {
            "window_seasons": 3, "shrink_k": 50, "recency_half_life": 400,
            "aging_curves": True, "conditioning": False,
            "gate_min_ev": 0.11, "gate_min_ev_hr": 0.15, "gate_min_gp": 150,
            "market_anchor": True, "max_fair_ratio": 1.5, "max_fair_diff": 0.15,
            "max_decimal": 15, "max_ev": 0.50,
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
            "window_seasons": 3, "shrink_k": 50, "recency_half_life": 400,
            "aging_curves": True, "conditioning": True, "start_pa": 3,
            "exp_pa_window": 30, "k_park": 2000,
            "conditioned_markets": "all except batter_home_runs",
            "gate_min_ev": 0.11, "gate_min_ev_hr": 0.15, "gate_min_gp": 150,
            "market_anchor": True, "max_fair_ratio": 1.5, "max_fair_diff": 0.15,
            "max_decimal": 15, "max_ev": 0.50,
        },
        "artifacts": ["models/props/aging_curves.json",
                      "models/props/park_factors.json"],
    },
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mlflow-tracking-uri", default=DEFAULT_URI)
    ap.add_argument("--set-champion", action="store_true",
                    help=f"point @champion at the {CHAMPION} version")
    args = ap.parse_args()

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient(tracking_uri=args.mlflow_tracking_uri)

    try:
        client.create_registered_model(
            MODEL_NAME,
            tags={"collection": "prop_models"},
            description="Batter prop per-game rate estimator lineage (rule-based: "
                        "trailing rates -> decay -> aging -> EB shrink -> "
                        "conditioning). Live P&L in mlb.prop_paper_bets joins on "
                        "the strategy_version tag.",
        )
        print(f"created registered model {MODEL_NAME}")
    except Exception:
        print(f"registered model {MODEL_NAME} exists")

    existing = {
        v.tags.get("strategy_version")
        for v in client.search_model_versions(f"name = '{MODEL_NAME}'")
    }

    repo = Path(__file__).parent.parent
    frame = load_lines()
    metrics = lineage_metrics(frame)

    champion_version = None
    for strategy_version in VERSIONS:
        if strategy_version in existing:
            print(f"skip {strategy_version}: already registered")
            if strategy_version == CHAMPION:
                for v in client.search_model_versions(f"name = '{MODEL_NAME}'"):
                    if v.tags.get("strategy_version") == CHAMPION:
                        champion_version = v.version
            continue
        spec = SPECS[strategy_version]
        with mlflow.start_run(run_name=strategy_version) as run:
            mlflow.log_params(spec["params"])
            mlflow.log_metrics(metrics[strategy_version])
            mlflow.set_tags({"strategy_version": strategy_version,
                             "collection": "prop_models"})
            contract = {
                "strategy_version": strategy_version,
                "description": spec["description"],
                "params": spec["params"],
                "artifacts": spec["artifacts"],
                "ledger_join": "mlb.prop_paper_bets.strategy_version",
                "implementation": "scripts/shop_batter_props.py",
                "calibration": "scripts/calibrate_prop_estimator.py "
                               "(held-out 2024+2025 starts)",
            }
            with tempfile.TemporaryDirectory() as td:
                cpath = Path(td) / "estimator_contract.json"
                cpath.write_text(json.dumps(contract, indent=1))
                mlflow.log_artifact(str(cpath), artifact_path="estimator")
            for rel in spec["artifacts"]:
                mlflow.log_artifact(str(repo / rel), artifact_path="estimator")
            version = client.create_model_version(
                name=MODEL_NAME,
                source=f"{run.info.artifact_uri}/estimator",
                run_id=run.info.run_id,
                tags={"strategy_version": strategy_version,
                      "collection": "prop_models"},
                description=spec["description"],
            )
            client.update_model_version(
                MODEL_NAME, version.version, description=spec["description"]
            )
            print(f"registered {MODEL_NAME} v{version.version} = {strategy_version}")
            if strategy_version == CHAMPION:
                champion_version = version.version

    if args.set_champion and champion_version is not None:
        client.set_registered_model_alias(MODEL_NAME, "champion", champion_version)
        print(f"@champion -> v{champion_version} ({CHAMPION})")


if __name__ == "__main__":
    main()
