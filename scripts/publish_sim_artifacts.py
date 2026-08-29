"""Publish the generated sim input artifacts to the shared MLflow store.

The simulator's inputs (pitch mixes, location pools, base-out tables, team
bullpen hands, calibrations) are generated locally on the iMac, which holds
Postgres and the raw feed archive. Publishing them as a tagged run in the
shared experiment lets any machine bootstrap through the MLflow artifact
proxy instead of copying files by hand.

    uv run python scripts/publish_sim_artifacts.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mlb.ml.mlflow_utils import (
    DEFAULT_MLFLOW_EXPERIMENT,
    configure_mlflow,
    resolve_mlflow_tracking_uri,
)
from mlb.sim.artifacts import SIM_ARTIFACT_KIND, SIM_INPUT_FILES

SIM_DIR = Path("models/sim")


def main() -> None:
    import mlflow

    parser = argparse.ArgumentParser(description="Publish sim inputs to MLflow.")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument(
        "--mlflow-experiment", type=str, default=DEFAULT_MLFLOW_EXPERIMENT
    )
    args = parser.parse_args()

    present = [name for name in SIM_INPUT_FILES if (SIM_DIR / name).exists()]
    missing = sorted(set(SIM_INPUT_FILES) - set(present))
    if not present:
        raise SystemExit(f"No sim inputs under {SIM_DIR}; generate them first")

    tracking_uri = resolve_mlflow_tracking_uri(args.mlflow_tracking_uri)
    configure_mlflow(
        args.mlflow_experiment, tracking_uri, require_tracking_uri=True
    )
    run_name = f"sim-inputs-{time.strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"artifact_kind": SIM_ARTIFACT_KIND})
        for name in present:
            mlflow.log_artifact(str(SIM_DIR / name), artifact_path="sim")
        mlflow.log_params({"files": ", ".join(present)})
    print(f"Published {len(present)} sim inputs as {run_name} ({run.info.run_id})")
    if missing:
        print(f"Not present (skipped): {', '.join(missing)}")


if __name__ == "__main__":
    main()
