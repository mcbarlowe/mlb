"""Resolve generated simulator inputs from the shared MLflow store.

Production passes the immutable simulator-input run pinned by the paired
outcome champion. Local development without a pinned run keeps using existing
files and only downloads the latest published input set when required files are
missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

SIM_ARTIFACT_KIND = "sim-inputs"

# Everything the simulator loads from models/sim. The calibration files are
# optional at runtime but published/bootstrapped together with the rest.
SIM_INPUT_FILES = [
    "pitch_mix.parquet",
    "pitch_locations.parquet",
    "base_out_tables.parquet",
    "team_bullpens.json",
    "sim_calibration.json",
    "win_calibration.json",
]

# Files without which the simulator cannot run at all.
REQUIRED_SIM_FILES = [
    "pitch_mix.parquet",
    "pitch_locations.parquet",
    "base_out_tables.parquet",
]

SIM_DIR = Path("models/sim")


def ensure_sim_artifacts(
    tracking_uri: str | None = None,
    sim_dir: Path = SIM_DIR,
    *,
    run_id: str | None = None,
) -> Path:
    """Ensure the selected simulator input release is available locally."""
    manifest_path = sim_dir / ".mlflow_sim_inputs.json"
    local_complete = all((sim_dir / name).exists() for name in REQUIRED_SIM_FILES)
    if local_complete and run_id is None:
        return sim_dir
    if local_complete and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if isinstance(manifest, dict) and manifest.get("run_id") == run_id:
            return sim_dir

    from mlflow.artifacts import download_artifacts
    from mlflow.tracking import MlflowClient

    from src.ml.mlflow_utils import (
        DEFAULT_MLFLOW_EXPERIMENT,
        resolve_mlflow_tracking_uri,
    )

    resolved_uri = resolve_mlflow_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=resolved_uri)
    if run_id is None:
        experiment = client.get_experiment_by_name(DEFAULT_MLFLOW_EXPERIMENT)
        if experiment is None:
            raise RuntimeError(
                f"Sim inputs missing under {sim_dir} and shared experiment "
                f"{DEFAULT_MLFLOW_EXPERIMENT!r} not found at {resolved_uri}"
            )
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=(
                "attributes.status = 'FINISHED' "
                f"and tags.artifact_kind = '{SIM_ARTIFACT_KIND}'"
            ),
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise RuntimeError(
                f"Sim inputs missing under {sim_dir} and no "
                f"{SIM_ARTIFACT_KIND!r} run was published"
            )
        run_id = runs[0].info.run_id
    else:
        run = client.get_run(run_id)
        if run.info.status != "FINISHED":
            raise RuntimeError(f"Pinned sim-inputs run {run_id} is not finished")
        if run.data.tags.get("artifact_kind") != SIM_ARTIFACT_KIND:
            raise RuntimeError(f"Pinned run {run_id} is not a sim-inputs release")

    sim_dir.parent.mkdir(parents=True, exist_ok=True)
    sim_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".sim-inputs-", dir=sim_dir.parent) as temp_dir:
        downloaded = Path(
            download_artifacts(
                run_id=run_id,
                artifact_path="sim",
                dst_path=temp_dir,
                tracking_uri=resolved_uri,
            )
        )
        for item in downloaded.iterdir():
            if item.is_file():
                copy2(item, sim_dir / item.name)

    missing = [name for name in REQUIRED_SIM_FILES if not (sim_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Shared sim inputs incomplete; missing {missing}")
    manifest_path.write_text(json.dumps({"run_id": run_id}, indent=2))
    print(f"Fetched sim inputs from MLflow run {run_id}")
    return sim_dir
