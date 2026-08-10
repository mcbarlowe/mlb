"""Resolve generated sim input artifacts from the shared MLflow store.

Consumers call ``ensure_sim_artifacts`` before loading ``models/sim/*``:
when the local files are missing (fresh clone, cleaned checkout), the
latest ``sim-inputs`` run in the shared experiment is downloaded through
the MLflow artifact proxy. Local files always win when present, so the
generating machine (the iMac) keeps using its freshly built copies.
"""

from __future__ import annotations

from pathlib import Path

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
) -> Path:
    """Make sure the sim input files exist locally; pull from MLflow if not.

    Returns ``sim_dir``. Raises RuntimeError when required files are absent
    both locally and in the shared store.
    """
    if all((sim_dir / name).exists() for name in REQUIRED_SIM_FILES):
        return sim_dir

    from mlflow.artifacts import download_artifacts
    from mlflow.tracking import MlflowClient

    from src.ml.mlflow_utils import (
        DEFAULT_MLFLOW_EXPERIMENT,
        resolve_mlflow_tracking_uri,
    )

    resolved_uri = resolve_mlflow_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=resolved_uri)
    experiment = client.get_experiment_by_name(DEFAULT_MLFLOW_EXPERIMENT)
    if experiment is None:
        raise RuntimeError(
            f"Sim inputs missing under {sim_dir} and shared experiment "
            f"{DEFAULT_MLFLOW_EXPERIMENT!r} not found at {resolved_uri}"
        )
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.artifact_kind = '{SIM_ARTIFACT_KIND}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            f"Sim inputs missing under {sim_dir} and no {SIM_ARTIFACT_KIND!r} "
            f"run published in {DEFAULT_MLFLOW_EXPERIMENT!r}; run "
            "scripts/publish_sim_artifacts.py on the generating machine"
        )
    run = runs[0]
    sim_dir.mkdir(parents=True, exist_ok=True)
    downloaded = download_artifacts(
        run_id=run.info.run_id,
        artifact_path="sim",
        dst_path=str(sim_dir.parent / "_sim_download"),
        tracking_uri=resolved_uri,
    )
    for item in Path(downloaded).iterdir():
        item.replace(sim_dir / item.name)
    print(f"Fetched sim inputs from MLflow run {run.info.run_id}")
    missing = [name for name in REQUIRED_SIM_FILES if not (sim_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Shared sim inputs incomplete; missing {missing}")
    return sim_dir
