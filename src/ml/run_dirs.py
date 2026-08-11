"""Resolve local pitch prediction training output directories.

Shared by the MLflow import script and the live pipeline so "auto" means the
same release everywhere: the newest complete non-quick pitch type run and the
newest complete location run.
"""

from __future__ import annotations

import json
from pathlib import Path

PITCH_TYPE_FILES = ("final_model.pt", "feature_engine.json", "results.json")
LOCATION_FILES = (
    "pitch_type_location_model.pt",
    "config.json",
    "test_metrics.json",
    "feature_engine.pt",
)


def _explicit_dir(raw: str) -> Path:
    path = Path(raw)
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def resolve_pitch_type_run_dir(raw: str) -> Path:
    if raw != "auto":
        return _explicit_dir(raw)
    for run_dir in sorted(Path("models/pitch_type").glob("run_*"), reverse=True):
        if not all((run_dir / name).is_file() for name in PITCH_TYPE_FILES):
            continue
        results = json.loads((run_dir / "results.json").read_text())
        if results.get("args", {}).get("quick"):
            continue
        return run_dir
    raise FileNotFoundError(
        "No complete non-quick models/pitch_type/run_* directory found; "
        "pass an explicit pitch type run directory"
    )


def resolve_location_run_dir(raw: str) -> Path:
    if raw != "auto":
        return _explicit_dir(raw)
    for run_dir in sorted(
        Path("models/pitch_type_location").glob("pitch_type_location_*"),
        reverse=True,
    ):
        if all((run_dir / name).is_file() for name in LOCATION_FILES):
            return run_dir
    raise FileNotFoundError(
        "No complete models/pitch_type_location/pitch_type_location_* "
        "directory found; pass an explicit location run directory"
    )
