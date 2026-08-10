"""Register trained pitch prediction models in shared MLflow.

Imports the on-disk pitch type (LSTM+attention) and pitch-type-conditioned
location releases into the shared experiment as native MLflow pytorch models,
creating one immutable registered-model version per release. ``--set-champion``
advances each ``champion`` alias to the newly registered version.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlflow

from src.ml.mlflow_registry import (
    PITCH_CHAMPION_ALIAS,
    PITCH_LOCATION_SPEC,
    PITCH_TYPE_SPEC,
    load_pitch_location_release,
    load_pitch_type_release,
    log_registered_pitch_model,
)
from src.ml.mlflow_utils import DEFAULT_MLFLOW_EXPERIMENT, configure_mlflow

PITCH_TYPE_FILES = ("final_model.pt", "feature_engine.json", "results.json")
LOCATION_FILES = (
    "pitch_type_location_model.pt",
    "config.json",
    "test_metrics.json",
    "feature_engine.pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register trained pitch prediction models in shared MLflow."
    )
    parser.add_argument(
        "--pitch-type-run-dir",
        type=str,
        default="auto",
        help=(
            "Pitch type training output directory (default: newest complete "
            "non-quick models/pitch_type/run_*)."
        ),
    )
    parser.add_argument(
        "--location-run-dir",
        type=str,
        default="auto",
        help=(
            "Location model training output directory (default: newest "
            "complete models/pitch_type_location/pitch_type_location_*)."
        ),
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=DEFAULT_MLFLOW_EXPERIMENT,
    )
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument(
        "--set-champion",
        action="store_true",
        help="Point each champion alias at the newly registered version.",
    )
    return parser.parse_args()


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
        "pass --pitch-type-run-dir explicitly"
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
        "directory found; pass --location-run-dir explicitly"
    )


def _run_stamp(run_dir: Path) -> str:
    return run_dir.name.removeprefix("pitch_type_location_").removeprefix("run_")


def main() -> None:
    args = parse_args()
    tracking_uri = configure_mlflow(
        args.mlflow_experiment,
        args.mlflow_tracking_uri,
        require_tracking_uri=True,
    )
    print(f"MLflow tracking URI: {tracking_uri}")

    releases = (
        (PITCH_TYPE_SPEC, "import-pitch-type", resolve_pitch_type_run_dir(args.pitch_type_run_dir), load_pitch_type_release),
        (PITCH_LOCATION_SPEC, "import-pitch-location", resolve_location_run_dir(args.location_run_dir), load_pitch_location_release),
    )
    for spec, run_prefix, run_dir, loader in releases:
        print(f"\nImporting {spec.model_family} from {run_dir}")
        loaded = loader(run_dir)
        with mlflow.start_run(run_name=f"{run_prefix}-{_run_stamp(run_dir)}"):
            mlflow.set_tags(
                {
                    "source": "local-import",
                    "imported_from_path": str(run_dir),
                }
            )
            registration = log_registered_pitch_model(
                spec=spec,
                loaded=loaded,
                set_champion=args.set_champion,
            )
        alias_note = (
            f" (alias @{PITCH_CHAMPION_ALIAS})" if args.set_champion else ""
        )
        print(
            f"Registered {registration.registered_model_name} "
            f"v{registration.version}{alias_note} from run {registration.run_id}"
        )


if __name__ == "__main__":
    main()
