"""Register trained pitch prediction models in shared MLflow.

Imports the on-disk pitch type (LSTM+attention) and pitch-type-conditioned
location releases into the shared experiment as native MLflow pytorch models,
creating one immutable registered-model version per release. ``--set-champion``
advances each ``champion`` alias to the newly registered version.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlflow

from mlb.ml.mlflow_registry import (
    PITCH_CHAMPION_ALIAS,
    PITCH_LOCATION_SPEC,
    PITCH_TYPE_SPEC,
    load_pitch_location_release,
    load_pitch_type_release,
    log_registered_pitch_model,
)
from mlb.ml.mlflow_utils import DEFAULT_MLFLOW_EXPERIMENT, configure_mlflow
from mlb.ml.run_dirs import (
    resolve_location_run_dir,
    resolve_pitch_type_run_dir,
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
        "--models",
        type=str,
        choices=("both", "pitch-type", "location"),
        default="both",
        help="Which releases to register (default: both).",
    )
    parser.add_argument(
        "--set-champion",
        action="store_true",
        help="Point each champion alias at the newly registered version.",
    )
    return parser.parse_args()




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

    releases = []
    if args.models in ("both", "pitch-type"):
        releases.append((PITCH_TYPE_SPEC, "import-pitch-type", resolve_pitch_type_run_dir(args.pitch_type_run_dir), load_pitch_type_release))
    if args.models in ("both", "location"):
        releases.append((PITCH_LOCATION_SPEC, "import-pitch-location", resolve_location_run_dir(args.location_run_dir), load_pitch_location_release))
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
