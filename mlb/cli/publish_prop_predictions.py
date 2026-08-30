#!/usr/bin/env python3
"""Publish an odds-free batch of MLB batter-prop probabilities as JSON."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from mlb.data_contracts.prop_predictions import (
    CONTRACT_VERSION,
    build_prop_prediction_artifact,
    resolve_adjustments,
)
from mlb.paths import state_root


def _read_request(path: Path, prediction_date: date) -> dict[str, object]:
    try:
        payload: Any = json.loads(path.read_text())
    except OSError as exc:
        raise ValueError(f"Could not read request JSON {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid request JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Prop prediction request must be a JSON object")
    requested_date = payload.get("prediction_date")
    if requested_date not in (None, prediction_date.isoformat()):
        raise ValueError(
            f"Request prediction_date {requested_date!r} does not match --date "
            f"{prediction_date.isoformat()!r}"
        )
    payload["prediction_date"] = prediction_date.isoformat()
    payload.setdefault("contract_version", CONTRACT_VERSION)
    return payload


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=date.fromisoformat)
    parser.add_argument("--request-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--model-source", choices=("registry", "local"), default="registry")
    parser.add_argument("--mlflow-tracking-uri")
    parser.add_argument("--aging-curves", type=Path)
    parser.add_argument("--park-factors", type=Path)
    parser.add_argument("--shrink-k", type=float, default=50.0)
    parser.add_argument("--recency-half-life", type=float, default=400.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        request = _read_request(args.request_json, args.date)
        curves, parks, provenance = resolve_adjustments(
            repo=state_root(),
            model_source=args.model_source,
            tracking_uri=args.mlflow_tracking_uri,
            aging_curves_path=args.aging_curves,
            park_factors_path=args.park_factors,
        )
        artifact = build_prop_prediction_artifact(
            request,
            prediction_date=args.date,
            aging_curves=curves,
            park_factors=parks,
            shrink_k=args.shrink_k,
            recency_half_life=args.recency_half_life,
            model_provenance=provenance,
        )
        _atomic_write_json(args.output_json, artifact)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"published {len(artifact['predictions'])} prop predictions to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
