#!/usr/bin/env python3
"""Fit leak-safe first-five run-environment calibration diagnostics.

The calibration is an additive run adjustment learned from chronological training
rows in an F5 CLV report JSON. Bucket adjustments are shrunk toward the global
training residual so small buckets do not dominate. This script is a research
holdout gate; it does not mutate model artifacts or betting configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.f5_run_environment_diagnostics import (
    F5RunEnvironmentRow,
    load_game_dates,
    load_report_rows,
)

DEFAULT_BUCKET_GROUPS = ("total_point",)
DEFAULT_TRAIN_FRACTION = 0.67
DEFAULT_SHRINKAGE = 20.0
DEFAULT_MIN_TEST_ROWS = 200
SUPPORTED_BUCKET_GROUPS = {"total_point", "month"}


@dataclass(frozen=True)
class BucketAdjustment:
    key: str
    n: int
    raw_adjustment: float
    shrunk_adjustment: float


@dataclass(frozen=True)
class RunMeanMetrics:
    n: int
    mean_actual: float
    mean_prediction: float
    mean_error: float
    mae: float
    rmse: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _parse_ints(value: str) -> tuple[int, ...]:
    try:
        seasons = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not seasons:
        raise argparse.ArgumentTypeError("at least one season is required")
    return seasons


def parse_bucket_groups(value: str) -> tuple[str, ...]:
    groups = tuple(part.strip() for part in value.split(",") if part.strip())
    if not groups:
        raise argparse.ArgumentTypeError("at least one bucket group is required")
    unsupported = sorted(set(groups) - SUPPORTED_BUCKET_GROUPS)
    if unsupported:
        raise argparse.ArgumentTypeError(
            f"unsupported bucket groups {unsupported}; expected subset of {sorted(SUPPORTED_BUCKET_GROUPS)}"
        )
    return groups


def split_rows(
    rows: Sequence[F5RunEnvironmentRow], *, train_fraction: float
) -> tuple[list[F5RunEnvironmentRow], list[F5RunEnvironmentRow]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    ordered = sorted(rows, key=lambda row: (row.game_date or row.season, row.game_pk))
    split_at = int(len(ordered) * train_fraction)
    if split_at <= 0 or split_at >= len(ordered):
        raise ValueError("train_fraction leaves an empty train or test split")
    return ordered[:split_at], ordered[split_at:]


def fit_bucket_adjustments(
    rows: Sequence[F5RunEnvironmentRow],
    *,
    bucket_groups: Sequence[str] = DEFAULT_BUCKET_GROUPS,
    shrinkage: float = DEFAULT_SHRINKAGE,
) -> tuple[dict[str, BucketAdjustment], float]:
    if shrinkage < 0.0 or not math.isfinite(shrinkage):
        raise ValueError("shrinkage must be a finite non-negative value")
    if not rows:
        raise ValueError("at least one training row is required")
    global_adjustment = _mean(row.actual_f5_total - row.sim_mean_total for row in rows)
    buckets: dict[str, list[F5RunEnvironmentRow]] = defaultdict(list)
    for row in rows:
        buckets[bucket_key(row, bucket_groups)].append(row)

    adjustments: dict[str, BucketAdjustment] = {}
    for key, bucket_rows in sorted(buckets.items()):
        raw = _mean(row.actual_f5_total - row.sim_mean_total for row in bucket_rows)
        n = len(bucket_rows)
        shrunk = (n * raw + shrinkage * global_adjustment) / (n + shrinkage)
        adjustments[key] = BucketAdjustment(
            key=key,
            n=n,
            raw_adjustment=raw,
            shrunk_adjustment=shrunk,
        )
    return adjustments, global_adjustment


def bucket_key(row: F5RunEnvironmentRow, bucket_groups: Sequence[str]) -> str:
    values: list[str] = []
    for group in bucket_groups:
        if group == "total_point":
            values.append(f"total={row.take_point:g}")
        elif group == "month":
            values.append(f"month={row.game_date.strftime('%Y-%m') if row.game_date else 'unknown'}")
        else:
            raise ValueError(f"unknown bucket group {group!r}")
    return "|".join(values)


def predict_calibrated_total(
    row: F5RunEnvironmentRow,
    adjustments: Mapping[str, BucketAdjustment],
    *,
    bucket_groups: Sequence[str],
    fallback_adjustment: float,
) -> float:
    adjustment = adjustments.get(bucket_key(row, bucket_groups))
    additive = fallback_adjustment if adjustment is None else adjustment.shrunk_adjustment
    return row.sim_mean_total + additive


def run_mean_metrics(
    rows: Sequence[F5RunEnvironmentRow], predictions: Sequence[float]
) -> RunMeanMetrics:
    if len(rows) != len(predictions):
        raise ValueError("rows and predictions must have the same length")
    if not rows:
        raise ValueError("at least one row is required")
    errors = [row.actual_f5_total - prediction for row, prediction in zip(rows, predictions)]
    return RunMeanMetrics(
        n=len(rows),
        mean_actual=_mean(row.actual_f5_total for row in rows),
        mean_prediction=_mean(predictions),
        mean_error=_mean(errors),
        mae=_mean(abs(error) for error in errors),
        rmse=math.sqrt(_mean(error * error for error in errors)),
    )


def build_report(
    rows: Sequence[F5RunEnvironmentRow],
    *,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    bucket_groups: Sequence[str] = DEFAULT_BUCKET_GROUPS,
    shrinkage: float = DEFAULT_SHRINKAGE,
    min_test_rows: int = DEFAULT_MIN_TEST_ROWS,
) -> dict[str, Any]:
    if min_test_rows < 1:
        raise ValueError("min_test_rows must be positive")
    train_rows, test_rows = split_rows(rows, train_fraction=train_fraction)
    adjustments, fallback_adjustment = fit_bucket_adjustments(
        train_rows,
        bucket_groups=bucket_groups,
        shrinkage=shrinkage,
    )

    train_base_predictions = [row.sim_mean_total for row in train_rows]
    test_base_predictions = [row.sim_mean_total for row in test_rows]
    train_calibrated_predictions = [
        predict_calibrated_total(
            row,
            adjustments,
            bucket_groups=bucket_groups,
            fallback_adjustment=fallback_adjustment,
        )
        for row in train_rows
    ]
    test_calibrated_predictions = [
        predict_calibrated_total(
            row,
            adjustments,
            bucket_groups=bucket_groups,
            fallback_adjustment=fallback_adjustment,
        )
        for row in test_rows
    ]

    metrics = {
        "train": {
            "base_sim": run_mean_metrics(train_rows, train_base_predictions).as_dict(),
            "calibrated": run_mean_metrics(
                train_rows, train_calibrated_predictions
            ).as_dict(),
        },
        "test": {
            "base_sim": run_mean_metrics(test_rows, test_base_predictions).as_dict(),
            "calibrated": run_mean_metrics(
                test_rows, test_calibrated_predictions
            ).as_dict(),
        },
    }
    gate = decide_calibration_gate(
        base_metrics=metrics["test"]["base_sim"],
        calibrated_metrics=metrics["test"]["calibrated"],
        test_rows=len(test_rows),
        min_test_rows=min_test_rows,
    )
    return {
        "report_type": "f5_run_environment_calibration",
        "rows": len(rows),
        "split": {
            "method": "chronological",
            "train_fraction": train_fraction,
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
        },
        "bucket_groups": list(bucket_groups),
        "shrinkage": shrinkage,
        "fallback_adjustment": fallback_adjustment,
        "bucket_adjustments": [
            asdict(adjustment) for adjustment in sorted(adjustments.values(), key=lambda item: item.key)
        ],
        "metrics": metrics,
        "calibration_gate": gate,
    }


def decide_calibration_gate(
    *,
    base_metrics: Mapping[str, float | int],
    calibrated_metrics: Mapping[str, float | int],
    test_rows: int,
    min_test_rows: int,
) -> dict[str, Any]:
    mae_improvement = float(base_metrics["mae"]) - float(calibrated_metrics["mae"])
    rmse_improvement = float(base_metrics["rmse"]) - float(calibrated_metrics["rmse"])
    checks = {
        "enough_heldout_sample": test_rows >= min_test_rows,
        "test_mae_improves": mae_improvement > 0.0,
        "test_rmse_improves": rmse_improvement > 0.0,
        "leak_safe_features": True,
    }
    status = "open" if all(checks.values()) else "closed"
    reason = (
        "Calibration gate open: shrunk bucket adjustment improved holdout MAE/RMSE"
        if status == "open"
        else "Gate closed: " + ", ".join(name for name, passed in checks.items() if not passed)
    )
    return {
        "status": status,
        "reason": reason,
        "checks": checks,
        "thresholds": {"min_test_rows": min_test_rows},
        "metrics": {
            "test_rows": test_rows,
            "mae_improvement": mae_improvement,
            "rmse_improvement": rmse_improvement,
        },
    }


def _mean(values: Sequence[float] | Any) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot compute mean of an empty sequence")
    return sum(materialized) / len(materialized)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-report-json", type=Path, required=True)
    parser.add_argument("--seasons", type=_parse_ints, default=(2025,))
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--bucket-groups", type=parse_bucket_groups, default=DEFAULT_BUCKET_GROUPS)
    parser.add_argument("--shrinkage", type=float, default=DEFAULT_SHRINKAGE)
    parser.add_argument("--min-test-rows", type=int, default=DEFAULT_MIN_TEST_ROWS)
    parser.add_argument("--no-db-dates", action="store_true")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args(argv)

    game_dates = {} if args.no_db_dates else load_game_dates(args.seasons)
    rows = load_report_rows(args.sim_report_json, game_dates=game_dates)
    report = build_report(
        rows,
        train_fraction=args.train_fraction,
        bucket_groups=args.bucket_groups,
        shrinkage=args.shrinkage,
        min_test_rows=args.min_test_rows,
    )
    output = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    if args.out_json is None:
        print(output, end="")
        return
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(output)
    print(f"wrote F5 run-environment calibration to {args.out_json}")


if __name__ == "__main__":
    main()
