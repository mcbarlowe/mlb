#!/usr/bin/env python3
"""Fit a constrained market-anchored F5 totals probability blend.

The blend is:

    p_blend = p_market + lambda * (p_sim - p_market)

Default lambda search is constrained to [0, 1], so the model can only move from
market toward the simulation. Lambda=0 means use the market exactly; lambda=1
means use the raw simulation. Chronological holdout metrics decide whether the
simulation adds incremental signal.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.f5_residual_calibration import (
    DEFAULT_TRAIN_FRACTION,
    EPS,
    F5ResidualRow,
    load_f5_market_rows,
    load_sim_probabilities_from_report,
    merge_sim_probabilities,
    probability_metrics,
    split_rows,
)

Objective = Literal["log_loss", "brier"]
DEFAULT_LAMBDA_MIN = 0.0
DEFAULT_LAMBDA_MAX = 1.0
DEFAULT_LAMBDA_STEPS = 101
DEFAULT_MIN_TEST_ROWS = 500


def _parse_ints(value: str) -> tuple[int, ...]:
    try:
        seasons = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not seasons:
        raise argparse.ArgumentTypeError("at least one season is required")
    return seasons


def blend_probability(row: F5ResidualRow, lambda_value: float) -> float:
    sim_probability = _required_sim_probability(row)
    probability = row.market_prob_over + lambda_value * (
        sim_probability - row.market_prob_over
    )
    return min(max(probability, EPS), 1.0 - EPS)


def lambda_grid(lambda_min: float, lambda_max: float, steps: int) -> list[float]:
    if steps < 2:
        raise ValueError("lambda steps must be at least 2")
    if not math.isfinite(lambda_min) or not math.isfinite(lambda_max):
        raise ValueError("lambda bounds must be finite")
    if lambda_min > lambda_max:
        raise ValueError("lambda_min must be <= lambda_max")
    step = (lambda_max - lambda_min) / (steps - 1)
    return [lambda_min + index * step for index in range(steps)]


def fit_lambda(
    rows: Sequence[F5ResidualRow],
    *,
    objective: Objective = "log_loss",
    lambda_min: float = DEFAULT_LAMBDA_MIN,
    lambda_max: float = DEFAULT_LAMBDA_MAX,
    steps: int = DEFAULT_LAMBDA_STEPS,
) -> tuple[float, dict[str, float]]:
    if not rows:
        raise ValueError("at least one row with sim probabilities is required")
    outcomes = [row.actual_over for row in rows]
    best_lambda: float | None = None
    best_score: float | None = None
    scores: dict[str, float] = {}
    for candidate in lambda_grid(lambda_min, lambda_max, steps):
        probabilities = [blend_probability(row, candidate) for row in rows]
        metrics = probability_metrics(probabilities, outcomes)
        score = metrics.log_loss if objective == "log_loss" else metrics.brier
        key = _lambda_key(candidate)
        scores[key] = score
        if best_score is None or score < best_score - 1e-12:
            best_lambda = candidate
            best_score = score
    if best_lambda is None:
        raise RuntimeError("failed to fit lambda")
    return best_lambda, scores


def build_report(
    rows: Sequence[F5ResidualRow],
    *,
    seasons: Sequence[int],
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    line_type: str = "open",
    sim_report_json: str | None = None,
    objective: Objective = "log_loss",
    lambda_min: float = DEFAULT_LAMBDA_MIN,
    lambda_max: float = DEFAULT_LAMBDA_MAX,
    lambda_steps: int = DEFAULT_LAMBDA_STEPS,
    min_test_rows: int = DEFAULT_MIN_TEST_ROWS,
) -> dict[str, Any]:
    sim_rows = [row for row in rows if row.sim_prob_over is not None]
    if not sim_rows:
        raise ValueError("no rows have sim probabilities; pass --sim-report-json")
    train_rows, test_rows = split_rows(sim_rows, train_fraction=train_fraction)
    selected_lambda, train_scores = fit_lambda(
        train_rows,
        objective=objective,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        steps=lambda_steps,
    )
    metrics = {
        "train": _metric_block(train_rows, selected_lambda),
        "test": _metric_block(test_rows, selected_lambda),
    }
    return {
        "report_type": "f5_market_anchor_blend",
        "inputs": {
            "seasons": list(seasons),
            "line_type": line_type,
            "sim_report_json": sim_report_json,
        },
        "rows": {
            "market": len(rows),
            "sim_merged": len(sim_rows),
            "pushes_dropped": True,
        },
        "split": {
            "method": "chronological",
            "train_fraction": train_fraction,
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
        },
        "lambda_grid": {
            "objective": objective,
            "min": lambda_min,
            "max": lambda_max,
            "steps": lambda_steps,
            "train_scores": train_scores,
        },
        "selected_lambda": selected_lambda,
        "metrics": metrics,
        "probability_gate": decide_probability_gate(
            market_metrics=metrics["test"]["market"],
            blended_metrics=metrics["test"]["blended"],
            selected_lambda=selected_lambda,
            test_rows=len(test_rows),
            min_test_rows=min_test_rows,
        ),
    }


def decide_probability_gate(
    *,
    market_metrics: Mapping[str, float | int | None],
    blended_metrics: Mapping[str, float | int | None],
    selected_lambda: float,
    test_rows: int,
    min_test_rows: int,
) -> dict[str, Any]:
    brier_improvement = _metric_float(market_metrics, "brier") - _metric_float(
        blended_metrics, "brier"
    )
    log_loss_improvement = _metric_float(
        market_metrics, "log_loss"
    ) - _metric_float(blended_metrics, "log_loss")
    checks = {
        "enough_heldout_sample": test_rows >= min_test_rows,
        "lambda_moves_toward_sim": selected_lambda > 0.0,
        "test_brier_improves_vs_market": brier_improvement > 0.0,
        "test_log_loss_improves_vs_market": log_loss_improvement > 0.0,
    }
    status = "open" if all(checks.values()) else "closed"
    reason = (
        "Probability gate open: market-anchored blend beat market Brier/log loss on enough holdout rows"
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
            "brier_improvement_vs_market": brier_improvement,
            "log_loss_improvement_vs_market": log_loss_improvement,
        },
    }


def _metric_block(rows: Sequence[F5ResidualRow], lambda_value: float) -> dict[str, dict[str, float | int | None]]:
    outcomes = [row.actual_over for row in rows]
    market = [row.market_prob_over for row in rows]
    sim = [_required_sim_probability(row) for row in rows]
    blended = [blend_probability(row, lambda_value) for row in rows]
    return {
        "market": probability_metrics(market, outcomes).as_dict(),
        "sim": probability_metrics(sim, outcomes).as_dict(),
        "blended": probability_metrics(blended, outcomes).as_dict(),
    }


def _required_sim_probability(row: F5ResidualRow) -> float:
    if row.sim_prob_over is None:
        raise ValueError(f"game_pk={row.game_pk} is missing sim_prob_over")
    return row.sim_prob_over


def _metric_float(metrics: Mapping[str, float | int | None], key: str) -> float:
    value = metrics[key]
    if value is None:
        raise ValueError(f"metric {key!r} is None")
    return float(value)


def _lambda_key(value: float) -> str:
    return f"{value:.6g}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=_parse_ints, default=(2025,))
    parser.add_argument("--line-type", choices=("open", "current", "close"), default="open")
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--sim-report-json", type=Path, required=True)
    parser.add_argument("--objective", choices=("log_loss", "brier"), default="log_loss")
    parser.add_argument("--lambda-min", type=float, default=DEFAULT_LAMBDA_MIN)
    parser.add_argument("--lambda-max", type=float, default=DEFAULT_LAMBDA_MAX)
    parser.add_argument("--lambda-steps", type=int, default=DEFAULT_LAMBDA_STEPS)
    parser.add_argument("--min-test-rows", type=int, default=DEFAULT_MIN_TEST_ROWS)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = load_f5_market_rows(args.seasons, line_type=args.line_type)
    rows = merge_sim_probabilities(rows, load_sim_probabilities_from_report(args.sim_report_json))
    report = build_report(
        rows,
        seasons=args.seasons,
        train_fraction=args.train_fraction,
        line_type=args.line_type,
        sim_report_json=str(args.sim_report_json),
        objective=args.objective,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        lambda_steps=args.lambda_steps,
        min_test_rows=args.min_test_rows,
    )
    output = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    if args.out_json is None:
        print(output, end="")
        return
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(output)
    print(f"wrote F5 market-anchor blend report to {args.out_json}")


if __name__ == "__main__":
    main()
