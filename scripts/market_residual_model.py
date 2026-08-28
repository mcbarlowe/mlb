"""Leak-free residual market model diagnostics for totals logs.

Reads existing ``sim_totals_eval`` log lines of the form::

    <game_pk> pt=<point> sim_over=<p> mkt_over=<p> actual=<runs>

Pushes are dropped. The residual model is fit only on ``--train-log`` rows and
then evaluated on ``--eval-log`` rows. Season ordering is inferred from log file
names such as ``sim_totals_2024_v3.log`` and enforced as train season(s) strictly
less than the eval season.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-9
FEATURE_NAMES = ("market_logit", "sim_minus_market_logit")
_FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
LINE_RE = re.compile(
    rf"(?P<game_pk>\d+)\s+pt=(?P<point>{_FLOAT_RE})\s+"
    rf"sim_over=(?P<sim>{_FLOAT_RE})\s+"
    rf"mkt_over=(?P<mkt>{_FLOAT_RE})\s+actual=(?P<actual>{_FLOAT_RE})"
)
SEASON_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


@dataclass(frozen=True)
class MarketResidualRow:
    game_pk: int
    season: int
    point: float
    sim_over: float
    market_over: float
    actual_total: float
    actual_over: int


@dataclass(frozen=True)
class ProbabilityMetrics:
    n: int
    brier: float
    log_loss: float
    mean_probability: float
    outcome_rate: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "brier": self.brier,
            "log_loss": self.log_loss,
            "mean_probability": self.mean_probability,
            "outcome_rate": self.outcome_rate,
        }


def infer_season_from_path(path: str | Path) -> int:
    """Infer a season from a log path containing a four-digit MLB season."""
    match = SEASON_RE.search(str(path))
    if match is None:
        raise ValueError(f"Could not infer season from log path: {path}")
    return int(match.group(1))


def parse_log_spec(spec: str) -> tuple[int | None, Path]:
    """Parse either ``PATH`` or the backward-compatible ``SEASON:PATH`` form."""
    season_match = re.match(r"^(20\d{2}):(.*)$", spec)
    if season_match is None:
        return None, Path(spec)
    return int(season_match.group(1)), Path(season_match.group(2))


def parse_totals_eval_lines(
    lines: Iterable[str], *, season: int
) -> list[MarketResidualRow]:
    """Parse totals-eval log lines, dropping pushes and ignoring unrelated lines."""
    rows: list[MarketResidualRow] = []
    for line_number, line in enumerate(lines, start=1):
        match = LINE_RE.search(line)
        if match is None:
            continue
        point = float(match.group("point"))
        sim_over = float(match.group("sim"))
        market_over = float(match.group("mkt"))
        actual_total = float(match.group("actual"))
        _validate_probability(sim_over, line_number=line_number, field="sim_over")
        _validate_probability(market_over, line_number=line_number, field="mkt_over")
        if math.isclose(actual_total, point):
            continue
        rows.append(
            MarketResidualRow(
                game_pk=int(match.group("game_pk")),
                season=season,
                point=point,
                sim_over=sim_over,
                market_over=market_over,
                actual_total=actual_total,
                actual_over=1 if actual_total > point else 0,
            )
        )
    return rows


def parse_totals_eval_log(path: str | Path, *, season: int | None = None) -> list[MarketResidualRow]:
    """Read and parse one totals-eval log file."""
    log_path = Path(path)
    resolved_season = infer_season_from_path(log_path) if season is None else season
    return parse_totals_eval_lines(log_path.read_text().splitlines(), season=resolved_season)


def read_log_specs(specs: Sequence[str]) -> tuple[list[MarketResidualRow], list[str]]:
    """Read repeated CLI log specs into one row list and normalized path list."""
    rows: list[MarketResidualRow] = []
    paths: list[str] = []
    for spec in specs:
        season, path = parse_log_spec(spec)
        rows.extend(parse_totals_eval_log(path, season=season))
        paths.append(str(path))
    return rows, paths


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean Brier score for binary probabilities."""
    probs, y = _aligned_arrays(probabilities, outcomes)
    return float(np.mean((probs - y) ** 2))


def log_loss_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean binary log loss for probabilities, clipped for numerical safety."""
    probs, y = _aligned_arrays(probabilities, outcomes)
    clipped = np.clip(probs, EPS, 1.0 - EPS)
    return float(np.mean(-(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))))


def probability_metrics(
    probabilities: Sequence[float], outcomes: Sequence[int]
) -> ProbabilityMetrics:
    probs, y = _aligned_arrays(probabilities, outcomes)
    return ProbabilityMetrics(
        n=len(probs),
        brier=float(np.mean((probs - y) ** 2)),
        log_loss=log_loss_score(probabilities, outcomes),
        mean_probability=float(np.mean(probs)),
        outcome_rate=float(np.mean(y)),
    )


def fit_residual_model(rows: Sequence[MarketResidualRow]) -> LogisticRegression:
    """Fit a deterministic market-logit plus sim-residual logistic model."""
    if not rows:
        raise ValueError("at least one training row is required")
    outcomes = np.array([row.actual_over for row in rows], dtype=int)
    if len(set(outcomes.tolist())) < 2:
        raise ValueError("training rows must contain both over and under outcomes")
    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    model.fit(_feature_matrix(rows), outcomes)
    return model


def predict_residual_probabilities(
    model: LogisticRegression, rows: Sequence[MarketResidualRow]
) -> list[float]:
    """Predict residual-model probabilities for ``actual_total > point``."""
    if not rows:
        return []
    return [float(prob) for prob in model.predict_proba(_feature_matrix(rows))[:, 1]]


def evaluate_residual_model(
    model: LogisticRegression,
    rows: Sequence[MarketResidualRow],
) -> dict[str, Any]:
    """Evaluate fitted residual-model probabilities against raw market probabilities."""
    if not rows:
        raise ValueError("at least one evaluation row is required")
    outcomes = [row.actual_over for row in rows]
    model_probs = predict_residual_probabilities(model, rows)
    market_probs = [row.market_over for row in rows]
    model_metrics = probability_metrics(model_probs, outcomes)
    market_metrics = probability_metrics(market_probs, outcomes)
    return {
        "model": model_metrics.as_dict(),
        "market": market_metrics.as_dict(),
        "gaps": metric_gaps(model_metrics, market_metrics),
    }


def metric_gaps(
    model_metrics: ProbabilityMetrics, market_metrics: ProbabilityMetrics
) -> dict[str, float]:
    """Return signed model-minus-market gaps and positive improvement aliases."""
    brier_gap = model_metrics.brier - market_metrics.brier
    log_loss_gap = model_metrics.log_loss - market_metrics.log_loss
    return {
        "brier_model_minus_market": brier_gap,
        "log_loss_model_minus_market": log_loss_gap,
        "brier_improvement": -brier_gap,
        "log_loss_improvement": -log_loss_gap,
    }



def model_coefficients(model: LogisticRegression) -> dict[str, float | dict[str, float]]:
    """Return JSON-safe fitted coefficient values."""
    intercept = float(np.ravel(model.intercept_)[0])
    coefs = np.ravel(model.coef_)
    return {
        "intercept": intercept,
        "features": {name: float(coef) for name, coef in zip(FEATURE_NAMES, coefs)},
    }


def validate_leak_free_split(
    train_rows: Sequence[MarketResidualRow], eval_rows: Sequence[MarketResidualRow]
) -> None:
    """Require every train season to be strictly earlier than every eval season."""
    if not train_rows:
        raise ValueError("no usable non-push rows parsed from train logs")
    if not eval_rows:
        raise ValueError("no usable non-push rows parsed from eval log")
    train_seasons = {row.season for row in train_rows}
    eval_seasons = {row.season for row in eval_rows}
    if max(train_seasons) >= min(eval_seasons):
        raise ValueError(
            "leak-free split violation: train seasons must be strictly earlier "
            f"than eval seasons (train={sorted(train_seasons)}, "
            f"eval={sorted(eval_seasons)})"
        )



def build_report(
    train_rows: Sequence[MarketResidualRow],
    eval_rows: Sequence[MarketResidualRow],
    *,
    train_logs: Sequence[str] = (),
    eval_log: str | None = None,
) -> dict[str, Any]:
    """Fit on train rows, evaluate held-out rows, and build a JSON-safe report."""
    validate_leak_free_split(train_rows, eval_rows)
    model = fit_residual_model(train_rows)
    train_seasons = sorted({row.season for row in train_rows})
    eval_seasons = sorted({row.season for row in eval_rows})
    return {
        "model_type": "sklearn.linear_model.LogisticRegression",
        "feature_names": list(FEATURE_NAMES),
        "train": {
            "logs": list(train_logs),
            "seasons": train_seasons,
            "rows": len(train_rows),
            "metrics": evaluate_residual_model(model, train_rows),
        },
        "eval": {
            "log": eval_log,
            "seasons": eval_seasons,
            "rows": len(eval_rows),
        },
        "metrics": evaluate_residual_model(model, eval_rows),
        "coefficients": model_coefficients(model),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-log",
        action="append",
        required=True,
        help="repeatable path, or optional SEASON:PATH, to training totals log",
    )
    parser.add_argument("--eval-log", required=True, help="held-out totals log path")
    parser.add_argument("--out-json", required=True, help="path for JSON report")
    args = parser.parse_args(argv)

    train_rows, train_paths = read_log_specs(args.train_log)
    eval_season, eval_path = parse_log_spec(args.eval_log)
    eval_rows = parse_totals_eval_log(eval_path, season=eval_season)
    report = build_report(
        train_rows,
        eval_rows,
        train_logs=train_paths,
        eval_log=str(eval_path),
    )
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _validate_probability(value: float, *, line_number: int, field: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(
            f"{field} on line {line_number} must be a finite probability in [0, 1]"
        )


def _aligned_arrays(
    probabilities: Sequence[float], outcomes: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have the same length")
    if len(probabilities) == 0:
        raise ValueError("at least one probability/outcome pair is required")
    probs = np.array(probabilities, dtype=float)
    y = np.array(outcomes, dtype=float)
    if np.any(~np.isfinite(probs)) or np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if np.any((y != 0.0) & (y != 1.0)):
        raise ValueError("outcomes must be binary 0/1 values")
    return probs, y


def _feature_matrix(rows: Sequence[MarketResidualRow]) -> np.ndarray:
    market = np.array([row.market_over for row in rows], dtype=float)
    sim = np.array([row.sim_over for row in rows], dtype=float)
    market_logit = _logit_array(market)
    sim_logit = _logit_array(sim)
    residual_logit = sim_logit - market_logit
    return np.column_stack([market_logit, residual_logit])


def _logit_array(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


if __name__ == "__main__":
    main()
