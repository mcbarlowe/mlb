#!/usr/bin/env python3
"""Diagnose first-five totals residual calibration against F5 market rows.

This is a research diagnostic. It reads historical F5 totals from ``mlb.f5_odds``
and first-five actuals from ``mlb.linescore`` without writing to Postgres. When a
JSON report from ``scripts/f5_clv_report.py`` is supplied, sim probabilities are
merged by ``game_pk`` and a residual calibration model is evaluated on a
chronological holdout.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
from psycopg import sql
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.f5_clv import F5BookLine, consensus_f5_totals_line
from src.database import PostgresConfig

EPS = 1e-6
DEFAULT_TRAIN_FRACTION = 0.67
DEFAULT_MIN_TEST_ROWS = 500
MARKET_BASE_FEATURE_NAMES = ("market_logit", "point")
RESIDUAL_BASE_FEATURE_NAMES = (
    "market_logit",
    "sim_minus_market_logit",
    "point",
)


@dataclass(frozen=True)
class F5ResidualRow:
    game_pk: int
    season: int
    game_date: date
    point: float
    market_prob_over: float
    actual_total: float
    actual_over: int
    book_count: int | None = None
    sim_prob_over: float | None = None


@dataclass(frozen=True)
class ProbabilityMetrics:
    n: int
    brier: float
    log_loss: float
    mean_probability: float
    actual_rate: float
    auc: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "n": self.n,
            "brier": self.brier,
            "log_loss": self.log_loss,
            "mean_probability": self.mean_probability,
            "actual_rate": self.actual_rate,
            "auc": self.auc,
        }


def _parse_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one season is required")
    return parsed


def _connect(config: PostgresConfig):
    return psycopg.connect(
        dbname=config.dbname,
        user=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
        connect_timeout=15,
    )


def _date_value(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError(f"expected date-compatible value, got {type(value).__name__}")


def _validate_probability(value: float, *, field: str) -> float:
    probability = float(value)
    if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError(f"{field} must be a finite probability in [0, 1]")
    return probability


def load_f5_actual_totals(game_pks: Sequence[int], *, prefix_innings: int) -> dict[int, float]:
    """Read completed first-five actual totals from Postgres linescore rows."""
    if not game_pks:
        return {}
    config = PostgresConfig.from_env()
    totals: dict[int, float] = {}
    with _connect(config) as conn, conn.cursor() as cursor:
        query = sql.SQL(
            """
            SELECT game_pk,
                   SUM(runs) FILTER (WHERE team_type = 'away')::float AS away_runs,
                   SUM(runs) FILTER (WHERE team_type = 'home')::float AS home_runs,
                   COUNT(*) FILTER (
                       WHERE team_type = 'away' AND runs IS NOT NULL
                   ) AS away_rows,
                   COUNT(*) FILTER (
                       WHERE team_type = 'home' AND runs IS NOT NULL
                   ) AS home_rows
            FROM {}.linescore
            WHERE game_pk = ANY(%s)
              AND inning BETWEEN 1 AND %s
            GROUP BY game_pk
            """
        ).format(sql.Identifier(config.schema))
        cursor.execute(query, (list(game_pks), prefix_innings))
        for game_pk, away_runs, home_runs, away_rows, home_rows in cursor.fetchall():
            if away_rows < prefix_innings or home_rows < prefix_innings:
                continue
            if away_runs is None or home_runs is None:
                continue
            totals[int(game_pk)] = float(away_runs) + float(home_runs)
    return totals


def build_f5_market_rows(
    *,
    lines_by_game: Mapping[int, Sequence[F5BookLine]],
    game_meta: Mapping[int, tuple[int, date]],
    actual_totals: Mapping[int, float],
) -> list[F5ResidualRow]:
    """Build non-push consensus market rows from grouped F5 book lines."""
    rows: list[F5ResidualRow] = []
    for game_pk, lines in lines_by_game.items():
        actual_total = actual_totals.get(game_pk)
        if actual_total is None:
            continue
        consensus = consensus_f5_totals_line(lines)
        if consensus is None or math.isclose(actual_total, consensus.point):
            continue
        season, game_date = game_meta[game_pk]
        rows.append(
            F5ResidualRow(
                game_pk=game_pk,
                season=season,
                game_date=game_date,
                point=consensus.point,
                market_prob_over=consensus.prob_over,
                actual_total=float(actual_total),
                actual_over=int(actual_total > consensus.point),
                book_count=len(lines),
            )
        )
    return sorted(rows, key=lambda row: (row.game_date, row.game_pk))


def load_f5_market_rows(
    seasons: Sequence[int],
    *,
    line_type: str = "open",
    prefix_innings: int = 5,
) -> list[F5ResidualRow]:
    """Load non-push F5 totals rows from Postgres for a single market line type."""
    config = PostgresConfig.from_env()
    lines_by_game: dict[int, list[F5BookLine]] = defaultdict(list)
    game_meta: dict[int, tuple[int, date]] = {}
    query = sql.SQL(
        """
        SELECT g.season::int, g.game_date, f.game_pk, f.bookmaker,
               f.total_point, f.over_ml, f.under_ml
        FROM {}.f5_odds f
        JOIN {}.games g USING(game_pk)
        WHERE g.season::int = ANY(%s)
          AND g.game_type = 'R'
          AND f.line_type = %s
          AND f.total_point IS NOT NULL
          AND f.over_ml IS NOT NULL
          AND f.under_ml IS NOT NULL
        ORDER BY g.game_date, f.game_pk, f.bookmaker
        """
    ).format(sql.Identifier(config.schema), sql.Identifier(config.schema))
    with _connect(config) as conn, conn.cursor() as cursor:
        cursor.execute(query, (list(seasons), line_type))
        for season, game_date, game_pk, bookmaker, point, over_ml, under_ml in cursor.fetchall():
            pk = int(game_pk)
            game_meta[pk] = (int(season), _date_value(game_date))
            lines_by_game[pk].append(
                F5BookLine(
                    bookmaker=str(bookmaker),
                    point=float(point),
                    over_ml=float(over_ml),
                    under_ml=float(under_ml),
                )
            )
    actual_totals = load_f5_actual_totals(tuple(lines_by_game), prefix_innings=prefix_innings)
    return build_f5_market_rows(
        lines_by_game=lines_by_game,
        game_meta=game_meta,
        actual_totals=actual_totals,
    )


def load_sim_probabilities_from_report(path: Path) -> dict[int, float]:
    """Extract per-game sim over probabilities from an F5 CLV report JSON file.

    Current ``f5_clv_report.py`` JSON exposes selected-side ``model_prob`` in
    ``bets`` rows. For edge-threshold 0.0 rows, that selected side is present for
    every simulated game; under probabilities are converted back to over
    probabilities. If future reports add game-level ``sim_prob_over`` style
    fields, those are preferred.
    """
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError("F5 CLV report JSON must contain an object")

    probabilities = _sim_probabilities_from_games(payload.get("games"))
    for game_pk, probability in _sim_probabilities_from_zero_edge_bets(
        payload.get("bets")
    ).items():
        probabilities.setdefault(game_pk, probability)
    return dict(sorted(probabilities.items()))


def merge_sim_probabilities(
    rows: Sequence[F5ResidualRow], sim_probabilities: Mapping[int, float]
) -> list[F5ResidualRow]:
    """Attach sim probabilities by ``game_pk`` while preserving all market rows."""
    merged: list[F5ResidualRow] = []
    for row in rows:
        sim_probability = sim_probabilities.get(row.game_pk)
        if sim_probability is None:
            merged.append(row)
            continue
        merged.append(
            replace(
                row,
                sim_prob_over=_validate_probability(
                    sim_probability,
                    field=f"sim probability for game_pk={row.game_pk}",
                ),
            )
        )
    return merged


def split_rows(
    rows: Sequence[F5ResidualRow], *, train_fraction: float
) -> tuple[list[F5ResidualRow], list[F5ResidualRow]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    ordered = sorted(rows, key=lambda row: (row.game_date, row.game_pk))
    split_at = int(len(ordered) * train_fraction)
    if split_at <= 0 or split_at >= len(ordered):
        raise ValueError("train_fraction leaves an empty train or test split")
    return ordered[:split_at], ordered[split_at:]


def probability_metrics(
    probabilities: Sequence[float], outcomes: Sequence[int]
) -> ProbabilityMetrics:
    probs, y = _aligned_arrays(probabilities, outcomes)
    clipped = np.clip(probs, EPS, 1.0 - EPS)
    auc = None
    if len(np.unique(y)) > 1:
        auc = float(roc_auc_score(y, probs))
    return ProbabilityMetrics(
        n=len(probs),
        brier=float(np.mean((probs - y) ** 2)),
        log_loss=float(np.mean(-(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))),
        mean_probability=float(np.mean(probs)),
        actual_rate=float(np.mean(y)),
        auc=auc,
    )


def fit_calibration_model(
    rows: Sequence[F5ResidualRow], *, include_sim: bool
) -> tuple[LogisticRegression, tuple[str, ...]]:
    if not rows:
        raise ValueError("at least one training row is required")
    outcomes = [row.actual_over for row in rows]
    if len(set(outcomes)) < 2:
        raise ValueError("training rows must contain both over and under outcomes")
    feature_names = _calibration_feature_names(rows, include_sim=include_sim)
    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    model.fit(_feature_matrix(rows, feature_names), np.array(outcomes, dtype=int))
    return model, feature_names


def predict_probabilities(
    model: LogisticRegression,
    rows: Sequence[F5ResidualRow],
    *,
    feature_names: Sequence[str],
) -> list[float]:
    if not rows:
        return []
    return [float(prob) for prob in model.predict_proba(_feature_matrix(rows, feature_names))[:, 1]]


def build_report(
    rows: Sequence[F5ResidualRow],
    *,
    seasons: Sequence[int],
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    line_type: str = "open",
    prefix_innings: int = 5,
    sim_report_json: str | None = None,
    min_test_rows: int = DEFAULT_MIN_TEST_ROWS,
) -> dict[str, Any]:
    """Fit market and optional sim-residual calibrators and return JSON-safe output."""
    if not rows:
        raise ValueError("at least one non-push market row is required")
    train_rows, test_rows = split_rows(rows, train_fraction=train_fraction)

    market_model, market_features = fit_calibration_model(train_rows, include_sim=False)
    market_train = [row.market_prob_over for row in train_rows]
    market_test = [row.market_prob_over for row in test_rows]
    train_outcomes = [row.actual_over for row in train_rows]
    test_outcomes = [row.actual_over for row in test_rows]
    metrics: dict[str, dict[str, dict[str, float | int | None]]] = {
        "train": {
            "market": probability_metrics(market_train, train_outcomes).as_dict(),
            "market_calibrated": probability_metrics(
                predict_probabilities(market_model, train_rows, feature_names=market_features),
                train_outcomes,
            ).as_dict(),
        },
        "test": {
            "market": probability_metrics(market_test, test_outcomes).as_dict(),
            "market_calibrated": probability_metrics(
                predict_probabilities(market_model, test_rows, feature_names=market_features),
                test_outcomes,
            ).as_dict(),
        },
    }
    coefficients: dict[str, Any] = {
        "market_calibrated": _model_coefficients(market_model, market_features)
    }
    feature_names: dict[str, list[str]] = {"market_calibrated": list(market_features)}

    sim_rows = [row for row in rows if row.sim_prob_over is not None]
    sim_train_rows: list[F5ResidualRow] = []
    sim_test_rows: list[F5ResidualRow] = []
    comparison_metrics: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None
    if sim_rows:
        sim_train_rows, sim_test_rows = split_rows(sim_rows, train_fraction=train_fraction)
        residual_model, residual_features = fit_calibration_model(
            sim_train_rows, include_sim=True
        )
        feature_names["residual_calibrated"] = list(residual_features)
        coefficients["residual_calibrated"] = _model_coefficients(
            residual_model, residual_features
        )
        sim_train_outcomes = [row.actual_over for row in sim_train_rows]
        sim_test_outcomes = [row.actual_over for row in sim_test_rows]
        sim_train_market = [row.market_prob_over for row in sim_train_rows]
        sim_test_market = [row.market_prob_over for row in sim_test_rows]
        sim_train = [_required_sim_probability(row) for row in sim_train_rows]
        sim_test = [_required_sim_probability(row) for row in sim_test_rows]
        metrics["sim_train"] = {
            "market": probability_metrics(sim_train_market, sim_train_outcomes).as_dict(),
            "sim": probability_metrics(sim_train, sim_train_outcomes).as_dict(),
            "residual_calibrated": probability_metrics(
                predict_probabilities(
                    residual_model,
                    sim_train_rows,
                    feature_names=residual_features,
                ),
                sim_train_outcomes,
            ).as_dict(),
        }
        metrics["sim_test"] = {
            "market": probability_metrics(sim_test_market, sim_test_outcomes).as_dict(),
            "sim": probability_metrics(sim_test, sim_test_outcomes).as_dict(),
            "residual_calibrated": probability_metrics(
                predict_probabilities(
                    residual_model,
                    sim_test_rows,
                    feature_names=residual_features,
                ),
                sim_test_outcomes,
            ).as_dict(),
        }
        comparison_metrics = (
            metrics["sim_test"]["market"],
            metrics["sim_test"]["residual_calibrated"],
        )

    report = {
        "model_type": "f5_totals_residual_calibration",
        "inputs": {
            "seasons": list(seasons),
            "line_type": line_type,
            "prefix_innings": prefix_innings,
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
            "sim_train_rows": len(sim_train_rows),
            "sim_test_rows": len(sim_test_rows),
        },
        "feature_names": feature_names,
        "metrics": metrics,
        "coefficients": coefficients,
    }
    report["betting_gate"] = decide_betting_gate(
        comparison_metrics=comparison_metrics,
        residual_test_rows=len(sim_test_rows),
        min_test_rows=min_test_rows,
    )
    return report


def decide_betting_gate(
    *,
    comparison_metrics: tuple[Mapping[str, Any], Mapping[str, Any]] | None,
    residual_test_rows: int,
    min_test_rows: int,
) -> dict[str, Any]:
    """Keep the research gate closed unless held-out residual scores beat market."""
    has_residual_calibration = comparison_metrics is not None
    enough_heldout_sample = residual_test_rows >= min_test_rows
    brier_improvement: float | None = None
    log_loss_improvement: float | None = None
    test_brier_improves = False
    test_log_loss_improves = False
    if comparison_metrics is not None:
        market_metrics, residual_metrics = comparison_metrics
        brier_improvement = float(market_metrics["brier"]) - float(residual_metrics["brier"])
        log_loss_improvement = float(market_metrics["log_loss"]) - float(residual_metrics["log_loss"])
        test_brier_improves = brier_improvement > 0.0
        test_log_loss_improves = log_loss_improvement > 0.0

    checks = {
        "has_residual_calibration": has_residual_calibration,
        "enough_heldout_sample": enough_heldout_sample,
        "test_brier_improves_vs_market": test_brier_improves,
        "test_log_loss_improves_vs_market": test_log_loss_improves,
    }
    status = "open" if all(checks.values()) else "closed"
    if status == "open":
        reason = "Residual-calibrated F5 totals beat market Brier/log loss on enough held-out rows"
    else:
        failed = [name for name, passed in checks.items() if not passed]
        reason = "Gate closed: " + ", ".join(failed)
    return {
        "status": status,
        "reason": reason,
        "checks": checks,
        "thresholds": {"min_test_rows": min_test_rows},
        "metrics": {
            "comparison_model": "residual_calibrated",
            "test_rows": residual_test_rows,
            "brier_improvement_vs_market": brier_improvement,
            "log_loss_improvement_vs_market": log_loss_improvement,
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=_parse_ints, default=(2025,))
    parser.add_argument("--line-type", choices=("open", "current", "close"), default="open")
    parser.add_argument("--prefix-innings", type=int, default=5)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument(
        "--sim-report-json",
        type=Path,
        default=None,
        help="optional JSON emitted by scripts/f5_clv_report.py",
    )
    parser.add_argument("--min-test-rows", type=int, default=DEFAULT_MIN_TEST_ROWS)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = load_f5_market_rows(
        args.seasons,
        line_type=args.line_type,
        prefix_innings=args.prefix_innings,
    )
    if not rows:
        raise SystemExit("No non-push F5 totals market rows found")
    if args.sim_report_json is not None:
        rows = merge_sim_probabilities(
            rows,
            load_sim_probabilities_from_report(args.sim_report_json),
        )
    report = build_report(
        rows,
        seasons=args.seasons,
        train_fraction=args.train_fraction,
        line_type=args.line_type,
        prefix_innings=args.prefix_innings,
        sim_report_json=str(args.sim_report_json) if args.sim_report_json else None,
        min_test_rows=args.min_test_rows,
    )
    output = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    if args.out_json is None:
        print(output, end="")
        return
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(output)
    print(f"wrote F5 residual calibration report to {args.out_json}")


def _aligned_arrays(
    probabilities: Sequence[float], outcomes: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have the same length")
    if not probabilities:
        raise ValueError("at least one probability/outcome pair is required")
    probs = np.array(probabilities, dtype=float)
    y = np.array(outcomes, dtype=float)
    if np.any(~np.isfinite(probs)) or np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if np.any((y != 0.0) & (y != 1.0)):
        raise ValueError("outcomes must be binary 0/1 values")
    return probs, y


def _calibration_feature_names(
    rows: Sequence[F5ResidualRow], *, include_sim: bool
) -> tuple[str, ...]:
    names: list[str] = list(
        RESIDUAL_BASE_FEATURE_NAMES if include_sim else MARKET_BASE_FEATURE_NAMES
    )
    if all(row.book_count is not None for row in rows):
        names.append("book_count")
    return tuple(names)


def _feature_matrix(rows: Sequence[F5ResidualRow], feature_names: Sequence[str]) -> np.ndarray:
    columns: list[list[float]] = []
    for feature_name in feature_names:
        if feature_name == "market_logit":
            columns.append([_logit(row.market_prob_over) for row in rows])
        elif feature_name == "sim_minus_market_logit":
            columns.append(
                [
                    _logit(_required_sim_probability(row)) - _logit(row.market_prob_over)
                    for row in rows
                ]
            )
        elif feature_name == "point":
            columns.append([row.point for row in rows])
        elif feature_name == "book_count":
            columns.append([_required_book_count(row) for row in rows])
        else:
            raise ValueError(f"unknown feature {feature_name!r}")
    return np.column_stack(columns)


def _model_coefficients(
    model: LogisticRegression, feature_names: Sequence[str]
) -> dict[str, float | dict[str, float]]:
    return {
        "intercept": float(np.ravel(model.intercept_)[0]),
        "features": {
            feature_name: float(coef)
            for feature_name, coef in zip(feature_names, np.ravel(model.coef_))
        },
    }


def _sim_probabilities_from_games(games: object) -> dict[int, float]:
    if not isinstance(games, list):
        return {}
    probabilities: dict[int, float] = {}
    for item in games:
        if not isinstance(item, dict) or "game_pk" not in item:
            continue
        probability = _first_present_probability(
            item,
            ("sim_prob_over", "model_prob_over", "sim_over", "model_over"),
        )
        if probability is None:
            continue
        probabilities[int(item["game_pk"])] = probability
    return probabilities


def _sim_probabilities_from_zero_edge_bets(bets: object) -> dict[int, float]:
    if not isinstance(bets, list):
        return {}
    probabilities: dict[int, float] = {}
    for item in bets:
        if not isinstance(item, dict):
            continue
        if float(item.get("edge_threshold", math.nan)) != 0.0:
            continue
        if "game_pk" not in item or "side" not in item or "model_prob" not in item:
            continue
        side = str(item["side"]).lower()
        model_probability = _validate_probability(
            float(item["model_prob"]), field="bets.model_prob"
        )
        if side == "over":
            probability = model_probability
        elif side == "under":
            probability = 1.0 - model_probability
        else:
            continue
        game_pk = int(item["game_pk"])
        existing = probabilities.get(game_pk)
        if existing is not None and not math.isclose(existing, probability, abs_tol=1e-12):
            raise ValueError(f"conflicting edge=0 sim probabilities for game_pk={game_pk}")
        probabilities[game_pk] = probability
    return probabilities


def _first_present_probability(
    item: Mapping[str, Any], field_names: Sequence[str]
) -> float | None:
    for field_name in field_names:
        if field_name in item and item[field_name] is not None:
            return _validate_probability(float(item[field_name]), field=field_name)
    return None


def _required_sim_probability(row: F5ResidualRow) -> float:
    if row.sim_prob_over is None:
        raise ValueError(f"game_pk={row.game_pk} is missing sim_prob_over")
    return row.sim_prob_over


def _required_book_count(row: F5ResidualRow) -> float:
    if row.book_count is None:
        raise ValueError(f"game_pk={row.game_pk} is missing book_count")
    return float(row.book_count)


def _logit(probability: float) -> float:
    probability = min(max(probability, EPS), 1.0 - EPS)
    return math.log(probability / (1.0 - probability))


if __name__ == "__main__":
    main()
