#!/usr/bin/env python3
"""Fit and evaluate a first-five totals market calibration.

This is a research diagnostic, not a betting gate opener. It uses historical F5
open totals from ``mlb.f5_odds`` and first-five actuals from ``mlb.linescore`` to
measure whether a simple Platt transform of the devigged market probability
improves Brier/log-loss on a chronological holdout.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import psycopg
from psycopg import sql
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.f5_clv_report import load_f5_actual_totals
from src.betting.f5_clv import F5BookLine, consensus_f5_totals_line
from src.database import PostgresConfig

EPS = 1e-6


@dataclass(frozen=True)
class F5MarketRow:
    game_pk: int
    season: int
    game_date: date
    point: float
    market_prob_over: float
    actual_total: float
    actual_over: int
    book_count: int


@dataclass(frozen=True)
class Metrics:
    n: int
    mean_probability: float
    actual_rate: float
    brier: float
    log_loss: float
    auc: float | None


@dataclass(frozen=True)
class CalibrationReport:
    rows: int
    train_rows: int
    test_rows: int
    seasons: tuple[int, ...]
    train_fraction: float
    coefficients: dict[str, float]
    train: dict[str, Metrics]
    test: dict[str, Metrics]
    betting_gate: dict[str, object]


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


def _logit(probability: float) -> float:
    probability = min(max(probability, EPS), 1.0 - EPS)
    return math.log(probability / (1.0 - probability))


def _metrics(probabilities: np.ndarray, labels: np.ndarray) -> Metrics:
    auc = None
    if len(np.unique(labels)) > 1:
        auc = float(roc_auc_score(labels, probabilities))
    return Metrics(
        n=len(labels),
        mean_probability=float(np.mean(probabilities)),
        actual_rate=float(np.mean(labels)),
        brier=float(brier_score_loss(labels, probabilities)),
        log_loss=float(log_loss(labels, np.clip(probabilities, EPS, 1.0 - EPS))),
        auc=auc,
    )


def _feature_matrix(rows: Sequence[F5MarketRow]) -> np.ndarray:
    return np.array([[_logit(row.market_prob_over)] for row in rows], dtype=float)


def _labels(rows: Sequence[F5MarketRow]) -> np.ndarray:
    return np.array([row.actual_over for row in rows], dtype=int)


def _market_probabilities(rows: Sequence[F5MarketRow]) -> np.ndarray:
    return np.array([row.market_prob_over for row in rows], dtype=float)


def load_f5_market_rows(seasons: Sequence[int]) -> list[F5MarketRow]:
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
          AND f.line_type = 'open'
          AND f.total_point IS NOT NULL
          AND f.over_ml IS NOT NULL
          AND f.under_ml IS NOT NULL
        ORDER BY g.game_date, f.game_pk, f.bookmaker
        """
    ).format(sql.Identifier(config.schema), sql.Identifier(config.schema))
    with _connect(config) as conn, conn.cursor() as cursor:
        cursor.execute(query, (list(seasons),))
        for season, game_date, game_pk, bookmaker, point, over_ml, under_ml in cursor.fetchall():
            pk = int(game_pk)
            game_meta[pk] = (int(season), game_date)
            lines_by_game[pk].append(
                F5BookLine(
                    bookmaker=str(bookmaker),
                    point=float(point),
                    over_ml=float(over_ml),
                    under_ml=float(under_ml),
                )
            )
    actual_totals = load_f5_actual_totals(tuple(lines_by_game), prefix_innings=5)
    rows: list[F5MarketRow] = []
    for game_pk, lines in lines_by_game.items():
        actual_total = actual_totals.get(game_pk)
        if actual_total is None:
            continue
        consensus = consensus_f5_totals_line(lines)
        if consensus is None or math.isclose(actual_total, consensus.point):
            continue
        season, game_date = game_meta[game_pk]
        rows.append(
            F5MarketRow(
                game_pk=game_pk,
                season=season,
                game_date=game_date,
                point=consensus.point,
                market_prob_over=consensus.prob_over,
                actual_total=actual_total,
                actual_over=int(actual_total > consensus.point),
                book_count=len(lines),
            )
        )
    return sorted(rows, key=lambda row: (row.game_date, row.game_pk))


def split_rows(
    rows: Sequence[F5MarketRow], *, train_fraction: float
) -> tuple[list[F5MarketRow], list[F5MarketRow]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    split_at = int(len(rows) * train_fraction)
    if split_at <= 0 or split_at >= len(rows):
        raise ValueError("train_fraction leaves an empty train or test split")
    ordered = list(rows)
    return ordered[:split_at], ordered[split_at:]


def build_report(
    rows: Sequence[F5MarketRow], *, seasons: Sequence[int], train_fraction: float
) -> CalibrationReport:
    train_rows, test_rows = split_rows(rows, train_fraction=train_fraction)
    x_train = _feature_matrix(train_rows)
    y_train = _labels(train_rows)
    x_test = _feature_matrix(test_rows)
    y_test = _labels(test_rows)
    market_train = _market_probabilities(train_rows)
    market_test = _market_probabilities(test_rows)

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    calibrated_train = model.predict_proba(x_train)[:, 1]
    calibrated_test = model.predict_proba(x_test)[:, 1]

    train = {
        "market": _metrics(market_train, y_train),
        "calibrated": _metrics(calibrated_train, y_train),
    }
    test = {
        "market": _metrics(market_test, y_test),
        "calibrated": _metrics(calibrated_test, y_test),
    }
    brier_improves = test["calibrated"].brier < test["market"].brier
    log_loss_improves = test["calibrated"].log_loss < test["market"].log_loss
    return CalibrationReport(
        rows=len(rows),
        train_rows=len(train_rows),
        test_rows=len(test_rows),
        seasons=tuple(seasons),
        train_fraction=train_fraction,
        coefficients={
            "intercept": float(np.ravel(model.intercept_)[0]),
            "market_logit": float(np.ravel(model.coef_)[0]),
        },
        train=train,
        test=test,
        betting_gate={
            "status": "closed",
            "reason": (
                "Research-only F5 market calibration: partial single-season split. "
                "Use as a probability diagnostic, not a betting approval."
            ),
            "checks": {
                "test_brier_improves": brier_improves,
                "test_log_loss_improves": log_loss_improves,
                "multi_season_holdout": False,
            },
        },
    )


def _metrics_line(label: str, metrics: Metrics) -> str:
    auc = "n/a" if metrics.auc is None else f"{metrics.auc:.4f}"
    return (
        f"{label:<12} n={metrics.n:4d} mean_p={metrics.mean_probability:.4f} "
        f"actual={metrics.actual_rate:.4f} brier={metrics.brier:.4f} "
        f"log_loss={metrics.log_loss:.4f} auc={auc}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=_parse_ints, default=(2025,))
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_f5_market_rows(args.seasons)
    if not rows:
        raise SystemExit("No non-push F5 open-total rows found")
    report = build_report(rows, seasons=args.seasons, train_fraction=args.train_fraction)
    print("F5 market calibration")
    print(f"seasons={report.seasons} rows={report.rows} train={report.train_rows} test={report.test_rows}")
    print("coefficients", report.coefficients)
    print("\nTRAIN")
    print(_metrics_line("market", report.train["market"]))
    print(_metrics_line("calibrated", report.train["calibrated"]))
    print("\nTEST")
    print(_metrics_line("market", report.test["market"]))
    print(_metrics_line("calibrated", report.test["calibrated"]))
    print("betting_gate", report.betting_gate)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(asdict(report), indent=2, sort_keys=True, default=str)
        )
        print(f"wrote F5 market calibration report to {args.out_json}")


if __name__ == "__main__":
    main()
