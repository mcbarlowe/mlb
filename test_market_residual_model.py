from __future__ import annotations

import math

import pytest

from scripts.market_residual_model import (
    MarketResidualRow,
    brier_score,
    build_report,
    log_loss_score,
    parse_totals_eval_lines,
)


def test_parse_totals_eval_lines_drops_pushes_and_ignores_unrelated_lines() -> None:
    rows = parse_totals_eval_lines(
        [
            "warmup line without totals payload",
            "1001 pt=8.5 sim_over=0.61 mkt_over=0.52 actual=10",
            "1002 pt=8.0 sim_over=0.40 mkt_over=0.49 actual=7",
            "1003 pt=9.0 sim_over=0.55 mkt_over=0.50 actual=9",
        ],
        season=2025,
    )

    assert rows == [
        MarketResidualRow(
            game_pk=1001,
            season=2025,
            point=8.5,
            sim_over=0.61,
            market_over=0.52,
            actual_total=10.0,
            actual_over=1,
        ),
        MarketResidualRow(
            game_pk=1002,
            season=2025,
            point=8.0,
            sim_over=0.40,
            market_over=0.49,
            actual_total=7.0,
            actual_over=0,
        ),
    ]


def test_probability_metrics_match_binary_score_definitions() -> None:
    probabilities = [0.8, 0.2]
    outcomes = [1, 0]

    assert brier_score(probabilities, outcomes) == pytest.approx(0.04)
    assert log_loss_score(probabilities, outcomes) == pytest.approx(-math.log(0.8))


def test_build_report_contains_predictive_metrics_and_coefficients() -> None:
    train_rows = [
        MarketResidualRow(1, 2024, 8.5, 0.80, 0.50, 10.0, 1),
        MarketResidualRow(2, 2024, 8.5, 0.20, 0.50, 7.0, 0),
        MarketResidualRow(3, 2024, 8.5, 0.70, 0.55, 9.0, 1),
        MarketResidualRow(4, 2024, 8.5, 0.30, 0.45, 6.0, 0),
    ]
    eval_rows = [
        MarketResidualRow(5, 2025, 8.5, 0.70, 0.50, 10.0, 1),
        MarketResidualRow(6, 2025, 8.5, 0.30, 0.50, 7.0, 0),
        MarketResidualRow(7, 2025, 8.5, 0.60, 0.50, 7.0, 0),
    ]

    report = build_report(train_rows, eval_rows)

    assert report["metrics"]["model"]["n"] == 3
    assert report["metrics"]["market"]["n"] == 3
    assert "brier_model_minus_market" in report["metrics"]["gaps"]
    assert report["coefficients"]["features"].keys() == {
        "market_logit",
        "sim_minus_market_logit",
    }
    assert "edge_buckets" not in report
