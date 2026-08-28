from __future__ import annotations

from datetime import date, timedelta

import pytest

from scripts.f5_market_anchor_blend import (
    blend_probability,
    build_report,
    fit_lambda,
    lambda_grid,
)
from scripts.f5_residual_calibration import F5ResidualRow


def _row(
    index: int,
    *,
    market: float,
    sim: float,
    actual_over: int,
) -> F5ResidualRow:
    return F5ResidualRow(
        game_pk=950000 + index,
        season=2025,
        game_date=date(2025, 4, 1) + timedelta(days=index),
        point=4.5,
        market_prob_over=market,
        actual_total=5.0 if actual_over else 4.0,
        actual_over=actual_over,
        book_count=6,
        sim_prob_over=sim,
    )


def test_lambda_grid_validates_and_includes_bounds() -> None:
    assert lambda_grid(0.0, 1.0, 3) == [0.0, 0.5, 1.0]
    with pytest.raises(ValueError, match="at least 2"):
        lambda_grid(0.0, 1.0, 1)


def test_fit_lambda_moves_toward_sim_when_train_sim_is_better() -> None:
    rows = [
        _row(1, market=0.5, sim=0.2, actual_over=0),
        _row(2, market=0.5, sim=0.8, actual_over=1),
        _row(3, market=0.5, sim=0.25, actual_over=0),
        _row(4, market=0.5, sim=0.75, actual_over=1),
    ]

    selected, scores = fit_lambda(rows, lambda_min=0.0, lambda_max=1.0, steps=11)

    assert selected == pytest.approx(1.0)
    assert scores["1"] < scores["0"]


def test_fit_lambda_stays_at_market_when_sim_is_harmful() -> None:
    rows = [
        _row(1, market=0.5, sim=0.8, actual_over=0),
        _row(2, market=0.5, sim=0.2, actual_over=1),
        _row(3, market=0.5, sim=0.75, actual_over=0),
        _row(4, market=0.5, sim=0.25, actual_over=1),
    ]

    selected, scores = fit_lambda(rows, lambda_min=0.0, lambda_max=1.0, steps=11)

    assert selected == pytest.approx(0.0)
    assert scores["0"] < scores["1"]


def test_build_report_evaluates_probability_fit_on_holdout() -> None:
    rows = [
        _row(1, market=0.5, sim=0.2, actual_over=0),
        _row(2, market=0.5, sim=0.8, actual_over=1),
        _row(3, market=0.5, sim=0.25, actual_over=0),
        _row(4, market=0.5, sim=0.75, actual_over=1),
        _row(5, market=0.5, sim=0.3, actual_over=0),
        _row(6, market=0.5, sim=0.7, actual_over=1),
    ]

    report = build_report(
        rows,
        seasons=(2025,),
        train_fraction=4 / 6,
        lambda_steps=11,
        min_test_rows=10,
    )

    assert report["selected_lambda"] == pytest.approx(1.0)
    assert report["metrics"]["test"]["blended"]["brier"] < report["metrics"]["test"]["market"]["brier"]
    assert report["probability_gate"]["status"] == "closed"
    assert report["probability_gate"]["checks"]["enough_heldout_sample"] is False


def test_blend_probability_is_market_anchor_interpolation() -> None:
    row = _row(1, market=0.4, sim=0.8, actual_over=1)

    assert blend_probability(row, 0.0) == pytest.approx(0.4)
    assert blend_probability(row, 0.5) == pytest.approx(0.6)
    assert blend_probability(row, 1.0) == pytest.approx(0.8)
