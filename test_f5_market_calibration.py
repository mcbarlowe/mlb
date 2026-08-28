from __future__ import annotations

from datetime import date, timedelta

import pytest

from scripts.f5_market_calibration import F5MarketRow, build_report, split_rows


def _row(index: int, *, probability: float, actual_over: int) -> F5MarketRow:
    point = 4.5
    return F5MarketRow(
        game_pk=700000 + index,
        season=2025,
        game_date=date(2025, 4, 1) + timedelta(days=index),
        point=point,
        market_prob_over=probability,
        actual_total=5.0 if actual_over else 4.0,
        actual_over=actual_over,
        book_count=6,
    )


def test_split_rows_rejects_empty_train_or_test() -> None:
    rows = [_row(1, probability=0.5, actual_over=1)]

    with pytest.raises(ValueError, match="empty train or test"):
        split_rows(rows, train_fraction=0.5)


def test_build_report_fits_market_logit_calibration() -> None:
    rows = [
        _row(1, probability=0.40, actual_over=0),
        _row(2, probability=0.45, actual_over=0),
        _row(3, probability=0.55, actual_over=1),
        _row(4, probability=0.60, actual_over=1),
        _row(5, probability=0.42, actual_over=0),
        _row(6, probability=0.58, actual_over=1),
    ]

    report = build_report(rows, seasons=(2025,), train_fraction=4 / 6)

    assert report.rows == 6
    assert report.train_rows == 4
    assert report.test_rows == 2
    assert set(report.coefficients) == {"intercept", "market_logit"}
    assert report.train["market"].n == 4
    assert report.test["calibrated"].n == 2
