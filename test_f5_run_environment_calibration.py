from __future__ import annotations

import argparse
from datetime import date, timedelta

import pytest

from scripts.f5_run_environment_calibration import (
    bucket_key,
    build_report,
    fit_bucket_adjustments,
    parse_bucket_groups,
    split_rows,
)
from scripts.f5_run_environment_diagnostics import F5RunEnvironmentRow


def _row(
    index: int,
    *,
    point: float = 4.5,
    sim_mean: float,
    actual: float,
) -> F5RunEnvironmentRow:
    return F5RunEnvironmentRow(
        game_pk=900000 + index,
        season=2025,
        game_date=date(2025, 4, 1) + timedelta(days=index),
        take_point=point,
        sim_mean_total=sim_mean,
        actual_f5_total=actual,
    )


def test_split_rows_is_chronological() -> None:
    rows = [
        _row(3, sim_mean=4.0, actual=5.0),
        _row(1, sim_mean=4.0, actual=5.0),
        _row(2, sim_mean=4.0, actual=5.0),
    ]

    train, test = split_rows(rows, train_fraction=2 / 3)

    assert [row.game_pk for row in train] == [900001, 900002]
    assert [row.game_pk for row in test] == [900003]


def test_fit_bucket_adjustments_shrinks_toward_global_residual() -> None:
    rows = [
        _row(1, point=4.0, sim_mean=4.0, actual=5.0),
        _row(2, point=4.0, sim_mean=4.0, actual=5.0),
        _row(3, point=5.0, sim_mean=5.0, actual=5.0),
    ]

    adjustments, fallback = fit_bucket_adjustments(
        rows,
        bucket_groups=("total_point",),
        shrinkage=3.0,
    )

    assert fallback == pytest.approx(2 / 3)
    assert adjustments["total=4"].raw_adjustment == pytest.approx(1.0)
    assert adjustments["total=4"].shrunk_adjustment == pytest.approx(0.8)
    assert adjustments["total=5"].shrunk_adjustment == pytest.approx(0.5)


def test_build_report_calibration_gate_and_betting_gate_are_separate() -> None:
    rows = [
        _row(1, point=4.0, sim_mean=4.0, actual=5.0),
        _row(2, point=4.0, sim_mean=4.0, actual=5.0),
        _row(3, point=5.0, sim_mean=5.0, actual=4.0),
        _row(4, point=5.0, sim_mean=5.0, actual=4.0),
        _row(5, point=4.0, sim_mean=4.0, actual=5.0),
        _row(6, point=5.0, sim_mean=5.0, actual=4.0),
    ]

    report = build_report(
        rows,
        train_fraction=4 / 6,
        bucket_groups=("total_point",),
        shrinkage=0.0,
        min_test_rows=10,
    )

    assert report["split"] == {
        "method": "chronological",
        "train_fraction": 4 / 6,
        "train_rows": 4,
        "test_rows": 2,
    }
    assert report["metrics"]["test"]["calibrated"]["mae"] < report["metrics"]["test"]["base_sim"]["mae"]
    assert report["calibration_gate"]["status"] == "closed"
    assert report["calibration_gate"]["checks"]["enough_heldout_sample"] is False
    assert report["betting_gate"]["status"] == "closed"


def test_bucket_key_supports_month_and_rejects_unknown_group() -> None:
    row = _row(1, point=4.0, sim_mean=4.0, actual=5.0)

    assert bucket_key(row, ("total_point", "month")) == "total=4|month=2025-04"
    with pytest.raises(ValueError, match="unknown bucket group"):
        bucket_key(row, ("close_move",))


def test_parse_bucket_groups_rejects_leaky_close_move() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_bucket_groups("total_point,close_move")
