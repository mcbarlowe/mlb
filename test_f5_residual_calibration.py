from __future__ import annotations

from datetime import date, timedelta

import pytest

from scripts.f5_residual_calibration import (
    F5ResidualRow,
    build_f5_market_rows,
    build_report,
    load_sim_probabilities_from_report,
    merge_sim_probabilities,
    split_rows,
)
from src.betting.f5_clv import F5BookLine


def _row(
    index: int,
    *,
    market_prob_over: float,
    actual_over: int,
    sim_prob_over: float | None = None,
) -> F5ResidualRow:
    point = 4.5
    return F5ResidualRow(
        game_pk=800000 + index,
        season=2025,
        game_date=date(2025, 4, 1) + timedelta(days=index),
        point=point,
        market_prob_over=market_prob_over,
        actual_total=5.0 if actual_over else 4.0,
        actual_over=actual_over,
        book_count=5,
        sim_prob_over=sim_prob_over,
    )


def test_build_market_rows_drops_pushes_and_keeps_chronological_order() -> None:
    lines_by_game = {
        2: (F5BookLine("book", 4.5, -110, -110),),
        1: (F5BookLine("book", 4.5, -110, -110),),
        3: (F5BookLine("book", 4.5, -110, -110),),
    }
    game_meta = {
        1: (2025, date(2025, 4, 1)),
        2: (2025, date(2025, 4, 2)),
        3: (2025, date(2025, 4, 3)),
    }
    actual_totals = {1: 5.0, 2: 4.5, 3: 4.0}

    rows = build_f5_market_rows(
        lines_by_game=lines_by_game,
        game_meta=game_meta,
        actual_totals=actual_totals,
    )

    assert [row.game_pk for row in rows] == [1, 3]
    assert [row.actual_over for row in rows] == [1, 0]
    assert all(row.book_count == 1 for row in rows)


def test_load_sim_probabilities_from_current_f5_clv_report_shape(tmp_path) -> None:
    report_path = tmp_path / "f5_clv.json"
    report_path.write_text(
        """
        {
          "games": [
            {"game_pk": 10, "take_prob_over": 0.51},
            {"game_pk": 11, "sim_prob_over": 0.62}
          ],
          "bets": [
            {"game_pk": 10, "edge_threshold": 0.0, "staking": "flat", "side": "under", "model_prob": 0.58},
            {"game_pk": 10, "edge_threshold": 0.03, "staking": "flat", "side": "under", "model_prob": 0.58},
            {"game_pk": 12, "edge_threshold": 0.0, "staking": "flat", "side": "over", "model_prob": 0.57}
          ]
        }
        """
    )

    probabilities = load_sim_probabilities_from_report(report_path)

    assert probabilities == {10: pytest.approx(0.42), 11: pytest.approx(0.62), 12: pytest.approx(0.57)}


def test_split_rows_rejects_empty_sides() -> None:
    with pytest.raises(ValueError, match="empty train or test"):
        split_rows([_row(1, market_prob_over=0.5, actual_over=1)], train_fraction=0.5)


def test_build_report_without_sim_rows_reports_market_only_gate_closed() -> None:
    rows = [
        _row(1, market_prob_over=0.40, actual_over=0),
        _row(2, market_prob_over=0.45, actual_over=0),
        _row(3, market_prob_over=0.55, actual_over=1),
        _row(4, market_prob_over=0.60, actual_over=1),
        _row(5, market_prob_over=0.42, actual_over=0),
        _row(6, market_prob_over=0.58, actual_over=1),
    ]

    report = build_report(rows, seasons=(2025,), train_fraction=4 / 6)

    assert report["rows"]["market"] == 6
    assert report["rows"]["sim_merged"] == 0
    assert set(report["metrics"]) == {"train", "test"}
    assert set(report["metrics"]["test"]) == {"market", "market_calibrated"}
    assert report["betting_gate"]["status"] == "closed"
    assert report["betting_gate"]["checks"]["has_residual_calibration"] is False


def test_build_report_with_sim_rows_compares_sim_and_residual_on_holdout() -> None:
    rows = [
        _row(1, market_prob_over=0.50, sim_prob_over=0.20, actual_over=0),
        _row(2, market_prob_over=0.50, sim_prob_over=0.80, actual_over=1),
        _row(3, market_prob_over=0.50, sim_prob_over=0.25, actual_over=0),
        _row(4, market_prob_over=0.50, sim_prob_over=0.75, actual_over=1),
        _row(5, market_prob_over=0.50, sim_prob_over=0.30, actual_over=0),
        _row(6, market_prob_over=0.50, sim_prob_over=0.70, actual_over=1),
        _row(7, market_prob_over=0.50, sim_prob_over=0.35, actual_over=0),
        _row(8, market_prob_over=0.50, sim_prob_over=0.65, actual_over=1),
    ]

    report = build_report(
        rows,
        seasons=(2025,),
        train_fraction=0.5,
        min_test_rows=100,
    )

    assert report["split"]["sim_train_rows"] == 4
    assert report["split"]["sim_test_rows"] == 4
    assert set(report["metrics"]["sim_test"]) == {"market", "sim", "residual_calibrated"}
    assert report["feature_names"]["residual_calibrated"] == [
        "market_logit",
        "sim_minus_market_logit",
        "point",
        "book_count",
    ]
    assert report["metrics"]["sim_test"]["residual_calibrated"]["brier"] < report["metrics"]["sim_test"]["market"]["brier"]
    assert report["betting_gate"]["status"] == "closed"
    assert report["betting_gate"]["checks"]["enough_heldout_sample"] is False


def test_merge_sim_probabilities_validates_probability_bounds() -> None:
    rows = [_row(1, market_prob_over=0.50, actual_over=1)]

    with pytest.raises(ValueError, match="probability"):
        merge_sim_probabilities(rows, {800001: 1.2})
