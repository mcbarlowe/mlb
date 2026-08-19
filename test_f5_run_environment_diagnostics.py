from __future__ import annotations

from datetime import date

import pytest

from scripts.f5_run_environment_diagnostics import (
    build_report,
    rows_from_report_payload,
)


def test_rows_from_report_payload_uses_db_dates_and_orders_rows() -> None:
    payload = {
        "games": [
            {
                "game_pk": 2,
                "season": 2025,
                "take_point": 4.5,
                "close_point": 5.0,
                "sim_mean_total": 4.0,
                "actual_f5_total": 6.0,
            },
            {
                "game_pk": 1,
                "season": 2025,
                "take_point": 4.0,
                "close_point": 3.5,
                "sim_mean_total": 5.0,
                "actual_f5_total": 3.0,
            },
        ]
    }

    rows = rows_from_report_payload(
        payload,
        game_dates={1: date(2025, 3, 28), 2: date(2025, 4, 2)},
    )

    assert [row.game_pk for row in rows] == [1, 2]
    assert rows[0].game_date == date(2025, 3, 28)
    assert rows[1].close_point == pytest.approx(5.0)


def test_build_report_summarizes_overall_month_and_close_move_bias() -> None:
    rows = rows_from_report_payload(
        {
            "games": [
                {
                    "game_pk": 1,
                    "season": 2025,
                    "game_date": "2025-04-01",
                    "take_point": 4.0,
                    "close_point": 4.5,
                    "sim_mean_total": 3.5,
                    "actual_f5_total": 5.0,
                },
                {
                    "game_pk": 2,
                    "season": 2025,
                    "game_date": "2025-04-02",
                    "take_point": 4.0,
                    "close_point": 4.0,
                    "sim_mean_total": 4.5,
                    "actual_f5_total": 3.0,
                },
                {
                    "game_pk": 3,
                    "season": 2025,
                    "game_date": "2025-05-01",
                    "take_point": 5.0,
                    "close_point": 4.5,
                    "sim_mean_total": 5.5,
                    "actual_f5_total": 6.0,
                },
            ]
        }
    )

    report = build_report(
        rows,
        groups=("overall", "month", "close_move"),
        min_group_rows=1,
    )

    summaries = {(row["group"], row["value"]): row for row in report["summaries"]}
    assert summaries[("overall", "all")]["n"] == 3
    assert summaries[("overall", "all")]["actual_minus_sim"] == pytest.approx(1 / 6)
    assert summaries[("month", "2025-04")]["n"] == 2
    assert summaries[("close_move", "up")]["close_minus_take"] == pytest.approx(0.5)
    assert summaries[("close_move", "down")]["actual_minus_market"] == pytest.approx(1.0)
    assert report["calibration_candidate"]["status"] == "closed"


def test_build_report_filters_small_non_overall_groups() -> None:
    rows = rows_from_report_payload(
        {
            "games": [
                {
                    "game_pk": 1,
                    "season": 2025,
                    "game_date": "2025-04-01",
                    "take_point": 4.0,
                    "sim_mean_total": 3.5,
                    "actual_f5_total": 5.0,
                }
            ]
        }
    )

    report = build_report(rows, groups=("overall", "month"), min_group_rows=2)

    assert [(row["group"], row["value"]) for row in report["summaries"]] == [
        ("overall", "all")
    ]
