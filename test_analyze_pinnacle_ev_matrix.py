from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts import analyze_pinnacle_ev_matrix as matrix


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "season",
        "game_pk",
        "game_date",
        "horizon_hours",
        "home_won",
        "best_side",
        "best_book",
        "best_decimal",
        "best_prob",
        "best_ev",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_read_evaluated_and_settle_home_away(tmp_path: Path) -> None:
    path = tmp_path / "evaluated.csv"
    write_rows(
        path,
        [
            {
                "season": 2025,
                "game_pk": 1,
                "game_date": "2025-04-01",
                "horizon_hours": 24.0,
                "home_won": 1,
                "best_side": "home",
                "best_book": "book",
                "best_decimal": 2.1,
                "best_prob": 0.5,
                "best_ev": 0.05,
            },
            {
                "season": 2025,
                "game_pk": 2,
                "game_date": "2025-04-02",
                "horizon_hours": 24.0,
                "home_won": 1,
                "best_side": "away",
                "best_book": "book",
                "best_decimal": 1.8,
                "best_prob": 0.5,
                "best_ev": 0.04,
            },
        ],
    )

    offers = matrix.read_evaluated([f"MLB:h2h:{path}"])
    settled = matrix.settle(offers, 0.01)

    assert len(settled) == 2
    assert settled[0].ret == pytest.approx(1.1)
    assert settled[1].ret == -1.0
    assert offers[0].cluster == "MLB:h2h:date:2025-04-01"


def test_build_matrix_includes_stratum_and_pool(tmp_path: Path) -> None:
    path = tmp_path / "evaluated.csv"
    write_rows(
        path,
        [
            {
                "season": 2025,
                "game_pk": 1,
                "game_date": "2025-04-01",
                "horizon_hours": 4.0,
                "home_won": 1,
                "best_side": "home",
                "best_book": "book",
                "best_decimal": 2.0,
                "best_prob": 0.5,
                "best_ev": 0.02,
            },
            {
                "season": 2025,
                "game_pk": 1,
                "game_date": "2025-04-01",
                "horizon_hours": 24.0,
                "home_won": 1,
                "best_side": "home",
                "best_book": "book",
                "best_decimal": 2.0,
                "best_prob": 0.5,
                "best_ev": 0.02,
            },
        ],
    )
    offers = matrix.read_evaluated([f"MLB:h2h:{path}"])

    rows = matrix.build_matrix(offers, thresholds=(0.01,), samples=0)

    assert [(row.sport, row.market, row.horizon) for row in rows] == [
        ("MLB", "h2h", "4h"),
        ("MLB", "h2h", "24h"),
        ("ALL", "all", "all"),
    ]
    assert rows[0].bets == 1
    assert rows[0].roi == pytest.approx(1.0)
    assert rows[2].bets == 2
