from __future__ import annotations

from datetime import datetime

import polars as pl

from scripts.load_totals_to_db import build_raw_totals_rows, build_resolved_totals_rows


def _staged_totals() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "game_id": "api-1",
                "commence_time": "2025-04-01T23:05:00+00:00",
                "game_date_utc": "2025-04-01",
                "away_team": "Away",
                "home_team": "Home",
                "bookmaker": "book-a",
                "snapshot_time": "2025-04-01T16:30:00+00:00",
                "book_last_update": "2025-04-01T16:25:00+00:00",
                "total_point": 8.5,
                "over_ml": -110,
                "under_ml": -110,
            },
            {
                "game_id": "api-unmatched",
                "commence_time": "2025-04-02T23:05:00+00:00",
                "game_date_utc": "2025-04-02",
                "away_team": "Other Away",
                "home_team": "Other Home",
                "bookmaker": "book-b",
                "snapshot_time": "2025-04-02T16:30:00+00:00",
                "book_last_update": "2025-04-02T16:25:00+00:00",
                "total_point": 9.0,
                "over_ml": 100,
                "under_ml": -120,
            },
        ]
    )


def test_build_raw_totals_rows_preserves_every_staged_api_row() -> None:
    rows = build_raw_totals_rows(_staged_totals(), season=2025, line_type="open")

    assert len(rows) == 2
    assert rows[0] == (
        2025,
        "api-1",
        "open",
        datetime.fromisoformat("2025-04-01T23:05:00+00:00"),
        "2025-04-01",
        "Away",
        "Home",
        "book-a",
        datetime.fromisoformat("2025-04-01T16:30:00+00:00"),
        datetime.fromisoformat("2025-04-01T16:25:00+00:00"),
        8.5,
        -110,
        -110,
    )
    assert rows[1][1] == "api-unmatched"


def test_build_resolved_totals_rows_keeps_only_safe_game_pk_matches() -> None:
    rows = build_resolved_totals_rows(
        _staged_totals(),
        gid_to_pk={
            "api-1": (
                777001,
                112,
                138,
                datetime.fromisoformat("2025-04-01T23:05:00+00:00"),
            )
        },
        line_type="open",
    )

    assert rows == [
        (
            777001,
            datetime.fromisoformat("2025-04-01T23:05:00+00:00").date(),
            138,
            112,
            "book-a",
            "open",
            8.5,
            -110,
            -110,
            datetime.fromisoformat("2025-04-01T16:30:00+00:00"),
        )
    ]
