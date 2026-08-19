from __future__ import annotations

import pytest

from src.betting.h2h_odds_store import normalize_h2h_odds_row


def test_normalize_h2h_odds_row_defaults_to_open_h2h_source():
    row = normalize_h2h_odds_row(
        {
            "game_pk": "123",
            "game_date": "2026-08-16",
            "away_team_id": "1",
            "home_team_id": "2",
            "bookmaker": "book_a",
            "home_ml": "-125.0",
            "away_ml": "110.0",
            "snapshot_time": "2026-08-16T12:01:00+00:00",
        }
    )

    assert row == {
        "game_pk": 123,
        "game_date": "2026-08-16",
        "away_team_id": 1,
        "home_team_id": 2,
        "bookmaker": "book_a",
        "market": "h2h",
        "line_type": "open",
        "home_ml": -125,
        "away_ml": 110,
        "snapshot_time": "2026-08-16T12:01:00+00:00",
        "source": "the-odds-api",
    }


def test_normalize_h2h_odds_row_requires_bookmaker():
    with pytest.raises(ValueError, match="bookmaker"):
        normalize_h2h_odds_row(
            {
                "game_pk": 123,
                "home_ml": -125,
                "away_ml": 110,
                "snapshot_time": "2026-08-16T12:01:00+00:00",
            }
        )
