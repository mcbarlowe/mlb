from __future__ import annotations

from datetime import date

import pytest

from src.betting.results import (
    date_range,
    event_ids_from_core_payload,
    parse_espn_summary,
)


def summary_payload(*, completed: bool = True) -> dict:
    return {
        "header": {
            "id": "401585183",
            "season": {"year": 2024, "type": 2},
            "week": 12,
            "competitions": [
                {
                    "id": "401585183",
                    "date": "2024-01-15T18:00Z",
                    "neutralSite": False,
                    "status": {"type": {"name": "STATUS_FINAL" if completed else "STATUS_SCHEDULED", "completed": completed}},
                    "competitors": [
                        {
                            "homeAway": "home",
                            "winner": True,
                            "score": "124",
                            "team": {"displayName": "Philadelphia 76ers"},
                        },
                        {
                            "homeAway": "away",
                            "winner": False,
                            "score": "115",
                            "team": {"displayName": "Houston Rockets"},
                        },
                    ],
                }
            ],
        }
    }


def test_parse_espn_summary_completed_result() -> None:
    result = parse_espn_summary(summary_payload(), sport="NBA")

    assert result is not None
    assert result.sport == "NBA"
    assert result.source_event_id == "401585183"
    assert result.season == 2024
    assert result.week == 12
    assert result.game_date == date(2024, 1, 15)
    assert result.home_team == "Philadelphia 76ers"
    assert result.away_team == "Houston Rockets"
    assert result.home_score == 124
    assert result.away_score == 115
    assert result.home_won is True


def test_parse_espn_summary_skips_incomplete_by_default() -> None:
    assert parse_espn_summary(summary_payload(completed=False), sport="NBA") is None


def test_event_ids_from_core_payload_deduplicates_refs() -> None:
    payload = {
        "items": [
            {"$ref": "http://sports.core.api.espn.com/v2/sports/football/leagues/college-football/events/401628354?lang=en"},
            {"$ref": "http://sports.core.api.espn.com/v2/sports/football/leagues/college-football/events/401628354?lang=en"},
            {"$ref": "http://sports.core.api.espn.com/v2/sports/football/leagues/college-football/events/401628361?lang=en"},
        ]
    }

    assert event_ids_from_core_payload(payload) == ["401628354", "401628361"]


def test_date_range_is_inclusive() -> None:
    assert date_range(date(2024, 1, 1), date(2024, 1, 3)) == [
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 1, 3),
    ]


def test_date_range_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="end"):
        date_range(date(2024, 1, 2), date(2024, 1, 1))
