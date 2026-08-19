from __future__ import annotations

from datetime import UTC, datetime

from scripts.fetch_f5_odds_history import (
    SOURCE,
    HistoricalGame,
    _line_timestamp,
    _match_event,
    _rows_from_event_odds,
)


def _game() -> HistoricalGame:
    return HistoricalGame(
        game_pk=777001,
        season=2025,
        game_date="2025-07-01",
        game_time=datetime(2025, 7, 1, 23, 5, tzinfo=UTC),
        away_team="NYY",
        home_team="TOR",
        away_team_id=147,
        home_team_id=141,
    )


def test_line_timestamp_uses_open_and_close_offsets() -> None:
    game = _game()

    assert _line_timestamp(game, "open", 8.0, 10.0) == "2025-07-01T15:05:00Z"
    assert _line_timestamp(game, "close", 8.0, 10.0) == "2025-07-01T22:55:00Z"


def test_match_event_resolves_by_team_alias_and_nearest_start_time() -> None:
    game = _game()
    events = [
        {
            "id": "wrong-time",
            "away_team": "New York Yankees",
            "home_team": "Toronto Blue Jays",
            "commence_time": "2025-07-01T08:00:00Z",
        },
        {
            "id": "matched",
            "away_team": "New York Yankees",
            "home_team": "Toronto Blue Jays",
            "commence_time": "2025-07-01T23:07:00Z",
        },
    ]
    mapping = {
        "NEW YORK YANKEES": 147,
        "TORONTO BLUE JAYS": 141,
    }

    matched = _match_event(
        events=events,
        game=game,
        team_mapping=mapping,
        max_match_hours=12.0,
    )

    assert matched is not None
    assert matched["id"] == "matched"


def test_rows_from_event_odds_extracts_f5_total_book_rows() -> None:
    game = _game()
    payload = {
        "timestamp": "2025-07-01T22:55:00Z",
        "data": {
            "id": "event-1",
            "commence_time": "2025-07-01T23:05:00Z",
            "away_team": "New York Yankees",
            "home_team": "Toronto Blue Jays",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "markets": [
                        {
                            "key": "totals_1st_5_innings",
                            "last_update": "2025-07-01T22:50:00Z",
                            "outcomes": [
                                {"name": "Over", "point": 4.5, "price": -115},
                                {"name": "Under", "point": 4.5, "price": -105},
                            ],
                        }
                    ],
                }
            ],
        },
    }

    rows = _rows_from_event_odds(
        payload=payload,
        game=game,
        line_type="close",
        snapshot_time="2025-07-01T22:55:00Z",
    )

    assert rows == [
        {
            "game_pk": 777001,
            "season": 2025,
            "game_date": "2025-07-01",
            "game_time": "2025-07-01T23:05:00+00:00",
            "away_team": "NYY",
            "home_team": "TOR",
            "away_team_id": 147,
            "home_team_id": 141,
            "bookmaker": "fanduel",
            "line_type": "close",
            "snapshot_time": "2025-07-01T22:55:00Z",
            "h2h_last_update": None,
            "spreads_last_update": None,
            "totals_last_update": "2025-07-01T22:50:00Z",
            "home_ml": None,
            "away_ml": None,
            "home_spread": None,
            "home_spread_ml": None,
            "away_spread": None,
            "away_spread_ml": None,
            "total_point": 4.5,
            "over_ml": -115.0,
            "under_ml": -105.0,
            "source": SOURCE,
        }
    ]
