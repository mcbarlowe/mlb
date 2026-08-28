from __future__ import annotations

from src.market_data.f5_odds import parse_f5_odds_rows
from src.market_data.f5_odds_store import normalize_f5_odds_row


def _payload() -> list[dict[str, object]]:
    return [
        {
            "id": "event-1",
            "commence_time": "2026-08-14T18:20:00Z",
            "away_team": "St. Louis Cardinals",
            "home_team": "Chicago Cubs",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "h2h_1st_5_innings",
                            "last_update": "2026-08-14T12:00:00Z",
                            "outcomes": [
                                {"name": "St. Louis Cardinals", "price": 110},
                                {"name": "Chicago Cubs", "price": -130},
                            ],
                        },
                        {
                            "key": "spreads_1st_5_innings",
                            "last_update": "2026-08-14T12:01:00Z",
                            "outcomes": [
                                {"name": "St. Louis Cardinals", "point": 0.5, "price": -145},
                                {"name": "Chicago Cubs", "point": -0.5, "price": 120},
                            ],
                        },
                        {
                            "key": "totals_1st_5_innings",
                            "last_update": "2026-08-14T12:02:00Z",
                            "outcomes": [
                                {"name": "Over", "point": 4.5, "price": -115},
                                {"name": "Under", "point": 4.5, "price": -105},
                            ],
                        },
                    ],
                }
            ],
        }
    ]


def test_parse_f5_odds_rows_extracts_moneyline_spread_and_total() -> None:
    rows = parse_f5_odds_rows(_payload())

    assert len(rows) == 1
    row = rows[0]
    assert row.game_id == "event-1"
    assert row.bookmaker == "draftkings"
    assert row.away_ml == 110
    assert row.home_ml == -130
    assert row.away_spread == 0.5
    assert row.away_spread_ml == -145
    assert row.home_spread == -0.5
    assert row.home_spread_ml == 120
    assert row.total_point == 4.5
    assert row.over_ml == -115
    assert row.under_ml == -105
    assert row.h2h_last_update == "2026-08-14T12:00:00Z"
    assert row.spreads_last_update == "2026-08-14T12:01:00Z"
    assert row.totals_last_update == "2026-08-14T12:02:00Z"


def test_normalize_f5_odds_row_coerces_numeric_fields_and_defaults_source() -> None:
    normalized = normalize_f5_odds_row(
        {
            "game_pk": "123",
            "game_date": "2026-08-14",
            "game_time": "2026-08-14T18:20:00Z",
            "away_team": "STL",
            "home_team": "CHC",
            "away_team_id": "138",
            "home_team_id": "112",
            "bookmaker": "draftkings",
            "line_type": "current",
            "snapshot_time": "2026-08-14T12:05:00+00:00",
            "h2h_last_update": "2026-08-14T12:00:00Z",
            "spreads_last_update": "",
            "totals_last_update": None,
            "home_ml": "-130",
            "away_ml": "110",
            "home_spread": "-0.5",
            "home_spread_ml": "120",
            "away_spread": "0.5",
            "away_spread_ml": "-145",
            "total_point": "4.5",
            "over_ml": "-115",
            "under_ml": "-105",
        }
    )

    assert normalized["game_pk"] == 123
    assert normalized["away_team_id"] == 138
    assert normalized["home_ml"] == -130
    assert normalized["away_spread"] == 0.5
    assert normalized["spreads_last_update"] is None
    assert normalized["source"] == "the-odds-api"
