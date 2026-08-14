from __future__ import annotations

from src.betting.nrfi_yrfi_odds import parse_nrfi_yrfi_odds_rows
from src.betting.nrfi_yrfi_odds_store import normalize_nrfi_yrfi_odds_row


def test_parse_nrfi_yrfi_odds_rows_maps_first_inning_total_sides() -> None:
    payload = {
        "id": "event-1",
        "commence_time": "2026-08-14T18:20:00Z",
        "away_team": "St. Louis Cardinals",
        "home_team": "Chicago Cubs",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "totals_1st_1_innings",
                        "last_update": "2026-08-14T12:00:00Z",
                        "outcomes": [
                            {"name": "Over", "point": 0.5, "price": -135},
                            {"name": "Under", "point": 0.5, "price": 105},
                        ],
                    }
                ],
            }
        ],
    }

    rows = parse_nrfi_yrfi_odds_rows(payload)

    assert len(rows) == 1
    row = rows[0]
    assert row.game_id == "event-1"
    assert row.bookmaker == "draftkings"
    assert row.total_point == 0.5
    assert row.yrfi_ml == -135
    assert row.nrfi_ml == 105
    assert row.market_last_update == "2026-08-14T12:00:00Z"


def test_parse_nrfi_yrfi_odds_rows_accepts_yes_no_labels() -> None:
    payload = [
        {
            "id": "event-1",
            "away_team": "Away",
            "home_team": "Home",
            "bookmakers": [
                {
                    "key": "book",
                    "markets": [
                        {
                            "key": "totals_1st_1_innings",
                            "outcomes": [
                                {"name": "Yes", "point": 0.5, "price": -120},
                                {"name": "No", "point": 0.5, "price": 100},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    row = parse_nrfi_yrfi_odds_rows(payload)[0]

    assert row.yrfi_ml == -120
    assert row.nrfi_ml == 100


def test_normalize_nrfi_yrfi_odds_row_coerces_fields_and_defaults_market() -> None:
    normalized = normalize_nrfi_yrfi_odds_row(
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
            "market_last_update": "2026-08-14T12:00:00Z",
            "total_point": "0.5",
            "yrfi_ml": "-135",
            "nrfi_ml": "105",
        }
    )

    assert normalized["game_pk"] == 123
    assert normalized["away_team_id"] == 138
    assert normalized["market_key"] == "totals_1st_1_innings"
    assert normalized["total_point"] == 0.5
    assert normalized["yrfi_ml"] == -135
    assert normalized["nrfi_ml"] == 105
    assert normalized["source"] == "the-odds-api"
