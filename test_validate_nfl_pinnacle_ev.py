from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from scripts import validate_nfl_pinnacle_ev as validator


def quote(book: str, home: int, away: int, snap: datetime) -> validator.Quote:
    return validator.Quote(bookmaker=book, home_ml=home, away_ml=away, snapshot_time=snap)


def game(quotes: dict[str, validator.Quote], *, home_won: bool = True) -> validator.GameQuotes:
    return validator.GameQuotes(
        season=2025,
        game_id="2025_01_ARI_BUF",
        game_type="REG",
        week=1,
        commence_time=datetime(2025, 9, 7, 17, 0, tzinfo=UTC),
        horizon_hours=4.0,
        home_won=home_won,
        quotes=quotes,
    )


def test_quotes_from_event_maps_unordered_home_and_away_outcomes() -> None:
    event = {
        "home_team": "Buffalo Bills",
        "away_team": "Arizona Cardinals",
        "bookmakers": [
            {
                "key": "pinnacle",
                "last_update": "2024-09-08T15:55:00Z",
                "markets": [
                    {
                        "key": "h2h",
                        "last_update": "2024-09-08T15:56:00Z",
                        "outcomes": [
                            {"name": "Arizona Cardinals", "price": 250},
                            {"name": "Buffalo Bills", "price": -310},
                        ],
                    }
                ],
            }
        ],
    }

    quotes = validator.quotes_from_event(event, datetime(2024, 9, 8, 15, 55, tzinfo=UTC))

    assert quotes["pinnacle"].home_ml == -310
    assert quotes["pinnacle"].away_ml == 250
    assert quotes["pinnacle"].snapshot_time == datetime(2024, 9, 8, 15, 56, tzinfo=UTC)


def test_event_matching_accepts_washington_rename_alias() -> None:
    nfl_game = validator.NflResult(
        season=2021,
        game_id="2021_01_LAC_WAS",
        game_type="REG",
        week=1,
        commence_time=datetime(2021, 9, 12, 17, 0, tzinfo=UTC),
        home_team="Washington Commanders",
        away_team="Los Angeles Chargers",
        home_aliases=validator.aliases_for_team("WAS"),
        away_aliases=validator.aliases_for_team("LAC"),
        home_score=16,
        away_score=20,
    )
    event = {
        "home_team": "Washington Football Team",
        "away_team": "Los Angeles Chargers",
        "commence_time": "2021-09-12T17:00:00Z",
    }

    assert validator.event_matches_game(event, nfl_game, tolerance=timedelta(minutes=1))


def test_evaluate_games_uses_only_time_aligned_selected_us_books() -> None:
    snap = datetime(2025, 9, 7, 13, 0, tzinfo=UTC)
    evaluated, drops = validator.evaluate_games(
        [
            game(
                {
                    "pinnacle": quote("pinnacle", -110, -110, snap),
                    "aligned": quote("aligned", +120, -130, snap + timedelta(minutes=30)),
                    "stale": quote("stale", +500, -700, snap + timedelta(hours=8)),
                    "off_panel": quote("off_panel", +300, -400, snap),
                }
            )
        ],
        pinnacle_book="pinnacle",
        us_books=("aligned", "stale"),
        max_gap_hours=1.0,
        min_us_books=1,
        devig="proportional",
    )

    assert not drops
    assert len(evaluated) == 1
    assert evaluated[0].us_books == 1
    assert evaluated[0].best_book == "aligned"
    assert evaluated[0].best_side == "home"


def test_settle_uses_pinnacle_probability_for_ev_threshold() -> None:
    snap = datetime(2025, 9, 7, 13, 0, tzinfo=UTC)
    evaluated, _ = validator.evaluate_games(
        [
            game(
                {
                    "pinnacle": quote("pinnacle", -110, -110, snap),
                    "us": quote("us", +120, -130, snap),
                },
                home_won=True,
            )
        ],
        pinnacle_book="pinnacle",
        us_books=("us",),
        max_gap_hours=1.0,
        min_us_books=1,
        devig="proportional",
    )

    bet = validator.settle(evaluated, 0.05)[0]

    assert bet.side == "home"
    assert bet.ev == pytest.approx(0.10)
    assert bet.won is True
    assert bet.ret == pytest.approx(1.20)
    assert validator.settle(evaluated, 0.20) == []
