from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from scripts import validate_pinnacle_cross_sport_horizon_ev as validator


def test_event_match_accepts_alias_and_time_tolerance() -> None:
    game = validator.SportResult(
        sport="NBA",
        season=2024,
        game_id="1",
        game_date=date(2024, 1, 1),
        commence_time=datetime(2024, 1, 1, 23, 0, tzinfo=UTC),
        home_team="LA Clippers",
        away_team="Boston Celtics",
        home_aliases=validator.aliases_for_team("LA Clippers"),
        away_aliases=validator.aliases_for_team("Boston Celtics"),
        home_won=True,
    )
    event = {
        "home_team": "Los Angeles Clippers",
        "away_team": "Boston Celtics",
        "commence_time": "2024-01-01T23:03:00Z",
    }

    assert validator.event_match_delta(event, game, tolerance=validator.timedelta(minutes=5)) is not None
    assert validator.event_match_delta(event, game, tolerance=validator.timedelta(minutes=2)) is None


def test_quotes_from_event_parses_home_away_moneylines() -> None:
    event = {
        "home_team": "Philadelphia 76ers",
        "away_team": "Houston Rockets",
        "bookmakers": [
            {
                "key": "pinnacle",
                "last_update": "2024-01-01T20:00:00Z",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Philadelphia 76ers", "price": -150},
                            {"name": "Houston Rockets", "price": 130},
                        ],
                    }
                ],
            }
        ],
    }

    quotes = validator.quotes_from_event(event, datetime(2024, 1, 1, 20, 0, tzinfo=UTC))

    assert quotes["pinnacle"].home_ml == -150
    assert quotes["pinnacle"].away_ml == 130


def test_evaluate_games_chooses_best_plus_ev_side() -> None:
    game = validator.GameQuotes(
        sport="NBA",
        season=2024,
        game_id="1",
        game_date=date(2024, 1, 1),
        commence_time=datetime(2024, 1, 1, 23, 0, tzinfo=UTC),
        horizon_hours=1.0,
        home_won=True,
        quotes={
            "pinnacle": validator.Quote(
                bookmaker="pinnacle",
                home_ml=-110,
                away_ml=-110,
                snapshot_time=datetime(2024, 1, 1, 22, 0, tzinfo=UTC),
                last_update=None,
            ),
            "fanduel": validator.Quote(
                bookmaker="fanduel",
                home_ml=120,
                away_ml=-130,
                snapshot_time=datetime(2024, 1, 1, 22, 0, tzinfo=UTC),
                last_update=None,
            ),
        },
    )

    evaluated, drops = validator.evaluate_games(
        [game], pinnacle_book="pinnacle", us_books=("fanduel",), min_us_books=1, devig="proportional"
    )

    assert not drops
    assert evaluated[0].best_side == "home"
    assert evaluated[0].best_ev == pytest.approx(0.10)
