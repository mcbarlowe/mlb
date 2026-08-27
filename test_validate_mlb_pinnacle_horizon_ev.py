from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from scripts import validate_mlb_pinnacle_horizon_ev as validator


def quote(book: str, home: int, away: int) -> validator.Quote:
    snap = datetime(2025, 4, 1, 0, 0, tzinfo=UTC)
    return validator.Quote(
        bookmaker=book,
        home_ml=home,
        away_ml=away,
        snapshot_time=snap,
        last_update=snap,
    )


def game_quotes(
    quotes: dict[str, validator.Quote], *, home_won: bool = True
) -> validator.GameQuotes:
    return validator.GameQuotes(
        season=2025,
        game_pk=1,
        game_date=date(2025, 4, 1),
        commence_time=datetime(2025, 4, 2, 0, 0, tzinfo=UTC),
        horizon_hours=24.0,
        home_won=home_won,
        quotes=quotes,
    )


def settled(game_date: date, ret: float) -> validator.SettledBet:
    return validator.SettledBet(
        season=2025,
        game_pk=1,
        game_date=game_date,
        horizon_hours=24.0,
        side="home",
        book="us",
        decimal_odds=2.0,
        probability=0.5,
        ev=0.0,
        won=ret > 0,
        ret=ret,
    )


def test_game_quotes_from_snapshot_accepts_cleveland_indians_alias() -> None:
    game = validator.MlbResult(
        season=2021,
        game_pk=1,
        game_date=date(2021, 4, 1),
        commence_time=datetime(2021, 4, 2, 0, 0, tzinfo=UTC),
        home_team="Cleveland Guardians",
        away_team="Detroit Tigers",
        home_aliases=validator.aliases_for_team("Cleveland Guardians"),
        away_aliases=validator.aliases_for_team("Detroit Tigers"),
        home_won=True,
    )
    body = {
        "timestamp": "2021-04-01T00:00:00Z",
        "data": [
            {
                "commence_time": "2021-04-02T00:00:00Z",
                "home_team": "Cleveland Indians",
                "away_team": "Detroit Tigers",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "last_update": "2021-04-01T00:00:00Z",
                                "outcomes": [
                                    {"name": "Cleveland Indians", "price": -110},
                                    {"name": "Detroit Tigers", "price": -110},
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }

    result = validator.game_quotes_from_snapshot(
        game,
        body,
        horizon_hours=24.0,
        commence_tolerance_hours=1.0,
    )

    assert result is not None
    assert result.quotes["pinnacle"].home_ml == -110


def test_evaluate_games_uses_pinnacle_probability_for_ev_threshold() -> None:
    evaluated, drops = validator.evaluate_games(
        [
            game_quotes(
                {
                    "pinnacle": quote("pinnacle", -110, -110),
                    "us": quote("us", +120, -130),
                },
                home_won=True,
            )
        ],
        pinnacle_book="pinnacle",
        us_books=("us",),
        min_us_books=1,
        devig="proportional",
    )

    assert not drops
    bet = validator.settle(evaluated, 0.05)[0]
    assert bet.side == "home"
    assert bet.ev == pytest.approx(0.10)
    assert bet.ret == pytest.approx(1.20)
    assert validator.settle(evaluated, 0.20) == []


def test_date_bootstrap_resamples_whole_dates() -> None:
    bets = [
        settled(date(2025, 4, 1), +1.0),
        settled(date(2025, 4, 1), -1.0),
    ]

    mean, lo, hi = validator.boot_ci_by_date(bets, samples=100, seed=1)

    assert mean == 0.0
    assert lo == 0.0
    assert hi == 0.0


def test_required_n_uses_normal_approximation() -> None:
    assert validator.required_n(1.0, 0.10, prospective_power=False) == pytest.approx(
        384.16
    )
    assert validator.required_n(1.0, 0.10, prospective_power=True) == pytest.approx(
        784.0
    )
