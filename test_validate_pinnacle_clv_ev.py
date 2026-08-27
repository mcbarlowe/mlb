from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from scripts import validate_pinnacle_clv_ev as validator


def quote(book: str, home: int, away: int, snap: datetime) -> validator.Quote:
    return validator.Quote(bookmaker=book, home_ml=home, away_ml=away, snapshot_time=snap)


def game(quotes: dict[str, validator.Quote], *, home_won: bool = True) -> validator.GameQuotes:
    return validator.GameQuotes(
        season=2025,
        game_pk=1,
        game_datetime=datetime(2025, 4, 1, 23, 0, tzinfo=UTC),
        month=4,
        home_won=home_won,
        quotes=quotes,
    )


def test_evaluate_games_uses_only_time_aligned_us_quotes() -> None:
    snap = datetime(2025, 4, 1, 18, 0, tzinfo=UTC)
    evaluated, drops = validator.evaluate_games(
        [
            game(
                {
                    "pinnacle": quote("pinnacle", -110, -110, snap),
                    "aligned": quote("aligned", +120, -130, snap + timedelta(minutes=15)),
                    "stale": quote("stale", +500, -700, snap + timedelta(hours=8)),
                }
            )
        ],
        pinnacle_book="pinnacle",
        max_gap_hours=1.0,
        min_us_books=1,
        devig="proportional",
        exclude_months=set(),
    )

    assert not drops
    assert len(evaluated) == 1
    assert evaluated[0].us_books == 1
    assert evaluated[0].best_book == "aligned"
    assert evaluated[0].best_side == "home"


def test_settle_uses_pinnacle_probability_as_expected_roi_threshold() -> None:
    snap = datetime(2025, 4, 1, 18, 0, tzinfo=UTC)
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
        max_gap_hours=1.0,
        min_us_books=1,
        devig="proportional",
        exclude_months=set(),
    )

    bet = validator.settle(evaluated, 0.05)[0]
    assert bet.side == "home"
    assert bet.ev == pytest.approx(0.10)
    assert bet.won is True
    assert bet.ret == pytest.approx(1.20)
    assert validator.settle(evaluated, 0.20) == []
