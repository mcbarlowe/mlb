"""Parsing helpers for NRFI/YRFI odds from first-inning totals."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    "NRFI_YRFI_MARKET",
    "NrfiYrfiOddsApiRow",
    "odds_games",
    "parse_nrfi_yrfi_odds_rows",
]

NRFI_YRFI_MARKET = "totals_1st_1_innings"


@dataclass(frozen=True)
class NrfiYrfiOddsApiRow:
    game_id: str
    commence_time: str | None
    away_team: str
    home_team: str
    bookmaker: str
    market_last_update: str | None = None
    total_point: float | None = None
    yrfi_ml: float | None = None
    nrfi_ml: float | None = None


def odds_games(payload: object) -> list[Mapping[str, object]]:
    if isinstance(payload, Mapping):
        data = payload.get("data")
        if isinstance(data, list):
            return [item for item in data if isinstance(item, Mapping)]
        if payload.get("bookmakers") is not None and payload.get("id") is not None:
            return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    return []


def _market(book: Mapping[str, object]) -> Mapping[str, object] | None:
    markets = book.get("markets")
    if not isinstance(markets, list):
        return None
    for market in markets:
        if isinstance(market, Mapping) and market.get("key") == NRFI_YRFI_MARKET:
            return market
    return None


def _outcomes_by_name(market: Mapping[str, object] | None) -> dict[str, Mapping[str, object]]:
    if market is None:
        return {}
    outcomes = market.get("outcomes")
    if not isinstance(outcomes, list):
        return {}
    return {
        str(outcome.get("name")): outcome
        for outcome in outcomes
        if isinstance(outcome, Mapping)
    }


def _price(outcome: Mapping[str, object] | None) -> float | None:
    if outcome is None:
        return None
    value = outcome.get("price")
    if value is None:
        return None
    return float(str(value))


def _point(outcome: Mapping[str, object] | None) -> float | None:
    if outcome is None:
        return None
    value = outcome.get("point")
    if value is None:
        return None
    return float(str(value))


def parse_nrfi_yrfi_odds_rows(payload: object) -> list[NrfiYrfiOddsApiRow]:
    rows: list[NrfiYrfiOddsApiRow] = []
    for game in odds_games(payload):
        game_id = game.get("id")
        away_team = game.get("away_team")
        home_team = game.get("home_team")
        bookmakers = game.get("bookmakers")
        if (
            game_id is None
            or away_team is None
            or home_team is None
            or not isinstance(bookmakers, list)
        ):
            continue
        for book in bookmakers:
            if not isinstance(book, Mapping):
                continue
            market = _market(book)
            if market is None:
                continue
            outcomes = _outcomes_by_name(market)
            yrfi = outcomes.get("Over") or outcomes.get("Yes")
            nrfi = outcomes.get("Under") or outcomes.get("No")
            if yrfi is None and nrfi is None:
                continue
            rows.append(
                NrfiYrfiOddsApiRow(
                    game_id=str(game_id),
                    commence_time=str(game.get("commence_time"))
                    if game.get("commence_time") is not None
                    else None,
                    away_team=str(away_team),
                    home_team=str(home_team),
                    bookmaker=str(book.get("key") or book.get("title") or "unknown"),
                    market_last_update=str(market.get("last_update"))
                    if market.get("last_update") is not None
                    else None,
                    total_point=_point(yrfi) if _point(yrfi) is not None else _point(nrfi),
                    yrfi_ml=_price(yrfi),
                    nrfi_ml=_price(nrfi),
                )
            )
    return rows
