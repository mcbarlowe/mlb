"""Parsing helpers for MLB first-five-innings odds."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    "F5_H2H_MARKET",
    "F5_MARKETS",
    "F5_SPREADS_MARKET",
    "F5_TOTALS_MARKET",
    "F5OddsApiRow",
    "odds_games",
    "parse_f5_odds_rows",
]

F5_H2H_MARKET = "h2h_1st_5_innings"
F5_SPREADS_MARKET = "spreads_1st_5_innings"
F5_TOTALS_MARKET = "totals_1st_5_innings"
F5_MARKETS = (F5_H2H_MARKET, F5_SPREADS_MARKET, F5_TOTALS_MARKET)


@dataclass(frozen=True)
class F5OddsApiRow:
    game_id: str
    commence_time: str | None
    away_team: str
    home_team: str
    bookmaker: str
    h2h_last_update: str | None = None
    spreads_last_update: str | None = None
    totals_last_update: str | None = None
    home_ml: float | None = None
    away_ml: float | None = None
    home_spread: float | None = None
    home_spread_ml: float | None = None
    away_spread: float | None = None
    away_spread_ml: float | None = None
    total_point: float | None = None
    over_ml: float | None = None
    under_ml: float | None = None


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


def _market_by_key(book: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    markets = book.get("markets")
    if not isinstance(markets, list):
        return {}
    return {
        str(market.get("key")): market
        for market in markets
        if isinstance(market, Mapping) and market.get("key") in F5_MARKETS
    }


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


def parse_f5_odds_rows(payload: object) -> list[F5OddsApiRow]:
    rows: list[F5OddsApiRow] = []
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
            markets = _market_by_key(book)
            h2h = markets.get(F5_H2H_MARKET)
            spreads = markets.get(F5_SPREADS_MARKET)
            totals = markets.get(F5_TOTALS_MARKET)
            if h2h is None and spreads is None and totals is None:
                continue
            h2h_outcomes = _outcomes_by_name(h2h)
            spread_outcomes = _outcomes_by_name(spreads)
            total_outcomes = _outcomes_by_name(totals)
            home_h2h = h2h_outcomes.get(str(home_team))
            away_h2h = h2h_outcomes.get(str(away_team))
            home_spread = spread_outcomes.get(str(home_team))
            away_spread = spread_outcomes.get(str(away_team))
            over = total_outcomes.get("Over")
            under = total_outcomes.get("Under")
            rows.append(
                F5OddsApiRow(
                    game_id=str(game_id),
                    commence_time=str(game.get("commence_time"))
                    if game.get("commence_time") is not None
                    else None,
                    away_team=str(away_team),
                    home_team=str(home_team),
                    bookmaker=str(book.get("key") or book.get("title") or "unknown"),
                    h2h_last_update=str(h2h.get("last_update")) if h2h else None,
                    spreads_last_update=str(spreads.get("last_update")) if spreads else None,
                    totals_last_update=str(totals.get("last_update")) if totals else None,
                    home_ml=_price(home_h2h),
                    away_ml=_price(away_h2h),
                    home_spread=_point(home_spread),
                    home_spread_ml=_price(home_spread),
                    away_spread=_point(away_spread),
                    away_spread_ml=_price(away_spread),
                    total_point=_point(over),
                    over_ml=_price(over),
                    under_ml=_price(under),
                )
            )
    return rows
