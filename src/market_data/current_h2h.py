"""Normalize current MLB h2h boards into close-line market-data rows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime

import requests

from src.market_data.team_mapping import team_abbrev_to_id
from src.sim.slate import SlateGame

CURRENT_ODDS_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"

NAME_ALIASES = {
    "Cleveland Guardians": "Cleveland Indians",
    "Miami Marlins": "Florida Marlins",
}


@dataclass(frozen=True)
class PaperOddsLine:
    bookmaker: str
    home_ml: float
    away_ml: float
    last_update: str | None = None


def _parse_date(value: str | None) -> date:
    if value is None:
        return datetime.now(tz=UTC).astimezone().date()
    return date.fromisoformat(value)


def _parse_datetime(value: object | None) -> datetime | None:
    if value is None:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _fetch_current_odds(api_key: str, regions: str) -> tuple[object, dict[str, str]]:
    response = requests.get(
        CURRENT_ODDS_URL,
        params={
            "apiKey": api_key,
            "regions": regions,
            "markets": "h2h",
            "oddsFormat": "american",
        },
        timeout=30,
    )
    response.raise_for_status()
    headers = {
        "x-requests-last": response.headers.get("x-requests-last", ""),
        "x-requests-remaining": response.headers.get("x-requests-remaining", ""),
        "x-requests-used": response.headers.get("x-requests-used", ""),
    }
    return response.json(), headers


def _odds_games(payload: object) -> list[Mapping[str, object]]:
    if isinstance(payload, Mapping):
        data = payload.get("data")
        if isinstance(data, list):
            return [item for item in data if isinstance(item, Mapping)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    return []


def _resolve_team_id(name: object, mapping: Mapping[str, int]) -> int | None:
    if name is None:
        return None
    text = str(name).strip()
    alias = NAME_ALIASES.get(text, text)
    return mapping.get(alias.upper())


def _market_outcomes(book: Mapping[str, object]) -> Mapping[str, object] | None:
    markets = book.get("markets")
    if not isinstance(markets, list):
        return None
    for market in markets:
        if isinstance(market, Mapping) and market.get("key") == "h2h":
            return market
    return None


def _match_slate_game(
    *,
    odds_game: Mapping[str, object],
    slate_by_pair: Mapping[tuple[int, int], Sequence[SlateGame]],
    team_mapping: Mapping[str, int],
    max_hours: float,
) -> SlateGame | None:
    away_id = _resolve_team_id(odds_game.get("away_team"), team_mapping)
    home_id = _resolve_team_id(odds_game.get("home_team"), team_mapping)
    if away_id is None or home_id is None:
        return None
    candidates = slate_by_pair.get((away_id, home_id), ())
    if not candidates:
        return None
    odds_time = _parse_datetime(odds_game.get("commence_time"))
    if odds_time is None:
        return candidates[0]
    scored: list[tuple[float, SlateGame]] = []
    for game in candidates:
        game_time = _parse_datetime(game.game_datetime)
        if game_time is None:
            continue
        diff_hours = abs((game_time - odds_time).total_seconds()) / 3600.0
        scored.append((diff_hours, game))
    if not scored:
        return candidates[0]
    diff, game = min(scored, key=lambda item: item[0])
    return game if diff <= max_hours else None


def _odds_by_game_pk(
    *,
    payload: object,
    slate_games: Sequence[SlateGame],
    max_match_hours: float,
) -> dict[int, list[PaperOddsLine]]:
    team_mapping = team_abbrev_to_id()
    slate_by_pair: dict[tuple[int, int], list[SlateGame]] = defaultdict(list)
    for game in slate_games:
        slate_by_pair[(game.away_team_id, game.home_team_id)].append(game)

    out: dict[int, list[PaperOddsLine]] = defaultdict(list)
    for odds_game in _odds_games(payload):
        slate_game = _match_slate_game(
            odds_game=odds_game,
            slate_by_pair=slate_by_pair,
            team_mapping=team_mapping,
            max_hours=max_match_hours,
        )
        if slate_game is None:
            continue
        home_name = odds_game.get("home_team")
        away_name = odds_game.get("away_team")
        bookmakers = odds_game.get("bookmakers")
        if not isinstance(bookmakers, list):
            continue
        for book in bookmakers:
            if not isinstance(book, Mapping):
                continue
            market = _market_outcomes(book)
            if market is None:
                continue
            outcomes = market.get("outcomes")
            if not isinstance(outcomes, list):
                continue
            prices = {
                str(item.get("name")): item.get("price")
                for item in outcomes
                if isinstance(item, Mapping)
            }
            home_ml = prices.get(str(home_name))
            away_ml = prices.get(str(away_name))
            if home_ml is None or away_ml is None:
                continue
            out[slate_game.game_pk].append(
                PaperOddsLine(
                    bookmaker=str(book.get("key") or book.get("title") or "unknown"),
                    home_ml=float(home_ml),
                    away_ml=float(away_ml),
                    last_update=str(market.get("last_update") or ""),
                )
            )
    return dict(out)


def _h2h_odds_rows(
    *,
    slate_games: Sequence[SlateGame],
    odds_by_game: Mapping[int, Sequence[PaperOddsLine]],
    snapshot_time: str,
    line_type: str,
    source: str,
) -> list[dict[str, object]]:
    slate_by_pk = {game.game_pk: game for game in slate_games}
    rows: list[dict[str, object]] = []
    for game_pk, lines in odds_by_game.items():
        game = slate_by_pk.get(game_pk)
        if game is None:
            continue
        for line in lines:
            rows.append(
                {
                    "game_pk": game.game_pk,
                    "game_date": game.slate_date,
                    "away_team_id": game.away_team_id,
                    "home_team_id": game.home_team_id,
                    "bookmaker": line.bookmaker,
                    "market": "h2h",
                    "line_type": line_type,
                    "home_ml": line.home_ml,
                    "away_ml": line.away_ml,
                    "snapshot_time": snapshot_time,
                    "source": source,
                }
            )
    return rows
