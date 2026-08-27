"""Validate fixed-horizon Pinnacle-fair +EV line shopping on MLB moneylines.

This avoids the historical ``mlb.odds`` timestamp-collapse problem by requesting
The Odds API at an explicit pregame horizon for each game and caching the raw
snapshot locally. One API snapshot can serve multiple games when first-pitch
minutes match; cached snapshots are reused on reruns.

Example:
    uv run python scripts/validate_mlb_pinnacle_horizon_ev.py \
      --seasons 2021 2022 2023 2024 2025 \
      --hours-before 24 --thresholds 0.02
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import aiohttp
import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_moneyline_lineshop import PANEL_PRIORITY, PANEL_TOP5, PANEL_TOP6
from scripts.validate_pinnacle_clv_ev import (
    brier,
    calibration_rows,
    parse_thresholds,
    z_score,
)
from src.betting.odds import american_to_decimal, american_to_prob, no_vig_two_way
from src.database import PostgresConfig

SPORT_KEY = "baseball_mlb"
ODDS_URL = f"https://api.the-odds-api.com/v4/historical/sports/{SPORT_KEY}/odds"
PANELS = {
    "top5": PANEL_TOP5,
    "top6": PANEL_TOP6,
    "priority5": PANEL_PRIORITY[:5],
    "priority6": PANEL_PRIORITY[:6],
}
TEAM_NAME_VARIANTS: Mapping[str, tuple[str, ...]] = {
    "Cleveland Guardians": ("Cleveland Guardians", "Cleveland Indians"),
    "Miami Marlins": ("Miami Marlins", "Florida Marlins"),
    "Los Angeles Angels": ("Los Angeles Angels", "Los Angeles Angels of Anaheim"),
    "Athletics": ("Athletics", "Oakland Athletics"),
    "Oakland Athletics": ("Oakland Athletics", "Athletics"),
}


@dataclass(frozen=True)
class MlbResult:
    season: int
    game_pk: int
    game_date: date
    commence_time: datetime
    home_team: str
    away_team: str
    home_aliases: tuple[str, ...]
    away_aliases: tuple[str, ...]
    home_won: bool


@dataclass(frozen=True)
class Quote:
    bookmaker: str
    home_ml: int
    away_ml: int
    snapshot_time: datetime
    last_update: datetime | None


@dataclass(frozen=True)
class GameQuotes:
    season: int
    game_pk: int
    game_date: date
    commence_time: datetime
    horizon_hours: float
    home_won: bool
    quotes: dict[str, Quote]


@dataclass(frozen=True)
class EvaluatedGame:
    season: int
    game_pk: int
    game_date: date
    commence_time: datetime
    horizon_hours: float
    home_won: bool
    pin_home: float
    us_home: float
    pin_hold: float
    us_books: int
    snapshot_time: datetime
    best_side: str
    best_book: str
    best_decimal: float
    best_prob: float
    best_ev: float


@dataclass(frozen=True)
class SettledBet:
    season: int
    game_pk: int
    game_date: date
    horizon_hours: float
    side: str
    book: str
    decimal_odds: float
    probability: float
    ev: float
    won: bool
    ret: float


@dataclass
class FetchStats:
    api_calls: int = 0
    api_credits: int = 0
    cached_snapshots: int = 0
    last_remaining: int | None = None


class OddsApiError(RuntimeError):
    pass


def parse_seasons(values: Sequence[str]) -> tuple[int, ...]:
    seasons: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                seasons.append(int(part))
    if not seasons:
        raise SystemExit("at least one season is required")
    return tuple(dict.fromkeys(seasons))


def parse_books(value: str) -> tuple[str, ...] | None:
    key = value.strip().lower()
    if key == "all":
        return None
    if key in PANELS:
        return tuple(PANELS[key])
    books = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not books:
        raise SystemExit("--us-books must be a panel name, 'all', or comma-separated books")
    return books


def parse_game_types(value: str) -> tuple[str, ...]:
    game_types = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    if not game_types:
        raise SystemExit("--game-types must include at least one MLB game type")
    return tuple(dict.fromkeys(game_types))


def sql_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise SystemExit(f"unsafe SQL identifier: {value!r}")
    return value


def as_utc(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def aliases_for_team(team_name: str) -> tuple[str, ...]:
    variants = TEAM_NAME_VARIANTS.get(team_name, (team_name,))
    normalized = tuple(dict.fromkeys(normalize_name(name) for name in variants))
    if not normalized:
        raise ValueError(f"no team aliases for {team_name!r}")
    return normalized


def cache_name(value: str) -> str:
    digest = hashlib.sha1(value.encode()).hexdigest()[:12]
    safe = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return f"{safe}_{digest}.json"


def odds_cache_path(cache_dir: Path, params: Mapping[str, str]) -> Path:
    ordered = urlencode(sorted(params.items()))
    return cache_dir / "snapshots" / cache_name(ordered)


def load_results(
    *, seasons: tuple[int, ...], game_types: tuple[str, ...]
) -> tuple[list[MlbResult], Counter[str]]:
    cfg = PostgresConfig.from_env()
    schema = sql_identifier(cfg.schema)
    drops: Counter[str] = Counter()
    results: list[MlbResult] = []
    query = f"""
        SELECT
            g.season::int,
            g.game_pk,
            g.game_date::date,
            g.game_datetime,
            home.team_name AS home_team,
            away.team_name AS away_team,
            SUM(l.runs) FILTER (WHERE l.team_type = 'away')::int AS away_runs,
            SUM(l.runs) FILTER (WHERE l.team_type = 'home')::int AS home_runs
        FROM {schema}.games g
        JOIN {schema}.teams home ON home.team_id = g.home_team_id
        JOIN {schema}.teams away ON away.team_id = g.away_team_id
        JOIN {schema}.linescore l USING (game_pk)
        WHERE g.season::int = ANY(%s)
          AND g.game_type = ANY(%s)
          AND g.abstract_game_state = 'Final'
          AND g.game_datetime IS NOT NULL
        GROUP BY g.season, g.game_pk, g.game_date, g.game_datetime,
                 home.team_name, away.team_name
        ORDER BY g.game_datetime, g.game_pk
    """
    with psycopg.connect(
        dbname=cfg.dbname,
        user=cfg.user,
        password=cfg.password,
        host=cfg.host,
        port=cfg.port,
        connect_timeout=15,
    ) as conn, conn.cursor() as cur:
        cur.execute(query, (list(seasons), list(game_types)))
        for season, game_pk, game_date, game_dt, home, away, away_runs, home_runs in cur:
            if home_runs == away_runs:
                drops["tie"] += 1
                continue
            try:
                results.append(
                    MlbResult(
                        season=int(season),
                        game_pk=int(game_pk),
                        game_date=game_date,
                        commence_time=as_utc(game_dt),
                        home_team=str(home),
                        away_team=str(away),
                        home_aliases=aliases_for_team(str(home)),
                        away_aliases=aliases_for_team(str(away)),
                        home_won=home_runs > away_runs,
                    )
                )
            except ValueError as exc:
                drops[f"team_alias_error:{exc}"] += 1
    return results, drops


async def request_odds_snapshot(
    *,
    session: aiohttp.ClientSession,
    limiter: asyncio.Semaphore,
    cache_dir: Path,
    api_key: str,
    request_time: datetime,
    regions: str,
    markets: str,
    refresh: bool,
    timeout: float,
    stats: FetchStats,
) -> Mapping[str, Any]:
    request_time = request_time.astimezone(UTC)
    public_params = {
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
        "date": request_time.isoformat().replace("+00:00", "Z"),
    }
    path = odds_cache_path(cache_dir, public_params)
    if path.exists() and not refresh:
        stats.cached_snapshots += 1
        return json.loads(path.read_text())

    params = {**public_params, "apiKey": api_key}
    async with limiter:
        last_error: str | None = None
        for attempt in range(5):
            try:
                async with session.get(ODDS_URL, params=params, timeout=timeout) as response:
                    text = await response.text()
                    if response.status == 200:
                        stats.api_calls += 1
                        stats.api_credits += int(response.headers.get("x-requests-last") or 0)
                        remaining = response.headers.get("x-requests-remaining")
                        stats.last_remaining = (
                            int(remaining) if remaining and remaining.isdigit() else None
                        )
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(json.loads(text), indent=2, sort_keys=True))
                        return json.loads(path.read_text())
                    if response.status == 429:
                        await asyncio.sleep(float(response.headers.get("retry-after") or 5))
                        continue
                    if response.status in (401, 422):
                        raise OddsApiError(f"{response.status}: {text[:500]}")
                    last_error = f"{response.status}: {text[:500]}"
            except (TimeoutError, aiohttp.ClientError) as exc:
                last_error = str(exc)
            await asyncio.sleep(2**attempt)
    raise OddsApiError(f"snapshot {public_params['date']} failed: {last_error}")


def outcome_prices(event: Mapping[str, Any], book: Mapping[str, Any]) -> tuple[int, int] | None:
    home = normalize_name(str(event.get("home_team", "")))
    away = normalize_name(str(event.get("away_team", "")))
    for market in book.get("markets", []):
        if market.get("key") != "h2h":
            continue
        prices: dict[str, int] = {}
        for outcome in market.get("outcomes", []):
            name = normalize_name(str(outcome.get("name", "")))
            price = outcome.get("price")
            if price in (None, 0):
                continue
            try:
                prices[name] = round(float(price))
            except (TypeError, ValueError):
                continue
        if home in prices and away in prices:
            return prices[home], prices[away]
    return None


def quote_last_update(book: Mapping[str, Any]) -> datetime | None:
    for market in book.get("markets", []):
        if market.get("key") == "h2h" and market.get("last_update"):
            return as_utc(str(market["last_update"]))
    if book.get("last_update"):
        return as_utc(str(book["last_update"]))
    return None


def quotes_from_event(event: Mapping[str, Any], response_time: datetime) -> dict[str, Quote]:
    quotes: dict[str, Quote] = {}
    for book in event.get("bookmakers", []):
        bookmaker = str(book.get("key", "")).lower()
        if not bookmaker:
            continue
        prices = outcome_prices(event, book)
        if prices is None:
            continue
        home_ml, away_ml = prices
        quotes[bookmaker] = Quote(
            bookmaker=bookmaker,
            home_ml=home_ml,
            away_ml=away_ml,
            snapshot_time=response_time,
            last_update=quote_last_update(book),
        )
    return quotes


def event_match_delta(
    event: Mapping[str, Any], game: MlbResult, *, tolerance: timedelta
) -> timedelta | None:
    event_home = normalize_name(str(event.get("home_team", "")))
    event_away = normalize_name(str(event.get("away_team", "")))
    if event_home not in game.home_aliases or event_away not in game.away_aliases:
        return None
    if not event.get("commence_time"):
        return None
    delta = abs(as_utc(str(event["commence_time"])) - game.commence_time)
    return delta if delta <= tolerance else None


def game_quotes_from_snapshot(
    game: MlbResult,
    body: Mapping[str, Any],
    *,
    horizon_hours: float,
    commence_tolerance_hours: float,
) -> GameQuotes | None:
    response_time = as_utc(str(body["timestamp"]))
    tolerance = timedelta(hours=commence_tolerance_hours)
    best_event: Mapping[str, Any] | None = None
    best_delta: timedelta | None = None
    for event in body.get("data", []):
        delta = event_match_delta(event, game, tolerance=tolerance)
        if delta is None:
            continue
        if best_delta is None or delta < best_delta:
            best_event = event
            best_delta = delta
    if best_event is None:
        return None
    return GameQuotes(
        season=game.season,
        game_pk=game.game_pk,
        game_date=game.game_date,
        commence_time=game.commence_time,
        horizon_hours=horizon_hours,
        home_won=game.home_won,
        quotes=quotes_from_event(best_event, response_time),
    )


async def collect_game_quotes(
    *,
    games: Sequence[MlbResult],
    hours_before: tuple[float, ...],
    cache_dir: Path,
    api_key: str,
    regions: str,
    refresh_odds: bool,
    timeout: float,
    commence_tolerance_hours: float,
    limit_snapshots: int | None,
    max_concurrency: int,
) -> tuple[list[GameQuotes], Counter[str], FetchStats]:
    drops: Counter[str] = Counter()
    stats = FetchStats()
    by_request: dict[datetime, list[tuple[MlbResult, float]]] = defaultdict(list)
    for game in games:
        for hours in hours_before:
            request_time = game.commence_time - timedelta(hours=hours)
            request_time = request_time.replace(second=0, microsecond=0)
            by_request[request_time].append((game, hours))

    request_items = sorted(by_request.items())
    if limit_snapshots is not None:
        limited = request_items[limit_snapshots:]
        for _request_time, grouped_games in limited:
            drops["limited_snapshot"] += len(grouped_games)
        request_items = request_items[:limit_snapshots]

    quotes: list[GameQuotes] = []
    limiter = asyncio.Semaphore(max_concurrency)
    async with aiohttp.ClientSession() as session:

        async def fetch_one(
            idx: int, request_time: datetime, grouped_games: list[tuple[MlbResult, float]]
        ) -> tuple[int, datetime, list[tuple[MlbResult, float]], Mapping[str, Any]]:
            body = await request_odds_snapshot(
                session=session,
                limiter=limiter,
                cache_dir=cache_dir,
                api_key=api_key,
                request_time=request_time,
                regions=regions,
                markets="h2h",
                refresh=refresh_odds,
                timeout=timeout,
                stats=stats,
            )
            return idx, request_time, grouped_games, body

        tasks = [
            asyncio.create_task(fetch_one(idx, request_time, grouped_games))
            for idx, (request_time, grouped_games) in enumerate(request_items, start=1)
        ]
        total = len(tasks)
        completed = 0
        for task in asyncio.as_completed(tasks):
            idx, request_time, grouped_games, body = await task
            completed += 1
            for game, hours in grouped_games:
                game_quotes = game_quotes_from_snapshot(
                    game,
                    body,
                    horizon_hours=hours,
                    commence_tolerance_hours=commence_tolerance_hours,
                )
                if game_quotes is None:
                    drops["not_in_snapshot"] += 1
                    continue
                if not game_quotes.quotes:
                    drops["no_quotes"] += 1
                    continue
                quotes.append(game_quotes)
            if completed % 200 == 0 or completed == total:
                print(
                    f"  snapshots {completed:,}/{total:,} | quotes {len(quotes):,} | "
                    f"api_calls {stats.api_calls:,} | credits {stats.api_credits:,} | "
                    f"cached {stats.cached_snapshots:,}",
                    flush=True,
                )
            _ = idx, request_time
    return quotes, drops, stats


def two_way_hold(home_ml: int, away_ml: int) -> float:
    return american_to_prob(home_ml) + american_to_prob(away_ml) - 1.0


def evaluate_games(
    games: Sequence[GameQuotes],
    *,
    pinnacle_book: str,
    us_books: tuple[str, ...] | None,
    min_us_books: int,
    devig: str,
) -> tuple[list[EvaluatedGame], Counter[str]]:
    evaluated: list[EvaluatedGame] = []
    drops: Counter[str] = Counter()
    selected_books = set(us_books) if us_books is not None else None
    for game in games:
        pin = game.quotes.get(pinnacle_book)
        if pin is None:
            drops["no_pinnacle"] += 1
            continue
        us_quotes = [
            q
            for book, q in game.quotes.items()
            if book != pinnacle_book and (selected_books is None or book in selected_books)
        ]
        if not us_quotes:
            drops["no_us_quote"] += 1
            continue
        if len(us_quotes) < min_us_books:
            drops["min_books"] += 1
            continue

        pin_home, pin_away = no_vig_two_way(pin.home_ml, pin.away_ml, method=devig)
        us_home = statistics.median(
            no_vig_two_way(q.home_ml, q.away_ml, method=devig)[0] for q in us_quotes
        )
        candidates: list[tuple[float, str, Quote, float, float]] = []
        for quote in us_quotes:
            home_decimal = american_to_decimal(quote.home_ml)
            away_decimal = american_to_decimal(quote.away_ml)
            candidates.append((pin_home * home_decimal - 1.0, "home", quote, home_decimal, pin_home))
            candidates.append((pin_away * away_decimal - 1.0, "away", quote, away_decimal, pin_away))
        best_ev, side, quote, decimal_odds, probability = max(candidates, key=lambda row: row[0])
        evaluated.append(
            EvaluatedGame(
                season=game.season,
                game_pk=game.game_pk,
                game_date=game.game_date,
                commence_time=game.commence_time,
                horizon_hours=game.horizon_hours,
                home_won=game.home_won,
                pin_home=pin_home,
                us_home=us_home,
                pin_hold=two_way_hold(pin.home_ml, pin.away_ml),
                us_books=len(us_quotes),
                snapshot_time=quote.snapshot_time,
                best_side=side,
                best_book=quote.bookmaker,
                best_decimal=decimal_odds,
                best_prob=probability,
                best_ev=best_ev,
            )
        )
    return evaluated, drops


def settle(evaluated: Sequence[EvaluatedGame], threshold: float) -> list[SettledBet]:
    bets: list[SettledBet] = []
    for game in evaluated:
        if game.best_ev < threshold:
            continue
        won = game.home_won if game.best_side == "home" else not game.home_won
        ret = game.best_decimal - 1.0 if won else -1.0
        bets.append(
            SettledBet(
                season=game.season,
                game_pk=game.game_pk,
                game_date=game.game_date,
                horizon_hours=game.horizon_hours,
                side=game.best_side,
                book=game.best_book,
                decimal_odds=game.best_decimal,
                probability=game.best_prob,
                ev=game.best_ev,
                won=won,
                ret=ret,
            )
        )
    return bets


def boot_ci_by_date(
    bets: Sequence[SettledBet], *, samples: int, seed: int
) -> tuple[float, float, float]:
    if not bets:
        return math.nan, math.nan, math.nan
    returns = [bet.ret for bet in bets]
    mean = float(np.mean(returns))
    if samples <= 0:
        return mean, math.nan, math.nan
    by_date: dict[date, list[float]] = defaultdict(list)
    for bet in bets:
        by_date[bet.game_date].append(bet.ret)
    dates = sorted(by_date)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        sampled_dates = [dates[rng.randrange(len(dates))] for _ in dates]
        sampled_returns = [ret for sampled_date in sampled_dates for ret in by_date[sampled_date]]
        draws.append(float(np.mean(sampled_returns)))
    draws.sort()
    return mean, draws[int(0.025 * samples)], draws[int(0.975 * samples)]


def required_n(sigma: float, target_roi: float, *, prospective_power: bool) -> float:
    if sigma <= 0.0 or target_roi <= 0.0 or not math.isfinite(sigma):
        return math.nan
    z = 1.96 + (0.84 if prospective_power else 0.0)
    return (z * sigma / target_roi) ** 2


def grouped_by_horizon(games: Iterable[EvaluatedGame]) -> dict[float, list[EvaluatedGame]]:
    grouped: dict[float, list[EvaluatedGame]] = defaultdict(list)
    for game in games:
        grouped[game.horizon_hours].append(game)
    return dict(sorted(grouped.items()))


def print_brier_by_horizon(evaluated: Sequence[EvaluatedGame]) -> None:
    if not evaluated:
        return
    print("\nBrier, home-win probability on eligible games")
    print(f"  {'horizon':>8} {'rows':>6} {'Pinnacle':>10} {'US median':>10}")
    print(f"  {'-' * 42}")
    for horizon, rows in grouped_by_horizon(evaluated).items():
        y = np.array([1.0 if game.home_won else 0.0 for game in rows])
        pin = np.array([game.pin_home for game in rows])
        us = np.array([game.us_home for game in rows])
        print(f"  {horizon:7.1f}h {len(rows):6d} {brier(pin, y):10.6f} {brier(us, y):10.6f}")


def print_calibration(
    title: str,
    probabilities: list[float],
    outcomes: list[bool],
    *,
    bins: int,
) -> None:
    rows = calibration_rows(probabilities, outcomes, bins=bins)
    if not rows:
        return
    total = sum(n for _, n, _, _, _ in rows)
    ece = sum(n * abs(diff) for _, n, _, _, diff in rows) / total
    print(f"\nCalibration: {title}")
    print(f"  {'bin':>9} {'n':>6} {'avg_p':>8} {'actual':>8} {'actual-p':>9}")
    print(f"  {'-' * 48}")
    for label, n, avg_p, actual, diff in rows:
        print(f"  {label:>9} {n:6d} {avg_p:8.3f} {actual:8.3f} {diff:+9.3f}")
    print(f"  ECE: {ece:.3f}")


def print_coverage(
    *,
    results: Sequence[MlbResult],
    quotes: Sequence[GameQuotes],
    evaluated: Sequence[EvaluatedGame],
    result_drops: Counter[str],
    fetch_drops: Counter[str],
    eval_drops: Counter[str],
    stats: FetchStats,
) -> None:
    print("Coverage")
    print(f"  final games in requested seasons/types: {len(results):,}")
    print(f"  game-horizon quote rows found: {len(quotes):,}")
    print(f"  eligible game-horizon rows after filters: {len(evaluated):,}")
    print(f"  Odds API calls made: {stats.api_calls:,}")
    print(f"  Odds API credits used: {stats.api_credits:,}")
    print(f"  cached snapshots reused: {stats.cached_snapshots:,}")
    if stats.last_remaining is not None:
        print(f"  credits remaining after last API call: {stats.last_remaining:,}")
    drops = result_drops + fetch_drops + eval_drops
    if drops:
        print("  dropped:")
        for reason, n in sorted(drops.items()):
            print(f"    {reason}: {n:,}")
    if not evaluated:
        return
    print("\nEligible coverage by horizon")
    print(f"  {'horizon':>8} {'rows':>6} {'avg_us_books':>12} {'median_hold':>12}")
    print(f"  {'-' * 46}")
    for horizon, rows in grouped_by_horizon(evaluated).items():
        print(
            f"  {horizon:7.1f}h {len(rows):6d} "
            f"{statistics.mean(g.us_books for g in rows):12.2f} "
            f"{statistics.median(g.pin_hold for g in rows):12.2%}"
        )


def print_required_n(bets: Sequence[SettledBet]) -> None:
    if len(bets) < 2:
        return
    returns = np.array([bet.ret for bet in bets])
    sigma = float(np.std(returns, ddof=1))
    roi = float(np.mean(returns))
    avg_ev = statistics.mean(bet.ev for bet in bets)
    targets = [("avg_EV", avg_ev), ("5%", 0.05), ("10%", 0.10), ("observed_ROI", roi)]
    print("\nRequired independent-bet N from observed per-bet variance")
    print(f"  per-bet sd: {sigma:.3f}u")
    print(f"  {'target':>12} {'roi':>8} {'95% lower>0':>14} {'80% powered':>13}")
    print(f"  {'-' * 54}")
    for label, target in targets:
        if target <= 0.0:
            print(f"  {label:>12} {target:+8.2%} {'n/a':>14} {'n/a':>13}")
            continue
        n95 = required_n(sigma, target, prospective_power=False)
        n80 = required_n(sigma, target, prospective_power=True)
        print(f"  {label:>12} {target:+8.2%} {math.ceil(n95):14,} {math.ceil(n80):13,}")


def print_strategy(
    evaluated: Sequence[EvaluatedGame], thresholds: tuple[float, ...], *, samples: int
) -> None:
    print("\nStrategy: one best US bet per game-horizon using Pinnacle no-vig fair probability")
    print(
        f"  {'min_EV':>7} {'bets':>6} {'win%':>7} {'avg_EV':>8} {'ROI':>8} "
        f"{'date-block 95% CI':>23} {'z':>7} {'Brier':>8}"
    )
    print(f"  {'-' * 102}")
    for threshold in thresholds:
        bets = settle(evaluated, threshold)
        if not bets:
            print(f"  {threshold:7.1%} {0:6d} {'-':>7} {'-':>8} {'-':>8} {'-':>23} {'-':>7} {'-':>8}")
            continue
        returns = np.array([bet.ret for bet in bets])
        probs = np.array([bet.probability for bet in bets])
        wins = np.array([1.0 if bet.won else 0.0 for bet in bets])
        mean, lo, hi = boot_ci_by_date(bets, samples=samples, seed=41)
        print(
            f"  {threshold:7.1%} {len(bets):6d} {float(wins.mean()):7.1%} "
            f"{statistics.mean(bet.ev for bet in bets):8.2%} {mean:+8.2%} "
            f"[{lo:+7.2%}, {hi:+7.2%}] {z_score(returns):+7.2f} "
            f"{brier(probs, wins):8.4f}"
        )


def print_season_breakdown(
    evaluated: Sequence[EvaluatedGame], threshold: float, *, samples: int
) -> None:
    bets = settle(evaluated, threshold)
    if not bets:
        return
    by_season: dict[int, list[SettledBet]] = defaultdict(list)
    for bet in bets:
        by_season[bet.season].append(bet)
    print(f"\nSeason breakdown at min_EV {threshold:.1%}")
    print(f"  {'season':>6} {'bets':>6} {'win%':>7} {'avg_EV':>8} {'ROI':>8} {'date-block 95% CI':>23}")
    print(f"  {'-' * 78}")
    for season in sorted(by_season):
        rows = by_season[season]
        mean, lo, hi = boot_ci_by_date(rows, samples=samples, seed=53 + season)
        print(
            f"  {season:6d} {len(rows):6d} {statistics.mean(bet.won for bet in rows):7.1%} "
            f"{statistics.mean(bet.ev for bet in rows):8.2%} {mean:+8.2%} "
            f"[{lo:+7.2%}, {hi:+7.2%}]"
        )


def write_evaluated(path: Path, evaluated: Sequence[EvaluatedGame]) -> None:
    if not evaluated:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "season",
        "game_pk",
        "game_date",
        "commence_time",
        "horizon_hours",
        "snapshot_time",
        "home_won",
        "pin_home",
        "us_home",
        "pin_hold",
        "us_books",
        "best_side",
        "best_book",
        "best_decimal",
        "best_prob",
        "best_ev",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for game in evaluated:
            writer.writerow(
                {
                    "season": game.season,
                    "game_pk": game.game_pk,
                    "game_date": game.game_date.isoformat(),
                    "commence_time": game.commence_time.isoformat(),
                    "horizon_hours": game.horizon_hours,
                    "snapshot_time": game.snapshot_time.isoformat(),
                    "home_won": int(game.home_won),
                    "pin_home": game.pin_home,
                    "us_home": game.us_home,
                    "pin_hold": game.pin_hold,
                    "us_books": game.us_books,
                    "best_side": game.best_side,
                    "best_book": game.best_book,
                    "best_decimal": game.best_decimal,
                    "best_prob": game.best_prob,
                    "best_ev": game.best_ev,
                }
            )
    print(f"\nwrote evaluated rows -> {path}")


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", nargs="+", default=["2021", "2022", "2023", "2024", "2025"])
    ap.add_argument("--hours-before", nargs="+", type=positive_float, default=[24.0])
    ap.add_argument("--game-types", default="R")
    ap.add_argument("--pinnacle-book", default="pinnacle")
    ap.add_argument(
        "--us-books",
        default="top5",
        help="Panel name (top5/top6/priority5/priority6), 'all', or comma-separated books.",
    )
    ap.add_argument("--regions", default="us,eu")
    ap.add_argument("--min-us-books", type=int, default=1)
    ap.add_argument("--devig", default="proportional", choices=("proportional", "shin"))
    ap.add_argument("--thresholds", nargs="+", default=["0", "0.01", "0.02", "0.03", "0.05"])
    ap.add_argument("--primary-threshold", type=float, default=0.02)
    ap.add_argument("--calibration-bins", type=int, default=10)
    ap.add_argument("--bootstrap-samples", type=int, default=4000)
    ap.add_argument("--commence-tolerance-hours", type=float, default=4.0)
    ap.add_argument("--cache-dir", type=Path, default=Path("data/odds_history/mlb_pinnacle_ev"))
    ap.add_argument("--output", type=Path, default=Path("output/mlb_pinnacle_ev/evaluated.csv"))
    ap.add_argument("--refresh-odds", action="store_true")
    ap.add_argument("--limit-snapshots", type=int)
    ap.add_argument("--max-concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=45.0)
    args = ap.parse_args()

    if args.min_us_books < 1:
        raise SystemExit("--min-us-books must be at least 1")
    if args.calibration_bins < 2:
        raise SystemExit("--calibration-bins must be at least 2")
    if args.bootstrap_samples < 0:
        raise SystemExit("--bootstrap-samples must be non-negative")
    if args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be at least 1")

    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise SystemExit("ODDS_API_KEY is not set")

    seasons = parse_seasons(args.seasons)
    hours_before = tuple(dict.fromkeys(float(value) for value in args.hours_before))
    game_types = parse_game_types(args.game_types)
    us_books = parse_books(args.us_books)
    thresholds = parse_thresholds(args.thresholds)
    if args.primary_threshold not in thresholds:
        thresholds = tuple(sorted((*thresholds, args.primary_threshold)))

    selected_books = "all non-Pinnacle" if us_books is None else ",".join(us_books)
    print("MLB fixed-horizon Pinnacle-fair +EV validation")
    print(f"  seasons: {','.join(str(season) for season in seasons)}")
    print(f"  game_types: {','.join(game_types)}")
    print(f"  hours_before: {','.join(f'{hours:g}' for hours in hours_before)}")
    print(f"  Pinnacle book: {args.pinnacle_book}")
    print(f"  US books: {selected_books}")
    print(f"  regions: {args.regions}")
    print(f"  min US books: {args.min_us_books}")
    print(f"  de-vig: {args.devig}")
    print(f"  cache_dir: {args.cache_dir}")
    print()

    results, result_drops = load_results(seasons=seasons, game_types=game_types)
    quotes, fetch_drops, stats = asyncio.run(
        collect_game_quotes(
            games=results,
            hours_before=hours_before,
            cache_dir=args.cache_dir,
            api_key=api_key,
            regions=args.regions,
            refresh_odds=args.refresh_odds,
            timeout=args.timeout,
            commence_tolerance_hours=args.commence_tolerance_hours,
            limit_snapshots=args.limit_snapshots,
            max_concurrency=args.max_concurrency,
        )
    )
    evaluated, eval_drops = evaluate_games(
        quotes,
        pinnacle_book=args.pinnacle_book,
        us_books=us_books,
        min_us_books=args.min_us_books,
        devig=args.devig,
    )
    evaluated = sorted(
        evaluated,
        key=lambda game: (game.season, game.game_date, game.game_pk, game.horizon_hours),
    )

    print_coverage(
        results=results,
        quotes=quotes,
        evaluated=evaluated,
        result_drops=result_drops,
        fetch_drops=fetch_drops,
        eval_drops=eval_drops,
        stats=stats,
    )
    print_brier_by_horizon(evaluated)
    if evaluated:
        print_calibration(
            "Pinnacle home-win probability on all eligible game-horizons",
            [game.pin_home for game in evaluated],
            [game.home_won for game in evaluated],
            bins=args.calibration_bins,
        )
    print_strategy(evaluated, thresholds, samples=args.bootstrap_samples)
    print_season_breakdown(evaluated, args.primary_threshold, samples=args.bootstrap_samples)
    primary_bets = settle(evaluated, args.primary_threshold)
    print_required_n(primary_bets)
    if primary_bets:
        print_calibration(
            f"Pinnacle probability of chosen side, bets at min_EV {args.primary_threshold:.1%}",
            [bet.probability for bet in primary_bets],
            [bet.won for bet in primary_bets],
            bins=args.calibration_bins,
        )
    write_evaluated(args.output, evaluated)


if __name__ == "__main__":
    main()
