"""Validate fixed-horizon Pinnacle-fair +EV line shopping for ESPN-result sports.

This covers sports where the historical final score source is
``scripts/fetch_sports_results.py`` and the price source is The Odds API
historical h2h endpoint. It emits the same evaluated CSV contract consumed by
``scripts/analyze_pinnacle_ev_matrix.py``.

Example:
    uv run python scripts/validate_pinnacle_cross_sport_horizon_ev.py \
      --sport nba \
      --results data/sports_results/nba_2024.csv \
      --hours-before 1 4 24 168 \
      --output output/nba_pinnacle_ev/evaluated.csv
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_moneyline_lineshop import PANEL_PRIORITY, PANEL_TOP5, PANEL_TOP6
from scripts.validate_pinnacle_clv_ev import (
    brier,
    calibration_rows,
    parse_thresholds,
    z_score,
)
from src.betting.odds import american_to_decimal, american_to_prob, no_vig_two_way

SPORT_ODDS_KEYS = {
    "nba": "basketball_nba",
    "ncaaf": "americanfootball_ncaaf",
    "nhl": "icehockey_nhl",
}
SPORT_CODES = {"nba": "NBA", "ncaaf": "NCAAF", "nhl": "NHL"}
PANELS = {
    "top5": PANEL_TOP5,
    "top6": PANEL_TOP6,
    "priority5": PANEL_PRIORITY[:5],
    "priority6": PANEL_PRIORITY[:6],
}
TEAM_NAME_VARIANTS: Mapping[str, tuple[str, ...]] = {
    "LA Clippers": ("LA Clippers", "Los Angeles Clippers"),
    "Los Angeles Clippers": ("Los Angeles Clippers", "LA Clippers"),
    "LA Kings": ("LA Kings", "Los Angeles Kings"),
    "Los Angeles Kings": ("Los Angeles Kings", "LA Kings"),
    "Ole Miss Rebels": ("Ole Miss Rebels", "Mississippi Rebels"),
    "Miami (OH) RedHawks": ("Miami (OH) RedHawks", "Miami Ohio RedHawks"),
    "UTSA Roadrunners": ("UTSA Roadrunners", "Texas-San Antonio Roadrunners"),
    "UConn Huskies": ("UConn Huskies", "Connecticut Huskies"),
    "UMass Minutemen": ("UMass Minutemen", "Massachusetts Minutemen"),
}


@dataclass(frozen=True)
class SportResult:
    sport: str
    season: int
    game_id: str
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
    sport: str
    season: int
    game_id: str
    game_date: date
    commence_time: datetime
    horizon_hours: float
    home_won: bool
    quotes: dict[str, Quote]


@dataclass(frozen=True)
class EvaluatedGame:
    sport: str
    season: int
    game_id: str
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
    game_id: str
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


def odds_cache_path(cache_dir: Path, sport: str, params: Mapping[str, str]) -> Path:
    ordered = urlencode(sorted(params.items()))
    return cache_dir / sport / "snapshots" / cache_name(ordered)


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


def parse_sports(values: Sequence[str]) -> tuple[str, ...]:
    sports: list[str] = []
    for value in values:
        for part in value.split(","):
            sport = part.strip().lower()
            if not sport:
                continue
            if sport not in SPORT_ODDS_KEYS:
                raise SystemExit(f"unsupported sport: {sport}")
            sports.append(sport)
    if not sports:
        raise SystemExit("at least one sport is required")
    return tuple(dict.fromkeys(sports))


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "t", "yes", "y"}


def load_results(
    paths: Sequence[Path], *, sports: tuple[str, ...], start: date | None, end: date | None
) -> tuple[list[SportResult], Counter[str]]:
    wanted_codes = {SPORT_CODES[sport] for sport in sports}
    drops: Counter[str] = Counter()
    results: list[SportResult] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        with path.open(newline="") as file:
            for row in csv.DictReader(file):
                sport = row["sport"].upper()
                if sport not in wanted_codes:
                    continue
                if row.get("status") != "STATUS_FINAL":
                    drops["not_final"] += 1
                    continue
                game_date = date.fromisoformat(row["game_date"])
                if start is not None and game_date < start:
                    continue
                if end is not None and game_date > end:
                    continue
                home_score = int(row["home_score"])
                away_score = int(row["away_score"])
                if home_score == away_score:
                    drops["tie"] += 1
                    continue
                game_id = row["source_event_id"]
                key = (sport, game_id)
                if key in seen:
                    drops["duplicate"] += 1
                    continue
                seen.add(key)
                try:
                    results.append(
                        SportResult(
                            sport=sport,
                            season=int(row["season"]) if row.get("season") else game_date.year,
                            game_id=game_id,
                            game_date=game_date,
                            commence_time=as_utc(row["commence_time"]),
                            home_team=row["home_team"],
                            away_team=row["away_team"],
                            home_aliases=aliases_for_team(row["home_team"]),
                            away_aliases=aliases_for_team(row["away_team"]),
                            home_won=parse_bool(row["home_won"]),
                        )
                    )
                except ValueError as exc:
                    drops[f"row_error:{exc}"] += 1
    return sorted(results, key=lambda game: (game.sport, game.commence_time, game.game_id)), drops


async def request_odds_snapshot(
    *,
    session: aiohttp.ClientSession,
    limiter: asyncio.Semaphore,
    cache_dir: Path,
    sport: str,
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
    path = odds_cache_path(cache_dir, sport, public_params)
    if path.exists() and not refresh:
        stats.cached_snapshots += 1
        return json.loads(path.read_text())

    params = {**public_params, "apiKey": api_key}
    url = f"https://api.the-odds-api.com/v4/historical/sports/{SPORT_ODDS_KEYS[sport]}/odds"
    async with limiter:
        last_error: str | None = None
        for attempt in range(5):
            try:
                async with session.get(url, params=params, timeout=timeout) as response:
                    text = await response.text()
                    if response.status == 200:
                        stats.api_calls += 1
                        stats.api_credits += int(response.headers.get("x-requests-last") or 0)
                        remaining = response.headers.get("x-requests-remaining")
                        stats.last_remaining = int(remaining) if remaining and remaining.isdigit() else None
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
    raise OddsApiError(f"{sport} snapshot {public_params['date']} failed: {last_error}")


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


def event_match_delta(event: Mapping[str, Any], game: SportResult, *, tolerance: timedelta) -> timedelta | None:
    event_home = normalize_name(str(event.get("home_team", "")))
    event_away = normalize_name(str(event.get("away_team", "")))
    if event_home not in game.home_aliases or event_away not in game.away_aliases:
        return None
    if not event.get("commence_time"):
        return None
    delta = abs(as_utc(str(event["commence_time"])) - game.commence_time)
    return delta if delta <= tolerance else None


def game_quotes_from_snapshot(
    game: SportResult,
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
        sport=game.sport,
        season=game.season,
        game_id=game.game_id,
        game_date=game.game_date,
        commence_time=game.commence_time,
        horizon_hours=horizon_hours,
        home_won=game.home_won,
        quotes=quotes_from_event(best_event, response_time),
    )


async def collect_game_quotes(
    *,
    games: Sequence[SportResult],
    sport_key: str,
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
    by_request: dict[datetime, list[tuple[SportResult, float]]] = defaultdict(list)
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
            idx: int, request_time: datetime, grouped_games: list[tuple[SportResult, float]]
        ) -> tuple[int, datetime, list[tuple[SportResult, float]], Mapping[str, Any]]:
            body = await request_odds_snapshot(
                session=session,
                limiter=limiter,
                cache_dir=cache_dir,
                sport=sport_key,
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
                sport=game.sport,
                season=game.season,
                game_id=game.game_id,
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
                game_id=game.game_id,
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


def grouped_by_horizon(games: Iterable[EvaluatedGame]) -> dict[float, list[EvaluatedGame]]:
    grouped: dict[float, list[EvaluatedGame]] = defaultdict(list)
    for game in games:
        grouped[game.horizon_hours].append(game)
    return dict(sorted(grouped.items()))


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
    results: Sequence[SportResult],
    quotes: Sequence[GameQuotes],
    evaluated: Sequence[EvaluatedGame],
    result_drops: Counter[str],
    fetch_drops: Counter[str],
    eval_drops: Counter[str],
    stats: FetchStats,
) -> None:
    print("Coverage")
    print(f"  final games in requested scope: {len(results):,}")
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
            f"{statistics.mean(game.us_books for game in rows):12.2f} "
            f"{statistics.median(game.pin_hold for game in rows):12.2%}"
        )


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


def write_evaluated(path: Path, evaluated: Sequence[EvaluatedGame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "season",
        "game_id",
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
                    "game_id": game.game_id,
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


def run_one_sport(args: argparse.Namespace, sport: str, results: Sequence[SportResult]) -> None:
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise SystemExit("ODDS_API_KEY is not set")

    sport_results = [result for result in results if result.sport == SPORT_CODES[sport]]
    hours_before = tuple(dict.fromkeys(float(value) for value in args.hours_before))
    us_books = parse_books(args.us_books)
    thresholds = parse_thresholds(args.thresholds)
    if args.primary_threshold not in thresholds:
        thresholds = tuple(sorted((*thresholds, args.primary_threshold)))

    selected_books = "all non-Pinnacle" if us_books is None else ",".join(us_books)
    print(f"{SPORT_CODES[sport]} fixed-horizon Pinnacle-fair +EV validation")
    print(f"  final results: {len(sport_results):,}")
    print(f"  hours_before: {','.join(f'{hours:g}' for hours in hours_before)}")
    print(f"  Pinnacle book: {args.pinnacle_book}")
    print(f"  US books: {selected_books}")
    print(f"  regions: {args.regions}")
    print(f"  min US books: {args.min_us_books}")
    print(f"  de-vig: {args.devig}")
    print(f"  cache_dir: {args.cache_dir}")
    print()

    quotes, fetch_drops, stats = asyncio.run(
        collect_game_quotes(
            games=sport_results,
            sport_key=sport,
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
        key=lambda game: (game.season, game.game_date, game.game_id, game.horizon_hours),
    )

    print_coverage(
        results=sport_results,
        quotes=quotes,
        evaluated=evaluated,
        result_drops=Counter(),
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

    output = args.output
    if len(args.sports) > 1:
        output = output.with_name(f"{output.stem}_{sport}{output.suffix}")
    write_evaluated(output, evaluated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sport", "--sports", dest="sports", nargs="+", required=True)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--start", type=parse_date)
    parser.add_argument("--end", type=parse_date)
    parser.add_argument("--hours-before", nargs="+", type=positive_float, default=[1.0, 4.0, 24.0, 168.0])
    parser.add_argument("--pinnacle-book", default="pinnacle")
    parser.add_argument(
        "--us-books",
        default="top5",
        help="Panel name (top5/top6/priority5/priority6), 'all', or comma-separated books.",
    )
    parser.add_argument("--regions", default="us,eu")
    parser.add_argument("--min-us-books", type=int, default=1)
    parser.add_argument("--devig", default="proportional", choices=("proportional", "shin"))
    parser.add_argument("--thresholds", nargs="+", default=["0.005", "0.01", "0.02"])
    parser.add_argument("--primary-threshold", type=float, default=0.01)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--commence-tolerance-hours", type=float, default=4.0)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/odds_history/cross_sport_pinnacle_ev"))
    parser.add_argument("--output", type=Path, default=Path("output/cross_sport_pinnacle_ev/evaluated.csv"))
    parser.add_argument("--refresh-odds", action="store_true")
    parser.add_argument("--limit-snapshots", type=int)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()

    if args.min_us_books < 1:
        raise SystemExit("--min-us-books must be at least 1")
    if args.calibration_bins < 2:
        raise SystemExit("--calibration-bins must be at least 2")
    if args.bootstrap_samples < 0:
        raise SystemExit("--bootstrap-samples must be non-negative")
    if args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be at least 1")

    args.sports = parse_sports(args.sports)
    results, result_drops = load_results(
        args.results,
        sports=args.sports,
        start=args.start,
        end=args.end,
    )
    if result_drops:
        print("Result load drops:")
        for reason, n in sorted(result_drops.items()):
            print(f"  {reason}: {n:,}")
    for sport in args.sports:
        run_one_sport(args, sport, results)


if __name__ == "__main__":
    main()
