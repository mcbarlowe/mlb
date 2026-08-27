"""Validate Pinnacle-fair +EV line shopping on NFL moneylines.

This is the football analogue of ``validate_pinnacle_clv_ev.py``, but it does
not use MLB tables. It uses staged local files only:

1. Load NFL final scores from nflverse ``games.csv``.
2. Fetch The Odds API historical ``americanfootball_nfl`` h2h snapshots.
3. At fixed pre-game horizons, compare Pinnacle no-vig fair probabilities to
   US book prices from the same API snapshot.
4. Bet the single best US side/book per game when expected ROI clears each
   threshold.
5. Report coverage, Brier score, calibration, realized ROI, and credit usage.

Raw Odds API responses are cached by requested snapshot timestamp, so reruns do
not spend credits unless ``--refresh-odds`` is supplied.

Example:

    uv run python scripts/validate_nfl_pinnacle_ev.py \
      --seasons 2021 2022 2023 2024 2025 \
      --hours-before 24 4 1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_moneyline_lineshop import PANEL_PRIORITY, PANEL_TOP5, PANEL_TOP6
from scripts.validate_pinnacle_clv_ev import (
    boot_ci,
    brier,
    calibration_rows,
    parse_thresholds,
    z_score,
)
from src.betting.odds import american_to_decimal, american_to_prob, no_vig_two_way

SPORT_KEY = "americanfootball_nfl"
ODDS_URL = f"https://api.the-odds-api.com/v4/historical/sports/{SPORT_KEY}/odds"
SCHEDULE_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
EASTERN = "America/New_York"
REGULAR_GAME_TYPES = ("REG",)
POSTSEASON_GAME_TYPES = ("WC", "DIV", "CON", "SB")
PANELS = {
    "top5": PANEL_TOP5,
    "top6": PANEL_TOP6,
    "priority5": PANEL_PRIORITY[:5],
    "priority6": PANEL_PRIORITY[:6],
}

TEAM_NAME_VARIANTS: Mapping[str, tuple[str, ...]] = {
    "ARI": ("Arizona Cardinals",),
    "ATL": ("Atlanta Falcons",),
    "BAL": ("Baltimore Ravens",),
    "BUF": ("Buffalo Bills",),
    "CAR": ("Carolina Panthers",),
    "CHI": ("Chicago Bears",),
    "CIN": ("Cincinnati Bengals",),
    "CLE": ("Cleveland Browns",),
    "DAL": ("Dallas Cowboys",),
    "DEN": ("Denver Broncos",),
    "DET": ("Detroit Lions",),
    "GB": ("Green Bay Packers",),
    "HOU": ("Houston Texans",),
    "IND": ("Indianapolis Colts",),
    "JAX": ("Jacksonville Jaguars",),
    "KC": ("Kansas City Chiefs",),
    "LA": ("Los Angeles Rams",),
    "LAC": ("Los Angeles Chargers",),
    "LV": ("Las Vegas Raiders",),
    "MIA": ("Miami Dolphins",),
    "MIN": ("Minnesota Vikings",),
    "NE": ("New England Patriots",),
    "NO": ("New Orleans Saints",),
    "NYG": ("New York Giants",),
    "NYJ": ("New York Jets",),
    "PHI": ("Philadelphia Eagles",),
    "PIT": ("Pittsburgh Steelers",),
    "SEA": ("Seattle Seahawks",),
    "SF": ("San Francisco 49ers",),
    "TB": ("Tampa Bay Buccaneers",),
    "TEN": ("Tennessee Titans",),
    "WAS": ("Washington Commanders", "Washington Football Team"),
}


@dataclass(frozen=True)
class NflResult:
    season: int
    game_id: str
    game_type: str
    week: int
    commence_time: datetime
    home_team: str
    away_team: str
    home_aliases: tuple[str, ...]
    away_aliases: tuple[str, ...]
    home_score: int
    away_score: int

    @property
    def home_won(self) -> bool:
        return self.home_score > self.away_score

    @property
    def tied(self) -> bool:
        return self.home_score == self.away_score


@dataclass(frozen=True)
class Quote:
    bookmaker: str
    home_ml: int
    away_ml: int
    snapshot_time: datetime


@dataclass(frozen=True)
class GameQuotes:
    season: int
    game_id: str
    game_type: str
    week: int
    commence_time: datetime
    horizon_hours: float
    home_won: bool
    quotes: dict[str, Quote]


@dataclass(frozen=True)
class EvaluatedGame:
    season: int
    game_id: str
    game_type: str
    week: int
    horizon_hours: float
    home_won: bool
    pin_home: float
    us_home: float
    pin_hold: float
    us_books: int
    max_gap_hours: float
    best_side: str
    best_book: str
    best_decimal: float
    best_prob: float
    best_ev: float


@dataclass(frozen=True)
class SettledBet:
    season: int
    game_id: str
    game_type: str
    week: int
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


def parse_game_types(values: Sequence[str]) -> tuple[str, ...]:
    game_types: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip().upper()
            if part == "POST":
                game_types.extend(POSTSEASON_GAME_TYPES)
            elif part:
                game_types.append(part)
    if not game_types:
        raise SystemExit("at least one game type is required")
    return tuple(dict.fromkeys(game_types))


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


def aliases_for_team(code: str) -> tuple[str, ...]:
    try:
        return tuple(normalize_name(name) for name in TEAM_NAME_VARIANTS[code])
    except KeyError as exc:
        raise ValueError(f"No NFL team-name mapping for {code!r}") from exc


def display_team(code: str) -> str:
    return TEAM_NAME_VARIANTS[code][0]


def cache_name(value: str) -> str:
    digest = hashlib.sha1(value.encode()).hexdigest()[:12]
    safe = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return f"{safe}_{digest}.json"


def download_schedule(cache_dir: Path, *, refresh: bool) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "nflverse_games.csv"
    if path.exists() and not refresh:
        return path
    response = requests.get(SCHEDULE_URL, timeout=60)
    response.raise_for_status()
    path.write_text(response.text)
    return path


def kickoff_utc(gameday: str, gametime: str) -> datetime:
    timestamp = f"{gameday} {gametime or '00:00'}"
    return (
        datetime.strptime(timestamp, "%Y-%m-%d %H:%M")
        .replace(tzinfo=ZoneInfo(EASTERN))
        .astimezone(UTC)
    )


def load_results(
    *,
    cache_dir: Path,
    seasons: tuple[int, ...],
    game_types: tuple[str, ...],
    refresh_schedule: bool,
) -> tuple[list[NflResult], Counter[str]]:
    schedule_path = download_schedule(cache_dir, refresh=refresh_schedule)
    drops: Counter[str] = Counter()
    results: list[NflResult] = []
    with schedule_path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            season_text = row.get("season") or ""
            if not season_text:
                drops["missing_season"] += 1
                continue
            season = int(season_text)
            if season not in seasons:
                continue
            game_type = (row.get("game_type") or "").upper()
            if game_type not in game_types:
                continue
            away_score = row.get("away_score") or ""
            home_score = row.get("home_score") or ""
            if not away_score or not home_score:
                drops["missing_score"] += 1
                continue
            away_team = row["away_team"]
            home_team = row["home_team"]
            gametime = row.get("gametime") or "00:00"
            try:
                result = NflResult(
                    season=season,
                    game_id=row["game_id"],
                    game_type=game_type,
                    week=int(row["week"]),
                    commence_time=kickoff_utc(row["gameday"], gametime),
                    home_team=display_team(home_team),
                    away_team=display_team(away_team),
                    home_aliases=aliases_for_team(home_team),
                    away_aliases=aliases_for_team(away_team),
                    home_score=int(float(home_score)),
                    away_score=int(float(away_score)),
                )
            except (KeyError, ValueError) as exc:
                drops[f"schedule_parse_error:{exc}"] += 1
                continue
            if result.tied:
                drops["tie"] += 1
                continue
            results.append(result)
    return results, drops


def odds_cache_path(cache_dir: Path, params: Mapping[str, str]) -> Path:
    ordered = urlencode(sorted(params.items()))
    return cache_dir / "snapshots" / cache_name(ordered)


def request_odds_snapshot(
    *,
    cache_dir: Path,
    api_key: str,
    request_time: datetime,
    regions: str,
    markets: str,
    refresh: bool,
    timeout: float,
    stats: FetchStats,
) -> Mapping[str, Any]:
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
        "date": request_time.astimezone(UTC).isoformat().replace("+00:00", "Z"),
    }
    path = odds_cache_path(cache_dir, params)
    if path.exists() and not refresh:
        stats.cached_snapshots += 1
        return json.loads(path.read_text())

    response = requests.get(ODDS_URL, params=params, timeout=timeout)
    if response.status_code != 200:
        raise OddsApiError(f"{response.status_code}: {response.text[:500]}")
    last_cost = int(response.headers.get("x-requests-last") or 0)
    stats.api_calls += 1
    stats.api_credits += last_cost
    remaining = response.headers.get("x-requests-remaining")
    stats.last_remaining = int(remaining) if remaining and remaining.isdigit() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(response.json(), indent=2, sort_keys=True))
    return json.loads(path.read_text())


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


def quote_time(response_time: datetime, book: Mapping[str, Any]) -> datetime:
    for market in book.get("markets", []):
        if market.get("key") == "h2h" and market.get("last_update"):
            return as_utc(str(market["last_update"]))
    if book.get("last_update"):
        return as_utc(str(book["last_update"]))
    return response_time


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
            snapshot_time=quote_time(response_time, book),
        )
    return quotes


def event_matches_game(
    event: Mapping[str, Any], game: NflResult, *, tolerance: timedelta
) -> bool:
    event_home = normalize_name(str(event.get("home_team", "")))
    event_away = normalize_name(str(event.get("away_team", "")))
    if event_home not in game.home_aliases or event_away not in game.away_aliases:
        return False
    commence = as_utc(str(event["commence_time"]))
    return abs(commence - game.commence_time) <= tolerance


def game_quotes_from_snapshot(
    game: NflResult,
    body: Mapping[str, Any],
    *,
    horizon_hours: float,
    commence_tolerance_hours: float,
) -> GameQuotes | None:
    response_time = as_utc(str(body["timestamp"]))
    tolerance = timedelta(hours=commence_tolerance_hours)
    for event in body.get("data", []):
        if event_matches_game(event, game, tolerance=tolerance):
            return GameQuotes(
                season=game.season,
                game_id=game.game_id,
                game_type=game.game_type,
                week=game.week,
                commence_time=game.commence_time,
                horizon_hours=horizon_hours,
                home_won=game.home_won,
                quotes=quotes_from_event(event, response_time),
            )
    return None


def collect_game_quotes(
    *,
    games: Sequence[NflResult],
    hours_before: tuple[float, ...],
    cache_dir: Path,
    api_key: str,
    regions: str,
    refresh_odds: bool,
    timeout: float,
    commence_tolerance_hours: float,
    limit_snapshots: int | None,
) -> tuple[list[GameQuotes], Counter[str], FetchStats]:
    drops: Counter[str] = Counter()
    stats = FetchStats()
    by_request: dict[datetime, list[tuple[NflResult, float]]] = defaultdict(list)
    for game in games:
        for hours in hours_before:
            request_time = game.commence_time - timedelta(hours=hours)
            request_time = request_time.replace(second=0, microsecond=0)
            by_request[request_time].append((game, hours))

    quotes: list[GameQuotes] = []
    for idx, request_time in enumerate(sorted(by_request), start=1):
        if limit_snapshots is not None and idx > limit_snapshots:
            drops["limited_snapshot"] += len(by_request[request_time])
            continue
        body = request_odds_snapshot(
            cache_dir=cache_dir,
            api_key=api_key,
            request_time=request_time,
            regions=regions,
            markets="h2h",
            refresh=refresh_odds,
            timeout=timeout,
            stats=stats,
        )
        for game, hours in by_request[request_time]:
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
    return quotes, drops, stats


def two_way_hold(home_ml: int, away_ml: int) -> float:
    return american_to_prob(home_ml) + american_to_prob(away_ml) - 1.0


def evaluate_games(
    games: Sequence[GameQuotes],
    *,
    pinnacle_book: str,
    us_books: tuple[str, ...] | None,
    max_gap_hours: float,
    min_us_books: int,
    devig: str,
) -> tuple[list[EvaluatedGame], Counter[str]]:
    evaluated: list[EvaluatedGame] = []
    drops: Counter[str] = Counter()
    for game in games:
        pin = game.quotes.get(pinnacle_book)
        if pin is None:
            drops["no_pinnacle"] += 1
            continue
        selected_books = set(us_books) if us_books is not None else None
        us_quotes = [
            q
            for book, q in game.quotes.items()
            if book != pinnacle_book and (selected_books is None or book in selected_books)
        ]
        if not us_quotes:
            drops["no_us_quote"] += 1
            continue
        aligned = [
            q
            for q in us_quotes
            if abs((q.snapshot_time - pin.snapshot_time).total_seconds()) / 3600.0
            <= max_gap_hours
        ]
        if len(aligned) < min_us_books:
            drops["timing_or_min_books"] += 1
            continue

        pin_home, pin_away = no_vig_two_way(pin.home_ml, pin.away_ml, method=devig)
        us_home = statistics.median(
            no_vig_two_way(q.home_ml, q.away_ml, method=devig)[0] for q in aligned
        )
        candidates: list[tuple[float, str, Quote, float, float]] = []
        for quote in aligned:
            home_decimal = american_to_decimal(quote.home_ml)
            away_decimal = american_to_decimal(quote.away_ml)
            candidates.append(
                (pin_home * home_decimal - 1.0, "home", quote, home_decimal, pin_home)
            )
            candidates.append(
                (pin_away * away_decimal - 1.0, "away", quote, away_decimal, pin_away)
            )
        best_ev, side, quote, decimal_odds, probability = max(
            candidates, key=lambda row: row[0]
        )
        max_gap = max(
            abs((q.snapshot_time - pin.snapshot_time).total_seconds()) / 3600.0
            for q in aligned
        )
        evaluated.append(
            EvaluatedGame(
                season=game.season,
                game_id=game.game_id,
                game_type=game.game_type,
                week=game.week,
                horizon_hours=game.horizon_hours,
                home_won=game.home_won,
                pin_home=pin_home,
                us_home=us_home,
                pin_hold=two_way_hold(pin.home_ml, pin.away_ml),
                us_books=len(aligned),
                max_gap_hours=max_gap,
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
                game_type=game.game_type,
                week=game.week,
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


def grouped_by_horizon(games: Iterable[EvaluatedGame]) -> dict[float, list[EvaluatedGame]]:
    grouped: dict[float, list[EvaluatedGame]] = defaultdict(list)
    for game in games:
        grouped[game.horizon_hours].append(game)
    return dict(sorted(grouped.items()))


def print_coverage(
    *,
    results: Sequence[NflResult],
    quotes: Sequence[GameQuotes],
    evaluated: Sequence[EvaluatedGame],
    schedule_drops: Counter[str],
    fetch_drops: Counter[str],
    eval_drops: Counter[str],
    stats: FetchStats,
) -> None:
    print("Coverage")
    print(f"  final non-tie games in requested seasons/types: {len(results):,}")
    print(f"  game-horizon quote rows found: {len(quotes):,}")
    print(f"  eligible game-horizon rows after filters: {len(evaluated):,}")
    print(f"  Odds API calls made: {stats.api_calls:,}")
    print(f"  Odds API credits used: {stats.api_credits:,}")
    print(f"  cached snapshots reused: {stats.cached_snapshots:,}")
    if stats.last_remaining is not None:
        print(f"  credits remaining after last API call: {stats.last_remaining:,}")
    drops = schedule_drops + fetch_drops + eval_drops
    if drops:
        print("  dropped:")
        for reason, n in sorted(drops.items()):
            print(f"    {reason}: {n:,}")

    by_horizon = grouped_by_horizon(evaluated)
    if not by_horizon:
        return
    print("\nEligible coverage by horizon")
    print(
        f"  {'hours_before':>12} {'rows':>6} {'avg_us_books':>12} "
        f"{'median_gap_m':>13} {'median_hold':>12}"
    )
    print(f"  {'-' * 64}")
    for horizon, rows in by_horizon.items():
        print(
            f"  {horizon:12.1f} {len(rows):6d} "
            f"{statistics.mean(g.us_books for g in rows):12.2f} "
            f"{statistics.median(g.max_gap_hours for g in rows) * 60:13.1f} "
            f"{statistics.median(g.pin_hold for g in rows):12.2%}"
        )


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


def print_brier(evaluated: Sequence[EvaluatedGame], *, samples: int) -> None:
    if not evaluated:
        return
    print("\nBrier, home-win probability on eligible games")
    print(
        f"  {'hours_before':>12} {'rows':>6} {'Pinnacle':>10} "
        f"{'US median':>10} {'Pin-US diff 95% CI':>31}"
    )
    print(f"  {'-' * 77}")
    for horizon, rows in grouped_by_horizon(evaluated).items():
        y = np.array([1.0 if game.home_won else 0.0 for game in rows])
        pin = np.array([game.pin_home for game in rows])
        us = np.array([game.us_home for game in rows])
        diff = (pin - y) ** 2 - (us - y) ** 2
        mean, lo, hi = boot_ci(diff, samples=samples, seed=19 + int(horizon * 10))
        print(
            f"  {horizon:12.1f} {len(rows):6d} {brier(pin, y):10.6f} "
            f"{brier(us, y):10.6f} {mean:+.6f} [{lo:+.6f}, {hi:+.6f}]"
        )


def print_strategy(
    evaluated: Sequence[EvaluatedGame], thresholds: tuple[float, ...], *, samples: int
) -> None:
    print("\nStrategy: one best US bet per game using Pinnacle no-vig fair probability")
    for horizon, rows in grouped_by_horizon(evaluated).items():
        print(f"\n  Horizon: {horizon:.1f} hours before kickoff")
        print(
            f"  {'min_EV':>7} {'bets':>6} {'win%':>7} {'avg_EV':>8} {'ROI':>8} "
            f"{'95% CI':>23} {'z':>7} {'Brier':>8}"
        )
        print(f"  {'-' * 91}")
        for threshold in thresholds:
            bets = settle(rows, threshold)
            if not bets:
                print(
                    f"  {threshold:7.1%} {0:6d} {'-':>7} {'-':>8} "
                    f"{'-':>8} {'-':>23} {'-':>7} {'-':>8}"
                )
                continue
            returns = np.array([bet.ret for bet in bets])
            probs = np.array([bet.probability for bet in bets])
            wins = np.array([1.0 if bet.won else 0.0 for bet in bets])
            seed = 41 + int(horizon * 10) + int(threshold * 10_000)
            mean, lo, hi = boot_ci(returns, samples=samples, seed=seed)
            print(
                f"  {threshold:7.1%} {len(bets):6d} {float(wins.mean()):7.1%} "
                f"{statistics.mean(bet.ev for bet in bets):8.2%} {mean:+8.2%} "
                f"[{lo:+7.2%}, {hi:+7.2%}] {z_score(returns):+7.2f} "
                f"{brier(probs, wins):8.4f}"
            )


def print_season_breakdown(
    evaluated: Sequence[EvaluatedGame], threshold: float, *, samples: int
) -> None:
    for horizon, rows in grouped_by_horizon(evaluated).items():
        bets = settle(rows, threshold)
        if not bets:
            continue
        by_season: dict[int, list[SettledBet]] = defaultdict(list)
        for bet in bets:
            by_season[bet.season].append(bet)
        print(f"\nSeason breakdown: {horizon:.1f}h before, min_EV {threshold:.1%}")
        print(
            f"  {'season':>6} {'bets':>6} {'win%':>7} {'avg_EV':>8} "
            f"{'ROI':>8} {'95% CI':>23}"
        )
        print(f"  {'-' * 67}")
        for season in sorted(by_season):
            season_bets = by_season[season]
            returns = np.array([bet.ret for bet in season_bets])
            mean, lo, hi = boot_ci(returns, samples=samples, seed=53 + season)
            print(
                f"  {season:6d} {len(season_bets):6d} "
                f"{statistics.mean(bet.won for bet in season_bets):7.1%} "
                f"{statistics.mean(bet.ev for bet in season_bets):8.2%} "
                f"{mean:+8.2%} [{lo:+7.2%}, {hi:+7.2%}]"
            )


def write_evaluated(path: Path, evaluated: Sequence[EvaluatedGame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "season",
                "game_id",
                "game_type",
                "week",
                "horizon_hours",
                "home_won",
                "pin_home",
                "us_home",
                "pin_hold",
                "us_books",
                "max_gap_hours",
                "best_side",
                "best_book",
                "best_decimal",
                "best_prob",
                "best_ev",
            ],
        )
        writer.writeheader()
        for game in evaluated:
            writer.writerow(
                {
                    "season": game.season,
                    "game_id": game.game_id,
                    "game_type": game.game_type,
                    "week": game.week,
                    "horizon_hours": game.horizon_hours,
                    "home_won": game.home_won,
                    "pin_home": game.pin_home,
                    "us_home": game.us_home,
                    "pin_hold": game.pin_hold,
                    "us_books": game.us_books,
                    "max_gap_hours": game.max_gap_hours,
                    "best_side": game.best_side,
                    "best_book": game.best_book,
                    "best_decimal": game.best_decimal,
                    "best_prob": game.best_prob,
                    "best_ev": game.best_ev,
                }
            )


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", nargs="+", default=["2021", "2022", "2023", "2024", "2025"])
    ap.add_argument("--game-types", nargs="+", default=list(REGULAR_GAME_TYPES))
    ap.add_argument("--hours-before", nargs="+", type=positive_float, default=[24.0, 4.0, 1.0])
    ap.add_argument("--pinnacle-book", default="pinnacle")
    ap.add_argument(
        "--us-books",
        default="top5",
        help="Panel name (top5/top6/priority5/priority6), 'all', or comma-separated books.",
    )
    ap.add_argument("--max-gap-hours", type=float, default=1.0)
    ap.add_argument("--min-us-books", type=int, default=1)
    ap.add_argument("--devig", default="proportional", choices=("proportional", "shin"))
    ap.add_argument("--thresholds", nargs="+", default=["0", "0.01", "0.02", "0.03", "0.05"])
    ap.add_argument("--calibration-bins", type=int, default=10)
    ap.add_argument("--bootstrap-samples", type=int, default=4000)
    ap.add_argument("--calibration-threshold", type=float, default=0.0)
    ap.add_argument("--regions", default="us,eu")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/odds_history/nfl_pinnacle_ev"))
    ap.add_argument("--output", type=Path, default=Path("output/nfl_pinnacle_ev/evaluated.csv"))
    ap.add_argument("--refresh-schedule", action="store_true")
    ap.add_argument("--refresh-odds", action="store_true")
    ap.add_argument("--limit-games", type=int)
    ap.add_argument("--limit-snapshots", type=int)
    ap.add_argument("--request-timeout", type=float, default=60.0)
    ap.add_argument("--commence-tolerance-hours", type=float, default=12.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seasons = parse_seasons(args.seasons)
    game_types = parse_game_types(args.game_types)
    hours_before = tuple(sorted(dict.fromkeys(args.hours_before), reverse=True))
    thresholds = parse_thresholds(args.thresholds)
    us_books = parse_books(args.us_books)
    if args.max_gap_hours < 0:
        raise SystemExit("--max-gap-hours must be non-negative")
    if args.min_us_books < 1:
        raise SystemExit("--min-us-books must be at least 1")
    if args.calibration_bins < 2:
        raise SystemExit("--calibration-bins must be at least 2")

    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("ODDS_API_KEY is required")

    selected_books = "all non-Pinnacle" if us_books is None else ",".join(us_books)
    print("NFL Pinnacle-fair +EV validation")
    print(f"  sport: {SPORT_KEY}")
    print(f"  seasons: {','.join(str(s) for s in seasons)}")
    print(f"  game_types: {','.join(game_types)}")
    print(f"  hours_before: {','.join(f'{h:g}' for h in hours_before)}")
    print(f"  Pinnacle book: {args.pinnacle_book}")
    print(f"  US books: {selected_books}")
    print(f"  max book update gap: {args.max_gap_hours:.2f}h")
    print(f"  min aligned US books: {args.min_us_books}")
    print(f"  de-vig: {args.devig}")
    print(f"  raw cache: {args.cache_dir}")
    print()

    results, schedule_drops = load_results(
        cache_dir=args.cache_dir,
        seasons=seasons,
        game_types=game_types,
        refresh_schedule=args.refresh_schedule,
    )
    if args.limit_games is not None:
        results = results[: args.limit_games]
    if args.dry_run:
        unique_snapshots = {
            (game.commence_time - timedelta(hours=hours)).replace(second=0, microsecond=0)
            for game in results
            for hours in hours_before
        }
        print(f"Dry run games: {len(results):,}")
        print(f"Dry run unique snapshots: {len(unique_snapshots):,}")
        print(f"Estimated credits without cache: {len(unique_snapshots) * 20:,}")
        return

    quotes, fetch_drops, stats = collect_game_quotes(
        games=results,
        hours_before=hours_before,
        cache_dir=args.cache_dir,
        api_key=api_key or "",
        regions=args.regions,
        refresh_odds=args.refresh_odds,
        timeout=args.request_timeout,
        commence_tolerance_hours=args.commence_tolerance_hours,
        limit_snapshots=args.limit_snapshots,
    )
    evaluated, eval_drops = evaluate_games(
        quotes,
        pinnacle_book=args.pinnacle_book,
        us_books=us_books,
        max_gap_hours=args.max_gap_hours,
        min_us_books=args.min_us_books,
        devig=args.devig,
    )
    write_evaluated(args.output, evaluated)

    print_coverage(
        results=results,
        quotes=quotes,
        evaluated=evaluated,
        schedule_drops=schedule_drops,
        fetch_drops=fetch_drops,
        eval_drops=eval_drops,
        stats=stats,
    )
    print_brier(evaluated, samples=args.bootstrap_samples)
    for horizon, rows in grouped_by_horizon(evaluated).items():
        print_calibration(
            f"Pinnacle home-win probability, {horizon:.1f}h before kickoff",
            [game.pin_home for game in rows],
            [game.home_won for game in rows],
            bins=args.calibration_bins,
        )
    print_strategy(evaluated, thresholds, samples=args.bootstrap_samples)
    if thresholds:
        print_season_breakdown(evaluated, thresholds[0], samples=args.bootstrap_samples)
    for horizon, rows in grouped_by_horizon(evaluated).items():
        calibration_bets = settle(rows, args.calibration_threshold)
        if calibration_bets:
            print_calibration(
                "Pinnacle probability of chosen side, "
                f"{horizon:.1f}h before, min_EV {args.calibration_threshold:.1%}",
                [bet.probability for bet in calibration_bets],
                [bet.won for bet in calibration_bets],
                bins=args.calibration_bins,
            )
    print(f"\nWrote evaluated rows to {args.output}")


if __name__ == "__main__":
    main()
