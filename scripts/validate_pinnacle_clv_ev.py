"""Validate Pinnacle-fair +EV line shopping against US moneylines.

This answers the CLV/+EV question directly and uses the database only:

1. Pull final outcomes from ``mlb.games`` + ``mlb.linescore``.
2. Pull Pinnacle and US book prices from ``mlb.odds`` at the requested line type.
3. Keep only games where the US quote used for execution is close enough in time to
   Pinnacle's quote.
4. Treat Pinnacle's no-vig probability as fair value.
5. Bet the single best US side/book per game when expected ROI clears each threshold.
6. Report realized ROI, Brier score, and calibration.

No model probabilities are used here. This is purely: "if Pinnacle is fair, were US
books hanging +EV lines at the same time, and did those bets win?"

Example:

    uv run python scripts/validate_pinnacle_clv_ev.py --seasons 2021 2022 2023 2024 2025
"""

from __future__ import annotations

import argparse
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_moneyline_lineshop import PANEL_PRIORITY, PANEL_TOP5, PANEL_TOP6
from src.betting.odds import american_to_decimal, american_to_prob, no_vig_two_way
from src.database import PostgresConfig

PANELS = {
    "top5": PANEL_TOP5,
    "top6": PANEL_TOP6,
    "priority5": PANEL_PRIORITY[:5],
    "priority6": PANEL_PRIORITY[:6],
}


@dataclass(frozen=True)
class Quote:
    bookmaker: str
    home_ml: int
    away_ml: int
    snapshot_time: datetime


@dataclass(frozen=True)
class GameQuotes:
    season: int
    game_pk: int
    game_datetime: datetime
    month: int
    home_won: bool
    quotes: dict[str, Quote]


@dataclass(frozen=True)
class EvaluatedGame:
    season: int
    game_pk: int
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
    game_pk: int
    side: str
    book: str
    decimal_odds: float
    probability: float
    ev: float
    won: bool
    ret: float


def parse_seasons(values: list[str]) -> tuple[int, ...]:
    seasons: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                seasons.append(int(part))
    if not seasons:
        raise SystemExit("at least one season is required")
    return tuple(dict.fromkeys(seasons))


def parse_thresholds(values: list[str]) -> tuple[float, ...]:
    thresholds: list[float] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                threshold = float(part)
                if threshold < 0:
                    raise SystemExit("EV thresholds must be non-negative")
                thresholds.append(threshold)
    return tuple(sorted(dict.fromkeys(thresholds)))


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


def sql_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise SystemExit(f"unsafe SQL identifier: {value!r}")
    return value


def as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def two_way_hold(home_ml: int, away_ml: int) -> float:
    return american_to_prob(home_ml) + american_to_prob(away_ml) - 1.0


def fetch_game_quotes(
    *,
    seasons: tuple[int, ...],
    line_type: str,
    pinnacle_book: str,
    us_books: tuple[str, ...] | None,
) -> list[GameQuotes]:
    cfg = PostgresConfig.from_env()
    schema = sql_identifier(cfg.schema)
    book_filter = "o.bookmaker <> %s" if us_books is None else "o.bookmaker = ANY(%s)"
    params: list[Any] = [list(seasons), line_type, pinnacle_book]
    params.append(list(us_books) if us_books is not None else pinnacle_book)

    query = f"""
        WITH outcomes AS (
            SELECT
                g.season::int AS season,
                g.game_pk,
                COALESCE(g.game_datetime, g.game_date)::timestamptz AS game_datetime,
                EXTRACT(MONTH FROM COALESCE(g.game_datetime, g.game_date))::int AS month,
                SUM(l.runs) FILTER (WHERE l.team_type = 'away')::int AS away_runs,
                SUM(l.runs) FILTER (WHERE l.team_type = 'home')::int AS home_runs
            FROM {schema}.games g
            JOIN {schema}.linescore l USING (game_pk)
            WHERE g.season::int = ANY(%s)
              AND g.game_type = 'R'
              AND g.abstract_game_state = 'Final'
            GROUP BY g.season, g.game_pk, g.game_datetime, g.game_date
        )
        SELECT
            outcomes.season,
            outcomes.game_pk,
            outcomes.game_datetime,
            outcomes.month,
            outcomes.home_runs > outcomes.away_runs AS home_won,
            o.bookmaker,
            o.home_ml,
            o.away_ml,
            o.snapshot_time
        FROM outcomes
        JOIN {schema}.odds o ON o.game_pk = outcomes.game_pk
        WHERE outcomes.home_runs <> outcomes.away_runs
          AND o.market = 'h2h'
          AND o.line_type = %s
          AND o.home_ml IS NOT NULL
          AND o.away_ml IS NOT NULL
          AND o.snapshot_time IS NOT NULL
          AND (o.bookmaker = %s OR {book_filter})
        ORDER BY outcomes.season, outcomes.game_datetime, outcomes.game_pk, o.bookmaker
    """

    grouped: dict[int, GameQuotes] = {}
    with psycopg.connect(
        dbname=cfg.dbname,
        user=cfg.user,
        password=cfg.password,
        host=cfg.host,
        port=cfg.port,
        connect_timeout=15,
    ) as conn, conn.cursor() as cur:
        cur.execute(query, params)
        for season, game_pk, game_dt, month, home_won, book, home_ml, away_ml, snap in cur:
            pk = int(game_pk)
            if pk not in grouped:
                grouped[pk] = GameQuotes(
                    season=int(season),
                    game_pk=pk,
                    game_datetime=as_datetime(game_dt),
                    month=int(month),
                    home_won=bool(home_won),
                    quotes={},
                )
            grouped[pk].quotes[str(book)] = Quote(
                bookmaker=str(book),
                home_ml=int(home_ml),
                away_ml=int(away_ml),
                snapshot_time=as_datetime(snap),
            )
    return list(grouped.values())


def evaluate_games(
    games: list[GameQuotes],
    *,
    pinnacle_book: str,
    max_gap_hours: float,
    min_us_books: int,
    devig: str,
    exclude_months: set[int],
) -> tuple[list[EvaluatedGame], Counter[str]]:
    evaluated: list[EvaluatedGame] = []
    drops: Counter[str] = Counter()
    for game in games:
        if game.month in exclude_months:
            drops["excluded_month"] += 1
            continue
        pin = game.quotes.get(pinnacle_book)
        if pin is None:
            drops["no_pinnacle"] += 1
            continue
        us_quotes = [q for book, q in game.quotes.items() if book != pinnacle_book]
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
            candidates.append((pin_home * home_decimal - 1.0, "home", quote, home_decimal, pin_home))
            candidates.append((pin_away * away_decimal - 1.0, "away", quote, away_decimal, pin_away))
        best_ev, side, quote, decimal_odds, probability = max(candidates, key=lambda row: row[0])
        max_gap = max(
            abs((q.snapshot_time - pin.snapshot_time).total_seconds()) / 3600.0 for q in aligned
        )
        evaluated.append(
            EvaluatedGame(
                season=game.season,
                game_pk=game.game_pk,
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


def settle(evaluated: list[EvaluatedGame], threshold: float) -> list[SettledBet]:
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


def boot_ci(values: np.ndarray, *, samples: int, seed: int) -> tuple[float, float, float]:
    if len(values) == 0:
        return math.nan, math.nan, math.nan
    if samples <= 0:
        mean = float(values.mean())
        return mean, math.nan, math.nan
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(float(np.mean([values[rng.randrange(n)] for _ in range(n)])) for _ in range(samples))
    return float(values.mean()), draws[int(0.025 * samples)], draws[int(0.975 * samples)]


def z_score(values: np.ndarray) -> float:
    if len(values) < 2:
        return math.nan
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    return float(values.mean() / se) if se else math.nan


def brier(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probabilities - outcomes) ** 2))


def print_brier(evaluated: list[EvaluatedGame], *, samples: int) -> None:
    if not evaluated:
        return
    y = np.array([1.0 if game.home_won else 0.0 for game in evaluated])
    pin = np.array([game.pin_home for game in evaluated])
    us = np.array([game.us_home for game in evaluated])
    diff = (pin - y) ** 2 - (us - y) ** 2
    mean, lo, hi = boot_ci(diff, samples=samples, seed=19)
    print("\nBrier, home-win probability on eligible games")
    print(f"  {'source':<18} {'brier':>9}")
    print(f"  {'-' * 28}")
    print(f"  {'Pinnacle no-vig':<18} {brier(pin, y):9.6f}")
    print(f"  {'US median no-vig':<18} {brier(us, y):9.6f}")
    print(
        f"  Pinnacle minus US: {mean:+.6f} "
        f"95% CI [{lo:+.6f}, {hi:+.6f}]"
    )


def calibration_rows(
    probabilities: list[float], outcomes: list[bool], *, bins: int
) -> list[tuple[str, int, float, float, float]]:
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have equal length")
    bucketed: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for probability, outcome in zip(probabilities, outcomes, strict=True):
        idx = min(int(probability * bins), bins - 1)
        bucketed[idx].append((probability, outcome))
    rows = []
    for idx, bucket in enumerate(bucketed):
        if not bucket:
            continue
        lo = idx / bins
        hi = (idx + 1) / bins
        avg_p = statistics.mean(p for p, _ in bucket)
        actual = statistics.mean(1.0 if outcome else 0.0 for _, outcome in bucket)
        rows.append((f"{lo:.1f}-{hi:.1f}", len(bucket), avg_p, actual, actual - avg_p))
    return rows


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
    games: list[GameQuotes], evaluated: list[EvaluatedGame], drops: Counter[str]) -> None:
    print("Coverage")
    print(f"  queried games with at least one requested quote: {len(games):,}")
    print(f"  eligible games after Pinnacle/time/book filters: {len(evaluated):,}")
    if drops:
        print("  dropped:")
        for reason, n in sorted(drops.items()):
            print(f"    {reason}: {n:,}")
    if not evaluated:
        return
    print("\nEligible coverage by season")
    print(f"  {'season':>6} {'games':>6} {'avg_us_books':>12} {'median_gap_h':>13} {'median_hold':>12}")
    print(f"  {'-' * 58}")
    by_season: dict[int, list[EvaluatedGame]] = defaultdict(list)
    for game in evaluated:
        by_season[game.season].append(game)
    for season in sorted(by_season):
        rows = by_season[season]
        print(
            f"  {season:6d} {len(rows):6d} "
            f"{statistics.mean(g.us_books for g in rows):12.2f} "
            f"{statistics.median(g.max_gap_hours for g in rows):13.2f} "
            f"{statistics.median(g.pin_hold for g in rows):12.2%}"
        )


def print_strategy(evaluated: list[EvaluatedGame], thresholds: tuple[float, ...], *, samples: int) -> None:
    print("\nStrategy: one best US bet per game using Pinnacle no-vig fair probability")
    print(
        f"  {'min_EV':>7} {'bets':>6} {'win%':>7} {'avg_EV':>8} {'ROI':>8} "
        f"{'95% CI':>23} {'z':>7} {'Brier':>8}"
    )
    print(f"  {'-' * 91}")
    for threshold in thresholds:
        bets = settle(evaluated, threshold)
        if not bets:
            print(f"  {threshold:7.1%} {0:6d} {'-':>7} {'-':>8} {'-':>8} {'-':>23} {'-':>7} {'-':>8}")
            continue
        returns = np.array([bet.ret for bet in bets])
        probs = np.array([bet.probability for bet in bets])
        wins = np.array([1.0 if bet.won else 0.0 for bet in bets])
        mean, lo, hi = boot_ci(returns, samples=samples, seed=41)
        print(
            f"  {threshold:7.1%} {len(bets):6d} {float(wins.mean()):7.1%} "
            f"{statistics.mean(bet.ev for bet in bets):8.2%} {mean:+8.2%} "
            f"[{lo:+7.2%}, {hi:+7.2%}] {z_score(returns):+7.2f} "
            f"{brier(probs, wins):8.4f}"
        )


def print_season_breakdown(
    evaluated: list[EvaluatedGame], threshold: float, *, samples: int
) -> None:
    bets = settle(evaluated, threshold)
    if not bets:
        return
    by_season: dict[int, list[SettledBet]] = defaultdict(list)
    for bet in bets:
        by_season[bet.season].append(bet)
    print(f"\nSeason breakdown at min_EV {threshold:.1%}")
    print(f"  {'season':>6} {'bets':>6} {'win%':>7} {'avg_EV':>8} {'ROI':>8} {'95% CI':>23}")
    print(f"  {'-' * 67}")
    for season in sorted(by_season):
        rows = by_season[season]
        returns = np.array([bet.ret for bet in rows])
        mean, lo, hi = boot_ci(returns, samples=samples, seed=53 + season)
        print(
            f"  {season:6d} {len(rows):6d} {statistics.mean(bet.won for bet in rows):7.1%} "
            f"{statistics.mean(bet.ev for bet in rows):8.2%} {mean:+8.2%} "
            f"[{lo:+7.2%}, {hi:+7.2%}]"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", nargs="+", default=["2021", "2022", "2023", "2024", "2025"])
    ap.add_argument("--line-type", default="open", choices=("open", "close", "true_close"))
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
    ap.add_argument("--exclude-month", action="append", type=int, default=[])
    ap.add_argument("--calibration-bins", type=int, default=10)
    ap.add_argument("--bootstrap-samples", type=int, default=4000)
    ap.add_argument(
        "--calibration-threshold",
        type=float,
        default=0.0,
        help="EV threshold for the bet-side calibration table.",
    )
    args = ap.parse_args()

    seasons = parse_seasons(args.seasons)
    thresholds = parse_thresholds(args.thresholds)
    us_books = parse_books(args.us_books)
    if args.max_gap_hours < 0:
        raise SystemExit("--max-gap-hours must be non-negative")
    if args.min_us_books < 1:
        raise SystemExit("--min-us-books must be at least 1")
    if args.calibration_bins < 2:
        raise SystemExit("--calibration-bins must be at least 2")

    selected_books = "all non-Pinnacle" if us_books is None else ",".join(us_books)
    print("Pinnacle-fair +EV validation")
    print(f"  seasons: {','.join(str(s) for s in seasons)}")
    print(f"  line_type: {args.line_type}")
    print(f"  Pinnacle book: {args.pinnacle_book}")
    print(f"  US books: {selected_books}")
    print(f"  max snapshot gap: {args.max_gap_hours:.2f}h")
    print(f"  min aligned US books: {args.min_us_books}")
    print(f"  de-vig: {args.devig}")
    if args.exclude_month:
        print(f"  excluded months: {','.join(str(m) for m in sorted(set(args.exclude_month)))}")
    print()

    games = fetch_game_quotes(
        seasons=seasons,
        line_type=args.line_type,
        pinnacle_book=args.pinnacle_book,
        us_books=us_books,
    )
    evaluated, drops = evaluate_games(
        games,
        pinnacle_book=args.pinnacle_book,
        max_gap_hours=args.max_gap_hours,
        min_us_books=args.min_us_books,
        devig=args.devig,
        exclude_months=set(args.exclude_month),
    )

    print_coverage(games, evaluated, drops)
    print_brier(evaluated, samples=args.bootstrap_samples)
    if evaluated:
        print_calibration(
            "Pinnacle home-win probability on all eligible games",
            [game.pin_home for game in evaluated],
            [game.home_won for game in evaluated],
            bins=args.calibration_bins,
        )
    print_strategy(evaluated, thresholds, samples=args.bootstrap_samples)
    if thresholds:
        print_season_breakdown(evaluated, thresholds[0], samples=args.bootstrap_samples)
    calibration_bets = settle(evaluated, args.calibration_threshold)
    if calibration_bets:
        print_calibration(
            f"Pinnacle probability of chosen side, bets at min_EV {args.calibration_threshold:.1%}",
            [bet.probability for bet in calibration_bets],
            [bet.won for bet in calibration_bets],
            bins=args.calibration_bins,
        )


if __name__ == "__main__":
    main()
