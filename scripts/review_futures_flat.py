"""Futures betting review: flat-stake ROI, per-season consistency, and threshold monotonicity.

``scripts/backtest_futures.py`` reports quarter-Kelly ROI. That is not comparable to the flat-1u
figures used everywhere else in this project, and it is badly distorted here: Kelly sizes heavy
favourites to ~0.000 units, so in 2017 the six winning ``miss_playoffs`` bets all carried zero
stake while the five losers carried all of it, producing an ROI of exactly -100% from a 55% win
rate. Any market whose bets cluster at long odds gets its ROI decided by a handful of positions.

This re-stakes every bet at flat 1u and applies the guards the rest of the project uses:

  * bootstrap confidence interval on pooled ROI
  * per-season sign consistency, which is what distinguished June from the other months
  * threshold monotonicity, which refuted the CLV claim
  * odds-bucket split, to check whether any result is just longshot or favourite bias
  * complementary-pair coherence: ``make_playoffs`` and ``miss_playoffs`` are opposite sides of
    the same question, so a real directional bias should show up as opposite signs, while both
    sides losing indicates the vig is simply being paid twice

Bet selection is untouched; only staking changes. The de-vig uses ``MARKET_SLOTS`` from the
audited backtest, so multi-winner markets normalise to the correct number of winning slots.

    uv run python scripts/review_futures_flat.py
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_futures import _run_season_backtest
from src.database import PostgresConfig, PostgresHandler

BOOT = 4000
SEASONS = {
    "division": [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025],
    "make_playoffs": [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025],
    "miss_playoffs": [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025],
}


def flat_profit(bet) -> float:
    return (bet.decimal_odds - 1.0) if bet.actual_win else -1.0


def boot_ci(arr: np.ndarray, seed: int = 23):
    rng = random.Random(seed)
    n = len(arr)
    draws = sorted(
        float(np.mean([arr[rng.randrange(n)] for _ in range(n)])) for _ in range(BOOT)
    )
    return float(arr.mean()), draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]


def summarise(bets, label: str, indent: str = "") -> tuple[float, float, float] | None:
    if len(bets) < 10:
        print(f"{indent}{label:>22}: {len(bets)} bets, too few")
        return None
    arr = np.array([flat_profit(b) for b in bets])
    m, lo, hi = boot_ci(arr)
    wins = sum(1 for b in bets if b.actual_win)
    flag = "excludes zero" if (lo > 0 or hi < 0) else ""
    print(f"{indent}{label:>22}: {len(bets):4d} bets | {wins / len(bets):5.1%} win | "
          f"ROI {m:+7.2%} | 95% CI [{lo:+7.2%}, {hi:+7.2%}] {flag}")
    return m, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edge-threshold", type=float, default=0.05)
    ap.add_argument("--markets", default="division,make_playoffs,miss_playoffs")
    args = ap.parse_args()
    markets = [m for m in args.markets.split(",") if m]

    cfg = PostgresConfig.from_env()
    store: dict[str, list] = {}
    with PostgresHandler(cfg) as pg:
        for market in markets:
            allbets = []
            for season in SEASONS.get(market, []):
                try:
                    allbets += _run_season_backtest(
                        season, market, args.edge_threshold, 0.25, 0.05, pg
                    )
                except Exception as exc:
                    print(f"  {season} {market}: {type(exc).__name__}: {str(exc)[:70]}")
            store[market] = allbets

    print()
    print("=" * 96)
    print(f"FLAT 1u STAKING, edge threshold {args.edge_threshold:.0%}")
    print("=" * 96)
    print()
    pooled = {}
    for market in markets:
        res = summarise(store[market], market)
        if res:
            pooled[market] = res
    print()

    print("Per season, flat 1u. Sign consistency is the guard that isolated June.")
    for market in markets:
        by_season: dict[int, list] = defaultdict(list)
        for b in store[market]:
            by_season[b.season].append(b)
        if not by_season:
            continue
        print(f"\n  {market}")
        print(f"  {'season':>7} | {'bets':>5} | {'win%':>6} | {'ROI':>9} | {'net':>8}")
        print("  " + "-" * 48)
        signs = ""
        for season in sorted(by_season):
            sub = by_season[season]
            arr = np.array([flat_profit(b) for b in sub])
            wins = sum(1 for b in sub if b.actual_win)
            signs += "+" if arr.mean() > 0 else "-"
            print(f"  {season:>7} | {len(sub):5d} | {wins / len(sub):5.1%} | "
                  f"{arr.mean():+8.2%} | {arr.sum():+7.2f}u")
        pos = signs.count("+")
        print(f"  sign pattern {signs}  ({pos}/{len(signs)} positive)")
    print()

    print("Threshold monotonicity. A real edge should not appear at one cut only.")
    for market in markets:
        if not store[market]:
            continue
        print(f"\n  {market}")
        print(f"  {'min edge':>9} | {'bets':>5} | {'win%':>6} | {'ROI':>9} | {'95% CI':>22}")
        print("  " + "-" * 62)
        for thr in (0.02, 0.05, 0.08, 0.12, 0.20):
            sub = [b for b in store[market] if b.edge >= thr]
            if len(sub) < 10:
                print(f"  {thr:>9.0%} | {len(sub):5d} | too few")
                continue
            arr = np.array([flat_profit(b) for b in sub])
            m, lo, hi = boot_ci(arr)
            wins = sum(1 for b in sub if b.actual_win)
            print(f"  {thr:>9.0%} | {len(sub):5d} | {wins / len(sub):5.1%} | {m:+8.2%} | "
                  f"[{lo:+7.2%}, {hi:+7.2%}]")
    print()

    print("Odds buckets. Separates genuine edge from longshot or favourite bias.")
    buckets = (
        ("heavy fav <= -500", lambda o: o <= -500),
        ("fav -500..-150", lambda o: -500 < o <= -150),
        ("near even -150..+150", lambda o: -150 < o <= 150),
        ("longshot > +150", lambda o: o > 150),
    )
    for market in markets:
        if not store[market]:
            continue
        print(f"\n  {market}")
        for name, pred in buckets:
            sub = [b for b in store[market] if pred(b.best_odds)]
            summarise(sub, name, indent="  ")
    print()

    if "make_playoffs" in pooled and "miss_playoffs" in pooled:
        mk = pooled["make_playoffs"][0]
        ms = pooled["miss_playoffs"][0]
        print("Complementary-pair coherence")
        print(f"  make_playoffs {mk:+.2%}   miss_playoffs {ms:+.2%}")
        if mk < 0 and ms > 0:
            print("  Opposite signs: consistent with a real directional bias, the model "
                  "overrating playoff chances.")
        elif mk < 0 and ms < 0:
            print("  Both negative: no directional bias, the vig is simply paid on both "
                  "sides.")
        else:
            print("  Both positive is impossible without a pricing error; inspect the de-vig.")


if __name__ == "__main__":
    main()
