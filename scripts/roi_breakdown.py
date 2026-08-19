"""Consolidated flat-1u ROI by season and bet type.

Every market in this project has been measured, but the numbers live in six different scripts with
different staking conventions, which makes them hard to compare. This assembles one table at a
single convention: flat 1 unit per bet, best available price, walk-forward model fits.

Bet types:

  moneyline            game-by-game, close, edge >= 3%, best of a five-book panel
  moneyline ex-June    the same with June removed, June being the one month the model
                       reliably loses in and where the market is measurably sharper
  division             preseason division futures, era-aware de-vig
  make_playoffs        preseason playoff-field futures
  miss_playoffs        the fade side of the same market
  win_totals           preseason season win totals at assumed -110, no odds are stored

Futures ROI is not comparable to moneyline ROI without care. A moneyline bet resolves in hours; a
futures bet ties capital from March to October, and there are roughly ten qualifying futures bets
per season against a thousand moneyline bets. Per-bet variance is also far larger because futures
payouts run from -650 to +600. The table reports bet counts alongside every figure so the reader
can see which cells carry any weight.

    uv run python scripts/roi_breakdown.py
    uv run python scripts/roi_breakdown.py --skip-futures    # moneyline only, much faster
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_futures import _run_season_backtest
from scripts.futures_outcomes import season_records
from scripts.test_june_scoped import load_season
from src.database import PostgresConfig, PostgresHandler

BOOT = 4000
ML_SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
FUTURES_SEASONS = {
    "division": [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025],
    "championship": [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025],
    "make_playoffs": [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025],
    "miss_playoffs": [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025],
}
# al_pennant and nl_pennant are excluded: the model's league_championship_prob sums to 4.0 across
# the league where 2.0 is required, so pennant edges are inflated roughly 2x and the market is not
# measurable until that is fixed.
WIN_TOTALS = Path("resources/season_win_totals_2022_2025.csv")
PROJECTIONS = Path("output/prior_baseline_backtest.csv")


def boot_ci(arr: np.ndarray, seed: int = 23):
    rng = random.Random(seed)
    n = len(arr)
    draws = sorted(
        float(np.mean([arr[rng.randrange(n)] for _ in range(n)])) for _ in range(BOOT)
    )
    return float(arr.mean()), draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]


def settle_moneyline(rows, edge: float, drop_june: bool):
    """Flat 1u returns for one season of game-by-game bets."""
    out = []
    for r in rows:
        if drop_june and r["month"] == 6:
            continue
        signed = r["model"] - r["fair"]
        if abs(signed) < edge:
            continue
        home_side = signed >= 0
        dec = r["best_home"] if home_side else r["best_away"]
        won = r["home_won"] == home_side
        out.append((dec - 1.0) if won else -1.0)
    return np.array(out)


def settle_win_totals(conn, schema: str, edge: float):
    """Flat 1u returns per season for preseason win totals at -110."""
    lines = {}
    for row in csv.DictReader(WIN_TOTALS.open()):
        lines[(int(row["season"]), row["abbreviation"])] = float(row["win_total"])
    proj = {}
    for row in csv.DictReader(PROJECTIONS.open()):
        if row["as_of_bucket"] != "opening_day" or row["projection_type"] != "model":
            continue
        proj[(int(row["season"]), row["abbreviation"])] = (
            int(row["team_id"]),
            float(row["expected_wins"]),
        )
    seasons = sorted({s for s, _ in lines})
    actual = {}
    for season in seasons:
        wins, _losses, _h2h = season_records(conn, schema, season)
        for team_id, w in wins.items():
            actual[(season, int(team_id))] = int(w)

    payout = 100.0 / 110.0
    by_season: dict[int, list[float]] = defaultdict(list)
    for key, line in lines.items():
        if key not in proj:
            continue
        team_id, expected = proj[key]
        season = key[0]
        if (season, team_id) not in actual:
            continue
        wins = actual[(season, team_id)]
        if wins == line or abs(expected - line) < edge:
            continue
        side_over = expected > line
        won = (wins > line) == side_over
        by_season[season].append(payout if won else -1.0)
    return {s: np.array(v) for s, v in by_season.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edge", type=float, default=0.03, help="Moneyline edge threshold.")
    ap.add_argument("--futures-edge", type=float, default=0.05)
    ap.add_argument("--wt-edge", type=float, default=0.0, help="Win-total edge in wins.")
    ap.add_argument("--skip-futures", action="store_true")
    args = ap.parse_args()

    cfg = PostgresConfig.from_env()
    results: dict[str, dict[int, np.ndarray]] = defaultdict(dict)

    conn = psycopg.connect(
        dbname=cfg.dbname, user=cfg.user, password=cfg.password,
        host=cfg.host, port=cfg.port, connect_timeout=15,
    )
    for season in ML_SEASONS:
        rows = load_season(conn, cfg.schema, season, 2018)
        results["moneyline"][season] = settle_moneyline(rows, args.edge, False)
        results["moneyline ex-June"][season] = settle_moneyline(rows, args.edge, True)
        print(f"  moneyline {season}: {len(rows)} games")
    for season, arr in settle_win_totals(conn, cfg.schema, args.wt_edge).items():
        results["win_totals"][season] = arr
    conn.close()

    if not args.skip_futures:
        with PostgresHandler(cfg) as pg:
            for market, seasons in FUTURES_SEASONS.items():
                for season in seasons:
                    try:
                        bets = _run_season_backtest(
                            season, market, args.futures_edge, 0.25, 0.05, pg
                        )
                    except Exception as exc:
                        print(f"  {market} {season}: {type(exc).__name__}")
                        continue
                    if bets:
                        results[market][season] = np.array(
                            [(b.decimal_odds - 1.0) if b.actual_win else -1.0
                             for b in bets]
                        )
                print(f"  {market}: {len(results[market])} seasons")

    order = [k for k in (
        "moneyline", "moneyline ex-June", "division", "championship",
        "make_playoffs", "miss_playoffs", "win_totals",
    ) if results.get(k)]
    seasons = sorted({s for k in order for s in results[k]})

    print()
    print("=" * 104)
    print("ROI BY SEASON AND BET TYPE, flat 1u, best price")
    print("=" * 104)
    print()
    head = f"{'season':>7} |" + "".join(f" {k:>18} |" for k in order)
    print(head)
    print("-" * len(head))
    for season in seasons:
        line = f"{season:>7} |"
        for k in order:
            arr = results[k].get(season)
            cell = "-" if arr is None or not len(arr) else f"{arr.mean():+.1%} ({len(arr)})"
            line += f" {cell:>18} |"
        print(line)
    print("-" * len(head))
    line = f"{'POOLED':>7} |"
    for k in order:
        arr = np.concatenate([v for v in results[k].values() if len(v)])
        line += f" {f'{arr.mean():+.2%} ({len(arr)})':>18} |"
    print(line)
    print()

    print("Pooled detail with intervals and season consistency:")
    print(f"{'bet type':>19} | {'bets':>5} | {'win%':>6} | {'ROI':>8} | {'95% CI':>22} | "
          f"{'net':>9} | seasons +")
    print("-" * 104)
    for k in order:
        per = {s: v for s, v in results[k].items() if len(v)}
        arr = np.concatenate(list(per.values()))
        m, lo, hi = boot_ci(arr)
        wins = float((arr > 0).mean())
        pos = sum(1 for v in per.values() if v.mean() > 0)
        # A sample with no winners has zero variance, so every bootstrap resample returns the
        # same mean and the interval collapses to a point. That is an inadequate sample, not a
        # significant result, so it must not be flagged as one.
        degenerate = wins == 0.0 or wins == 1.0 or hi - lo < 1e-9
        if degenerate:
            star = "  <-- degenerate, no variance to resample"
        elif lo > 0 or hi < 0:
            star = "  <-- CI excludes zero"
        else:
            star = ""
        print(f"{k:>19} | {len(arr):5d} | {wins:5.1%} | {m:+7.2%} | "
              f"[{lo:+7.2%}, {hi:+7.2%}] | {arr.sum():+8.1f}u | "
              f"{pos}/{len(per)}{star}")
    print()

    print("Bets per season, which is what limits every futures conclusion:")
    for k in order:
        per = [len(v) for v in results[k].values() if len(v)]
        if per:
            print(f"  {k:>19}: median {int(np.median(per)):4d} bets/season "
                  f"over {len(per)} seasons")


if __name__ == "__main__":
    main()
