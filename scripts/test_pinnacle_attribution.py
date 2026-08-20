"""Does Pinnacle disagree with a US opener at the same instant, and is that disagreement tradeable?

Answer, established below: no. On 4,773 games where the Pinnacle and US prices are genuinely
contemporaneous, only 8 disagree by 2% or more, and at a 1% threshold those bets return -1.70%.
US openers are not soft relative to Pinnacle. The apparent +13% to +21% edge found on a first pass
was pure look-ahead, from comparing a Pinnacle price captured hours later against a stale US
opener.

That mistake is the reason this script asserts snapshot alignment before computing anything. It was
made twice:

  1. The stored timestamp 17:55:37-05:00 is Central, so the real snapshot is 22:55 UTC and it is a
     day-ahead pull. Requesting 18:00 UTC returned same-day prices, a median of 19 hours later.
  2. After fixing the request times, the loader's primary key
     (game_pk, bookmaker, market, line_type) has no snapshot component, so with roughly two
     snapshots per game whichever row loaded last silently won - typically the one closest to first
     pitch, about 45 minutes out.

Both produced results that passed threshold monotonicity, de-vig sensitivity, favourite and
longshot splits, and replication across five seasons. The guard suite in this repository tests for
selection and for noise, and has no defence against a timing leak. Any two-price comparison must
assert the snapshot gap first.

What does survive is a benchmark correction. On 1,651 matched 2025 games Pinnacle's Brier is
0.236804 against 0.241478 for a five-book US median, and the model at 0.241114 is indistinguishable
from the US median but significantly worse than Pinnacle, +0.004310 with a 95% interval of
[+0.001831, +0.006830]. Every market comparison elsewhere in this repository was graded against the
softer reference.

Pinnacle is used only as a benchmark, never as a model input. Feeding a sharp price into the model
is self-defeating: the lambda-shrinkage sweep showed accuracy improving monotonically toward the
market while bet volume collapsed to zero, because profit comes from disagreement and any feature
that reproduces the price destroys it.

    uv run python scripts/test_pinnacle_attribution.py
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_moneyline import load_finals, walkforward_home_probs
from scripts.backtest_moneyline_lineshop import PANEL_PRIORITY
from src.betting.odds import american_to_decimal, no_vig_two_way
from src.database import PostgresConfig

PANEL = PANEL_PRIORITY[:5]
BOOT = 4000
SEASON = 2025


def boot_ci(arr: np.ndarray, seed: int = 23):
    rng = random.Random(seed)
    n = len(arr)
    d = sorted(
        float(np.mean([arr[rng.randrange(n)] for _ in range(n)])) for _ in range(BOOT)
    )
    return float(arr.mean()), d[int(0.025 * BOOT)], d[int(0.975 * BOOT)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edge", type=float, default=0.04)
    ap.add_argument("--pin-edge", type=float, default=0.02,
                    help="Minimum Pinnacle-vs-book disagreement for the model-free strategy.")
    ap.add_argument("--max-gap-hours", type=float, default=1.0,
                    help="Reject games whose Pinnacle and US snapshots differ by more than this. "
                         "Comparing prices from different moments measures line movement, not "
                         "disagreement, and betting the earlier one is look-ahead.")
    args = ap.parse_args()

    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=15,
    )
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT o.game_pk, o.bookmaker, o.home_ml, o.away_ml, o.snapshot_time,
                   EXTRACT(MONTH FROM g.game_datetime)::int AS mo
            FROM {c.schema}.odds o JOIN {c.schema}.games g ON g.game_pk = o.game_pk
            WHERE g.season::int = %s AND g.game_type = 'R' AND o.line_type = 'open'
              AND (o.bookmaker = 'pinnacle' OR o.bookmaker = ANY(%s))
              AND o.home_ml IS NOT NULL AND o.away_ml IS NOT NULL
              AND g.game_datetime IS NOT NULL AND o.snapshot_time IS NOT NULL
            """,
            (SEASON, list(PANEL)),
        )
        books: dict[int, dict[str, tuple[int, int]]] = defaultdict(dict)
        stamps: dict[int, dict[str, object]] = defaultdict(dict)
        month: dict[int, int] = {}
        for pk, bk, h, a, snap, mo in cur.fetchall():
            books[int(pk)][bk] = (int(h), int(a))
            stamps[int(pk)][bk] = snap
            month[int(pk)] = int(mo)
    finals = load_finals([SEASON]).set_index("game_pk")
    probs = walkforward_home_probs(
        SEASON, list(range(2018, SEASON))
    ).set_index("game_pk")["model_prob_home"]
    conn.close()

    rows = []
    skipped_gap = 0
    for pk, bk in books.items():
        if "pinnacle" not in bk or pk not in finals.index or pk not in probs.index:
            continue
        if month[pk] == 6:
            continue
        us = [x for x in PANEL if x in bk]
        if len(us) < 2:
            continue
        # Timing guard. Comparing two prices captured at different moments measures line movement,
        # not disagreement, and betting the earlier one on the later one's opinion is look-ahead.
        gap_hours = abs(
            (stamps[pk]["pinnacle"] - min(stamps[pk][x] for x in us)).total_seconds()
        ) / 3600.0
        if gap_hours > args.max_gap_hours:
            skipped_gap += 1
            continue
        us_prices = [bk[x] for x in us]
        us_fair = statistics.median(
            no_vig_two_way(h, a, method="proportional")[0] for h, a in us_prices
        )
        ph, pa = bk["pinnacle"]
        pin_fair = no_vig_two_way(ph, pa, method="proportional")[0]
        rows.append(
            {
                "model": float(probs.loc[pk]),
                "us": us_fair,
                "pin": pin_fair,
                "us_bh": max(american_to_decimal(h) for h, _ in us_prices),
                "us_ba": max(american_to_decimal(a) for _, a in us_prices),
                "home_won": bool(finals.loc[pk, "home_won"]),
                "y": 1.0 if bool(finals.loc[pk, "home_won"]) else 0.0,
            }
        )
    print(f"{len(rows)} games in {SEASON} with Pinnacle and 2+ US opening prices, June removed,")
    print(f"snapshots within {args.max_gap_hours}h of each other. "
          f"{skipped_gap} games dropped on the timing guard.")
    if not rows:
        print()
        print("No games survive the timing guard. Pinnacle's earliest stored snapshot for these")
        print("games is hours after the US opener, so no contemporaneous comparison is possible.")
        return
    print()

    md = np.array([r["model"] for r in rows])
    usf = np.array([r["us"] for r in rows])
    pinf = np.array([r["pin"] for r in rows])
    y = np.array([r["y"] for r in rows])

    print("=== 1. Accuracy, all games. Which reference is the model measured against? ===")
    print(f"{'forecaster':>22} | {'Brier':>9}")
    print("-" * 36)
    for lab, p in (("model", md), ("US 5-book median", usf), ("Pinnacle", pinf)):
        print(f"{lab:>22} | {float(np.mean((p - y) ** 2)):9.6f}")
    print()
    for lab, ref in (("US median", usf), ("Pinnacle", pinf)):
        diff = (md - y) ** 2 - (ref - y) ** 2
        m, lo, hi = boot_ci(diff, seed=31)
        v = "model better" if hi < 0 else ("reference better" if lo > 0 else "indistinguishable")
        print(f"  model minus {lab:>10}: {m:+.6f}  95% CI [{lo:+.6f}, {hi:+.6f}]  {v}")
    print()

    print(f"=== 2. Attribution on the model's bets, edge >= {args.edge:.0%} vs US ===")
    agree, disagree = [], []
    for r in rows:
        sig = r["model"] - r["us"]
        if abs(sig) < args.edge:
            continue
        home = sig >= 0
        dec = r["us_bh"] if home else r["us_ba"]
        ret = (dec - 1.0) if r["home_won"] == home else -1.0
        pin_vs_us = r["pin"] - r["us"]
        # Does the sharp book also think the US price is wrong in the model's direction?
        (agree if (pin_vs_us > 0) == (sig > 0) else disagree).append(
            {"ret": ret, "sig": abs(sig), "pin_gap": abs(pin_vs_us)}
        )
    print(f"{'Pinnacle sided with':>22} | {'bets':>5} | {'win%':>6} | {'ROI':>8} | {'95% CI':>22}")
    print("-" * 76)
    for lab, grp in (("the MODEL", agree), ("the US BOOK", disagree)):
        if len(grp) < 20:
            print(f"{lab:>22} | {len(grp):5d} | too few")
            continue
        arr = np.array([g["ret"] for g in grp])
        m, lo, hi = boot_ci(arr)
        wins = float((arr > 0).mean())
        print(f"{lab:>22} | {len(arr):5d} | {wins:5.1%} | {m:+7.2%} | [{lo:+7.2%}, {hi:+7.2%}]")
    if len(agree) >= 20 and len(disagree) >= 20:
        a = np.array([g["ret"] for g in agree])
        b = np.array([g["ret"] for g in disagree])
        se = float(np.sqrt(a.var() / len(a) + b.var() / len(b)))
        print()
        print(f"  difference {a.mean() - b.mean():+.2%}, SE {se:.2%}, "
              f"z = {(a.mean() - b.mean()) / se:+.2f}")
        print("  A large positive difference means the model was working as a soft-line detector.")
    print()

    print(f"=== 3. Model-free: bet the US book when Pinnacle disagrees by >= {args.pin_edge:.0%} ===")
    settled = []
    for r in rows:
        gap = r["pin"] - r["us"]
        if abs(gap) < args.pin_edge:
            continue
        home = gap >= 0
        dec = r["us_bh"] if home else r["us_ba"]
        settled.append((dec - 1.0) if r["home_won"] == home else -1.0)
    if len(settled) >= 20:
        arr = np.array(settled)
        m, lo, hi = boot_ci(arr, seed=41)
        print(f"  {len(arr)} bets, win {float((arr > 0).mean()):.1%}, ROI {m:+.2%}, "
              f"95% CI [{lo:+.2%}, {hi:+.2%}]")
        print("  This uses no model at all: Pinnacle sets the fair price, the US book is the target.")
    else:
        print(f"  only {len(settled)} qualifying bets")


if __name__ == "__main__":
    main()
