"""Does the moneyline model perform differently by month, and is postseason included?

Segmenting by month is a subgroup search, so it invites the error this project has repeatedly
caught: a month will look good by chance. Three guards are applied.

  1. Bonferroni correction for the number of months tested.
  2. A permutation test on the spread across months, shuffling bets between months while holding
     each month's bet count fixed. This asks whether the observed month-to-month dispersion
     exceeds what identical per-bet odds produce by chance, the same instrument that showed the
     season-to-season spread was noise at p = 0.211.
  3. A per-season consistency check for any month that survives. A real calendar effect should
     recur across seasons; noise will not.

There is a plausible prior in both directions, which is why it is worth measuring rather than
assuming. Early season carries stale carry-over Elo and small samples. Late season carries
tanking, call-ups, rest days and clinched teams, none of which the features represent.

Also reports the postseason position: every backtest in this project filters ``game_type = 'R'``,
so postseason games have never been included.

    uv run python scripts/analyze_monthly_performance.py
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
MONTHS = {3: "Mar/Apr", 4: "Mar/Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep/Oct",
          10: "Sep/Oct"}


def postseason_position(conn, schema: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT g.game_type, COUNT(*) AS games,
                   COUNT(DISTINCT o.game_pk) AS with_odds
            FROM {schema}.games g
            LEFT JOIN {schema}.odds o ON o.game_pk = g.game_pk
            WHERE g.season::int BETWEEN 2020 AND 2026
            GROUP BY g.game_type ORDER BY 2 DESC
            """
        )
        print("Games by type, 2020-2026, and how many carry moneyline odds:")
        for gt, games, with_odds in cur.fetchall():
            label = {"R": "regular season", "P": "postseason", "S": "spring training",
                     "F": "wild card", "D": "division series", "L": "league series",
                     "W": "world series", "A": "all-star", "E": "exhibition"}.get(
                         str(gt), str(gt))
            print(f"  {gt!s:3s} {label:16s} {games:7d} games, {with_odds:6d} with odds")
    print()
    print("Every backtest in this project filters game_type = 'R', so postseason is excluded.")
    print()


def load_bets(conn, schema: str, season: int, edge: float, train_start: int):
    """Settled bets for one season, tagged with calendar month."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT o.game_pk, o.home_ml, o.away_ml,
                   EXTRACT(MONTH FROM g.game_datetime)::int AS month
            FROM {schema}.odds o JOIN {schema}.games g ON g.game_pk = o.game_pk
            WHERE g.season::int = %s AND g.game_type = 'R' AND o.line_type = 'close'
              AND o.bookmaker = ANY(%s)
              AND o.home_ml IS NOT NULL AND o.away_ml IS NOT NULL
              AND g.game_datetime IS NOT NULL
            """,
            (season, list(PANEL)),
        )
        rows: dict[int, list[tuple[int, int]]] = defaultdict(list)
        month: dict[int, int] = {}
        for pk, home, away, mo in cur.fetchall():
            rows[int(pk)].append((int(home), int(away)))
            month[int(pk)] = int(mo)

    finals = load_finals([season]).set_index("game_pk")
    probs = walkforward_home_probs(season, list(range(train_start, season))).set_index(
        "game_pk"
    )["model_prob_home"]

    out = []
    for pk, prices in rows.items():
        if len(prices) < 2 or pk not in finals.index or pk not in probs.index:
            continue
        fair = statistics.median(
            no_vig_two_way(h, a, method="proportional")[0] for h, a in prices
        )
        model = float(probs.loc[pk])
        signed = model - fair
        if abs(signed) < edge:
            continue
        home_side = signed >= 0
        dec = (
            max(american_to_decimal(h) for h, _ in prices) if home_side
            else max(american_to_decimal(a) for _, a in prices)
        )
        won = bool(finals.loc[pk, "home_won"]) == home_side
        out.append((month[pk], 1.0, (dec - 1.0) if won else -1.0, int(won)))
    return out


def roi(bets) -> float:
    return sum(b[2] for b in bets) / sum(b[1] for b in bets)


def ci(bets, seed=31):
    n = len(bets)
    rng = random.Random(seed)
    draws = sorted(
        sum(bets[rng.randrange(n)][2] for _ in range(n)) / n for _ in range(BOOT)
    )
    return draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edge", type=float, default=0.03)
    ap.add_argument("--seasons", default="2021,2022,2023,2024,2025,2026")
    ap.add_argument("--train-start", type=int, default=2018)
    args = ap.parse_args()

    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=15,
    )
    postseason_position(conn, c.schema)

    seasons = [int(s) for s in args.seasons.split(",")]
    per_season: dict[int, list] = {}
    for season in seasons:
        per_season[season] = load_bets(conn, c.schema, season, args.edge, args.train_start)
        print(f"{season}: {len(per_season[season])} qualifying bets")
    conn.close()

    allbets = [b for v in per_season.values() for b in v]
    pooled = roi(allbets)
    print()
    print(f"Pooled {len(allbets)} bets, ROI {pooled:+.2%}, "
          f"train {args.train_start}..N-1, edge >= {args.edge:.0%}")
    print()

    by_month: dict[str, list] = defaultdict(list)
    for b in allbets:
        by_month[MONTHS.get(b[0], str(b[0]))].append(b)
    order = ["Mar/Apr", "May", "Jun", "Jul", "Aug", "Sep/Oct"]
    order = [m for m in order if m in by_month]
    k = len(order)
    z_bonf = {1: 1.96, 2: 2.24, 3: 2.39, 4: 2.50, 5: 2.58, 6: 2.64}.get(k, 2.64)

    print(f"{'month':>8} | {'bets':>5} | {'win%':>6} | {'ROI':>8} | {'95% CI':>20} | "
          f"{'z':>6} | {'survives':>8}")
    print("-" * 78)
    for m in order:
        bets = by_month[m]
        r = roi(bets)
        lo, hi = ci(bets)
        sd = statistics.stdev(b[2] for b in bets)
        se = sd / len(bets) ** 0.5
        wins = sum(b[3] for b in bets) / len(bets)
        z = r / se
        print(f"{m:>8} | {len(bets):5d} | {wins:5.1%} | {r:+7.2%} | "
              f"[{lo:+6.2%}, {hi:+6.2%}] | {z:+6.2f} | "
              f"{'YES' if abs(z) > z_bonf else 'no':>8}")
    print(f"\nBonferroni threshold for {k} months: |z| > {z_bonf:.2f}")

    # Permutation test on the spread across months.
    obs = [roi(by_month[m]) for m in order]
    obs_range = max(obs) - min(obs)
    sizes = [len(by_month[m]) for m in order]
    rng = random.Random(99)
    ge = 0
    for _ in range(BOOT):
        pool = allbets[:]
        rng.shuffle(pool)
        cut, rois = 0, []
        for n in sizes:
            chunk = pool[cut:cut + n]
            cut += n
            rois.append(sum(b[2] for b in chunk) / n)
        if (max(rois) - min(rois)) >= obs_range:
            ge += 1
    p = (ge + 1) / (BOOT + 1)
    print()
    print(f"Permutation test on month-to-month spread: observed range {obs_range:.2%}, "
          f"p = {p:.3f}")
    if p > 0.05:
        print("  The spread is consistent with chance. No month effect to exploit.")
    else:
        print("  The spread exceeds chance; check per-season consistency below.")

    # Per-season consistency for the best and worst month.
    best = order[int(np.argmax(obs))]
    worst = order[int(np.argmin(obs))]
    print()
    print("Per-season consistency (a real calendar effect should recur):")
    print(f"{'season':>7} | {best:>10} | {worst:>10}")
    print("-" * 34)
    for season in seasons:
        sb = [b for b in per_season[season] if MONTHS.get(b[0]) == best]
        sw = [b for b in per_season[season] if MONTHS.get(b[0]) == worst]
        fb = f"{roi(sb):+.2%} ({len(sb)})" if sb else "n/a"
        fw = f"{roi(sw):+.2%} ({len(sw)})" if sw else "n/a"
        print(f"{season:7d} | {fb:>10} | {fw:>10}")


if __name__ == "__main__":
    main()
