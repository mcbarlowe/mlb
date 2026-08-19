"""Is the full-game totals market biased, and what does it cost?

The survey put full-game totals at a 2.91% shopped hold, so breakeven needs an edge above 1.46%,
making it the cheapest untested market that asks a different question from win probability. The
odds were already in ``mlb.odds_totals`` for 2024 and 2025, so no API spend was required.

Same instrument that killed pitcher strikeout props in one pass:

  - hold measured inside a single (game, line_type, point, book) group, never pooled across books
    or across line points
  - fair probability is the median across books of each book's own de-vigged pair
  - primary line only, meaning the point quoted by the most books
  - pushes dropped rather than assigned
  - per season as well as pooled
  - model-free strategies settled at the best shopped price

Outcomes are total runs by both teams across all innings from ``mlb.linescore``.

    uv run python scripts/test_totals_market_bias.py --line-type close
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

from src.betting.odds import american_to_decimal, no_vig_two_way
from src.database import PostgresConfig

BOOT = 4000


def game_totals(conn, schema: str) -> dict[int, int]:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT game_pk, SUM(runs)
            FROM {schema}.linescore
            GROUP BY game_pk
            HAVING COUNT(DISTINCT team_type) = 2 AND SUM(runs) IS NOT NULL
            """
        )
        return {int(pk): int(r) for pk, r in cur.fetchall()}


def boot_ci(settled, seed=9):
    n = len(settled)
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(sum(settled[i][1] for i in idx) / sum(settled[i][0] for i in idx))
    draws.sort()
    return draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--line-type", default="close")
    args = ap.parse_args()

    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=15,
    )
    actual = game_totals(conn, c.schema)
    with conn.cursor() as cur:
        cur.execute(f"SELECT DISTINCT line_type FROM {c.schema}.odds_totals")
        print("line_type values:", sorted(r[0] for r in cur.fetchall()))
        cur.execute(
            f"""
            SELECT o.game_pk, g.season::int, o.bookmaker, o.total_point::float,
                   o.over_ml, o.under_ml
            FROM {c.schema}.odds_totals o JOIN {c.schema}.games g ON g.game_pk = o.game_pk
            WHERE o.line_type = %s AND g.game_type = 'R'
              AND o.total_point IS NOT NULL
              AND o.over_ml IS NOT NULL AND o.under_ml IS NOT NULL
            """,
            (args.line_type,),
        )
        rows = cur.fetchall()
    conn.close()
    print(f"graded games available: {len(actual):,}")
    print(f"{args.line_type} quotes: {len(rows):,}")

    quotes: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    counts: dict[tuple, dict[float, int]] = defaultdict(lambda: defaultdict(int))
    season_of: dict[int, int] = {}
    per_book: list[float] = []
    for pk, season, _book, point, over, under in rows:
        pk = int(pk)
        quotes[(pk, float(point))].append((int(over), int(under)))
        counts[pk][float(point)] += 1
        season_of[pk] = season
        per_book.append(
            1.0 / american_to_decimal(int(over)) + 1.0 / american_to_decimal(int(under)) - 1.0
        )

    primary = {pk: max(cn, key=cn.get) for pk, cn in counts.items()}
    graded, shopped, depth = [], [], []
    pushes = 0
    for pk, point in primary.items():
        q = quotes[(pk, point)]
        if len(q) < 2 or pk not in actual:
            continue
        if actual[pk] == point:
            pushes += 1
            continue
        best_over = max(american_to_decimal(o) for o, _ in q)
        best_under = max(american_to_decimal(u) for _, u in q)
        shopped.append(1.0 / best_over + 1.0 / best_under - 1.0)
        depth.append(len(q))
        graded.append({
            "season": season_of[pk], "point": point, "n_books": len(q),
            "fair_over": statistics.median(
                no_vig_two_way(o, u, method="proportional")[0] for o, u in q
            ),
            "best_over_dec": best_over, "best_under_dec": best_under,
            "over_hit": 1 if actual[pk] > point else 0,
        })

    pb = statistics.median(per_book)
    sh = statistics.median(shopped)
    print()
    print(f"per-book hold {pb:.2%}, shopped hold {sh:.2%} "
          f"({statistics.median(depth):.0f} books at the primary line), "
          f"breakeven {sh / 2:.2%}")
    print(f"graded non-push games: {len(graded):,} (pushes dropped: {pushes:,})")
    print()

    print(f"{'season':>7} | {'n':>5} | {'implied':>8} | {'actual':>7} | {'bias':>7} | "
          f"{'SE':>6} | {'z':>6}")
    print("-" * 62)
    seasons = sorted({g["season"] for g in graded})
    for season in [*seasons, "pooled"]:
        sub = graded if season == "pooled" else [g for g in graded if g["season"] == season]
        if not sub:
            continue
        imp = statistics.mean(g["fair_over"] for g in sub)
        act = statistics.mean(g["over_hit"] for g in sub)
        se = (0.25 / len(sub)) ** 0.5
        print(f"{season!s:>7} | {len(sub):5d} | {imp:7.2%} | {act:6.2%} | "
              f"{act - imp:+6.2%} | {se:5.2%} | {(act - imp) / se:+6.2f}")

    fo = np.array([g["fair_over"] for g in graded])
    oh = np.array([g["over_hit"] for g in graded])
    print()
    print("calibration by implied-probability decile:")
    order = np.argsort(fo)
    for chunk in np.array_split(order, 10):
        print(f"  {fo[chunk].min():.0%}-{fo[chunk].max():.0%}".rjust(15)
              + f" | n={len(chunk):4d} | implied {fo[chunk].mean():7.2%} | "
              f"actual {oh[chunk].mean():6.2%} | err {oh[chunk].mean() - fo[chunk].mean():+6.2%}")

    print()
    print("model-free strategies at best shopped price:")
    for label, side in (("always over", 1), ("always under", 0)):
        settled = []
        for g in graded:
            dec = g["best_over_dec"] if side else g["best_under_dec"]
            settled.append((1.0, (dec - 1.0) if g["over_hit"] == side else -1.0))
        roi = sum(p for _, p in settled) / sum(s for s, _ in settled)
        lo, hi = boot_ci(settled)
        print(f"  {label:>12}: {len(settled):5d} bets  ROI {roi:+7.2%}  "
              f"95% CI [{lo:+7.2%}, {hi:+7.2%}]")


if __name__ == "__main__":
    main()
