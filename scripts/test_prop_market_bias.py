"""Is the pitcher-strikeout prop market biased, and is any bias tradeable?

This needs no model. If the market's de-vigged implied probability of the over systematically
misses the realised rate by more than half the shopped hold, betting the cheap side is profitable
on its own.

Shopped hold on this market measures 3.33% at the primary line with six books, so breakeven needs
an edge above 1.66%. A fourteen-day pilot showed the over hitting 52.1% against an implied 49.4%,
a 2.7 point lean that clears breakeven on the point estimate but carried z = +0.87 on 259 starts,
which establishes nothing. Three seasons give roughly 10,000 starts and a standard error near
0.5 points.

Method notes, each of which fixes a specific way this project has previously gone wrong:

  - Only the primary line is used, meaning the point quoted by the most books. Alternate-line
    ladders posted by a single book are excluded, since pairing an over at a low rung with an
    under at a high rung is not a placeable position.
  - Fair probability is the median across books of each book's own de-vigged pair, so pricing is
    never mixed across books.
  - Pushes, where strikeouts equal the line, are dropped rather than assigned.
  - Results are reported per season as well as pooled, because a pooled figure hid a defect
    earlier in this project.
  - Any tradeable claim is settled at the best shopped price, not at consensus.

    uv run python scripts/test_prop_market_bias.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.odds import american_to_decimal, no_vig_two_way
from src.database import PostgresConfig

FILES = {
    2023: "data/odds_history/props_2023-03-25_2023-09-29.parquet",
    2024: "data/odds_history/props_2024-03-25_2024-09-29.parquet",
    2025: "data/odds_history/props_2025-03-25_2025-09-29.parquet",
}
BOOT = 4000
BREAKEVEN = 0.0166


def actual_strikeouts(conn, schema: str) -> dict[tuple[str, dt.date], int]:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT p.player_name, DATE(g.game_datetime), p.strikeouts
            FROM {schema}.pitching p JOIN {schema}.games g ON g.game_pk = p.game_pk
            WHERE g.game_type = 'R' AND p.gamesstarted = 1 AND p.strikeouts IS NOT NULL
            """
        )
        return {(str(n).strip(), d): int(so) for n, d, so in cur.fetchall()}


def build(season: int, path: str, actual) -> list[dict]:
    frame = pl.read_parquet(path).drop_nulls(["point", "price", "side", "player"])
    frame = frame.filter(pl.col("market") == "pitcher_strikeouts")

    quotes: dict[tuple, dict[str, int]] = defaultdict(dict)
    counts: dict[tuple, dict[float, int]] = defaultdict(lambda: defaultdict(int))
    when: dict[tuple, dt.date] = {}
    for r in frame.iter_rows(named=True):
        side = str(r["side"]).lower()
        if side not in ("over", "under"):
            continue
        g = (r["event_id"], r["player"])
        quotes[(*g, float(r["point"]), r["bookmaker"])][side] = int(r["price"])
        counts[g][float(r["point"])] += 1
        when[g] = dt.datetime.fromisoformat(r["commence_time"]).date()

    primary = {g: max(c, key=c.get) for g, c in counts.items()}
    rows = []
    for g, point in primary.items():
        eid, player = g
        pairs = [
            v for (e, p, pt, _bk), v in quotes.items()
            if e == eid and p == player and pt == point and "over" in v and "under" in v
        ]
        if len(pairs) < 2:
            continue
        day = when[g]
        so = actual.get((player, day))
        if so is None:
            so = actual.get((player, day - dt.timedelta(days=1)))
        if so is None or so == point:
            continue
        rows.append({
            "season": season,
            "player": player,
            "point": point,
            "n_books": len(pairs),
            "fair_over": statistics.median(
                no_vig_two_way(v["over"], v["under"], method="proportional")[0]
                for v in pairs
            ),
            "best_over_dec": max(american_to_decimal(v["over"]) for v in pairs),
            "best_under_dec": max(american_to_decimal(v["under"]) for v in pairs),
            "over_hit": 1 if so > point else 0,
            "strikeouts": so,
        })
    return rows


def boot_ci(settled, seed=3):
    n = len(settled)
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(
            sum(settled[i][1] for i in idx) / sum(settled[i][0] for i in idx)
        )
    draws.sort()
    return draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=15,
    )
    actual = actual_strikeouts(conn, c.schema)
    conn.close()
    print(f"graded starts in database: {len(actual):,}")

    rows = []
    for season, path in FILES.items():
        if not Path(path).exists():
            print(f"{season}: missing {path}")
            continue
        r = build(season, path, actual)
        rows += r
        print(f"{season}: {len(r):,} graded pitcher-games")
    frame = pl.DataFrame(rows)
    print(f"pooled: {len(frame):,}")
    print()

    print(f"{'season':>7} | {'n':>5} | {'implied':>8} | {'actual':>7} | {'bias':>7} | "
          f"{'SE':>6} | {'z':>6}")
    print("-" * 62)
    for season in [*sorted(FILES), "pooled"]:
        sub = frame if season == "pooled" else frame.filter(pl.col("season") == season)
        if not len(sub):
            continue
        imp = float(sub["fair_over"].mean())
        act = float(sub["over_hit"].mean())
        se = (0.25 / len(sub)) ** 0.5
        print(f"{season!s:>7} | {len(sub):5d} | {imp:7.2%} | {act:6.2%} | "
              f"{act - imp:+6.2%} | {se:5.2%} | {(act - imp) / se:+6.2f}")

    print()
    print(f"Breakeven edge on this market is {BREAKEVEN:.2%} (half the 3.33% shopped hold).")
    print()
    print("Calibration by implied-probability decile, pooled:")
    fo = frame["fair_over"].to_numpy()
    oh = frame["over_hit"].to_numpy()
    order = np.argsort(fo)
    print(f"  {'range':>13} | {'n':>5} | {'implied':>8} | {'actual':>7} | {'err':>7}")
    for chunk in np.array_split(order, 10):
        print(f"  {fo[chunk].min():.0%}-{fo[chunk].max():.0%}".rjust(15)
              + f" | {len(chunk):5d} | {fo[chunk].mean():7.2%} | "
              f"{oh[chunk].mean():6.2%} | {oh[chunk].mean() - fo[chunk].mean():+6.2%}")

    print()
    print("Model-free strategies, settled at the best shopped price:")
    for label, side in (("always over", "over"), ("always under", "under")):
        settled = []
        for r in frame.iter_rows(named=True):
            dec = r["best_over_dec"] if side == "over" else r["best_under_dec"]
            won = r["over_hit"] == (1 if side == "over" else 0)
            settled.append((1.0, (dec - 1.0) if won else -1.0))
        roi = sum(p for _, p in settled) / sum(s for s, _ in settled)
        lo, hi = boot_ci(settled)
        print(f"  {label:>12}: {len(settled):5d} bets  ROI {roi:+7.2%}  "
              f"95% CI [{lo:+7.2%}, {hi:+7.2%}]")


if __name__ == "__main__":
    main()
