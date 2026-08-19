"""What do player prop markets cost, and how much does shopping recover?

The cheapest test of whether a market is worth modelling is its hold. A market can be visibly
inefficient and still untradeable if the vig exceeds the inefficiency, which is exactly what
happened with first-five totals: a real 1.67% under lean against a 3.60% shopped hold.

Hold is computed strictly within a single (event, market, player, book, line point) group. Prop
lines differ across books and across players, so pooling across points pairs an over at one line
with an under at another, which is not a placeable position and fabricates a near-zero hold. This
project has already made that mistake once on totals.

Reported per market:

  per-book hold        median overround on a single book's own two-sided price
  shopped hold         best over and best under among books quoting the SAME point
  shopping gain        the difference, comparable to +2.7pp measured on moneyline
  point dispersion     how often books disagree on the line itself, which is a second and
                       usually larger source of shopping value in prop markets

Benchmarks: full-game moneyline is 4.10% per book and about 1.40% shopped with five accounts.
First-five totals is 6.52% per book and 3.60% shopped.

    uv run python scripts/analyze_prop_hold.py --frame data/odds_history/props_2025-06-01_2025-06-14.parquet
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.odds import american_to_decimal, american_to_prob


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame", required=True)
    args = ap.parse_args()

    frame = pl.read_parquet(args.frame).drop_nulls(["point", "price", "side", "player"])
    print(f"{len(frame):,} prop quotes, {frame['event_id'].n_unique()} events")
    print()

    # Pair over and under within a single book at a single point.
    pairs: dict[tuple, dict[str, int]] = defaultdict(dict)
    for row in frame.iter_rows(named=True):
        side = str(row["side"]).lower()
        if side not in ("over", "under"):
            continue
        key = (row["event_id"], row["market"], row["player"], row["bookmaker"],
               float(row["point"]))
        pairs[key][side] = int(row["price"])

    per_book: dict[str, list[float]] = defaultdict(list)
    by_point: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    points_per_player: dict[tuple, set[float]] = defaultdict(set)
    for (eid, market, player, _book, point), sides in pairs.items():
        if "over" not in sides or "under" not in sides:
            continue
        over, under = sides["over"], sides["under"]
        per_book[market].append(
            american_to_prob(over) + american_to_prob(under) - 1.0
        )
        by_point[(eid, market, player, point)].append((over, under))
        points_per_player[(eid, market, player)].add(point)

    shopped: dict[str, list[float]] = defaultdict(list)
    depth: dict[str, list[int]] = defaultdict(list)
    for (_eid, market, _player, _point), quotes in by_point.items():
        if len(quotes) < 2:
            continue
        best_over = max(american_to_decimal(o) for o, _ in quotes)
        best_under = max(american_to_decimal(u) for _, u in quotes)
        shopped[market].append(1.0 / best_over + 1.0 / best_under - 1.0)
        depth[market].append(len(quotes))

    spread: dict[str, list[int]] = defaultdict(list)
    for (_eid, market, _player), pts in points_per_player.items():
        spread[market].append(len(pts))

    print(f"{'market':>20} | {'pairs':>6} | {'per-book':>9} | {'shopped':>8} | "
          f"{'gain':>7} | {'books/pt':>8} | {'distinct pts':>12}")
    print("-" * 92)
    for market in sorted(per_book):
        pb = statistics.median(per_book[market])
        sh = statistics.median(shopped[market]) if shopped[market] else float("nan")
        dp = statistics.median(depth[market]) if depth[market] else 0
        sp = statistics.mean(spread[market]) if spread[market] else 0
        print(f"{market:>20} | {len(per_book[market]):6d} | {pb:8.2%} | {sh:7.2%} | "
              f"{(pb - sh) * 100:+6.2f}pp | {dp:8.0f} | {sp:12.2f}")

    print()
    print(f"{'reference':>20} | {'':>6} | {'per-book':>9} | {'shopped':>8}")
    print(f"{'full-game moneyline':>20} | {'':>6} | {0.041:8.2%} | {0.014:7.2%}")
    print(f"{'first-five totals':>20} | {'':>6} | {0.0652:8.2%} | {0.0360:7.2%}")
    print()
    print("Breakeven on a two-way market needs edge above half the shopped hold.")
    for market in sorted(per_book):
        sh = statistics.median(shopped[market]) if shopped[market] else float("nan")
        print(f"  {market:>20}: needs edge > {sh / 2:.2%}")


if __name__ == "__main__":
    main()
