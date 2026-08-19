"""Interim analysis of the totals evaluation from its per-game log.

The evaluation writes database rows only at the end, but each per-game stdout line carries a
complete result, so progress is analysable at any point:

    <game_pk> pt=<line> sim_over=<p> mkt_over=<p> actual=<runs>

Reports the Brier comparison with a bootstrap interval on the gap, plus realised edge on the side
the simulation prefers. Realised edge is the decision-relevant quantity and is directly comparable
to the breakeven figure of 1.24% measured for this market, whereas a Brier win does not by itself
clear a hold. Prices are not in the log, so ROI proper waits for the completed run.

    uv run python scripts/analyze_totals_eval_log.py --log data/analysis/totals_eval_600g.log
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

LINE = re.compile(
    r"^(\d+)\s+pt=([\d.]+)\s+sim_over=([\d.]+)\s+mkt_over=([\d.]+)\s+actual=(\d+)\s*$"
)
BOOT = 4000
BREAKEVEN = 0.0124


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default="data/analysis/totals_eval_600g.log")
    ap.add_argument("--edges", default="0.02,0.03,0.05,0.07")
    args = ap.parse_args()

    rows = []
    pushes = 0
    for raw in Path(args.log).read_text().splitlines():
        m = LINE.match(raw.strip())
        if not m:
            continue
        _pk, pt, sim, mkt, actual = m.groups()
        pt, sim, mkt, actual = float(pt), float(sim), float(mkt), int(actual)
        if actual == pt:
            pushes += 1
            continue
        rows.append((sim, mkt, 1 if actual > pt else 0))

    if len(rows) < 30:
        print(f"only {len(rows)} usable games so far")
        return

    sim = np.array([r[0] for r in rows])
    mkt = np.array([r[1] for r in rows])
    y = np.array([r[2] for r in rows])
    n = len(rows)

    sb = (sim - y) ** 2
    mb = (mkt - y) ** 2
    eps = 1e-9
    sl = -(y * np.log(np.clip(sim, eps, 1)) + (1 - y) * np.log(np.clip(1 - sim, eps, 1)))
    ml = -(y * np.log(np.clip(mkt, eps, 1)) + (1 - y) * np.log(np.clip(1 - mkt, eps, 1)))

    print(f"games {n} (pushes dropped {pushes})   over rate {y.mean():.1%}")
    print(f"  sim    Brier {sb.mean():.4f}   log loss {sl.mean():.4f}")
    print(f"  market Brier {mb.mean():.4f}   log loss {ml.mean():.4f}")

    diff = sb - mb
    rng = random.Random(4)
    draws = sorted(
        float(np.mean([diff[rng.randrange(n)] for _ in range(n)])) for _ in range(BOOT)
    )
    lo, hi = draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]
    se = diff.std(ddof=1) / np.sqrt(n)
    print(f"  Brier gap (sim - market) {diff.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"z {diff.mean() / se:+.2f}   (negative means the sim is better)")
    print(f"  sim probability spread: sd {sim.std():.4f}, range "
          f"[{sim.min():.2f}, {sim.max():.2f}]")
    print(f"  market spread:          sd {mkt.std():.4f}, range "
          f"[{mkt.min():.2f}, {mkt.max():.2f}]")

    print()
    # Edge measured against the market probability is invalid for this market. The book sets the
    # line so P(over) is pinned near 50%, so "actual minus market probability" conflates the
    # simulation's skill with the sample's own over/under imbalance: in an under-heavy sample any
    # under-leaning model shows a large apparent edge against 50%. Each side is therefore compared
    # against its own blind baseline, and a directional accident shows up as one side carrying all
    # the gain.
    base_over = float(y.mean())
    print(f"Per-side hit rate against a blind baseline. Sample over rate {base_over:.1%}, "
          f"so blind under wins {1 - base_over:.1%}.")
    print(f"  {'edge':>6} | {'side':>5} | {'bets':>5} | {'won':>7} | {'blind':>7} | "
          f"{'delta':>7}")
    print("  " + "-" * 52)
    for thr in (float(x) for x in args.edges.split(",")):
        for label, mask, hit, blind in (
            ("over", sim - mkt >= thr, y, base_over),
            ("under", mkt - sim >= thr, 1 - y, 1.0 - base_over),
        ):
            k = int(mask.sum())
            if k < 10:
                print(f"  {thr:6.2f} | {label:>5} | {k:5d} |   too few")
                continue
            won = float(hit[mask].mean())
            print(f"  {thr:6.2f} | {label:>5} | {k:5d} | {won:6.1%} | {blind:6.1%} | "
                  f"{won - blind:+6.1%}")
    print()
    print("A Brier win does not by itself clear a hold. ROI at real prices, against the 1.24%")
    print(f"breakeven for this market, requires the completed run. (breakeven {BREAKEVEN:.2%})")


if __name__ == "__main__":
    main()
