"""Interim analysis of the totals evaluation from its per-game log.

The evaluation writes database rows only at the end, but each per-game stdout line carries a
complete result, so progress is analysable at any point:

    <game_pk> pt=<line> sim_over=<p> mkt_over=<p> actual=<runs>

Reports model and market Brier scores and log losses, with a bootstrap interval
on the Brier-score gap and probability-spread diagnostics.

    uv run python scripts/analyze_totals_eval_log.py --log data/analysis/totals_eval_600g.log
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np

LINE = re.compile(
    r"^(\d+)\s+pt=([\d.]+)\s+sim_over=([\d.]+)\s+mkt_over=([\d.]+)\s+actual=(\d+)\s*$"
)
BOOT = 4000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default="data/analysis/totals_eval_600g.log")
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


if __name__ == "__main__":
    main()
