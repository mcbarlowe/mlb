"""Pool the line-shopping backtest across seasons and sweep the number of accounts.

Answers two questions the per-season output cannot:
  1. What is the pooled ROI, where seasons are weighted by bet count rather than averaged?
  2. How does execution value scale with the number of books held (best-of-K)?

    uv run python scripts/backtest_moneyline_lineshop_pooled.py
    uv run python scripts/backtest_moneyline_lineshop_pooled.py --edges 0.05 --max-k 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_moneyline import load_finals, walkforward_home_probs
from scripts.backtest_moneyline_lineshop import PANEL_PRIORITY, load_quotes, run

SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=",".join(str(s) for s in SEASONS))
    ap.add_argument("--edges", default="0.0,0.02,0.03,0.05")
    ap.add_argument("--max-k", type=int, default=len(PANEL_PRIORITY))
    ap.add_argument("--line-type", default="close", choices=("open", "close"))
    ap.add_argument("--staking", default="flat", choices=("flat", "kelly"))
    ap.add_argument("--devig", default="proportional", choices=("proportional", "shin"))
    args = ap.parse_args()

    seasons = [int(s) for s in args.seasons.split(",")]
    thresholds = [float(x) for x in args.edges.split(",")]
    ks = list(range(1, args.max_k + 1))

    # Model probabilities and results are independent of the book panel, so fetch once.
    print("Fitting walk-forward models...", flush=True)
    probs, finals = {}, {}
    for season in seasons:
        train = list(range(2015, season))
        probs[season] = walkforward_home_probs(season, train).set_index("game_pk")[
            "model_prob_home"
        ]
        finals[season] = load_finals([season]).set_index("game_pk")

    print()
    print(f"Pooled line-shop backtest | bet at {args.line_type} | {args.staking} staking "
          f"| de-vig {args.devig}")
    print(f"Seasons {min(seasons)}-{max(seasons)} | accounts ranked "
          f"{', '.join(PANEL_PRIORITY[: args.max_k])}")
    print()

    for threshold in thresholds:
        print(f"--- edge >= {threshold:.2f} ---")
        print(f"{'K':>2} | {'bets':>6} | {'win%':>6} | {'ROI best':>9} | "
              f"{'ROI cons':>9} | {'shop gain':>10} | {'price impr':>10}")
        print("-" * 74)
        for k in ks:
            panel = PANEL_PRIORITY[:k]
            bets = wins = 0
            staked = staked_cons = 0.0
            profit_best = profit_cons = 0.0
            impr = 0.0
            for season in seasons:
                quotes = load_quotes(season, panel, args.line_type, args.devig)
                res, _ = run(
                    quotes, probs[season], finals[season], threshold,
                    args.staking, 0.25, 0.05,
                )
                bets += res.bets
                wins += res.wins
                staked += res.staked
                staked_cons += res.staked_cons
                profit_best += res.profit_best
                profit_cons += res.profit_cons
                impr += res.improvement_sum
            if not bets:
                print(f"{k:2d} | {'no bets':>6}")
                continue
            roi_best = profit_best / staked
            roi_cons = profit_cons / staked_cons
            print(f"{k:2d} | {bets:6d} | {wins / bets:5.1%} | {roi_best:+8.2%} | "
                  f"{roi_cons:+8.2%} | {(roi_best - roi_cons) * 100:+7.2f}pp | "
                  f"{impr / bets:+9.3%}")
        print()


if __name__ == "__main__":
    main()
