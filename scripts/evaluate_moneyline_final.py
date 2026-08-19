"""Consolidated moneyline evaluation at a single edge threshold.

Produces every number in one pass so per-season and pooled figures cannot drift apart
across code revisions. Reports two configurations side by side:

  A. shopped   - bet at the last strictly-pre-game snapshot, best price among a fixed
                 panel of books, edge vs the median of per-book de-vigged fair probs.
  B. consensus - identical bet selection, settled at the median price instead.

Confidence intervals are bootstrap percentile over settled bets (resampling bets, not
seasons), so they capture per-bet payout variance but not season-to-season regime change.

    uv run python scripts/evaluate_moneyline_final.py
    uv run python scripts/evaluate_moneyline_final.py --edge 0.03 --staking kelly
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_moneyline import load_finals, walkforward_home_probs
from scripts.backtest_moneyline_lineshop import PANEL_PRIORITY, load_quotes, run

SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)
BOOT = 4000


def subset_forecasts(
    quotes, probs, finals, edge_threshold: float
) -> list[tuple[float, float, bool]]:
    """(model_prob, fair_prob, won) on the backed side, for qualifying bets only.

    Aggregate reliability says nothing about the subset a strategy actually bets. A model
    can be well calibrated overall while being wrong precisely where it most disagrees with
    the market, which is the case that matters.
    """
    out = []
    for pk, q in quotes.items():
        if pk not in probs.index or pk not in finals.index:
            continue
        model_home = float(probs.loc[pk])
        edge = model_home - q.fair_home
        if abs(edge) < edge_threshold:
            continue
        home_won = bool(finals.loc[pk, "home_won"])
        if edge >= 0.0:
            out.append((model_home, q.fair_home, home_won))
        else:
            out.append((1.0 - model_home, 1.0 - q.fair_home, not home_won))
    return out


def ci(settled: list[tuple[float, float]], seed: int = 1234) -> tuple[float, float]:
    """Bootstrap percentile CI on stake-weighted ROI."""
    n = len(settled)
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        staked = sum(settled[i][0] for i in idx)
        profit = sum(settled[i][1] for i in idx)
        draws.append(profit / staked)
    draws.sort()
    return draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edge", type=float, default=0.05)
    ap.add_argument("--accounts", type=int, default=5)
    ap.add_argument("--staking", default="flat", choices=("flat", "kelly"))
    ap.add_argument("--line-type", default="close", choices=("open", "close"))
    ap.add_argument("--devig", default="proportional", choices=("proportional", "shin"))
    args = ap.parse_args()

    panel = PANEL_PRIORITY[: args.accounts]
    print(f"Moneyline evaluation | edge >= {args.edge:.0%} | {args.staking} staking | "
          f"bet at {args.line_type} | {args.accounts} accounts ({', '.join(panel)})")
    print("Model: champion recipe, logistic refit walk-forward (train 2015..N-1)")
    print()

    per_season = []
    all_shopped: list[tuple[float, float]] = []
    pooled_staked_cons = pooled_profit_cons = 0.0
    pooled_wins = 0
    forecasts: list[tuple[float, float, bool]] = []

    for season in SEASONS:
        probs = walkforward_home_probs(season, list(range(2015, season))).set_index(
            "game_pk"
        )["model_prob_home"]
        finals = load_finals([season]).set_index("game_pk")
        quotes = load_quotes(season, panel, args.line_type, args.devig)
        res, n_games = run(
            quotes, probs, finals, args.edge, args.staking, 0.25, 0.05
        )
        if not res.bets:
            continue
        roi = res.profit_best / res.staked
        lo, hi = ci(res.settled, seed=season)
        per_season.append(
            (season, n_games, res.bets, res.wins / res.bets, roi, lo, hi,
             res.profit_best, res.profit_cons / res.staked_cons)
        )
        all_shopped += res.settled
        pooled_staked_cons += res.staked_cons
        pooled_profit_cons += res.profit_cons
        pooled_wins += res.wins
        forecasts += subset_forecasts(quotes, probs, finals, args.edge)

    print(f"{'Season':>6} | {'Games':>5} | {'Bets':>5} | {'%slate':>6} | {'Win%':>6} | "
          f"{'ROI':>8} | {'95% CI':>19} | {'Net':>8}")
    print("-" * 88)
    for s, g, b, w, roi, lo, hi, net, _ in per_season:
        print(f"{s:6d} | {g:5d} | {b:5d} | {b / g:5.0%} | {w:5.1%} | {roi:+7.2%} | "
              f"[{lo:+6.2%}, {hi:+6.2%}] | {net:+7.1f}u")

    bets = sum(p[2] for p in per_season)
    staked = sum(st for st, _ in all_shopped)
    profit = sum(pr for _, pr in all_shopped)
    roi = profit / staked
    lo, hi = ci(all_shopped)
    roi_cons = pooled_profit_cons / pooled_staked_cons
    sd = statistics.stdev(pr for _, pr in all_shopped)
    se = sd / bets**0.5

    print("-" * 88)
    print(f"{'POOLED':>6} | {'':>5} | {bets:5d} | {'':>6} | {pooled_wins / bets:5.1%} | "
          f"{roi:+7.2%} | [{lo:+6.2%}, {hi:+6.2%}] | {profit:+7.1f}u")
    print()
    print(f"Consensus execution, identical selection: {roi_cons:+.2%}  "
          f"-> shopping worth {(roi - roi_cons) * 100:+.2f}pp")
    print(f"Per-bet sd {sd:.3f} units, SE {se:.2%}. "
          f"Seasons profitable: {sum(1 for p in per_season if p[4] > 0)}/{len(per_season)}")
    print(f"Zero inside 95% CI: {'yes' if lo <= 0.0 <= hi else 'no'}. "
          f"n for 2% ROI at 2 SE: {(sd / 0.01) ** 2:,.0f} bets "
          f"({(sd / 0.01) ** 2 / (bets / len(per_season)):,.0f} seasons at this rate)")

    # Forecast accuracy on the backed side of qualifying bets only.
    n = len(forecasts)
    model_p = statistics.mean(f[0] for f in forecasts)
    fair_p = statistics.mean(f[1] for f in forecasts)
    actual = sum(f[2] for f in forecasts) / n
    brier_model = statistics.mean((f[0] - f[2]) ** 2 for f in forecasts)
    brier_fair = statistics.mean((f[1] - f[2]) ** 2 for f in forecasts)
    claimed = model_p - fair_p
    realized = actual - fair_p

    print()
    print(f"Forecast accuracy on the {n} backed sides:")
    print(f"  model says {model_p:.1%} | market fair says {fair_p:.1%} | "
          f"actual {actual:.1%}")
    print(f"  model error {model_p - actual:+.1%} | market error {fair_p - actual:+.1%}")
    print(f"  Brier: model {brier_model:.4f} vs market {brier_fair:.4f} -> "
          f"{'model better' if brier_model < brier_fair else 'MARKET BETTER'}")
    print(f"  claimed edge {claimed:+.1%}, realized {realized:+.1%} "
          f"({realized / claimed:.0%} of claim)")


if __name__ == "__main__":
    main()
