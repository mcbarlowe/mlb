"""Does a market-anchor shrinkage weight survive out-of-sample selection?

Shrinking the model toward the market suppresses the winner's curse term while retaining the
edge term, so mild shrinkage plausibly improves per-bet quality even as bet volume falls. The
in-sample table in ``demo_accuracy_vs_profit.py`` hints at this, with realised edge rising from
+0.74% at lambda 0 to +2.52% at lambda 0.3. That table is not evidence: it is non-monotone, the
best cell is roughly 1.7 standard errors, and picking the best cell is the overfitting error.

This tests it honestly. For each held-out season, lambda is chosen to maximise pooled ROI on
**prior seasons only**, then applied unchanged to the held-out season. Results are pooled across
held-out seasons and bootstrapped.

Three comparisons matter:

  chosen        lambda picked out-of-sample, the honest estimate
  lambda = 0    no shrinkage, the baseline it must beat
  hindsight     lambda picked on the held-out season itself, an upper bound that shows how much
                of any apparent gain is selection rather than signal

A fourth check addresses a confound. Shrinking scales disagreement by roughly (1 - lambda), so
lambda may be nothing more than a higher edge threshold wearing a different hat. For the chosen
lambda, the threshold on the unshrunk model that reproduces the same bet count is found, and
their ROIs compared. If they agree, lambda is threshold tuning, and raising the threshold was
already measured to fail.

    uv run python scripts/test_market_anchor_lambda.py
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_moneyline import load_finals, walkforward_home_probs
from scripts.backtest_moneyline_lineshop import PANEL_PRIORITY, Quote, run
from src.betting.odds import american_to_decimal, no_vig_two_way
from src.database import PostgresConfig

SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)
LAMBDAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
PANEL = PANEL_PRIORITY[:5]
EPS = 1e-6
BOOT = 4000


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def load_quotes_for(conn, schema: str, season: int, line_type: str) -> dict[int, Quote]:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT o.game_pk, o.home_ml, o.away_ml
            FROM {schema}.odds o JOIN {schema}.games g ON o.game_pk = g.game_pk
            WHERE g.season::int = %s AND g.game_type = 'R' AND o.line_type = %s
              AND o.bookmaker = ANY(%s)
              AND o.home_ml IS NOT NULL AND o.away_ml IS NOT NULL
            """,
            (season, line_type, list(PANEL)),
        )
        rows: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for pk, home, away in cur.fetchall():
            rows[int(pk)].append((int(home), int(away)))
    out: dict[int, Quote] = {}
    for pk, prices in rows.items():
        if len(prices) < 2:
            continue
        out[pk] = Quote(
            game_pk=pk,
            fair_home=statistics.median(
                no_vig_two_way(h, a, method="proportional")[0] for h, a in prices
            ),
            best_home_dec=max(american_to_decimal(h) for h, _ in prices),
            best_away_dec=max(american_to_decimal(a) for _, a in prices),
            cons_home_dec=statistics.median([american_to_decimal(h) for h, _ in prices]),
            cons_away_dec=statistics.median([american_to_decimal(a) for _, a in prices]),
            n_books=len(prices),
        )
    return out


def roi(settled) -> float | None:
    staked = sum(s for s, _ in settled)
    return (sum(p for _, p in settled) / staked) if staked else None


def ci(settled, seed=7):
    n = len(settled)
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        st = sum(settled[i][0] for i in idx)
        draws.append(sum(settled[i][1] for i in idx) / st)
    draws.sort()
    return draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edge", type=float, default=0.05)
    ap.add_argument("--line-type", default="true_close",
                    choices=("open", "close", "true_close"))
    args = ap.parse_args()

    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=15,
    )
    data = {}
    for season in SEASONS:
        quotes = load_quotes_for(conn, c.schema, season, args.line_type)
        probs = walkforward_home_probs(season, list(range(2015, season))).set_index(
            "game_pk"
        )["model_prob_home"]
        finals = load_finals([season]).set_index("game_pk")
        keys = [k for k in quotes if k in probs.index and k in finals.index]
        data[season] = ({k: quotes[k] for k in keys},
                        {k: float(probs.loc[k]) for k in keys}, finals, keys)
    conn.close()

    # Cache settled bets for every (season, lambda).
    cache: dict[tuple[int, float], list[tuple[float, float]]] = {}
    for season, (quotes, model, finals, keys) in data.items():
        for lam in LAMBDAS:
            shrunk = {
                k: float(sigmoid((1 - lam) * logit(model[k])
                                 + lam * logit(quotes[k].fair_home)))
                for k in keys
            }
            res, _ = run(quotes, pd.Series(shrunk, name="model_prob_home"), finals,
                         args.edge, "flat", 0.25, 0.05)
            cache[(season, lam)] = res.settled

    print(f"Market-anchor lambda, walk-forward selection. Bet at {args.line_type}, "
          f"edge >= {args.edge:.0%}, flat 1u, best price, panel of 5.")
    print(f"Grid {LAMBDAS}. Lambda chosen on prior seasons only, applied unchanged.")
    print()
    print(f"{'test':>5} | {'selected on':>13} | {'chosen':>6} | {'bets':>5} | "
          f"{'held-out ROI':>12} | {'lam=0 ROI':>10} | {'hindsight':>9}")
    print("-" * 82)

    chosen_all, base_all, hind_all = [], [], []
    picks = []
    for i, test in enumerate(SEASONS):
        prior = SEASONS[:i]
        if len(prior) < 2:
            continue
        scored = {}
        for lam in LAMBDAS:
            pooled = [b for s in prior for b in cache[(s, lam)]]
            r = roi(pooled)
            if r is not None and len(pooled) >= 100:
                scored[lam] = r
        if not scored:
            continue
        lam_star = max(scored, key=scored.get)
        picks.append(lam_star)
        held = cache[(test, lam_star)]
        base = cache[(test, 0.0)]
        hind = max(
            (roi(cache[(test, lam)]) or -9.9 for lam in LAMBDAS),
        )
        chosen_all += held
        base_all += base
        hind_all += cache[(test, max(LAMBDAS, key=lambda L: roi(cache[(test, L)]) or -9.9))]
        print(f"{test:5d} | {f'{prior[0]}-{prior[-1]}':>13} | {lam_star:6.1f} | "
              f"{len(held):5d} | {(roi(held) or 0):+11.2%} | {(roi(base) or 0):+9.2%} | "
              f"{hind:+8.2%}")

    print("-" * 82)
    for label, settled in (("chosen OOS", chosen_all), ("lambda = 0", base_all),
                           ("hindsight", hind_all)):
        if not settled:
            continue
        lo, hi = ci(settled)
        print(f"{label:>12}: {len(settled):5d} bets  ROI {roi(settled):+7.2%}  "
              f"95% CI [{lo:+.2%}, {hi:+.2%}]")

    print()
    print(f"lambda picked: {picks}  "
          f"{'STABLE' if len(set(picks)) == 1 else 'UNSTABLE across seasons'}")

    # Confound check: is the chosen lambda just a higher threshold?
    print()
    print("Confound check. Shrinking scales disagreement by about (1 - lambda), so lambda may")
    print("act as a raised threshold. For each test season, the unshrunk threshold reproducing")
    print("the same bet count is found and its ROI compared.")
    print(f"{'test':>5} | {'lam':>4} | {'bets':>5} | {'shrunk ROI':>10} | "
          f"{'matched thr':>11} | {'unshrunk ROI':>12}")
    print("-" * 66)
    for i, test in enumerate(SEASONS):
        prior = SEASONS[:i]
        if len(prior) < 2 or not picks:
            continue
        idx = i - 2
        if idx >= len(picks):
            break
        lam_star = picks[idx]
        if lam_star == 0.0:
            continue
        quotes, model, finals, keys = data[test]
        target = len(cache[(test, lam_star)])
        best = None
        for thr in np.arange(args.edge, 0.30, 0.002):
            res, _ = run(quotes, pd.Series(model, name="model_prob_home"), finals,
                         float(thr), "flat", 0.25, 0.05)
            if best is None or abs(res.bets - target) < abs(best[1] - target):
                best = (float(thr), res.bets, res.settled)
            if res.bets <= target:
                break
        if best:
            print(f"{test:5d} | {lam_star:4.1f} | {target:5d} | "
                  f"{(roi(cache[(test, lam_star)]) or 0):+9.2%} | {best[0]:11.3f} | "
                  f"{(roi(best[2]) or 0):+11.2%}")


if __name__ == "__main__":
    main()
