"""Does dropping 2015-2017 from the training window improve moneyline ROI?

``docs/win_model_recency_analysis.md`` reports +1.40% average ROI from retraining on 2018 onward
instead of 2015 onward, validated on 2024, 2025 and 2026, and calls the result statistically
significant. Two problems motivate an independent check.

First, that document contradicts itself on leakage. Its appendix states "Training data: Seasons
strictly before test year", while its own caveat section states "Training data includes test
season: SVM and GBM were trained on 2018-test_season, not strictly before. Logistic regression
also does this." The deployed artefact is registered as trained on 2018-2025 while being credited
with +5.64% on 2025, which would be in-sample. Scoring in-sample is a failure this project has
already hit once: 2024 moneyline moved from +0.43% to -4.09% once walk-forward was enforced.

Second, the registry does not support the document. ``mlb-team-strength-win`` logs
``train_seasons = [2021, 2022, 2023, 2024]``, not the 2015-2023 the document attributes to it, and
``mlb-team-strength-win-recency`` v1 carries no run_id, so it has no logged training provenance at
all.

This script re-tests the hypothesis with leakage structurally impossible: ``walkforward_home_probs``
raises if the test season appears in the training seasons. Because both arms score the same games
against the same prices and differ only in training window, the comparison is **paired**, so the
difference is evaluated on per-bet profit differences rather than by comparing two independent
intervals. That is materially more powerful and is the correct test for this claim.

    uv run python scripts/test_training_window_recency.py
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_moneyline import load_finals, walkforward_home_probs
from scripts.backtest_moneyline_lineshop import PANEL_PRIORITY, Quote, run
from src.betting.odds import american_to_decimal, no_vig_two_way
from src.database import PostgresConfig

PANEL = PANEL_PRIORITY[:5]
BOOT = 4000


def load_quotes(conn, schema: str, season: int, line_type: str) -> dict[int, Quote]:
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
    return sum(p for _, p in settled) / staked if staked else None


def paired_ci(diffs: list[float], seed: int = 17) -> tuple[float, float, float]:
    """Bootstrap the mean per-bet profit difference between the two arms."""
    n = len(diffs)
    mean = sum(diffs) / n
    rng = random.Random(seed)
    draws = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOT)
    )
    return mean, draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edge", type=float, default=0.03)
    ap.add_argument("--seasons", default="2021,2022,2023,2024,2025")
    ap.add_argument("--line-type", default="close")
    ap.add_argument("--recency-start", type=int, default=2018)
    ap.add_argument("--baseline-start", type=int, default=2015)
    args = ap.parse_args()

    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=15,
    )

    print(f"Paired walk-forward test, edge >= {args.edge:.0%}, flat 1u, best price, "
          f"bet at {args.line_type}.")
    print(f"baseline trains {args.baseline_start}..N-1, recency trains "
          f"{args.recency_start}..N-1. Leakage impossible: the helper rejects a test season "
          f"present in training.")
    print()
    print(f"{'test':>5} | {'bets':>5} | {'baseline':>9} | {'recency':>9} | {'delta':>8}")
    print("-" * 52)

    all_diffs: list[float] = []
    base_all, rec_all = [], []
    for season in (int(s) for s in args.seasons.split(",")):
        quotes = load_quotes(conn, c.schema, season, args.line_type)
        finals = load_finals([season]).set_index("game_pk")
        arms = {}
        for label, start in (("base", args.baseline_start), ("rec", args.recency_start)):
            train = list(range(start, season))
            if len(train) < 3:
                continue
            probs = walkforward_home_probs(season, train).set_index("game_pk")[
                "model_prob_home"
            ]
            keys = [k for k in quotes if k in probs.index and k in finals.index]
            res, _ = run(
                {k: quotes[k] for k in keys},
                pd.Series({k: float(probs.loc[k]) for k in keys}, name="model_prob_home"),
                finals, args.edge, "flat", 0.25, 0.05,
            )
            arms[label] = res
        if len(arms) < 2:
            continue
        rb, rr = roi(arms["base"].settled), roi(arms["rec"].settled)
        # Pair on bet count where the two arms overlap; profits are per unit stake.
        n = min(len(arms["base"].settled), len(arms["rec"].settled))
        diffs = [
            arms["rec"].settled[i][1] - arms["base"].settled[i][1] for i in range(n)
        ]
        all_diffs += diffs
        base_all += arms["base"].settled
        rec_all += arms["rec"].settled
        print(f"{season:5d} | {n:5d} | {(rb or 0):+8.2%} | {(rr or 0):+8.2%} | "
              f"{((rr or 0) - (rb or 0)) * 100:+7.2f}pp")

    print("-" * 52)
    rb, rr = roi(base_all), roi(rec_all)
    print(f"{'POOL':>5} | {len(all_diffs):5d} | {rb:+8.2%} | {rr:+8.2%} | "
          f"{(rr - rb) * 100:+7.2f}pp")
    mean, lo, hi = paired_ci(all_diffs)
    print()
    print(f"Paired mean per-bet profit difference {mean:+.4f} units, "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  -> recency arm is {'BETTER' if lo > 0 else 'WORSE' if hi < 0 else 'INDISTINGUISHABLE'}"
          f" at the 5% level")
    print()
    print("The document reports +1.40% average and calls it statistically significant, "
          "without a test.")
    conn.close()


if __name__ == "__main__":
    main()
