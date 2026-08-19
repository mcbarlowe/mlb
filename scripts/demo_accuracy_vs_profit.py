"""Why a more accurate standalone forecast need not bet better.

Profit on a bet is driven by ``p_true - p_price``. The model does not appear in that
expression; it only decides which bets get placed. Writing
``p_model = p_true + e_model`` and ``p_market = p_true + e_market`` gives

    disagreement = p_model - p_market = e_model - e_market

so selecting on large disagreement selects a mixture of "the market is wrong" (real edge) and
"the model is wrong" (winner's curse), in proportion to the two error variances.

This script makes the consequence concrete. The model is shrunk toward the market in logit
space:

    p_lambda = sigmoid((1 - lambda) * logit(p_model) + lambda * logit(p_market))

Because the market is the more accurate forecaster here, increasing lambda **strictly improves
accuracy**. If better accuracy implied better betting, ROI would improve with lambda. Instead
the disagreement that generates bets shrinks by construction, so volume collapses. The
interesting question is whether per-bet quality improves enough to offset that, which is
measured rather than assumed.

    uv run python scripts/demo_accuracy_vs_profit.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_moneyline import load_finals, walkforward_home_probs
from scripts.backtest_moneyline_lineshop import PANEL_PRIORITY, Quote, run
from src.betting.odds import american_to_decimal, no_vig_two_way
from src.database import PostgresConfig

SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)
PANEL = PANEL_PRIORITY[:5]
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def load_quotes_for(conn, schema: str, season: int, line_type: str) -> dict[int, Quote]:
    """Same construction as the line-shop backtest: per-book de-vig, median fair, best price."""
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
        fair = statistics.median(
            no_vig_two_way(h, a, method="proportional")[0] for h, a in prices
        )
        home_decs = [american_to_decimal(h) for h, _ in prices]
        away_decs = [american_to_decimal(a) for _, a in prices]
        out[pk] = Quote(
            game_pk=pk, fair_home=fair,
            best_home_dec=max(home_decs), best_away_dec=max(away_decs),
            cons_home_dec=statistics.median(home_decs),
            cons_away_dec=statistics.median(away_decs),
            n_books=len(prices),
        )
    return out


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

    per_season = {}
    for season in SEASONS:
        quotes = load_quotes_for(conn, c.schema, season, args.line_type)
        probs = walkforward_home_probs(season, list(range(2015, season))).set_index(
            "game_pk"
        )["model_prob_home"]
        finals = load_finals([season]).set_index("game_pk")
        keys = [k for k in quotes if k in probs.index and k in finals.index]
        per_season[season] = (
            quotes,
            {k: float(probs.loc[k]) for k in keys},
            finals,
            keys,
        )
    conn.close()

    print(f"Shrinking the model toward the market. Bet at {args.line_type}, "
          f"edge >= {args.edge:.0%}, best price, panel of 5.")
    print("lambda 0 is the raw model; lambda 1 is the market itself.")
    print()
    print(f"{'lambda':>6} | {'Brier':>7} | {'better?':>7} | {'bets':>5} | {'ROI':>8} | "
          f"{'realised edge':>13} | {'net':>8}")
    print("-" * 76)

    baseline_brier = None
    for lam in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0):
        settled: list[tuple[float, float]] = []
        bets = wins = 0
        sq_err: list[float] = []
        edges: list[float] = []
        for season, (quotes, model, finals, keys) in per_season.items():
            import pandas as pd

            shrunk = {}
            for k in keys:
                z = (1 - lam) * logit(np.array([model[k]]))[0] + lam * logit(
                    np.array([quotes[k].fair_home])
                )[0]
                shrunk[k] = float(sigmoid(z))
                y = 1.0 if bool(finals.loc[k, "home_won"]) else 0.0
                sq_err.append((shrunk[k] - y) ** 2)
            series = pd.Series(shrunk, name="model_prob_home")
            res, _ = run(
                {k: quotes[k] for k in keys}, series, finals, args.edge, "flat", 0.25, 0.05
            )
            settled += res.settled
            bets += res.bets
            wins += res.wins
            # realised edge on the backed side, market fair versus outcome
            for k in keys:
                d = shrunk[k] - quotes[k].fair_home
                if abs(d) < args.edge:
                    continue
                home = d >= 0
                fair = quotes[k].fair_home if home else 1 - quotes[k].fair_home
                won = bool(finals.loc[k, "home_won"]) == home
                edges.append(float(won) - fair)

        brier = float(np.mean(sq_err))
        if baseline_brier is None:
            baseline_brier = brier
        better = "--" if lam == 0.0 else ("yes" if brier < baseline_brier else "no")
        if bets:
            staked = sum(s for s, _ in settled)
            profit = sum(p for _, p in settled)
            roi = f"{profit / staked:+7.2%}"
            net = f"{profit:+7.1f}u"
            re_ = f"{statistics.mean(edges):+12.2%}"
        else:
            roi, net, re_ = "n/a", "0.0u", "n/a"
        print(f"{lam:6.1f} | {brier:.5f} | {better:>7} | {bets:5d} | {roi:>8} | "
              f"{re_:>13} | {net:>8}")

    print()
    print("Accuracy improves monotonically in lambda because the market is the better")
    print("forecaster. Bet volume collapses because disagreement is what generates bets and")
    print("shrinking toward the market removes it. At lambda 1 the model IS the market: zero")
    print("disagreement, zero bets, zero profit, and the best accuracy in the table.")
    print()
    print("The lesson is not that accuracy is bad. It is that accuracy gained by reproducing")
    print("the market's own information cannot pay, because the disagreement it removes is the")
    print("only thing that places bets. Only accuracy gained from information the price lacks")
    print("both improves the forecast and preserves a reason to bet.")


if __name__ == "__main__":
    main()
