"""Ceiling on "predict the close, bet the open".

If a model converges toward the closing line, it can identify opening prices that will move, and
betting those at the open captures the movement. We do measurably predict movement: regressing
logit(true_close) on logit(open) and logit(model) gives the model a coefficient of +0.1156
+/- 0.0062, z = +18.70. So the mechanism is real.

The question is whether it is large enough to pay, and that has a computable ceiling. Replace the
model with an oracle that knows the closing line exactly, then bet at opening prices. No
achievable close-predictor can beat perfect foresight of the close, so this is a strict upper
bound on the whole family of strategies.

Two things are reported for each edge threshold:

  oracle   side chosen by the true closing line, executed at the best opening price
  actual   side chosen by our walk-forward model, executed the same way, for scale

If the oracle is not clearly profitable, the strategy is capped out and no model improvement
rescues it. Uses only seasons with genuine closing lines.

    uv run python scripts/test_beat_the_opener_ceiling.py
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_moneyline import load_finals, walkforward_home_probs
from scripts.backtest_moneyline_lineshop import PANEL_PRIORITY
from src.betting.odds import american_to_decimal, no_vig_two_way
from src.database import PostgresConfig

SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)
PANEL = PANEL_PRIORITY[:5]
BOOT = 4000


def load_market(conn, schema: str, season: int, line_type: str):
    """Per game: median per-book de-vigged home probability, and best price each side."""
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
    fair, best_home, best_away = {}, {}, {}
    for pk, prices in rows.items():
        if len(prices) < 2:
            continue
        fair[pk] = statistics.median(
            no_vig_two_way(h, a, method="proportional")[0] for h, a in prices
        )
        best_home[pk] = max(american_to_decimal(h) for h, _ in prices)
        best_away[pk] = max(american_to_decimal(a) for _, a in prices)
    return fair, best_home, best_away


def settle(signal, open_fair, best_home, best_away, won_home, threshold):
    """Bet the side the signal prefers over the opening fair price, at the best opening price."""
    out = []
    for pk, sig in signal.items():
        if pk not in open_fair or pk not in won_home:
            continue
        edge = sig - open_fair[pk]
        if abs(edge) < threshold:
            continue
        home = edge >= 0
        dec = best_home[pk] if home else best_away[pk]
        won = won_home[pk] == home
        out.append((1.0, (dec - 1.0) if won else -1.0))
    return out


def summarise(label, settled, seed):
    if not settled:
        return f"{label:>8}: no bets"
    staked = sum(s for s, _ in settled)
    profit = sum(p for _, p in settled)
    roi = profit / staked
    n = len(settled)
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(
            sum(settled[i][1] for i in idx) / sum(settled[i][0] for i in idx)
        )
    draws.sort()
    lo, hi = draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]
    wins = sum(1 for _, p in settled if p > 0)
    return (f"{label:>8}: {n:5d} bets  win {wins / n:5.1%}  ROI {roi:+7.2%}  "
            f"95% CI [{lo:+7.2%}, {hi:+7.2%}]  net {profit:+7.1f}u")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edges", default="0.0,0.01,0.02,0.03,0.05")
    args = ap.parse_args()

    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=15,
    )
    oracle_sig, model_sig = {}, {}
    open_fair, best_home, best_away, won_home = {}, {}, {}, {}
    move = []
    for season in SEASONS:
        of, bh, ba = load_market(conn, c.schema, season, "open")
        cf, _, _ = load_market(conn, c.schema, season, "true_close")
        finals = load_finals([season]).set_index("game_pk")
        probs = walkforward_home_probs(season, list(range(2015, season))).set_index(
            "game_pk"
        )["model_prob_home"]
        keys = [k for k in of if k in cf and k in finals.index and k in probs.index]
        for k in keys:
            open_fair[k] = of[k]
            best_home[k], best_away[k] = bh[k], ba[k]
            won_home[k] = bool(finals.loc[k, "home_won"])
            oracle_sig[k] = cf[k]
            model_sig[k] = float(probs.loc[k])
            move.append(abs(cf[k] - of[k]))
    conn.close()

    print(f"Ceiling on betting the open using foresight of the close. "
          f"{len(open_fair):,} games with both prices.")
    print(f"Mean |true_close - open| = {statistics.mean(move):.4f}, "
          f"median {statistics.median(move):.4f}")
    print("Oracle knows the closing line exactly. No achievable predictor beats it.")
    print()
    for thr in (float(x) for x in args.edges.split(",")):
        o = settle(oracle_sig, open_fair, best_home, best_away, won_home, thr)
        m = settle(model_sig, open_fair, best_home, best_away, won_home, thr)
        print(f"edge >= {thr:.2f}")
        print(f"  {summarise('oracle', o, 11)}")
        print(f"  {summarise('model', m, 12)}")
    print()
    print("The oracle result is the maximum any close-predicting strategy can earn against")
    print("opening prices in this market. If it does not clear the hold, converging harder on")
    print("the closing line cannot create value, because the close itself is barely more")
    print("accurate than the open: Brier 0.2389 against 0.2392 over the same games.")


if __name__ == "__main__":
    main()
