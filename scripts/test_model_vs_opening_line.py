"""Is the model more accurate than the OPENING line, rather than the closing line?

Every containment result in this project compared the model against the closing price and found the
model adds nothing: fitting the outcome on both gives the market a coefficient of +1.034 +/- 0.113
and the model -0.022 +/- 0.120. But the deployed paper strategy bets at the open, and the open is a
demonstrably weaker number - the oracle test showed that perfect foresight of the close, executed at
opening prices, returns +6.14%. So "contained by the close" does not automatically imply "contained
by the open", and the distinction decides whether the strategy has a basis.

Three tests on the same games, so the comparison is paired throughout:

  1. Paired Brier of model against the opening fair price, and against the closing fair price, on
     the identical game set. If the model beats the open while losing to the close, betting the
     open is the correct play and the earlier conclusion was too broad.
  2. Containment against the open. Fit the outcome on logit(open) and logit(model) together. A
     non-zero model coefficient means the model carries information the opening price lacks, which
     is exactly the claim.
  3. Realised ROI at both entry points on the same bet list, to check that any accuracy advantage
     survives the prices actually available.

The opening snapshot is a fixed-cadence pull at a median of 19-29 hours before first pitch. The
closing snapshot is the targeted one, a median of 4 minutes out.

    uv run python scripts/test_model_vs_opening_line.py
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_moneyline import load_finals, walkforward_home_probs
from scripts.backtest_moneyline_lineshop import PANEL_PRIORITY
from src.betting.odds import american_to_decimal, no_vig_two_way
from src.database import PostgresConfig

PANEL = PANEL_PRIORITY[:5]
BOOT = 4000
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def load_line(conn, schema: str, season: int, line_type: str):
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT o.game_pk, o.home_ml, o.away_ml,
                   EXTRACT(MONTH FROM g.game_datetime)::int AS mo
            FROM {schema}.odds o JOIN {schema}.games g ON g.game_pk = o.game_pk
            WHERE g.season::int = %s AND g.game_type = 'R' AND o.line_type = %s
              AND o.bookmaker = ANY(%s)
              AND o.home_ml IS NOT NULL AND o.away_ml IS NOT NULL
              AND g.game_datetime IS NOT NULL
            """,
            (season, line_type, list(PANEL)),
        )
        px: dict[int, list[tuple[int, int]]] = defaultdict(list)
        month: dict[int, int] = {}
        for pk, h, a, mo in cur.fetchall():
            px[int(pk)].append((int(h), int(a)))
            month[int(pk)] = int(mo)
    out = {}
    for pk, q in px.items():
        if len(q) < 2:
            continue
        out[pk] = {
            "fair": statistics.median(
                no_vig_two_way(h, a, method="proportional")[0] for h, a in q
            ),
            "bh": max(american_to_decimal(h) for h, _ in q),
            "ba": max(american_to_decimal(a) for _, a in q),
            "month": month[pk],
        }
    return out


def boot_ci(arr: np.ndarray, seed: int = 17):
    rng = random.Random(seed)
    n = len(arr)
    d = sorted(
        float(np.mean([arr[rng.randrange(n)] for _ in range(n)])) for _ in range(BOOT)
    )
    return float(arr.mean()), d[int(0.025 * BOOT)], d[int(0.975 * BOOT)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edge", type=float, default=0.03)
    ap.add_argument("--drop-june", action="store_true")
    args = ap.parse_args()

    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=15,
    )
    rows = []
    for season in SEASONS:
        op = load_line(conn, c.schema, season, "open")
        cl = load_line(conn, c.schema, season, "close")
        finals = load_finals([season]).set_index("game_pk")
        probs = walkforward_home_probs(
            season, list(range(2018, season))
        ).set_index("game_pk")["model_prob_home"]
        keys = set(op) & set(cl) & set(finals.index) & set(probs.index)
        for pk in keys:
            if args.drop_june and op[pk]["month"] == 6:
                continue
            rows.append(
                {
                    "season": season,
                    "model": float(probs.loc[pk]),
                    "open": op[pk]["fair"],
                    "close": cl[pk]["fair"],
                    "obh": op[pk]["bh"],
                    "oba": op[pk]["ba"],
                    "cbh": cl[pk]["bh"],
                    "cba": cl[pk]["ba"],
                    "home_won": bool(finals.loc[pk, "home_won"]),
                    "y": 1.0 if bool(finals.loc[pk, "home_won"]) else 0.0,
                }
            )
        print(f"  {season}: {len([r for r in rows if r['season'] == season])} games")
    conn.close()

    y = np.array([r["y"] for r in rows])
    md = np.array([r["model"] for r in rows])
    op_ = np.array([r["open"] for r in rows])
    cl_ = np.array([r["close"] for r in rows])
    june = " (June removed)" if args.drop_june else ""
    print()
    print(f"=== 1. Paired accuracy on {len(rows)} games with both prices{june} ===")
    print(f"{'comparison':>34} | {'gap':>10} | {'95% CI':>24} | verdict")
    print("-" * 88)

    def rep(label, a, b):
        diff = (np.clip(a, 1e-9, 1 - 1e-9) - y) ** 2 - (np.clip(b, 1e-9, 1 - 1e-9) - y) ** 2
        m, lo, hi = boot_ci(diff)
        if hi < 0:
            v = "FIRST is better"
        elif lo > 0:
            v = "second is better"
        else:
            v = "indistinguishable"
        print(f"{label:>34} | {m:+10.6f} | [{lo:+.6f}, {hi:+.6f}] | {v}")

    print(f"{'model Brier':>34} : {float(np.mean((md - y) ** 2)):.6f}")
    print(f"{'opening line Brier':>34} : {float(np.mean((op_ - y) ** 2)):.6f}")
    print(f"{'closing line Brier':>34} : {float(np.mean((cl_ - y) ** 2)):.6f}")
    print()
    rep("model vs OPENING line", md, op_)
    rep("model vs CLOSING line", md, cl_)
    rep("opening vs CLOSING line", op_, cl_)

    print()
    print("=== 2. Containment against the opening line ===")
    from sklearn.linear_model import LogisticRegression

    for label, ref in (("opening", op_), ("closing", cl_)):
        X = np.column_stack([[logit(p) for p in ref], [logit(p) for p in md]])
        fit = LogisticRegression(penalty=None, max_iter=2000).fit(X, y)
        pred = fit.predict_proba(X)[:, 1]
        w = pred * (1 - pred)
        cov = np.linalg.inv((X * w[:, None]).T @ X)
        se = np.sqrt(np.diag(cov))
        b = fit.coef_[0]
        print(f"  fit on logit({label}) + logit(model):")
        print(f"    logit({label}) coef {b[0]:+.3f} +/- {se[0]:.3f}  (z={b[0]/se[0]:+.1f})")
        print(f"    logit(model)  coef {b[1]:+.3f} +/- {se[1]:.3f}  (z={b[1]/se[1]:+.1f})"
              f"{'   <-- model adds information' if b[1] - 1.96 * se[1] > 0 else ''}")

    print()
    print(f"=== 3. Realised ROI at each entry point, edge >= {args.edge:.0%}, flat 1u ===")
    print(f"{'entry':>10} | {'bets':>5} | {'win%':>6} | {'ROI':>8} | {'95% CI':>22}")
    print("-" * 62)
    for label, ref, bh, ba in (
        ("open", "open", "obh", "oba"),
        ("close", "close", "cbh", "cba"),
    ):
        settled = []
        for r in rows:
            s = r["model"] - r[ref]
            if abs(s) < args.edge:
                continue
            home = s >= 0
            dec = r[bh] if home else r[ba]
            settled.append((dec - 1.0) if r["home_won"] == home else -1.0)
        arr = np.array(settled)
        m, lo, hi = boot_ci(arr, seed=23)
        print(f"{label:>10} | {len(arr):5d} | {float((arr > 0).mean()):5.1%} | {m:+7.2%} | "
              f"[{lo:+7.2%}, {hi:+7.2%}]")


if __name__ == "__main__":
    main()
