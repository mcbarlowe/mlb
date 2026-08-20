"""Threshold sweep for the best configuration: bet the opening line, skip June.

Betting the open rather than the close is worth about 2.4pp on identical games, and skipping June
removes the one month where the market is measurably sharper than usual. Together they give the
first positive full-sample result in this project, +1.16% on 3,629 bets across six seasons at an
edge threshold of 3%.

The deployed paper strategy uses 5%, so this sweeps the range. Two things to read:

  Monotonicity. If the edge is real, ROI should rise as the threshold rises, because the remaining
  bets are the ones where the model disagrees most with a stale price. Non-monotone behaviour is the
  signature that killed the CLV claim earlier in this project.

  Bet flow. A higher threshold means fewer bets, so resolution takes longer. The sweep reports how
  many seasons each threshold needs to prove its own observed edge, which is the practical cost of
  being selective.

Also reports the seasonal breakdown and the containment coefficient restricted to the bet subset at
each threshold, since a timing edge should show the model carrying information the opening price
lacks specifically on the games it chooses.

    uv run python scripts/sweep_open_threshold.py
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


def boot_ci(arr: np.ndarray, seed: int = 23):
    rng = random.Random(seed)
    n = len(arr)
    d = sorted(
        float(np.mean([arr[rng.randrange(n)] for _ in range(n)])) for _ in range(BOOT)
    )
    return float(arr.mean()), d[int(0.025 * BOOT)], d[int(0.975 * BOOT)]


def settle(rows, entry, thr):
    bh, ba = ("obh", "oba") if entry == "open" else ("cbh", "cba")
    out = []
    for r in rows:
        s = r["model"] - r[entry]
        if abs(s) < thr:
            continue
        home = s >= 0
        dec = r[bh] if home else r[ba]
        out.append(
            {
                "ret": (dec - 1.0) if r["home_won"] == home else -1.0,
                "season": r["season"],
                "ref": r[entry],
                "model": r["model"],
                "y": r["y"],
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thresholds", default="0.02,0.03,0.04,0.05,0.06,0.08,0.10")
    ap.add_argument("--focus", type=float, default=0.05)
    args = ap.parse_args()
    thresholds = [float(t) for t in args.thresholds.split(",")]

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
        for pk in set(op) & set(cl) & set(finals.index) & set(probs.index):
            if op[pk]["month"] == 6:
                continue
            rows.append(
                {
                    "season": season,
                    "model": float(probs.loc[pk]),
                    "open": op[pk]["fair"],
                    "close": cl[pk]["fair"],
                    "obh": op[pk]["bh"], "oba": op[pk]["ba"],
                    "cbh": cl[pk]["bh"], "cba": cl[pk]["ba"],
                    "home_won": bool(finals.loc[pk, "home_won"]),
                    "y": 1.0 if bool(finals.loc[pk, "home_won"]) else 0.0,
                }
            )
    conn.close()
    print(f"{len(rows)} games, June removed, both prices present")
    print()

    print("Threshold sweep, flat 1u, best price at the stated entry")
    print(f"{'thr':>5} | {'entry':>6} | {'bets':>5} | {'win%':>6} | {'ROI':>8} | "
          f"{'95% CI':>22} | {'z':>6} | {'seasons to prove':>16}")
    print("-" * 100)
    for thr in thresholds:
        for entry in ("open", "close"):
            b = settle(rows, entry, thr)
            if len(b) < 40:
                print(f"{thr:5.0%} | {entry:>6} | {len(b):5d} | too few")
                continue
            arr = np.array([x["ret"] for x in b])
            m, lo, hi = boot_ci(arr)
            se = arr.std() / len(arr) ** 0.5
            z = m / se
            per_season = len(b) / len(SEASONS)
            need = (arr.std() / (m / 1.96)) ** 2 if m > 0 else float("nan")
            seasons = f"{need / per_season:.1f}" if m > 0 else "n/a"
            star = "  <--" if entry == "open" and abs(thr - args.focus) < 1e-9 else ""
            print(f"{thr:5.0%} | {entry:>6} | {len(arr):5d} | "
                  f"{float((arr > 0).mean()):5.1%} | {m:+7.2%} | "
                  f"[{lo:+7.2%}, {hi:+7.2%}] | {z:+6.2f} | {seasons:>16}{star}")
        print()

    print(f"=== Seasonal breakdown at edge >= {args.focus:.0%}, open entry, June removed ===")
    b = settle(rows, "open", args.focus)
    print(f"{'season':>7} | {'bets':>5} | {'win%':>6} | {'ROI':>8} | {'net':>8}")
    print("-" * 46)
    signs = ""
    for s in SEASONS:
        sub = np.array([x["ret"] for x in b if x["season"] == s])
        if not len(sub):
            continue
        signs += "+" if sub.mean() > 0 else "-"
        print(f"{s:>7} | {len(sub):5d} | {float((sub > 0).mean()):5.1%} | "
              f"{sub.mean():+7.2%} | {sub.sum():+7.2f}u")
    print(f"  sign pattern {signs}  ({signs.count('+')}/{len(signs)} positive)")
    print()

    print("Containment on the bet subset only, fit outcome on logit(open) + logit(model):")
    from sklearn.linear_model import LogisticRegression

    print(f"  {'thr':>5} | {'bets':>5} | {'open coef':>18} | {'model coef':>18}")
    print("  " + "-" * 56)
    for thr in thresholds:
        b = settle(rows, "open", thr)
        if len(b) < 200:
            continue
        y = np.array([x["y"] for x in b])
        if y.min() == y.max():
            continue
        X = np.column_stack([[logit(x["ref"]) for x in b], [logit(x["model"]) for x in b]])
        fit = LogisticRegression(max_iter=2000, C=1e9).fit(X, y)
        p = fit.predict_proba(X)[:, 1]
        w = p * (1 - p)
        se = np.sqrt(np.diag(np.linalg.inv((X * w[:, None]).T @ X)))
        co = fit.coef_[0]
        flag = "  <-- adds info" if co[1] - 1.96 * se[1] > 0 else ""
        print(f"  {thr:5.0%} | {len(b):5d} | {co[0]:+8.3f} +/- {se[0]:.3f} | "
              f"{co[1]:+8.3f} +/- {se[1]:.3f}{flag}")


if __name__ == "__main__":
    main()
