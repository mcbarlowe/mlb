"""Real CLV: does the model anticipate the true closing line, and can it beat it?

With genuine closing prices loaded (``line_type='true_close'``, median 4 minutes before first
pitch) two distinct questions separate that the 2.5h proxy could not:

  A. outcome     ~ logit(true_close) + logit(model)
     Does the model know anything the closing price does not? This is the profitability
     question.

  B. logit(true_close) ~ logit(open) + logit(model)
     Does the model predict where the line moves, beyond what the opening price already
     implies? This is the CLV question asked properly. A positive model coefficient here is
     genuine anticipation of market movement, which is exploitable by betting early even if
     the model cannot beat the close. A coefficient of zero means the previously reported
     positive CLV was regression to the mean on a noisy opening price, as suspected.

Both are single-season (2025) because only that season has true closing lines, so treat the
standard errors accordingly.

    uv run python scripts/measure_real_clv.py
"""

from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_moneyline import load_finals, walkforward_home_probs
from scripts.backtest_moneyline_lineshop import PANEL_PRIORITY
from src.betting.odds import no_vig_two_way
from src.database import PostgresConfig

SEASON = 2025
PANEL = PANEL_PRIORITY[:5]
EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ols(x: np.ndarray, y: np.ndarray):
    xd = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(xd, y, rcond=None)
    resid = y - xd @ beta
    s2 = (resid**2).sum() / (len(y) - xd.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(xd.T @ xd)))
    return beta, se


def irls(x: np.ndarray, y: np.ndarray, iters: int = 60):
    xd = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(xd.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(xd @ beta)))
        w = np.clip(p * (1 - p), 1e-10, None)
        step = np.linalg.solve(xd.T @ (xd * w[:, None]), xd.T @ (y - p))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1.0 / (1.0 + np.exp(-(xd @ beta)))
    w = np.clip(p * (1 - p), 1e-10, None)
    cov = np.linalg.inv(xd.T @ (xd * w[:, None]))
    return beta, np.sqrt(np.diag(cov))


def fair_probs(conn, schema: str, line_type: str, season: int) -> dict[int, float]:
    """Median over panel books of each book's own de-vigged home probability."""
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
        buf: dict[int, list[float]] = defaultdict(list)
        for pk, home, away in cur.fetchall():
            buf[int(pk)].append(
                no_vig_two_way(int(home), int(away), method="proportional")[0]
            )
    return {pk: statistics.median(v) for pk, v in buf.items() if len(v) >= 2}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default="2020,2021,2022,2023,2024,2025")
    args = ap.parse_args()
    seasons = [int(s) for s in args.seasons.split(",")]

    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=15,
    )
    chunks: dict[str, list[np.ndarray]] = {k: [] for k in ("po", "pc", "pt", "pm", "y")}
    for season in seasons:
        prices = {
            lt: fair_probs(conn, c.schema, lt, season)
            for lt in ("open", "close", "true_close")
        }
        probs = walkforward_home_probs(season, list(range(2015, season))).set_index(
            "game_pk"
        )["model_prob_home"]
        finals = load_finals([season]).set_index("game_pk")
        keys = sorted(
            set(prices["open"]) & set(prices["close"]) & set(prices["true_close"])
            & set(probs.index) & set(finals.index)
        )
        print(f"{season}: open {len(prices['open']):,} close {len(prices['close']):,} "
              f"true_close {len(prices['true_close']):,} -> {len(keys):,} usable")
        chunks["po"].append(np.array([prices["open"][k] for k in keys]))
        chunks["pc"].append(np.array([prices["close"][k] for k in keys]))
        chunks["pt"].append(np.array([prices["true_close"][k] for k in keys]))
        chunks["pm"].append(np.array([float(probs.loc[k]) for k in keys]))
        chunks["y"].append(
            np.array([bool(finals.loc[k, "home_won"]) for k in keys]).astype(int)
        )
    conn.close()

    po, pc, pt, pm, y = (np.concatenate(chunks[k]) for k in ("po", "pc", "pt", "pm", "y"))
    print(f"\npooled games with all three prices, model, and result: {len(y):,}")

    print("\n" + "=" * 78)
    print("Market accuracy at each observation point")
    print(f"{'point':>12} | {'lead':>12} | {'Brier':>7} | {'slope':>15} | {'intercept':>15}")
    print("-" * 78)
    for name, p, lead in (
        ("open", po, "19-29h"), ("close (proxy)", pc, "~2.5h"), ("true_close", pt, "~4min")
    ):
        beta, se = irls(logit(p).reshape(-1, 1), y)
        print(f"{name:>12} | {lead:>12} | {np.mean((p - y) ** 2):.4f} | "
              f"{beta[1]:+.3f} +/- {se[1]:.3f} | {beta[0]:+.3f} +/- {se[0]:.3f}")
    print(f"{'model':>12} | {'n/a':>12} | {np.mean((pm - y) ** 2):.4f} |")

    print("\n" + "=" * 78)
    print("A. Does the model beat the TRUE close?  outcome ~ logit(true_close) + logit(model)")
    beta, se = irls(np.column_stack([logit(pt), logit(pm)]), y)
    print(f"   logit(true_close) {beta[1]:+.3f} +/- {se[1]:.3f}")
    print(f"   logit(model)      {beta[2]:+.3f} +/- {se[2]:.3f}  "
          f"(z = {beta[2] / se[2]:+.2f})")
    print(f"   -> {'model adds information' if abs(beta[2] / se[2]) > 1.96 else 'no incremental information'}")

    print("\n" + "=" * 78)
    print("B. Does the model predict LINE MOVEMENT?  "
          "logit(true_close) ~ logit(open) + logit(model)")
    beta, se = ols(np.column_stack([logit(po), logit(pm)]), logit(pt))
    z = beta[2] / se[2]
    print(f"   logit(open)  {beta[1]:+.4f} +/- {se[1]:.4f}")
    print(f"   logit(model) {beta[2]:+.4f} +/- {se[2]:.4f}  (z = {z:+.2f})")
    print(f"   -> {'GENUINE anticipation of line movement' if abs(z) > 1.96 else 'no anticipation beyond the opening price'}")

    move = logit(pt) - logit(po)
    print(f"\n   line movement open->true_close: mean {move.mean():+.4f} logits, "
          f"sd {move.std(ddof=1):.4f}")
    print(f"   corr(model - open, movement) = "
          f"{np.corrcoef(logit(pm) - logit(po), move)[0, 1]:+.4f}")

    print("\n" + "=" * 78)
    print("Naive CLV on selected bets, for comparison with the earlier reported figure")
    dis = pm - pt
    for label, ref in (("vs proxy close", pc), ("vs true close", pt)):
        d = pm - po
        sel = np.abs(d) >= 0.05
        back_home = d[sel] >= 0
        take = np.where(back_home, po[sel], 1 - po[sel])
        settle = np.where(back_home, ref[sel], 1 - ref[sel])
        clv = settle - take
        print(f"   bet at open, {label}: {sel.sum():,} bets, "
              f"mean CLV {clv.mean():+.4f}, beat-close {np.mean(clv > 0):.0%}")
    print(f"\n   (selection uses |model - open| >= 5%; {np.abs(dis).mean():.4f} mean "
          f"|model - true_close| for reference)")


if __name__ == "__main__":
    main()
