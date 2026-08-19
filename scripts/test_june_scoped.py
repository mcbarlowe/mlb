"""If we never bet June, how does the model compare to the market, and what is ROI by season?

Two scoping questions, one data load.

1. Brier scoped to bettable months. The pooled model-versus-market comparison includes June, the
   month the model reliably loses in (negative in 12/12 cells across two training windows and six
   seasons). If June is never bet, including it in the accuracy comparison measures a decision
   nobody would make.

2. Flat-bet ROI by season with June removed from the bet list, so the seasonal dispersion is
   visible rather than pooled away.

Scoping a metric is legitimate; it is not the same as creating edge. There is a specific trap:
removing a month changes *both* averages. If June is a month where the market is unusually sharp
rather than one where the model is unusually weak, the model's relative position was never about
model degradation. The decomposition at the end separates those by comparing each forecaster's
June performance against its own non-June baseline.

Paired date-block bootstrap throughout: model and market are scored on identical games, so the
per-game difference is the statistic and day-level clustering is preserved.

    uv run python scripts/test_june_scoped.py
"""

from __future__ import annotations

import argparse
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


def load_season(conn, schema: str, season: int, train_start: int):
    """Per game: model prob, market fair prob, best prices, outcome, date block, month."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT o.game_pk, o.home_ml, o.away_ml,
                   g.game_datetime::date AS game_date,
                   EXTRACT(MONTH FROM g.game_datetime)::int AS month
            FROM {schema}.odds o JOIN {schema}.games g ON g.game_pk = o.game_pk
            WHERE g.season::int = %s AND g.game_type = 'R' AND o.line_type = 'close'
              AND o.bookmaker = ANY(%s)
              AND o.home_ml IS NOT NULL AND o.away_ml IS NOT NULL
              AND g.game_datetime IS NOT NULL
            """,
            (season, list(PANEL)),
        )
        prices: dict[int, list[tuple[int, int]]] = defaultdict(list)
        meta: dict[int, tuple[str, int]] = {}
        for pk, home, away, date, month in cur.fetchall():
            prices[int(pk)].append((int(home), int(away)))
            meta[int(pk)] = (str(date), int(month))

    finals = load_finals([season]).set_index("game_pk")
    probs = walkforward_home_probs(
        season, list(range(train_start, season))
    ).set_index("game_pk")["model_prob_home"]

    rows = []
    for pk, quotes in prices.items():
        if len(quotes) < 2 or pk not in finals.index or pk not in probs.index:
            continue
        fair = statistics.median(
            no_vig_two_way(h, a, method="proportional")[0] for h, a in quotes
        )
        date, month = meta[pk]
        rows.append(
            {
                "model": float(probs.loc[pk]),
                "fair": fair,
                "best_home": max(american_to_decimal(h) for h, _ in quotes),
                "best_away": max(american_to_decimal(a) for _, a in quotes),
                "home_won": bool(finals.loc[pk, "home_won"]),
                "y": 1.0 if bool(finals.loc[pk, "home_won"]) else 0.0,
                "block": f"{season}:{date}",
                "month": month,
            }
        )
    return rows


def brier(p, y) -> float:
    return float(np.mean((np.clip(np.asarray(p), 1e-9, 1 - 1e-9) - np.asarray(y)) ** 2))


def block_ci(diff: np.ndarray, blocks: np.ndarray, seed: int = 17):
    uniq = sorted(set(blocks))
    index = {b: np.where(blocks == b)[0] for b in uniq}
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOT):
        picked = np.concatenate(
            [index[uniq[rng.randrange(len(uniq))]] for _ in range(len(uniq))]
        )
        draws.append(float(diff[picked].mean()))
    draws.sort()
    return float(diff.mean()), draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]


def report_brier(label: str, rows) -> None:
    model = np.array([r["model"] for r in rows])
    market = np.array([r["fair"] for r in rows])
    y = np.array([r["y"] for r in rows])
    blocks = np.array([r["block"] for r in rows])
    diff = (np.clip(model, 1e-9, 1 - 1e-9) - y) ** 2 - (market - y) ** 2
    mean, lo, hi = block_ci(diff, blocks)
    verdict = ("MODEL BEATS PRICE" if hi < 0 else
               "price beats model" if lo > 0 else "indistinguishable")
    print(f"{label:>15} | {len(rows):5d} | {brier(model, y):.6f} | {brier(market, y):.6f} | "
          f"{mean:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {verdict}")


def settle(rows, edge: float):
    out = []
    for r in rows:
        signed = r["model"] - r["fair"]
        if abs(signed) < edge:
            continue
        home_side = signed >= 0
        dec = r["best_home"] if home_side else r["best_away"]
        won = r["home_won"] == home_side
        out.append((dec - 1.0) if won else -1.0)
    return np.array(out)


def roi_ci(arr: np.ndarray, seed: int = 23):
    rng = random.Random(seed)
    n = len(arr)
    draws = sorted(
        float(np.mean([arr[rng.randrange(n)] for _ in range(n)])) for _ in range(BOOT)
    )
    return float(arr.mean()), draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default="2021,2022,2023,2024,2025,2026")
    ap.add_argument("--train-start", type=int, default=2018)
    ap.add_argument("--edge", type=float, default=0.03)
    ap.add_argument("--drop-months", default="6")
    args = ap.parse_args()
    seasons = [int(s) for s in args.seasons.split(",")]
    drop = {int(m) for m in args.drop_months.split(",") if m}
    months = ",".join(str(m) for m in sorted(drop))

    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=15,
    )
    per = {s: load_season(conn, c.schema, s, args.train_start) for s in seasons}
    conn.close()
    allrows = [r for v in per.values() for r in v]
    kept = [r for r in allrows if r["month"] not in drop]
    dropped = [r for r in allrows if r["month"] in drop]

    print(f"=== 1. Model vs market Brier, paired on identical games (drop month {months}) ===")
    print(f"{'scope':>15} | {'games':>5} | {'model':>8} | {'market':>8} | {'gap':>9} | "
          f"{'95% CI':>22} | verdict")
    print("-" * 106)
    report_brier("all months", allrows)
    report_brier("bettable only", kept)
    report_brier("June only", dropped)
    print()
    print("Per season, bettable months only:")
    print("-" * 106)
    for s in seasons:
        rows = [r for r in per[s] if r["month"] not in drop]
        if rows:
            report_brier(str(s), rows)
    print()

    print(f"=== 2. Flat-bet ROI by season, edge >= {args.edge:.0%}, best of 5 books ===")
    print(f"{'season':>7} | {'all months':>26} | {'June removed':>26}")
    print(f"{'':>7} | {'bets':>5} {'ROI':>8} {'net':>10} | {'bets':>5} {'ROI':>8} {'net':>10}")
    print("-" * 68)
    tot_all, tot_keep = [], []
    for s in seasons:
        a = settle(per[s], args.edge)
        k = settle([r for r in per[s] if r["month"] not in drop], args.edge)
        tot_all.append(a)
        tot_keep.append(k)
        print(f"{s:>7} | {len(a):5d} {a.mean():+7.2%} {a.sum():+9.1f}u | "
              f"{len(k):5d} {k.mean():+7.2%} {k.sum():+9.1f}u")
    A = np.concatenate(tot_all)
    K = np.concatenate(tot_keep)
    print("-" * 68)
    print(f"{'POOLED':>7} | {len(A):5d} {A.mean():+7.2%} {A.sum():+9.1f}u | "
          f"{len(K):5d} {K.mean():+7.2%} {K.sum():+9.1f}u")
    print()
    for label, arr in (("all months", A), ("June removed", K)):
        m, lo, hi = roi_ci(arr)
        print(f"  {label:>13}: {len(arr):5d} bets  ROI {m:+.2%}  95% CI [{lo:+.2%}, {hi:+.2%}]"
              f"  {'excludes zero' if lo > 0 else 'includes zero'}")
    wins = sum(1 for x in tot_keep if len(x) and x.mean() > 0)
    print(f"  seasons positive with June removed: {wins}/{len(seasons)}")
    print()

    print("=== 3. Why June looks bad: each forecaster against its own baseline ===")
    md_d = brier([r["model"] for r in dropped], [r["y"] for r in dropped])
    md_k = brier([r["model"] for r in kept], [r["y"] for r in kept])
    mk_d = brier([r["fair"] for r in dropped], [r["y"] for r in dropped])
    mk_k = brier([r["fair"] for r in kept], [r["y"] for r in kept])
    print(f"{'':>10} | {'June':>9} | {'non-June':>9} | {'own delta':>10}")
    print("-" * 46)
    print(f"{'model':>10} | {md_d:.6f} | {md_k:.6f} | {md_d - md_k:+.6f}")
    print(f"{'market':>10} | {mk_d:.6f} | {mk_k:.6f} | {mk_d - mk_k:+.6f}")
    model_share = md_d - md_k
    market_share = mk_k - mk_d
    total = model_share + market_share
    print()
    if total != 0:
        print(f"June gap is {market_share / total:+.0%} market sharpening, "
              f"{model_share / total:+.0%} model degradation.")
    y_all = [r["y"] for r in allrows]
    gap_all = brier([r["model"] for r in allrows], y_all) - brier(
        [r["fair"] for r in allrows], y_all
    )
    print(f"pooled gap all months    {gap_all:+.6f}")
    print(f"pooled gap bettable only {md_k - mk_k:+.6f}")


if __name__ == "__main__":
    main()
