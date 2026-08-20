"""Is the recent moneyline profit real, or is it two lucky seasons?

The claim under test is not a cherry-pick. It is that the model improved as training data
accumulated, so recent seasons are genuinely profitable. The pooled 2025-2026 figure with June
removed is +6.59% on 1,378 bets with a 95% interval of [+0.94%, +12.59%], which excludes zero, and
the paired accuracy test on those same bets puts the model nominally ahead of the market. Both
point the same way. That is the strongest evidence in this project and it deserves a real test.

Three tests, in increasing severity.

1. Within-season replication. Split each season chronologically in half. A season that is
   genuinely profitable should be profitable in both halves. Noise usually concentrates in one.
   This is the strongest test available without waiting for new data, because the two halves are
   independent samples from the same season and the same model.

2. Trend. Regress per-season ROI and per-season model-minus-market Brier gap on season index. If
   accumulating data improves the model, the Brier gap should shrink monotonically, not jump in
   one season.

3. Sample adequacy. Report how many bets are needed to resolve the observed edge, and what the
   interval looks like under the selection actually performed, choosing the best two of six
   seasons.

The accuracy instrument is reported alongside ROI everywhere, because ROI keeps only win or loss
while paired Brier keeps the probability, and on this bet volume the difference in power is large.

    uv run python scripts/test_recent_moneyline_profit.py
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
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]


def load_season(conn, schema: str, season: int, train_start: int):
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT o.game_pk, o.home_ml, o.away_ml,
                   g.game_datetime::date AS d,
                   EXTRACT(MONTH FROM g.game_datetime)::int AS mo
            FROM {schema}.odds o JOIN {schema}.games g ON g.game_pk = o.game_pk
            WHERE g.season::int = %s AND g.game_type = 'R' AND o.line_type = 'close'
              AND o.bookmaker = ANY(%s)
              AND o.home_ml IS NOT NULL AND o.away_ml IS NOT NULL
              AND g.game_datetime IS NOT NULL
            """,
            (season, list(PANEL)),
        )
        px: dict[int, list[tuple[int, int]]] = defaultdict(list)
        meta: dict[int, tuple[str, int]] = {}
        for pk, h, a, d, mo in cur.fetchall():
            px[int(pk)].append((int(h), int(a)))
            meta[int(pk)] = (str(d), int(mo))
    finals = load_finals([season]).set_index("game_pk")
    probs = walkforward_home_probs(
        season, list(range(train_start, season))
    ).set_index("game_pk")["model_prob_home"]
    rows = []
    for pk, q in px.items():
        if len(q) < 2 or pk not in finals.index or pk not in probs.index:
            continue
        d, mo = meta[pk]
        rows.append(
            {
                "date": d,
                "month": mo,
                "model": float(probs.loc[pk]),
                "fair": statistics.median(
                    no_vig_two_way(h, a, method="proportional")[0] for h, a in q
                ),
                "bh": max(american_to_decimal(h) for h, _ in q),
                "ba": max(american_to_decimal(a) for _, a in q),
                "home_won": bool(finals.loc[pk, "home_won"]),
                "y": 1.0 if bool(finals.loc[pk, "home_won"]) else 0.0,
            }
        )
    return sorted(rows, key=lambda r: r["date"])


def bets(rows, edge: float):
    out = []
    for r in rows:
        s = r["model"] - r["fair"]
        if abs(s) < edge:
            continue
        home = s >= 0
        dec = r["bh"] if home else r["ba"]
        won = r["home_won"] == home
        out.append(
            {
                "ret": (dec - 1.0) if won else -1.0,
                "model": r["model"],
                "fair": r["fair"],
                "y": r["y"],
                "date": r["date"],
            }
        )
    return out


def roi_ci(arr: np.ndarray, seed: int = 23):
    rng = random.Random(seed)
    n = len(arr)
    d = sorted(
        float(np.mean([arr[rng.randrange(n)] for _ in range(n)])) for _ in range(BOOT)
    )
    return float(arr.mean()), d[int(0.025 * BOOT)], d[int(0.975 * BOOT)]


def brier_gap(sub, seed: int = 31):
    m = np.array([b["model"] for b in sub])
    k = np.array([b["fair"] for b in sub])
    y = np.array([b["y"] for b in sub])
    diff = (np.clip(m, 1e-9, 1 - 1e-9) - y) ** 2 - (k - y) ** 2
    rng = random.Random(seed)
    n = len(diff)
    d = sorted(
        float(np.mean([diff[rng.randrange(n)] for _ in range(n)])) for _ in range(BOOT)
    )
    return float(diff.mean()), d[int(0.025 * BOOT)], d[int(0.975 * BOOT)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edge", type=float, default=0.03)
    ap.add_argument("--drop-june", action="store_true", default=True)
    args = ap.parse_args()

    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=15,
    )
    per = {}
    for s in SEASONS:
        rows = load_season(conn, c.schema, s, 2018)
        if args.drop_june:
            rows = [r for r in rows if r["month"] != 6]
        per[s] = bets(rows, args.edge)
        print(f"  {s}: {len(per[s])} bets")
    conn.close()

    print()
    print("=== 1. Within-season replication, June removed, chronological halves ===")
    print(f"{'season':>7} | {'H1 bets':>7} {'H1 ROI':>8} | {'H2 bets':>7} {'H2 ROI':>8} | "
          f"both same sign?")
    print("-" * 74)
    agree = []
    for s in SEASONS:
        b = per[s]
        if len(b) < 40:
            continue
        mid = len(b) // 2
        h1 = np.array([x["ret"] for x in b[:mid]])
        h2 = np.array([x["ret"] for x in b[mid:]])
        same = (h1.mean() > 0) == (h2.mean() > 0)
        agree.append((s, same, h1.mean(), h2.mean()))
        print(f"{s:>7} | {len(h1):7d} {h1.mean():+7.2%} | {len(h2):7d} {h2.mean():+7.2%} | "
              f"{'yes' if same else 'NO'}")
    print()
    print("  A genuinely profitable season should be profitable in both halves.")

    print()
    print("=== 2. Trend across seasons ===")
    print(f"{'season':>7} | {'bets':>5} | {'ROI':>8} | {'Brier gap':>10} | {'95% CI on gap':>24}")
    print("-" * 70)
    rois, gaps = [], []
    for s in SEASONS:
        b = per[s]
        if len(b) < 40:
            continue
        arr = np.array([x["ret"] for x in b])
        g, glo, ghi = brier_gap(b)
        rois.append(arr.mean())
        gaps.append(g)
        print(f"{s:>7} | {len(b):5d} | {arr.mean():+7.2%} | {g:+10.6f} | "
              f"[{glo:+.6f}, {ghi:+.6f}]")
    idx = np.arange(len(rois), dtype=float)
    print()
    print(f"  corr(season index, ROI)       = {np.corrcoef(idx, rois)[0, 1]:+.3f}")
    print(f"  corr(season index, Brier gap) = {np.corrcoef(idx, gaps)[0, 1]:+.3f}")
    print("  A model improving with data should show ROI rising and the gap falling, steadily.")

    print()
    print("=== 3. Pooled recent window versus the rest ===")
    late = [x for s in (2025, 2026) for x in per[s]]
    early = [x for s in (2021, 2022, 2023, 2024) for x in per[s]]
    for label, sub in (("2021-2024", early), ("2025-2026", late)):
        arr = np.array([x["ret"] for x in sub])
        m, lo, hi = roi_ci(arr)
        g, glo, ghi = brier_gap(sub)
        print(f"  {label}: {len(arr):5d} bets")
        print(f"      ROI       {m:+7.2%}  95% CI [{lo:+7.2%}, {hi:+7.2%}]"
              f"{'  excludes zero' if lo > 0 or hi < 0 else ''}")
        print(f"      Brier gap {g:+.6f}  95% CI [{glo:+.6f}, {ghi:+.6f}]"
              f"{'  excludes zero' if glo > 0 or ghi < 0 else ''}")

    print()
    print("=== 4. Sample adequacy ===")
    arr = np.array([x["ret"] for x in late])
    sd = float(arr.std())
    se = sd / len(arr) ** 0.5
    need = int((sd / (arr.mean() / 1.96)) ** 2) if arr.mean() > 0 else 0
    print(f"  observed edge {arr.mean():+.2%}, per-bet sd {sd:.3f}, SE {se:.2%}, "
          f"z = {arr.mean() / se:+.2f}")
    print(f"  bets needed for the observed edge to clear zero by 2 SE: {need:,}")
    print(f"  bets available in the window: {len(arr):,}  "
          f"({len(arr) / need:.0%} of that)" if need else "")
    print()
    print("  Selection cost: the window is the best 2 of 6 seasons. Under that choice the")
    print("  nominal interval understates the true one, because 15 two-season windows were")
    print("  available and the largest was reported.")


if __name__ == "__main__":
    main()
