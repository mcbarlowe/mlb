"""Is the first-five totals market as efficient as the full-game moneyline?

Full-game moneyline prices at calibration slope 1.015 +/- 0.048 with intercept 0.003, and
thirteen candidate features failed to beat it. The question is whether a thinner market prices
as well. ``mlb.f5_odds`` carries first-five totals for 2025 with both open and close, so this
also measures whether that market improves between the two.

Totals need care that moneyline does not: books post different total points for the same game,
so de-vigged probabilities cannot be pooled across books quoting different lines. Games are
therefore grouped by (game_pk, total_point) and the point with the most books quoting it is
taken as that game's market line, requiring at least two books at that point.

Outcome is first-five runs by both teams from ``mlb.linescore``, restricted to games with five
complete innings. Exact pushes are dropped rather than assigned.

    uv run python scripts/test_f5_totals_efficiency.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.odds import no_vig_two_way
from src.database import PostgresConfig

EPS = 1e-6
BINS = 8


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


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


def murphy(p: np.ndarray, y: np.ndarray, bins: int = BINS):
    base = y.mean()
    chunks = np.array_split(np.argsort(p), bins)
    rel = res = 0.0
    rows = []
    for idx in chunks:
        if len(idx) == 0:
            continue
        pk, ok = p[idx].mean(), y[idx].mean()
        rel += len(idx) * (pk - ok) ** 2
        res += len(idx) * (ok - base) ** 2
        rows.append((len(idx), p[idx].min(), p[idx].max(), pk, ok))
    return rel / len(p), res / len(p), base * (1 - base), rows


def f5_runs(conn, schema: str) -> dict[int, int]:
    """Total runs through five innings, only for games with five complete innings."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT game_pk, SUM(runs) AS runs
            FROM {schema}.linescore
            WHERE inning <= 5
            GROUP BY game_pk
            HAVING COUNT(DISTINCT inning) = 5
               AND COUNT(DISTINCT team_type) = 2
            """
        )
        return {int(pk): int(r) for pk, r in cur.fetchall() if r is not None}


def market_line(conn, schema: str, line_type: str) -> dict[int, tuple[float, float, int]]:
    """Per game: (total_point, median de-vigged P(over), n_books) at the best-supported point."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT game_pk, total_point, bookmaker, over_ml, under_ml
            FROM {schema}.f5_odds
            WHERE line_type = %s AND total_point IS NOT NULL
              AND over_ml IS NOT NULL AND under_ml IS NOT NULL
            """,
            (line_type,),
        )
        by_point: dict[tuple[int, float], list[float]] = defaultdict(list)
        for pk, point, _book, over, under in cur.fetchall():
            by_point[(int(pk), float(point))].append(
                no_vig_two_way(int(over), int(under), method="proportional")[0]
            )

    best: dict[int, tuple[float, float, int]] = {}
    for (pk, point), probs in by_point.items():
        if len(probs) < 2:
            continue
        # keep the point quoted by the most books; ties break toward the lower point
        if pk not in best or len(probs) > best[pk][2]:
            best[pk] = (point, float(np.median(probs)), len(probs))
    return best


def assess(name: str, p: np.ndarray, y: np.ndarray) -> None:
    beta, se = irls(logit(p).reshape(-1, 1), y)
    rel, res, unc, rows = murphy(p, y)
    brier = float(np.mean((p - y) ** 2))
    z_slope, z_int = (beta[1] - 1.0) / se[1], beta[0] / se[0]
    print(f"{name}  (n={len(p):,})")
    print(f"  slope     {beta[1]:+.3f} +/- {se[1]:.3f}  (z vs 1 = {z_slope:+.2f})")
    print(f"  intercept {beta[0]:+.3f} +/- {se[0]:.3f}  (z vs 0 = {z_int:+.2f})")
    print(f"  base rate {y.mean():.1%}, mean forecast {p.mean():.1%}")
    print(f"  Brier {brier:.4f} = reliability {rel:.4f} - resolution {res:.4f} "
          f"+ uncertainty {unc:.4f}")
    verdict = "calibrated" if abs(z_slope) < 1.96 and abs(z_int) < 1.96 else "MISCALIBRATED"
    print(f"  verdict: {verdict}")
    print(f"  {'n':>5} | {'range':>15} | {'mean fcst':>9} | {'observed':>9} | {'err':>7}")
    for n, lo, hi, pk_, ok in rows:
        print(f"  {n:5d} | [{lo:.3f}, {hi:.3f}] | {pk_:8.1%} | {ok:8.1%} | {pk_ - ok:+6.1%}")
    print()


def main() -> None:
    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=15,
    )
    runs = f5_runs(conn, c.schema)
    print(f"F5 run totals available for {len(runs):,} games (five complete innings)")
    print()
    print("Reference, full-game moneyline 2020-2025 (n=11,912):")
    print("  slope +1.015 +/- 0.048, intercept +0.003 +/- 0.020, resolution 0.0095")
    print("=" * 74)
    print()

    store = {}
    for line_type in ("open", "close"):
        lines = market_line(conn, c.schema, line_type)
        samples, pushes = [], 0
        for pk, (point, prob, _n) in lines.items():
            if pk not in runs:
                continue
            if runs[pk] == point:
                pushes += 1
                continue
            samples.append((prob, int(runs[pk] > point)))
        if len(samples) < 200:
            print(f"F5 totals {line_type}: only {len(samples)} usable games, skipped\n")
            continue
        p = np.array([s[0] for s in samples])
        y = np.array([s[1] for s in samples])
        store[line_type] = (p, y, dict(lines))
        assess(f"F5 totals, {line_type} (pushes dropped: {pushes})", p, y)
    conn.close()

    if "open" in store and "close" in store:
        po, yo, lo = store["open"]
        pc, yc, lc = store["close"]
        print("=" * 74)
        print("Does the F5 market improve from open to close?")
        print(f"  Brier open {np.mean((po - yo) ** 2):.4f}  "
              f"close {np.mean((pc - yc) ** 2):.4f}")
        common = set(lo) & set(lc)
        moved = sum(1 for pk in common if lo[pk][0] != lc[pk][0])
        print(f"  {len(common):,} games in both; total point moved in {moved:,} "
              f"({moved / max(len(common), 1):.0%})")
        print("  (full-game moneyline improved only 0.2397 -> 0.2395 across its two snapshots)")


if __name__ == "__main__":
    main()
