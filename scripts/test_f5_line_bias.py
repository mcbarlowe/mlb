"""Is the first-five totals line biased, and is any bias larger than the hold?

Calibration of P(over) is nearly uninformative for a totals market: books set the point so the
two sides are close to even, leaving resolution near zero and the slope estimate unusable. The
informative test uses the continuous outcome. If the line is unbiased then actual first-five
runs regress on the line with slope 1 and intercept 0, and the residual mean is zero.

Any bias only matters net of the hold, so the hold is measured on the same rows rather than
assumed. The comparison of interest is the mean residual in runs, converted to a probability
edge, against the per-book overround.

    uv run python scripts/test_f5_line_bias.py
"""

from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.odds import american_to_decimal, american_to_prob, no_vig_two_way
from src.database import PostgresConfig


def f5_runs(conn, schema: str) -> dict[int, int]:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT game_pk, SUM(runs)
            FROM {schema}.linescore
            WHERE inning <= 5
            GROUP BY game_pk
            HAVING COUNT(DISTINCT inning) = 5 AND COUNT(DISTINCT team_type) = 2
            """
        )
        return {int(pk): int(r) for pk, r in cur.fetchall() if r is not None}


def main() -> None:
    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=15,
    )
    runs = f5_runs(conn, c.schema)

    for line_type in ("open", "close"):
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT game_pk, total_point, bookmaker, over_ml, under_ml
                FROM {c.schema}.f5_odds
                WHERE line_type = %s AND total_point IS NOT NULL
                  AND over_ml IS NOT NULL AND under_ml IS NOT NULL
                """,
                (line_type,),
            )
            rows = cur.fetchall()

        # Everything must be computed inside a single (game, total_point) group. Pooling
        # across points pairs an over at one line with an under at another, which is not a
        # placeable position and fabricates a near-zero hold.
        by_point: dict[tuple[int, float], list[tuple[int, int]]] = defaultdict(list)
        holds: list[float] = []
        for pk, point, _book, over, under in rows:
            by_point[(int(pk), float(point))].append((int(over), int(under)))
            holds.append(american_to_prob(int(over)) + american_to_prob(int(under)) - 1.0)

        # Per game keep the point quoted by the most books, requiring at least two.
        chosen: dict[int, tuple[float, list[tuple[int, int]]]] = {}
        for (pk, point), quotes in by_point.items():
            if len(quotes) < 2:
                continue
            if pk not in chosen or len(quotes) > len(chosen[pk][1]):
                chosen[pk] = (point, quotes)

        line = {pk: point for pk, (point, _) in chosen.items()}
        over_devig = {
            pk: statistics.median(
                no_vig_two_way(o, u, method="proportional")[0] for o, u in quotes
            )
            for pk, (_, quotes) in chosen.items()
        }
        shopped_by_game = {
            pk: 1.0 / max(american_to_decimal(o) for o, _ in quotes)
            + 1.0 / max(american_to_decimal(u) for _, u in quotes)
            - 1.0
            for pk, (_, quotes) in chosen.items()
        }
        books_at_point = statistics.median(
            len(quotes) for _, (_, quotes) in chosen.items()
        )

        paired = [(line[pk], runs[pk]) for pk in line if pk in runs]
        if len(paired) < 200:
            print(f"{line_type}: only {len(paired)} usable games\n")
            continue

        x = np.array([p for p, _ in paired], dtype=float)
        y = np.array([r for _, r in paired], dtype=float)
        resid = y - x
        n = len(paired)
        se_resid = resid.std(ddof=1) / np.sqrt(n)
        xd = np.column_stack([np.ones(n), x])
        beta, *_ = np.linalg.lstsq(xd, y, rcond=None)
        fitted = xd @ beta
        s2 = ((y - fitted) ** 2).sum() / (n - 2)
        cov = s2 * np.linalg.inv(xd.T @ xd)
        se = np.sqrt(np.diag(cov))

        hold = statistics.median(holds)
        shopped = statistics.median(
            [shopped_by_game[pk] for pk in line if pk in runs]
        )

        print(f"F5 totals, {line_type}  (n={n:,})")
        print(f"  mean line {x.mean():.3f} runs, mean actual {y.mean():.3f} runs")
        print(f"  mean residual (actual - line) {resid.mean():+.4f} +/- {se_resid:.4f} "
              f"(z = {resid.mean() / se_resid:+.2f})")
        print(f"  regression actual ~ line: slope {beta[1]:+.3f} +/- {se[1]:.3f} "
              f"(z vs 1 = {(beta[1] - 1) / se[1]:+.2f}), "
              f"intercept {beta[0]:+.3f} +/- {se[0]:.3f}")
        print(f"  residual sd {resid.std(ddof=1):.3f} runs")
        print(f"  per-book hold {hold:.2%}, best-of-{books_at_point:.0f}-at-same-point "
              f"shopped hold {shopped:.2%}")

        # Convert the run bias into a probability edge on the under, using the empirical
        # distribution of residuals rather than a parametric assumption.
        under_rate = float(np.mean(y < x))
        over_rate = float(np.mean(y > x))
        push_rate = float(np.mean(y == x))
        priced_over = statistics.median(
            [over_devig[pk] for pk in line if pk in runs]
        )
        live = under_rate + over_rate
        print(f"  outcomes: over {over_rate:.1%}, under {under_rate:.1%}, "
              f"push {push_rate:.1%}")
        print(f"  excluding pushes: over {over_rate / live:.1%} vs de-vigged price "
              f"{priced_over:.1%}  -> edge on under {priced_over - over_rate / live:+.2%}")
        print(f"  breakeven needs edge > shopped hold / 2 = {shopped / 2:.2%}  -> "
              f"{'TRADEABLE' if (priced_over - over_rate / live) > shopped / 2 else 'not tradeable'}")
        print()

    conn.close()
    print("Reference: full-game moneyline resolution 0.0095, slope 1.015 +/- 0.048,")
    print("per-book hold 4.1%, shopped hold ~1.4% with 5 accounts.")


if __name__ == "__main__":
    main()
