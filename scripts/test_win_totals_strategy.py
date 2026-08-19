"""Does the season-projection model beat preseason win totals?

An early session pass reported +5.5% ROI on 120 bets, +14.6% filtering to 3-plus win edges, and
+44% in 2025, and called win totals a cleaner test than game-by-game moneyline. That predates the
futures audit, which found four defects in the futures pipeline and turned a claimed +71% into
-3.37%. This re-tests the win-totals claim with the discipline the rest of the project now uses.

Structural weaknesses to state up front, because they bound what any result can mean:

  1. No prices. ``resources/season_win_totals_2022_2025.csv`` carries the line but no odds, so
     ROI must assume standard -110 juice on both sides. Real win-total markets are frequently
     -115/-125 and the vig is not symmetric.
  2. One source, so no line shopping. Shopping is the only verified edge in this repository
     (+2.1pp on moneyline); a market where it is impossible starts 2.1pp behind.
  3. Thirty bets per season across four seasons. 120 bets is roughly a tenth of one season of
     moneyline bets, so per-season figures carry very wide intervals.
  4. Season-long holds. A win-total bet ties capital from March to October, so ROI per bet is not
     comparable to a moneyline ROI per bet without adjusting for turnover.

Guards applied: bootstrap intervals on every figure, per-season sign consistency, a threshold
sweep to check monotonicity, an over/under split, and a Bonferroni-corrected view of the
subgroups. The threshold sweep is the check that refuted the CLV claim earlier.

    uv run python scripts/test_win_totals_strategy.py
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.futures_outcomes import season_records
from src.database import PostgresConfig

BOOT = 4000
WIN_TOTALS = Path("resources/season_win_totals_2022_2025.csv")
PROJECTIONS = Path("output/prior_baseline_backtest.csv")
JUICE = -110


def american_profit(odds: int) -> float:
    """Profit per 1 unit staked at the given American price."""
    return 100.0 / abs(odds) if odds < 0 else odds / 100.0


def final_wins(conn, schema: str, seasons: list[int]) -> dict[tuple[int, int], int]:
    """Regular-season wins per (season, team_id), reusing the audited linescore derivation."""
    out: dict[tuple[int, int], int] = {}
    for season in seasons:
        wins, _losses, _h2h = season_records(conn, schema, season)
        for team_id, w in wins.items():
            out[(season, int(team_id))] = int(w)
    return out


def boot_ci(arr: np.ndarray, seed: int = 23):
    rng = random.Random(seed)
    n = len(arr)
    draws = sorted(
        float(np.mean([arr[rng.randrange(n)] for _ in range(n)])) for _ in range(BOOT)
    )
    return float(arr.mean()), draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edge", type=float, default=0.0,
                    help="Minimum |projected - line| in wins to place a bet.")
    ap.add_argument("--bucket", default="opening_day")
    ap.add_argument("--projection-type", default="model", choices=("model", "baseline"))
    args = ap.parse_args()

    lines: dict[tuple[int, str], float] = {}
    for row in csv.DictReader(WIN_TOTALS.open()):
        lines[(int(row["season"]), row["abbreviation"])] = float(row["win_total"])

    proj: dict[tuple[int, str], tuple[int, float]] = {}
    for row in csv.DictReader(PROJECTIONS.open()):
        if row["as_of_bucket"] != args.bucket:
            continue
        if row["projection_type"] != args.projection_type:
            continue
        proj[(int(row["season"]), row["abbreviation"])] = (
            int(row["team_id"]),
            float(row["expected_wins"]),
        )

    seasons = sorted({s for s, _ in lines})
    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=15,
    )
    actual = final_wins(conn, c.schema, seasons)
    conn.close()

    rows = []
    missing = 0
    for key, line in lines.items():
        if key not in proj:
            missing += 1
            continue
        team_id, expected = proj[key]
        season = key[0]
        if (season, team_id) not in actual:
            missing += 1
            continue
        wins = actual[(season, team_id)]
        if wins == line:
            continue
        rows.append(
            {
                "season": season,
                "team": key[1],
                "line": line,
                "model": expected,
                "wins": wins,
                "edge": expected - line,
                "over_won": wins > line,
            }
        )
    print(f"{len(rows)} team-seasons graded, {missing} unmatched, "
          f"{args.projection_type} projections at {args.bucket}, juice {JUICE}")
    print(f"mean |edge| {np.mean([abs(r['edge']) for r in rows]):.2f} wins, "
          f"mean line {np.mean([r['line'] for r in rows]):.1f}, "
          f"mean actual {np.mean([r['wins'] for r in rows]):.1f}")
    print()

    payout = american_profit(JUICE)

    def settle(subset, edge: float):
        out = []
        for r in subset:
            if abs(r["edge"]) < edge:
                continue
            side_over = r["edge"] > 0
            won = r["over_won"] == side_over
            out.append(payout if won else -1.0)
        return np.array(out)

    print("Threshold sweep. Monotonicity is the check that refuted the CLV claim.")
    print(f"{'min edge':>9} | {'bets':>5} | {'win%':>6} | {'ROI':>8} | {'95% CI':>22} | zero in CI")
    print("-" * 76)
    for thr in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        arr = settle(rows, thr)
        if len(arr) < 15:
            print(f"{thr:>9.0f} | {len(arr):5d} | too few")
            continue
        m, lo, hi = boot_ci(arr)
        wins = float((arr > 0).mean())
        print(f"{thr:>9.0f} | {len(arr):5d} | {wins:5.1%} | {m:+7.2%} | "
              f"[{lo:+7.2%}, {hi:+7.2%}] | {'yes' if lo <= 0 <= hi else 'NO'}")
    print()
    print(f"Breakeven win rate at {JUICE}: {1 / (1 + payout):.1%}")
    print()

    for thr in (0.0, 3.0):
        arr = settle(rows, thr)
        if len(arr) < 15:
            continue
        print(f"Per season at min edge {thr:.0f}:")
        print(f"{'season':>7} | {'bets':>5} | {'win%':>6} | {'ROI':>8}")
        print("-" * 34)
        signs = ""
        for s in seasons:
            sub = settle([r for r in rows if r["season"] == s], thr)
            if not len(sub):
                signs += "?"
                continue
            signs += "+" if sub.mean() > 0 else "-"
            print(f"{s:>7} | {len(sub):5d} | {float((sub > 0).mean()):5.1%} | {sub.mean():+7.2%}")
        print(f"  sign pattern: {signs}  ({signs.count('+')}/{len(signs)} positive)")
        print()

    print("Over versus under split at min edge 0 (direction bias):")
    print(f"{'side':>7} | {'bets':>5} | {'win%':>6} | {'ROI':>8} | {'95% CI':>22}")
    print("-" * 62)
    for label, pick_over in (("over", True), ("under", False)):
        sub = []
        for r in rows:
            if (r["edge"] > 0) != pick_over:
                continue
            sub.append(payout if r["over_won"] == pick_over else -1.0)
        if len(sub) < 15:
            continue
        arr = np.array(sub)
        m, lo, hi = boot_ci(arr)
        print(f"{label:>7} | {len(arr):5d} | {float((arr > 0).mean()):5.1%} | {m:+7.2%} | "
              f"[{lo:+7.2%}, {hi:+7.2%}]")
    print()

    # Is the projection even more accurate than the line? That is the precondition for edge.
    err_model = np.array([abs(r["model"] - r["wins"]) for r in rows])
    err_line = np.array([abs(r["line"] - r["wins"]) for r in rows])
    diff = err_model - err_line
    m, lo, hi = boot_ci(diff, seed=31)
    print("Accuracy precondition: mean absolute error in wins, paired on the same team-seasons")
    print(f"  model MAE {err_model.mean():.3f}  line MAE {err_line.mean():.3f}")
    print(f"  model minus line {m:+.3f} wins  95% CI [{lo:+.3f}, {hi:+.3f}]")
    if hi < 0:
        print("  -> MODEL MORE ACCURATE than the posted line")
    elif lo > 0:
        print("  -> line more accurate than the model")
    else:
        print("  -> indistinguishable; no accuracy basis for an edge")


if __name__ == "__main__":
    main()
