"""Does the season-projection model beat preseason win totals, at REAL prices?

An early session pass reported +5.5% ROI on 120 bets at assumed -110 juice and called win
totals a cleaner test than game-by-game moneyline. The first disciplined re-test kept the
-110 assumption because ``resources/season_win_totals_2022_2025.csv`` carried no prices.
This version settles at the real per-side prices scraped from Covers
(``resources/season_win_totals_odds.csv``, scripts/scrape_covers_win_totals.py: one
preseason snapshot per season, typically BetMGM, with over/under odds like -115/-105).

Remaining structural weaknesses, stated up front:

  1. One book, one snapshot, so no line shopping. Shopping is the only verified edge in
     this repository (+2.1pp on moneyline); a market where it is impossible starts behind.
  2. ~30 bets per season. Per-season figures carry very wide intervals.
  3. Season-long holds. A win-total bet ties capital from March to October, so ROI per bet
     is not comparable to a moneyline ROI per bet without adjusting for turnover.

Guards: outcomes derived from our own game data (Covers' published wins used only as an
integrity cross-check), bootstrap intervals on every figure, per-season sign consistency,
threshold-sweep monotonicity, and the accuracy precondition (is the projection more
accurate than the line at all?).

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
ODDS_CSV = Path("resources/season_win_totals_odds.csv")
PROJECTIONS = Path("output/prior_baseline_backtest.csv")
SHORT_SEASONS = {2020}  # 60-game schedule; totals not comparable


def american_profit(odds: int) -> float:
    """Profit per 1 unit staked at the given American price."""
    return 100.0 / abs(odds) if odds < 0 else odds / 100.0


def final_wins(conn, schema: str, seasons: list[int]) -> dict[tuple[int, int], int]:
    """Regular-season wins per (season, team_id), audited linescore derivation."""
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
    ap.add_argument("--bucket", default="opening_day")
    ap.add_argument("--projection-type", default="model", choices=("model", "baseline"))
    args = ap.parse_args()

    market: dict[tuple[int, str], dict] = {}
    for row in csv.DictReader(ODDS_CSV.open()):
        season = int(row["season"])
        if season in SHORT_SEASONS:
            continue
        market[(season, row["abbreviation"])] = {
            "team_id": int(row["team_id"]),
            "line": float(row["win_total"]),
            "over_odds": int(row["over_odds"]),
            "under_odds": int(row["under_odds"]),
            "covers_wins": int(row["actual_wins"]) if row["actual_wins"] else None,
        }

    proj: dict[tuple[int, str], float] = {}
    for row in csv.DictReader(PROJECTIONS.open()):
        if row["as_of_bucket"] != args.bucket:
            continue
        if row["projection_type"] != args.projection_type:
            continue
        proj[(int(row["season"]), row["abbreviation"])] = float(row["expected_wins"])

    seasons = sorted({s for s, _ in market})
    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=15,
    )
    db_seasons = [s for s in seasons if s >= 2015]  # game data starts 2015
    actual = final_wins(conn, c.schema, db_seasons)
    conn.close()

    # integrity: our derived wins vs Covers' published wins
    checked = mismatched = 0
    for (season, _ab), m in market.items():
        key = (season, m["team_id"])
        if m["covers_wins"] is None or key not in actual:
            continue
        checked += 1
        if actual[key] != m["covers_wins"]:
            mismatched += 1
    print(f"outcome integrity: {checked} team-seasons cross-checked vs Covers, "
          f"{mismatched} mismatches")

    graded = []
    for (season, ab), m in market.items():
        wins = actual.get((season, m["team_id"]), m["covers_wins"])
        if wins is None or wins == m["line"]:
            continue
        graded.append({
            "season": season, "team": ab, "line": m["line"], "wins": wins,
            "over_odds": m["over_odds"], "under_odds": m["under_odds"],
            "over_won": wins > m["line"],
            "model": proj.get((season, ab)),
        })
    holds = [
        1.0 / (1.0 + american_profit(g["over_odds"]))
        + 1.0 / (1.0 + american_profit(g["under_odds"])) - 1.0
        for g in graded
    ]
    print(f"{len(graded)} graded team-seasons {min(seasons)}-{max(seasons)} "
          f"(2020 excluded) | mean two-way hold {np.mean(holds):.2%}\n")

    # ---- market-only diagnostics at real prices, every season ----
    print("Model-free at real prices (blind sides, every season):")
    print(f"{'strategy':>13} | {'bets':>5} | {'win%':>6} | {'ROI':>8} | {'95% CI':>22}")
    print("-" * 68)
    for label, side_over in (("always over", True), ("always under", False)):
        arr = np.array([
            american_profit(g["over_odds"] if side_over else g["under_odds"])
            if g["over_won"] == side_over else -1.0
            for g in graded
        ])
        m, lo, hi = boot_ci(arr)
        print(f"{label:>13} | {len(arr):5d} | {float((arr > 0).mean()):5.1%} | "
              f"{m:+7.2%} | [{lo:+7.2%}, {hi:+7.2%}]")
    over_rate = np.mean([g["over_won"] for g in graded])
    print(f"  over hit rate {over_rate:.1%} across {len(graded)} team-seasons\n")

    # ---- model strategy at real prices (seasons with projections) ----
    rows = [g for g in graded if g["model"] is not None]
    if not rows:
        print("no seasons with model projections available")
        return
    for r in rows:
        r["edge"] = r["model"] - r["line"]
    print(f"Model strategy: {len(rows)} team-seasons with {args.projection_type} "
          f"projections at {args.bucket}, settled at real prices")

    def settle(subset, edge_thr: float):
        out = []
        for r in subset:
            if abs(r["edge"]) < edge_thr:
                continue
            side_over = r["edge"] > 0
            won = r["over_won"] == side_over
            price = r["over_odds"] if side_over else r["under_odds"]
            out.append(american_profit(price) if won else -1.0)
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

    proj_seasons = sorted({r["season"] for r in rows})
    for thr in (0.0, 3.0):
        arr = settle(rows, thr)
        if len(arr) < 15:
            continue
        print(f"Per season at min edge {thr:.0f} (real prices):")
        print(f"{'season':>7} | {'bets':>5} | {'win%':>6} | {'ROI':>8}")
        print("-" * 34)
        signs = ""
        for s in proj_seasons:
            sub = settle([r for r in rows if r["season"] == s], thr)
            if not len(sub):
                signs += "?"
                continue
            signs += "+" if sub.mean() > 0 else "-"
            print(f"{s:>7} | {len(sub):5d} | {float((sub > 0).mean()):5.1%} | {sub.mean():+7.2%}")
        print(f"  sign pattern: {signs}  ({signs.count('+')}/{len(signs)} positive)\n")

    # accuracy precondition: is the projection more accurate than the line at all?
    err_model = np.array([abs(r["model"] - r["wins"]) for r in rows])
    err_line = np.array([abs(r["line"] - r["wins"]) for r in rows])
    diff = err_model - err_line
    m, lo, hi = boot_ci(diff)
    print("Accuracy precondition: mean absolute error in wins, paired on the same team-seasons")
    print(f"  model MAE {err_model.mean():.3f}  line MAE {err_line.mean():.3f}")
    print(f"  model minus line {m:+.3f} wins  95% CI [{lo:+.3f}, {hi:+.3f}]")
    if lo <= 0 <= hi:
        print("  -> indistinguishable; no accuracy basis for an edge")
    elif m < 0:
        print("  -> model more accurate than the line")
    else:
        print("  -> LINE more accurate than the model")


if __name__ == "__main__":
    main()
