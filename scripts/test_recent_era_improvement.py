"""Did the model genuinely improve in recent seasons, or is this the 2025 artifact again?

The ROI breakdown shows make_playoffs, miss_playoffs and win_totals all positive in 2024 and 2025,
and 2024-2025 are the only two seasons where *both* sides of the playoff market are positive at
once. Betting both sides with different team selections is not arbitrage, so both winning means the
selections beat the vig twice. That is worth a real test.

ROI cannot run this test. There are 22 qualifying playoff-market bets per season, so a two-season
window is 44 bets and any interval spans tens of percentage points. The same claim has now been
made three times in this project in different forms, and each time the high-power instrument
disagreed with the ROI:

  * the recency training window, refuted by a paired walk-forward test at -0.68pp
  * the season-to-season ROI spread, refuted by a permutation test at p = 0.211
  * "2025 and 2026 both positive", refuted because 2025 was inside the training set

So this uses accuracy, on 30 team-seasons per year rather than 22 bets, and it uses the market as
the benchmark rather than a break-even line. The question is narrow: does the model's Brier gap
against the de-vigged futures price shrink in recent seasons?

Reports per-season Brier gap, an explicit early-versus-late comparison with a bootstrap interval on
the difference, and the same split for realised ROI so the two instruments can be compared directly.

    uv run python scripts/test_recent_era_improvement.py
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_futures import (
    _load_actual_outcomes,
    _run_season_backtest,
    market_slots,
    wildcards_per_league,
)
from src.betting.futures_odds_store import load_latest_futures_odds
from src.betting.odds import american_to_decimal
from src.database import PostgresConfig, PostgresHandler
from src.sim.season import load_season_schedule, load_team_info, simulate_season
from src.sim.team_strength import fit_strength_predictor, load_completed_games

BOOT = 4000
SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
LATE = {2024, 2025}


def boot_ci(arr: np.ndarray, seed: int = 23):
    rng = random.Random(seed)
    n = len(arr)
    draws = sorted(
        float(np.mean([arr[rng.randrange(n)] for _ in range(n)])) for _ in range(BOOT)
    )
    return float(arr.mean()), draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]


def diff_ci(a: np.ndarray, b: np.ndarray, seed: int = 29):
    """Bootstrap interval for mean(a) - mean(b) with independent resampling."""
    rng = random.Random(seed)
    na, nb = len(a), len(b)
    draws = sorted(
        float(np.mean([a[rng.randrange(na)] for _ in range(na)]))
        - float(np.mean([b[rng.randrange(nb)] for _ in range(nb)]))
        for _ in range(BOOT)
    )
    return (
        float(a.mean() - b.mean()),
        draws[int(0.025 * BOOT)],
        draws[int(0.975 * BOOT)],
    )


def devig(rows, market: str, season: int) -> dict[int, float]:
    best: dict[int, int] = {}
    for row in rows:
        team = int(row["team_id"])
        odds = int(row["american_odds"])
        best[team] = max(best.get(team, odds), odds)
    raw = {t: 1.0 / american_to_decimal(o) for t, o in best.items()}
    total = sum(raw.values())
    slots = market_slots(market, season)
    if total <= 0 or not slots:
        return {}
    overround = total / slots
    return {t: p / overround for t, p in raw.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=10_000)
    ap.add_argument("--futures-edge", type=float, default=0.05)
    args = ap.parse_args()

    cfg = PostgresConfig.from_env()
    per_season_acc: dict[int, list[tuple[float, float, float]]] = {}
    per_season_roi: dict[int, list[float]] = {}

    with PostgresHandler(cfg) as pg:
        for season in SEASONS:
            outcomes = _load_actual_outcomes(season, pg)
            made = outcomes.get("make_playoffs", set())
            odds_rows = load_latest_futures_odds(
                pg, season=season, market_type="make_playoffs"
            )
            if not made or not odds_rows:
                continue
            market = devig(odds_rows, "make_playoffs", season)

            games = load_season_schedule(season)
            teams = load_team_info()
            completed = load_completed_games(
                start_season=season - 5, end_season=season - 1
            )
            predictor, _ = fit_strength_predictor(
                completed,
                prediction_season=season,
                train_seasons=list(range(season - 4, season)),
            )
            proj = simulate_season(
                games=games,
                teams=teams,
                as_of_date=datetime(season, 3, 15, tzinfo=UTC).date(),
                trials=args.trials,
                predictor=predictor,
                wild_cards_per_league=wildcards_per_league(season),
            )
            acc = []
            for team in proj.teams:
                tid = int(team.team_id)
                if tid not in market:
                    continue
                acc.append((float(team.playoff_prob), market[tid],
                            1.0 if tid in made else 0.0))
            per_season_acc[season] = acc

            bets: list[float] = []
            for mkt in ("make_playoffs", "miss_playoffs"):
                for b in _run_season_backtest(
                    season, mkt, args.futures_edge, 0.25, 0.05, pg
                ):
                    bets.append((b.decimal_odds - 1.0) if b.actual_win else -1.0)
            per_season_roi[season] = bets
            print(f"  {season}: {len(acc)} priced teams, {len(bets)} bets")

    print()
    print("Per season: model vs de-vigged market accuracy on playoff probability")
    print(f"{'season':>7} | {'n':>3} | {'model':>8} | {'market':>8} | {'gap':>9} | "
          f"{'bets':>5} | {'ROI':>8} | era")
    print("-" * 82)
    for season in SEASONS:
        acc = per_season_acc.get(season)
        if not acc:
            continue
        m = np.array([a[0] for a in acc])
        k = np.array([a[1] for a in acc])
        y = np.array([a[2] for a in acc])
        bm = float(np.mean((m - y) ** 2))
        bk = float(np.mean((k - y) ** 2))
        bets = np.array(per_season_roi.get(season, []))
        roi = f"{bets.mean():+7.2%}" if len(bets) else "      -"
        print(f"{season:>7} | {len(acc):3d} | {bm:8.5f} | {bk:8.5f} | {bm - bk:+9.5f} | "
              f"{len(bets):5d} | {roi} | {'LATE' if season in LATE else 'early'}")

    early_acc = np.concatenate([
        (np.array([a[0] for a in per_season_acc[s]])
         - np.array([a[2] for a in per_season_acc[s]])) ** 2
        - (np.array([a[1] for a in per_season_acc[s]])
           - np.array([a[2] for a in per_season_acc[s]])) ** 2
        for s in per_season_acc if s not in LATE
    ])
    late_acc = np.concatenate([
        (np.array([a[0] for a in per_season_acc[s]])
         - np.array([a[2] for a in per_season_acc[s]])) ** 2
        - (np.array([a[1] for a in per_season_acc[s]])
           - np.array([a[2] for a in per_season_acc[s]])) ** 2
        for s in per_season_acc if s in LATE
    ])
    early_roi = np.concatenate([np.array(per_season_roi[s]) for s in per_season_roi
                               if s not in LATE])
    late_roi = np.concatenate([np.array(per_season_roi[s]) for s in per_season_roi
                              if s in LATE])

    print()
    print("Early (2016-2023) versus late (2024-2025)")
    print()
    print("Instrument 1: accuracy, model minus market Brier. Negative means model better.")
    for label, arr in (("early", early_acc), ("late", late_acc)):
        m, lo, hi = boot_ci(arr)
        print(f"  {label:>6}: n={len(arr):4d}  gap {m:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]")
    d, dlo, dhi = diff_ci(late_acc, early_acc)
    print(f"  late minus early: {d:+.5f}  95% CI [{dlo:+.5f}, {dhi:+.5f}]")
    print(f"  -> {'IMPROVEMENT is significant' if dhi < 0 else 'no significant improvement'}")
    print()
    print("Instrument 2: realised ROI, both sides of the playoff market.")
    for label, arr in (("early", early_roi), ("late", late_roi)):
        m, lo, hi = boot_ci(arr)
        print(f"  {label:>6}: n={len(arr):4d}  ROI {m:+.2%}  95% CI [{lo:+.2%}, {hi:+.2%}]")
    d, dlo, dhi = diff_ci(late_roi, early_roi)
    print(f"  late minus early: {d:+.2%}  95% CI [{dlo:+.2%}, {dhi:+.2%}]")
    print(f"  -> {'IMPROVEMENT is significant' if dlo > 0 else 'no significant improvement'}")
    print()
    print("If the instruments disagree, accuracy wins: it uses the probability rather than")
    print("discarding it, and here it carries roughly ten times the effective sample.")


if __name__ == "__main__":
    main()
