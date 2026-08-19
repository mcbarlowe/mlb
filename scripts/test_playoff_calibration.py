"""Is the model's playoff probability miscalibrated, and in which direction?

The flat-stake futures review found ``make_playoffs`` at -15.28% and ``miss_playoffs`` at +9.11%
over nine seasons. Opposite signs on two sides of the same question point at a directional bias
rather than at vig, so the hypothesis is that the model overrates playoff probability.

ROI cannot settle this. Futures produce roughly ten qualifying bets per season and per-bet variance
is large because payouts run from -650 to +600, so a 9% edge needs on the order of two thousand
bets to resolve. That is nearly two hundred seasons.

Calibration can settle it on far more data. Every team-season carries a model playoff probability
and a binary outcome, which is 30 teams times 9 seasons rather than 91 bets, and the test uses the
probability itself instead of discarding it. This is the same substitution that settled the
moneyline question: paired accuracy beats realised ROI as an instrument.

Projections are built exactly as ``scripts/backtest_futures.py`` builds them, including training
strictly on seasons before the projected one, so the numbers correspond to the bets.

    uv run python scripts/test_playoff_calibration.py
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_futures import (
    _load_actual_outcomes,
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


def boot_ci(arr: np.ndarray, seed: int = 23):
    rng = random.Random(seed)
    n = len(arr)
    draws = sorted(
        float(np.mean([arr[rng.randrange(n)] for _ in range(n)])) for _ in range(BOOT)
    )
    return float(arr.mean()), draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]


def devig(rows, market: str, season: int) -> dict[int, float]:
    """De-vigged market probability per team, normalised to the era's winning-slot count."""
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
    args = ap.parse_args()

    cfg = PostgresConfig.from_env()
    rows: list[dict] = []
    with PostgresHandler(cfg) as pg:
        for season in SEASONS:
            outcomes = _load_actual_outcomes(season, pg)
            made = outcomes.get("make_playoffs", set())
            if not made:
                print(f"  {season}: no playoff outcomes, skipping")
                continue
            odds_rows = load_latest_futures_odds(
                pg, season=season, market_type="make_playoffs"
            )
            market = devig(odds_rows, "make_playoffs", season) if odds_rows else {}

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
            for team in proj.teams:
                p = float(team.playoff_prob)
                if not np.isfinite(p):
                    continue
                rows.append(
                    {
                        "season": season,
                        "team_id": int(team.team_id),
                        "model": p,
                        "market": market.get(int(team.team_id)),
                        "made": int(team.team_id) in made,
                    }
                )
            print(f"  {season}: {len(proj.teams)} teams, {len(made)} playoff berths")

    if not rows:
        print("no rows produced")
        return

    model = np.array([r["model"] for r in rows])
    made = np.array([1.0 if r["made"] else 0.0 for r in rows])
    print()
    print(f"{len(rows)} team-seasons, {int(made.sum())} berths "
          f"({made.mean():.1%} base rate)")
    print()

    m, lo, hi = boot_ci(model - made)
    print("Overall bias, model probability minus realised outcome:")
    print(f"  mean model {model.mean():.4f}   actual {made.mean():.4f}")
    print(f"  bias {m:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    if lo > 0:
        print("  -> model SYSTEMATICALLY OVERRATES playoff probability")
    elif hi < 0:
        print("  -> model systematically UNDERRATES playoff probability")
    else:
        print("  -> no detectable directional bias")
    print()

    print("Calibration by model probability bucket:")
    print(f"  {'bucket':>12} | {'n':>4} | {'mean model':>10} | {'actual':>7} | {'bias':>8}")
    print("  " + "-" * 52)
    edges = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.01]
    for a, b in itertools.pairwise(edges):
        mask = (model >= a) & (model < b)
        if mask.sum() < 5:
            continue
        print(f"  {f'{a:.2f}-{b:.2f}':>12} | {int(mask.sum()):4d} | "
              f"{model[mask].mean():10.3f} | {made[mask].mean():7.3f} | "
              f"{model[mask].mean() - made[mask].mean():+8.3f}")
    print()

    print("Per season bias:")
    print(f"  {'season':>7} | {'n':>3} | {'mean model':>10} | {'actual':>7} | {'bias':>8}")
    print("  " + "-" * 46)
    signs = ""
    for season in SEASONS:
        sub = [r for r in rows if r["season"] == season]
        if not sub:
            continue
        mm = np.array([r["model"] for r in sub])
        yy = np.array([1.0 if r["made"] else 0.0 for r in sub])
        signs += "+" if mm.mean() > yy.mean() else "-"
        print(f"  {season:>7} | {len(sub):3d} | {mm.mean():10.3f} | {yy.mean():7.3f} | "
              f"{mm.mean() - yy.mean():+8.3f}")
    print(f"  sign pattern {signs}  ({signs.count('+')}/{len(signs)} overrating)")
    print()

    paired = [r for r in rows if r["market"] is not None]
    if len(paired) > 30:
        mm = np.array([r["model"] for r in paired])
        mk = np.array([r["market"] for r in paired])
        yy = np.array([1.0 if r["made"] else 0.0 for r in paired])
        bm = float(np.mean((mm - yy) ** 2))
        bk = float(np.mean((mk - yy) ** 2))
        d, dlo, dhi = boot_ci((mm - yy) ** 2 - (mk - yy) ** 2, seed=31)
        print(f"Paired Brier on {len(paired)} team-seasons priced by the market:")
        print(f"  model {bm:.5f}   market {bk:.5f}")
        print(f"  gap {d:+.5f}   95% CI [{dlo:+.5f}, {dhi:+.5f}]")
        if dhi < 0:
            print("  -> MODEL MORE ACCURATE than the futures market")
        elif dlo > 0:
            print("  -> futures market more accurate than the model")
        else:
            print("  -> indistinguishable")
        mb, mblo, mbhi = boot_ci(mk - yy, seed=41)
        print(f"  market's own bias {mb:+.4f} [{mblo:+.4f}, {mbhi:+.4f}] "
              f"(a correctly de-vigged market should sit near zero)")


if __name__ == "__main__":
    main()
