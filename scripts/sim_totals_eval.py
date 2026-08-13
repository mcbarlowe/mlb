"""Benchmark the Monte Carlo sim against the totals (O/U) market.

The sim's whole-game score distribution yields P(total > line) directly. We
compare that to the consensus de-vigged market P(over) against realized totals
(Brier + log loss + a simple flat-bet ROI), on a sample of games.

    uv run python scripts/sim_totals_eval.py --season 2025 --games 500 --sims 500
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.odds import no_vig_two_way
from src.database import PostgresConfig
from src.sim.contact_environment import ContactEnvironment, parse_weather
from src.sim.db_games import GameDataStore
from src.sim.slate import build_day_ahead_simulator

EPS = 1e-9


def market_totals(season: int) -> dict[int, tuple[float, float]]:
    """game_pk -> (consensus_point, consensus_devigged_p_over)."""
    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=15,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT o.game_pk, o.total_point, o.over_ml, o.under_ml
                    FROM {c.schema}.odds_totals o JOIN {c.schema}.games g USING(game_pk)
                    WHERE g.season::int=%s AND o.line_type='close'""",
                (season,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    pts: dict[int, list[float]] = {}
    povs: dict[int, list[float]] = {}
    for game_pk, point, over_ml, under_ml in rows:
        p_over, _ = no_vig_two_way(float(over_ml), float(under_ml))
        pts.setdefault(int(game_pk), []).append(float(point))
        povs.setdefault(int(game_pk), []).append(p_over)
    return {
        pk: (statistics.median(pts[pk]), sum(povs[pk]) / len(povs[pk]))
        for pk in pts
    }

def game_environments(season: int) -> dict[int, tuple]:
    """game_pk -> (venue_id, GameWeather) for the contact environment."""
    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=15,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT game_pk, venue_id, weather_temp, weather_wind,
                           weather_condition
                    FROM {c.schema}.games WHERE season::int=%s""",
                (season,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return {
        int(pk): (int(v) if v is not None else None, parse_weather(t, w, cond))
        for pk, v, t, w, cond in rows
    }


def brier(p, o):
    return (p - o) ** 2


def logloss(p, o):
    p = min(max(p, EPS), 1 - EPS)
    return -(o * math.log(p) + (1 - o) * math.log(1 - p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--sims", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--edge", type=float, default=0.03, help="flat-bet edge threshold")
    ap.add_argument("--pa-calibration", default=None,
                    help="off-season PA calibration path (OOS totals test)")
    args = ap.parse_args()

    import os

    market = market_totals(args.season)
    store = GameDataStore.load(args.season)
    simulator, _ = build_day_ahead_simulator(
        season=args.season, seed=args.seed, outcome_run_dir="auto",
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
        pa_calibration_path=args.pa_calibration,
    )
    contact_env = ContactEnvironment.load(args.season)
    envs = game_environments(args.season) if contact_env else {}
    print(f"contact environment: {'ON' if contact_env else 'OFF (no park factors)'}")

    candidates = [pk for pk in store.final_game_pks(args.seed, 10_000) if pk in market]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    rows = []
    for pk in candidates:
        if len(rows) >= args.games:
            break
        point, mkt_over = market[pk]
        try:
            away = store.lineup(pk, "away", individual_bullpen=True)
            home = store.lineup(pk, "home", individual_bullpen=True)
        except (ValueError, KeyError):
            continue
        environment = None
        if contact_env:
            venue_id, weather = envs.get(pk, (None, None))
            environment = contact_env.multipliers(venue_id, weather)
        results = simulator.simulate_many(
            away, home, args.sims, environment=environment
        )
        totals = [r.away_runs + r.home_runs for r in results]
        n = len(totals)
        over = sum(1 for t in totals if t > point)
        push = sum(1 for t in totals if t == point)
        sim_over = (over + 0.5 * push) / n
        a, h = store.final(pk)
        actual = a + h
        o = 1.0 if actual > point else (0.0 if actual < point else 0.5)
        rows.append((pk, point, sim_over, mkt_over, o))
        print(f"{pk} pt={point} sim_over={sim_over:.2f} mkt_over={mkt_over:.2f} actual={actual}", flush=True)

    m = len(rows)
    if m == 0:
        raise SystemExit("no games")
    sim_b = sum(brier(r[2], r[4]) for r in rows) / m
    mkt_b = sum(brier(r[3], r[4]) for r in rows) / m
    sim_ll = sum(logloss(r[2], r[4]) for r in rows) / m
    mkt_ll = sum(logloss(r[3], r[4]) for r in rows) / m

    # paired bootstrap on sim_brier - mkt_brier (negative => sim better)
    diffs = [brier(r[2], r[4]) - brier(r[3], r[4]) for r in rows]
    mean_diff = sum(diffs) / m
    rng2 = random.Random(0)
    boots = sorted(sum(diffs[rng2.randrange(m)] for _ in range(m)) / m for _ in range(4000))
    lo, hi = boots[100], boots[3899]

    # simple flat-bet ROI: bet the side the sim favors vs market by > edge; -110 price
    bets = wins = 0
    for _, _, so, mo, o in rows:
        if so - mo > args.edge:      # sim likes the over
            bets += 1; wins += o
        elif mo - so > args.edge:    # sim likes the under
            bets += 1; wins += (1 - o)
    roi = (wins * (100 / 110) - (bets - wins)) / bets if bets else float("nan")

    print(f"\nSeason {args.season}: {m} games, {args.sims} sims each")
    print(f"{'':8}{'Brier':>10}{'log loss':>10}")
    print(f"{'sim':8}{sim_b:10.4f}{sim_ll:10.4f}")
    print(f"{'market':8}{mkt_b:10.4f}{mkt_ll:10.4f}")
    print(f"Brier gap (sim - market): {mean_diff:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  (neg = sim better)")
    print(f"flat-bet @ edge>{args.edge}: {bets} bets, {wins:.1f} wins ({wins/bets:.1%}), ROI {roi:+.1%}" if bets else "no bets")


if __name__ == "__main__":
    main()
