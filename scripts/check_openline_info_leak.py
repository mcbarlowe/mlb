#!/usr/bin/env python
"""Information-cutoff leak check for open-line backtests.

At open execution the bet is placed when the line posts (earliest open
snapshot_time), but walk-forward features are built from every game BEFORE
first pitch. Any feature-relevant game that FINISHED after the open posted is
information the model had and the opening line could not have priced.

For every bet in the open+best-price configuration, mark it CONTAMINATED when
either team's most recent prior game ended (start + 4h, conservative) after
the game's earliest open snapshot. Report contamination rate and ROI on the
clean vs contaminated subsets, all months and June removed, 2021-2026.

If the clean-subset ROI matches the headline, the prior-evening leak is not
driving the result.
"""

from __future__ import annotations

import sys
from datetime import timedelta

sys.path.insert(0, "/Users/matthewbarlowe/code/python/mlb")

import numpy as np
import psycopg

from scripts.backtest_moneyline import load_finals, walkforward_home_probs
from scripts.backtest_moneyline_lineshop import PANEL_PRIORITY
from src.betting.odds import american_to_decimal, no_vig_two_way
from src.database import PostgresConfig

PANEL = PANEL_PRIORITY[:5]
EDGE = 0.03
SEASONS = (2021, 2022, 2023, 2024, 2025, 2026)
GAME_HOURS = 4.0  # conservative game-duration proxy


def load_bets(conn, schema: str, season: int):
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT o.game_pk, o.home_ml, o.away_ml, o.snapshot_time,
                   EXTRACT(MONTH FROM g.game_datetime)::int AS month
            FROM {schema}.odds o JOIN {schema}.games g ON g.game_pk = o.game_pk
            WHERE g.season::int = %s AND g.game_type = 'R' AND o.line_type = 'open'
              AND o.bookmaker = ANY(%s)
              AND o.home_ml IS NOT NULL AND o.away_ml IS NOT NULL
              AND g.game_datetime IS NOT NULL
            """,
            (season, list(PANEL)),
        )
        raw = cur.fetchall()
        cur.execute(
            f"""SELECT game_pk, game_datetime, home_team_id, away_team_id
                FROM {schema}.games
                WHERE season::int = %s AND game_type = 'R'
                  AND game_datetime IS NOT NULL""",
            (season,),
        )
        sched = cur.fetchall()

    finals = load_finals([season]).set_index("game_pk")["home_won"].to_dict()
    probs = walkforward_home_probs(season, list(range(2018, season))).set_index(
        "game_pk"
    )["model_prob_home"].to_dict()

    game_dt = {int(pk): dt for pk, dt, _, _ in sched}
    team_games: dict[int, list] = {}
    for pk, dt, hid, aid in sched:
        for tid in (int(hid), int(aid)):
            team_games.setdefault(tid, []).append(dt)
    for tid in team_games:
        team_games[tid].sort()
    game_teams = {int(pk): (int(hid), int(aid)) for pk, _, hid, aid in sched}

    per_game: dict[int, dict] = {}
    for pk, h, a, snap, month in raw:
        pk = int(pk)
        if pk not in finals or pk not in probs or pk not in game_dt:
            continue
        g = per_game.setdefault(pk, {"fairs": [], "best_h": 0.0, "best_a": 0.0,
                                     "month": int(month), "snap": snap})
        g["snap"] = min(g["snap"], snap)
        hd, ad = american_to_decimal(int(h)), american_to_decimal(int(a))
        g["best_h"] = max(g["best_h"], hd)
        g["best_a"] = max(g["best_a"], ad)
        g["fairs"].append(no_vig_two_way(int(h), int(a), "proportional")[0])

    rows = []
    for pk, g in per_game.items():
        if len(g["fairs"]) < 2:
            continue
        dt = game_dt[pk]
        contaminated = False
        for tid in game_teams[pk]:
            prior = [d for d in team_games[tid] if d < dt]
            if prior and prior[-1] + timedelta(hours=GAME_HOURS) > g["snap"]:
                contaminated = True
                break
        rows.append({
            "model": probs[pk], "fair": float(np.median(g["fairs"])),
            "best_home": g["best_h"], "best_away": g["best_a"],
            "home_won": bool(finals[pk]), "month": g["month"],
            "contaminated": contaminated,
        })
    return rows


def settle(rows, drop_june: bool, subset: str):
    out = []
    for r in rows:
        if drop_june and r["month"] == 6:
            continue
        if subset == "clean" and r["contaminated"]:
            continue
        if subset == "contaminated" and not r["contaminated"]:
            continue
        signed = r["model"] - r["fair"]
        if abs(signed) < EDGE:
            continue
        home = signed >= 0
        dec = r["best_home"] if home else r["best_away"]
        out.append((dec - 1.0) if r["home_won"] == home else -1.0)
    return np.array(out)


def main() -> None:
    c = PostgresConfig.from_env()
    conn = psycopg.connect(dbname=c.dbname, user=c.user, password=c.password,
                           host=c.host, port=c.port, connect_timeout=15)
    all_rows = []
    for season in SEASONS:
        rows = load_bets(conn, c.schema, season)
        all_rows.extend(rows)
        n_cont = sum(r["contaminated"] for r in rows)
        print(f"{season}: {len(rows)} games with open panel, "
              f"{n_cont} contaminated ({n_cont / max(len(rows), 1):.0%})")
    conn.close()

    print(f"\nOpen+best-price bets, edge >= {EDGE:.0%}, prior game end = start + {GAME_HOURS:g}h")
    print(f"{'scope':>22} | {'bets':>5} | {'ROI':>8} | {'se':>6} | {'net':>8}")
    print("-" * 62)
    for drop_june in (False, True):
        label = "June removed" if drop_june else "all months"
        for subset in ("all", "clean", "contaminated"):
            arr = settle(all_rows, drop_june, subset)
            if len(arr) < 10:
                continue
            se = arr.std(ddof=1) / np.sqrt(len(arr))
            print(f"{label + ' / ' + subset:>22} | {len(arr):5d} | {arr.mean():+8.2%} | "
                  f"{se:6.2%} | {arr.sum():+7.1f}u")


if __name__ == "__main__":
    main()
