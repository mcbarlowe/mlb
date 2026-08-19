#!/usr/bin/env python
"""Moneyline at best-of-5-book OPENING prices - the actual betting configuration.

Bets are placed at the best available price among the five-book panel at the
OPENING snapshot, edge measured against the median of per-book de-vigged fair
probabilities at the open, flat 1u, walk-forward model (train 2018..season-1).
All-months and June-removed variants, per season 2021-2026.
"""

from __future__ import annotations

import sys

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


def load_season_open(conn, schema: str, season: int):
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT o.game_pk, o.bookmaker, o.home_ml, o.away_ml,
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

    finals = load_finals([season]).set_index("game_pk")["home_won"].to_dict()
    train = list(range(2018, season))
    probs = walkforward_home_probs(season, train).set_index("game_pk")[
        "model_prob_home"
    ].to_dict()

    per_game: dict[int, dict] = {}
    for pk, book, h, a, month in raw:
        pk = int(pk)
        if pk not in finals or pk not in probs:
            continue
        g = per_game.setdefault(pk, {"fairs": [], "best_h": 0.0, "best_a": 0.0,
                                     "month": int(month)})
        hd, ad = american_to_decimal(int(h)), american_to_decimal(int(a))
        g["best_h"] = max(g["best_h"], hd)
        g["best_a"] = max(g["best_a"], ad)
        g["fairs"].append(no_vig_two_way(int(h), int(a), "proportional")[0])
    rows = []
    for pk, g in per_game.items():
        if len(g["fairs"]) < 2:
            continue
        rows.append({
            "model": probs[pk], "fair": float(np.median(g["fairs"])),
            "best_home": g["best_h"], "best_away": g["best_a"],
            "home_won": bool(finals[pk]), "month": g["month"],
        })
    return rows


def settle(rows, drop_june: bool):
    out = []
    for r in rows:
        if drop_june and r["month"] == 6:
            continue
        signed = r["model"] - r["fair"]
        if abs(signed) < EDGE:
            continue
        home = signed >= 0
        dec = r["best_home"] if home else r["best_away"]
        won = r["home_won"] == home
        out.append((dec - 1.0) if won else -1.0)
    return np.array(out)


def main() -> None:
    c = PostgresConfig.from_env()
    conn = psycopg.connect(dbname=c.dbname, user=c.user, password=c.password,
                           host=c.host, port=c.port, connect_timeout=15)
    print(f"BET AT OPEN, best of {len(PANEL)} books, edge >= {EDGE:.0%}, flat 1u, walk-forward\n")
    print(f"{'season':>7} | {'all: bets':>9} {'ROI':>8} {'net':>8} | "
          f"{'exJun: bets':>11} {'ROI':>8} {'net':>8}")
    print("-" * 72)
    pool_all, pool_ex = [], []
    for season in SEASONS:
        rows = load_season_open(conn, c.schema, season)
        a = settle(rows, False)
        e = settle(rows, True)
        pool_all.append(a)
        pool_ex.append(e)
        print(f"{season:>7} | {len(a):9d} {a.mean():+8.2%} {a.sum():+7.1f}u | "
              f"{len(e):11d} {e.mean():+8.2%} {e.sum():+7.1f}u")
    conn.close()
    for label, pools in (("ALL MONTHS", pool_all), ("JUNE REMOVED", pool_ex)):
        arr = np.concatenate(pools)
        se = arr.std(ddof=1) / np.sqrt(len(arr))
        print(f"\n{label}: {len(arr)} bets | ROI {arr.mean():+.2%} "
              f"(se {se:.2%}, ~CI [{arr.mean() - 1.96 * se:+.2%}, "
              f"{arr.mean() + 1.96 * se:+.2%}]) | net {arr.sum():+.1f}u")


if __name__ == "__main__":
    main()
