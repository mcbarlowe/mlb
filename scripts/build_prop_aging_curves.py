"""Build aging curves for the batter prop tool.

Delta-method aging curves on the log-odds of clearing each prop line
(stat > point), per (market, point): for every player with >= 60 games at
consecutive integer ages, average the within-player year-over-year logit
change (weighted by harmonic-mean games), then integrate into a cumulative
curve anchored at age 27. Written to ``models/props/aging_curves.json`` for the
model-only ``scripts/publish_prop_predictions.py`` producer.

Usage: uv run python scripts/build_prop_aging_curves.py [--start 2015] [--out ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from mlb.data_contracts.prop_predictions import STAT_COLUMNS, STAT_FNS
from mlb.database import PostgresConfig

MARKET_POINTS: dict[str, tuple[float, ...]] = {
    "batter_home_runs": (0.5, 1.5),
    "batter_hits": (0.5, 1.5, 2.5),
    "batter_total_bases": (0.5, 1.5, 2.5, 3.5),
    "batter_rbis": (0.5, 1.5, 2.5),
    "batter_runs_scored": (0.5, 1.5),
    "batter_walks": (0.5, 1.5),
    "batter_stolen_bases": (0.5, 1.5),
    "batter_strikeouts": (0.5, 1.5, 2.5),
    "batter_doubles": (0.5,),
    "batter_singles": (0.5, 1.5),
    "batter_hits_runs_rbis": (1.5, 2.5, 3.5),
}
CURVE_MIN_GP = 60
AGE_LO, AGE_HI = 21, 38


def load_lines(start_season: int, end_season: int) -> pd.DataFrame:
    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=10,
    )
    cols = ", ".join(f"COALESCE(b.{col}, 0)::int AS {col}" for col in STAT_COLUMNS)
    frame = pd.read_sql(
        f"""
        SELECT b.player_id, g.season::int AS season,
               g.game_date::date AS game_date, p.birth_date::date AS birth_date,
               {cols}
        FROM {c.schema}.batting b
        JOIN {c.schema}.games g USING (game_pk)
        JOIN {c.schema}.players p USING (player_id)
        WHERE g.game_type = 'R' AND g.abstract_game_state = 'Final'
          AND g.season::int BETWEEN %(a)s AND %(b)s
          AND COALESCE(b.plateappearances, 0) > 0
          AND p.birth_date IS NOT NULL
        """,
        conn,
        params={"a": start_season, "b": end_season},
    )
    conn.close()
    frame["iage"] = (
        (
            pd.to_datetime(frame["game_date"]) - pd.to_datetime(frame["birth_date"])
        ).dt.days / 365.25
    ).round().astype(int)
    return frame


def build_curve(frame: pd.DataFrame, market: str, point: float) -> dict[str, float]:
    fn = STAT_FNS[market]
    stats = {col: frame[col].to_numpy() for col in STAT_COLUMNS}
    outcome = (
        np.array(fn({col: stats[col] for col in STAT_COLUMNS})) > point
    ).astype(int)
    work = frame[["player_id", "season", "iage"]].assign(y=outcome)
    agg = (
        work.groupby(["player_id", "season", "iage"])["y"]
        .agg(succ="sum", n="count")
        .reset_index()
    )
    agg = agg[agg["n"] >= CURVE_MIN_GP]
    if agg.empty:
        return {}
    rate = (agg["succ"] + 0.5) / (agg["n"] + 1.0)
    agg = agg.assign(lo=np.log(rate / (1.0 - rate)))
    deltas: dict[int, list[tuple[float, float]]] = {}
    for _, g in agg.groupby("player_id"):
        g = g.sort_values("season")
        rows = list(g.itertuples(index=False))
        for a, b in pairwise(rows):
            if b.season == a.season + 1 and b.iage == a.iage + 1:
                w = 2.0 / (1.0 / a.n + 1.0 / b.n)
                deltas.setdefault(int(a.iage), []).append((float(b.lo - a.lo), w))
    step: dict[int, float] = {}
    for age, pairs in deltas.items():
        tw = sum(w for _, w in pairs)
        if tw:
            step[age] = sum(d * w for d, w in pairs) / tw
    curve = {27: 0.0}
    for age in range(27, AGE_HI):
        curve[age + 1] = curve[age] + step.get(age, 0.0)
    for age in range(27, AGE_LO, -1):
        curve[age - 1] = curve[age] - step.get(age - 1, 0.0)
    return {str(age): round(v, 5) for age, v in sorted(curve.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end", type=int, default=datetime.now(UTC).year)
    ap.add_argument("--out", default="models/props/aging_curves.json")
    args = ap.parse_args()

    frame = load_lines(args.start, args.end)
    print(f"loaded {len(frame):,} player-game lines {args.start}-{args.end}")

    curves: dict[str, dict[str, float]] = {}
    for market, points in MARKET_POINTS.items():
        for point in points:
            key = f"{market}|{point:g}"
            curve = build_curve(frame, market, point)
            if curve:
                curves[key] = curve
                peak = min(curve.items(), key=lambda kv: -kv[1])
                print(f"  {key:<32} ages {min(curve)}-{max(curve)} | "
                      f"peak age {peak[0]} | age-34 vs 27: {curve['34']:+.3f} logits")

    payload = {
        "_meta": {
            "built": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seasons": [args.start, args.end],
            "ref_age": 27,
            "method": "delta-method on logit(stat > point), harmonic-games weights",
        },
        "curves": curves,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"\nwrote {len(curves)} curves to {out}")


if __name__ == "__main__":
    main()
