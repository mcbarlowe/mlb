"""Build park factors for the batter prop tool.

For each (market, line) and venue (home team), the shrunk log-odds offset of
clearing the line at that park vs league, from the last three completed-ish
seasons of starts (PA >= 3): delta = logit(park rate) - logit(league rate),
shrunk by n/(n + 2000) games of prior weight. Written to
``models/props/park_factors.json`` for the model-only prop prediction producer.

Usage: uv run python scripts/build_prop_park_factors.py [--start 2024] [--out ...]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from build_prop_aging_curves import MARKET_POINTS

from src.data_contracts.prop_predictions import STAT_COLUMNS, STAT_FNS
from src.database import PostgresConfig

START_PA = 3
K_PARK = 2000.0


def load_lines(start_season: int, end_season: int) -> pd.DataFrame:
    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=10,
    )
    cols = ", ".join(f"COALESCE(b.{col}, 0)::int AS {col}" for col in STAT_COLUMNS)
    frame = pd.read_sql(
        f"""
        SELECT g.home_team_id, {cols}
        FROM {c.schema}.batting b
        JOIN {c.schema}.games g USING (game_pk)
        WHERE g.game_type = 'R' AND g.abstract_game_state = 'Final'
          AND g.season::int BETWEEN %(a)s AND %(b)s
          AND COALESCE(b.plateappearances, 0) >= %(pa)s
        """,
        conn,
        params={"a": start_season, "b": end_season, "pa": START_PA},
    )
    conn.close()
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=datetime.now(UTC).year - 2)
    ap.add_argument("--end", type=int, default=datetime.now(UTC).year)
    ap.add_argument("--out", default="models/props/park_factors.json")
    args = ap.parse_args()

    frame = load_lines(args.start, args.end)
    print(f"loaded {len(frame):,} starts {args.start}-{args.end}")

    stats = {col: frame[col].to_numpy() for col in STAT_COLUMNS}
    venues = frame["home_team_id"].to_numpy()
    factors: dict[str, dict[str, float]] = {}
    for market, points in MARKET_POINTS.items():
        fn = STAT_FNS[market]
        values = np.asarray(fn(stats))
        for point in points:
            outcome = (values > point).astype(float)
            overall = outcome.mean()
            if not 0.0 < overall < 1.0:
                continue
            lo_all = math.log(overall / (1.0 - overall))
            per_venue: dict[str, float] = {}
            for venue in np.unique(venues):
                mask = venues == venue
                n = int(mask.sum())
                rate = (outcome[mask].sum() + 0.5) / (n + 1.0)
                delta = math.log(rate / (1.0 - rate)) - lo_all
                per_venue[str(int(venue))] = round(delta * n / (n + K_PARK), 5)
            factors[f"{market}|{point:g}"] = per_venue

    payload = {
        "_meta": {
            "built": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seasons": [args.start, args.end],
            "start_pa": START_PA,
            "k_park": K_PARK,
            "method": "shrunk logit offset vs league, starts only",
        },
        "factors": factors,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    hits = factors.get("batter_hits|0.5", {})
    if hits:
        hi = max(hits, key=lambda k: hits[k])
        lo = min(hits, key=lambda k: hits[k])
        print(f"H|0.5 extremes: team {hi} {hits[hi]:+.3f} logits, team {lo} {hits[lo]:+.3f}")
    print(f"wrote {len(factors)} market-line park maps to {out}")


if __name__ == "__main__":
    main()
