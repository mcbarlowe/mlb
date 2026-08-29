"""Pre-game starter "stuff trend" features from pitch-level data.

Hypothesis under test: changes in a pitcher's physical stuff can become predictive
before results-based statistics reflect them. This producer derives a leak-free
feature for MLB model evaluation; downstream market comparison belongs in betting.

Strictly pre-game by construction. For each start, the feature compares the pitcher's recent
form window (his previous ``RECENT`` starts) against a baseline window (the ``BASELINE`` starts
before those). The current game contributes nothing, so there is no leakage.

Per-game features are differenced home minus away, matching the orientation of the home-win
target.

    uv run python scripts/build_starter_stuff_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import PostgresConfig

RECENT = 2      # starts forming the "current form" window
BASELINE = 6    # starts before those forming the comparison baseline
FASTBALLS = ("FF", "FA", "SI", "FT", "FC")
BREAKING = ("SL", "CU", "KC", "SV", "ST")
OUT = Path("data/analysis/starter_stuff.parquet")


def load_start_aggregates() -> pl.DataFrame:
    """One row per (game, team, starting pitcher) with mean velocity and break.

    The starting pitcher is the one who threw the first pitch that his team's side of the
    inning allowed, identified as the pitcher with the lowest at_bat_index in the game for
    that fielding team.
    """
    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=30,
    )
    query = f"""
        WITH first_ab AS (
            SELECT game_pk, pitcher_id,
                   MIN(at_bat_index) AS first_ab,
                   BOOL_OR(half_inning = 'top') AS pitched_top
            FROM {c.schema}.pitches
            WHERE game_type = 'R' AND pitcher_id IS NOT NULL
            GROUP BY game_pk, pitcher_id
        ),
        starters AS (
            -- lowest at_bat_index per game and per side identifies that side's starter
            SELECT game_pk, pitcher_id, pitched_top,
                   ROW_NUMBER() OVER (
                       PARTITION BY game_pk, pitched_top ORDER BY first_ab
                   ) AS rn
            FROM first_ab
        )
        SELECT p.game_pk,
               p.game_date,
               p.pitcher_id,
               s.pitched_top,
               AVG(p.pitch_start_speed) FILTER (
                   WHERE p.pitch_type_code = ANY(%(fb)s)
               ) AS fb_velo,
               AVG(p.break_vertical_induced) FILTER (
                   WHERE p.pitch_type_code = ANY(%(br)s)
               ) AS br_ivb,
               AVG(ABS(p.break_horizontal)) FILTER (
                   WHERE p.pitch_type_code = ANY(%(br)s)
               ) AS br_hb,
               COUNT(*) AS pitches
        FROM {c.schema}.pitches p
        JOIN starters s
          ON s.game_pk = p.game_pk
         AND s.pitcher_id = p.pitcher_id
         AND s.rn = 1
        WHERE p.game_type = 'R'
        GROUP BY p.game_pk, p.game_date, p.pitcher_id, s.pitched_top
        HAVING COUNT(*) >= 30
    """
    with conn.cursor() as cur:
        cur.execute(query, {"fb": list(FASTBALLS), "br": list(BREAKING)})
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    conn.close()
    return pl.DataFrame(
        {c_: [r[i] for r in rows] for i, c_ in enumerate(cols)}, strict=False
    )


def trend_features(starts: pl.DataFrame) -> pl.DataFrame:
    """Recent-window minus baseline-window deltas, shifted so the current start is excluded."""
    starts = starts.with_columns(
        pl.col("game_date").cast(pl.Date),
        pl.col("fb_velo").cast(pl.Float64),
        pl.col("br_ivb").cast(pl.Float64),
        pl.col("br_hb").cast(pl.Float64),
    ).sort(["pitcher_id", "game_date", "game_pk"])

    out = starts
    for col in ("fb_velo", "br_ivb", "br_hb"):
        out = out.with_columns(
            # shift(1) excludes the current start, so both windows are strictly historical
            pl.col(col).shift(1).rolling_mean(RECENT, min_samples=RECENT)
            .over("pitcher_id").alias(f"{col}_recent"),
            pl.col(col).shift(1 + RECENT).rolling_mean(BASELINE, min_samples=3)
            .over("pitcher_id").alias(f"{col}_base"),
        )
        out = out.with_columns(
            (pl.col(f"{col}_recent") - pl.col(f"{col}_base")).alias(f"{col}_delta")
        )
    return out


def main() -> None:
    starts = load_start_aggregates()
    print(f"starter-game rows: {len(starts):,}")

    feats = trend_features(starts)
    keep = ["game_pk", "pitcher_id", "pitched_top",
            "fb_velo_delta", "br_ivb_delta", "br_hb_delta"]
    feats = feats.select(keep).drop_nulls(["fb_velo_delta"])
    print(f"rows with a computable velocity trend: {len(feats):,}")

    # pitched_top True means the pitcher worked the top half, i.e. he is the HOME starter.
    home = feats.filter(pl.col("pitched_top")).drop("pitched_top", "pitcher_id")
    away = feats.filter(~pl.col("pitched_top")).drop("pitched_top", "pitcher_id")
    joined = home.join(away, on="game_pk", how="inner", suffix="_away")

    frame = joined.with_columns(
        (pl.col("fb_velo_delta") - pl.col("fb_velo_delta_away")).alias("fb_velo_edge"),
        (pl.col("br_ivb_delta") - pl.col("br_ivb_delta_away")).alias("br_ivb_edge"),
        (pl.col("br_hb_delta") - pl.col("br_hb_delta_away")).alias("br_hb_edge"),
    ).select("game_pk", "fb_velo_edge", "br_ivb_edge", "br_hb_edge")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(OUT)
    print(f"games with both starters' trends: {len(frame):,} -> {OUT}")
    print()
    print(frame.describe())


if __name__ == "__main__":
    main()
