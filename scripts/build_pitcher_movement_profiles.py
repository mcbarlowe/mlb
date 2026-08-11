"""Build leak-free pitcher movement profiles from mlb.pitches.

For every (pitcher, game) appearance, computes the trailing per-pitch-type
usage share and mean velocity / horizontal movement / vertical movement /
spin rate over the pitcher's previous appearances (strictly before the game,
so the store can never leak same-game information into features).

Outputs under --output-dir:
    pitcher_movement_profiles.parquet        long: one row per
                                             (pitcher, game, pitch type)
    pitcher_movement_profiles_wide.parquet   one row per (pitcher, game)
    league_default_profiles.json             fallbacks for pitchers with
                                             no measured history
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import polars as pl

from src.database import PostgresHandler
from src.ml.features import PITCH_TYPE_CODES
from src.ml.movement_profiles import (
    DEFAULT_WINDOW_GAMES,
    compute_trailing_profiles,
    league_default_profiles,
    pivot_profiles_wide,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build trailing pitcher movement profiles."
    )
    parser.add_argument("--output-dir", type=str, default="data/profiles")
    parser.add_argument(
        "--window-games",
        type=int,
        default=DEFAULT_WINDOW_GAMES,
        help="Trailing window in pitcher appearances",
    )
    parser.add_argument(
        "--min-season",
        type=int,
        default=None,
        help="Earliest season to include (default: all)",
    )
    return parser.parse_args()


def load_per_game_stats(min_season: int | None) -> pl.DataFrame:
    canonical = ", ".join(f"'{code}'" for code in PITCH_TYPE_CODES)
    season_filter = (
        f"AND p.season >= {int(min_season)}" if min_season is not None else ""
    )
    query = f"""
        SELECT
            p.pitcher_id,
            p.game_pk,
            g.game_date,
            CASE
                WHEN p.pitch_type_code IN ({canonical}) THEN p.pitch_type_code
                ELSE 'OTHER'
            END AS pitch_type_code,
            COUNT(*) AS n,
            AVG(p.pitch_start_speed) AS velo,
            AVG(p.pfxx) AS pfx_x,
            AVG(p.pfxz) AS pfx_z,
            AVG(p.spin_rate) AS spin_rate
        FROM mlb.pitches p
        JOIN mlb.games g USING (game_pk)
        WHERE p.pitcher_id IS NOT NULL
          AND p.pitch_type_code IS NOT NULL
          {season_filter}
        GROUP BY 1, 2, 3, 4
    """
    with PostgresHandler() as db:
        frame = db.query(query)
    return pl.from_pandas(frame, nan_to_null=True).with_columns(
        pl.col("pitcher_id").cast(pl.Int64),
        pl.col("game_pk").cast(pl.Int64),
        pl.col("game_date").cast(pl.String),
        pl.col("n").cast(pl.Int64),
        *[
            pl.col(stat).cast(pl.Float64)
            for stat in ("velo", "pfx_x", "pfx_z", "spin_rate")
        ],
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print("Loading per-game pitch stats from Postgres...")
    per_game = load_per_game_stats(args.min_season)
    print(
        f"  {len(per_game):,} (pitcher, game, type) rows | "
        f"{per_game['pitcher_id'].n_unique():,} pitchers | "
        f"{per_game['game_pk'].n_unique():,} games"
    )

    print(f"Computing trailing profiles (window={args.window_games} games)...")
    trailing = compute_trailing_profiles(per_game, window_games=args.window_games)

    long_path = output_dir / "pitcher_movement_profiles.parquet"
    trailing.write_parquet(long_path)
    print(f"  Wrote {len(trailing):,} rows -> {long_path}")

    wide = pivot_profiles_wide(trailing)
    wide_path = output_dir / "pitcher_movement_profiles_wide.parquet"
    wide.write_parquet(wide_path)
    print(f"  Wrote {len(wide):,} rows x {len(wide.columns)} cols -> {wide_path}")

    defaults = league_default_profiles(trailing)
    defaults_path = output_dir / "league_default_profiles.json"
    defaults_path.write_text(json.dumps(defaults, indent=2, sort_keys=True))
    print(f"  Wrote {len(defaults)} league defaults -> {defaults_path}")

    print(f"Done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
