"""Profile stores for live outcome inference.

Training computes rolling pitcher/batter features leak-free from history.
Live inference needs the *current* value of those same features, so we
export each entity's most recent profile row to parquet and look it up.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

PITCHER_PROFILE_COLUMNS = [
    "profile_speed_delta",
    "profile_spin_delta",
    "profile_ivb_delta",
    "profile_hb_delta",
    "profile_release_x",
    "profile_release_z",
    "pitcher_whiff_rate",
    "pitcher_hr_rate",
]

BATTER_PRIOR_COLUMNS = [
    "batter_swing_rate",
    "batter_whiff_rate",
    "batter_chase_rate",
    "batter_hr_rate",
    "batter_xbh_rate",
    "batter_hit_rate",
]

PITCHER_PROFILES_FILE = "pitcher_profiles.parquet"
BATTER_PRIORS_FILE = "batter_priors.parquet"

# Synthetic pitcher id used by the game simulator's generic bullpen arm.
# It gets league-median profile values so the outcome models see a plausible
# "average arm" instead of all-null features (CatBoost's missing-value
# routing made the null arm systematically soft).
LEAGUE_PITCHER_ID = 0


def build_profile_stores(
    frame: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Latest profile per pitcher × pitch type and priors per batter.

    ``frame`` is a full training frame (``build_training_frame`` output),
    chronologically sorted; the last row per group carries the most
    up-to-date expanding values.
    """
    pitcher_profiles = (
        frame.filter(pl.col("pitch_type").is_not_null())
        .group_by(["pitcher_id", "pitch_type"], maintain_order=True)
        .agg(pl.col(column).last() for column in PITCHER_PROFILE_COLUMNS)
    )
    league_rows = (
        pitcher_profiles.group_by("pitch_type")
        .agg(pl.col(column).median() for column in PITCHER_PROFILE_COLUMNS)
        .with_columns(pl.lit(LEAGUE_PITCHER_ID, dtype=pl.Int64).alias("pitcher_id"))
        .select(pitcher_profiles.columns)
    )
    pitcher_profiles = pl.concat([pitcher_profiles, league_rows])
    batter_priors = (
        frame.group_by("batter_id", maintain_order=True)
        .agg(pl.col(column).last() for column in BATTER_PRIOR_COLUMNS)
    )
    return pitcher_profiles, batter_priors


def save_profile_stores(
    pitcher_profiles: pl.DataFrame,
    batter_priors: pl.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pitcher_profiles.write_parquet(output_dir / PITCHER_PROFILES_FILE)
    batter_priors.write_parquet(output_dir / BATTER_PRIORS_FILE)


def load_profile_stores(directory: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    return (
        pl.read_parquet(directory / PITCHER_PROFILES_FILE),
        pl.read_parquet(directory / BATTER_PRIORS_FILE),
    )
