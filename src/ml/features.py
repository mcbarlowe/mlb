"""
Feature engineering for pitch prediction models.

This module transforms raw pitch data into features suitable for ML models.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl
import torch

# Pitch type codes and their indices for model encoding
PITCH_TYPE_CODES = [
    "FF",  # Four-Seam Fastball
    "SI",  # Sinker
    "FC",  # Cutter
    "CH",  # Changeup
    "SL",  # Slider
    "CU",  # Curveball
    "KC",  # Knuckle Curve
    "ST",  # Sweeper
    "FS",  # Splitter
    "KN",  # Knuckleball
    "OTHER",  # Catch-all for rare pitch types
]

PITCH_TYPE_TO_IDX = {code: idx for idx, code in enumerate(PITCH_TYPE_CODES)}
IDX_TO_PITCH_TYPE = {idx: code for code, idx in PITCH_TYPE_TO_IDX.items()}


def parse_count(count_str: str) -> tuple[int, int]:
    """Parse count string like '1-2' into (balls, strikes)."""
    if not count_str or "-" not in count_str:
        return 0, 0
    parts = count_str.split("-")
    return int(parts[0]), int(parts[1])


class PitchFeatureEngine:
    """
    Feature engineering for pitch prediction.

    Transforms raw pitch data into model-ready features including:
    - Game state features (count, outs, runners, score)
    - Sequence features (previous pitches in at-bat)
    - Categorical encodings for pitcher/batter IDs
    - Pitcher tendency features (repertoire, fastball percentage)
    - At-bat cumulative features
    - Pitch family × handedness interaction features
    """

    # Fastball pitch types for cumulative counting
    FASTBALL_TYPES: ClassVar[list[str]] = ["FF", "SI", "FC", "FA", "FT"]

    # Pitch family definitions for interaction features
    OFFSPEED_TYPES: ClassVar[list[str]] = ["CH", "FS"]
    BREAKING_TYPES: ClassVar[list[str]] = ["SL", "CU", "KC", "ST", "SV"]  # SV = slurve

    def __init__(
        self,
        data_path: Path | str | None = None,
        movement_profiles_dir: Path | str | None = None,
    ):
        """
        Initialize the feature engine.

        Args:
            data_path: Path to parquet files or the string ``"postgres"``.
            movement_profiles_dir: Optional directory produced by
                ``scripts/build_pitcher_movement_profiles.py``. When set,
                ``transform`` attaches trailing pitcher movement profile
                features and ``get_feature_columns`` includes them.
        """
        raw_data_path = data_path or Path("data/processed/livefeeds")
        self.use_postgres = str(raw_data_path) == "postgres"
        self.data_path = None if self.use_postgres else Path(raw_data_path)
        self.movement_profiles_dir = (
            Path(movement_profiles_dir) if movement_profiles_dir else None
        )
        self._movement_wide: pl.DataFrame | None = None
        self._movement_defaults: dict[str, float] | None = None
        self.pitcher_to_idx: dict[int, int] = {}
        self.batter_to_idx: dict[int, int] = {}
        self.pitcher_ff_pct: dict[int, float] = {}  # Pitcher fastball percentage
        self.pitcher_repertoire_size: dict[int, int] = {}  # Number of pitch types
        self._fitted = False

    def load_data(
        self,
        seasons: list[str] | None = None,
        sample_frac: float | None = None,
    ) -> pl.DataFrame:
        """
        Load pitch data from parquet files or PostgreSQL.

        Args:
            seasons: List of seasons to load (e.g., ["2023", "2024"]).
                    If None, loads all available seasons.
            sample_frac: Optional fraction to sample for faster iteration.

        Returns:
            Polars DataFrame with pitch data.
        """
        if self.use_postgres:
            from src.ml.postgres_data import load_pitches_from_postgres

            df = load_pitches_from_postgres(seasons=seasons)
        elif seasons:
            assert self.data_path is not None
            patterns = [self.data_path / season / "*.parquet" for season in seasons]
            dfs = []
            for pattern in patterns:
                if pattern.parent.exists():
                    dfs.append(pl.scan_parquet(str(pattern)))
            if not dfs:
                raise ValueError(f"No data found for seasons: {seasons}")
            df = pl.concat(dfs).collect()
        else:
            assert self.data_path is not None
            df = pl.scan_parquet(str(self.data_path / "**/*.parquet")).collect()

        if sample_frac and sample_frac < 1.0:
            df = df.sample(fraction=sample_frac, seed=42)

        return df

    def fit(self, df: pl.DataFrame) -> "PitchFeatureEngine":
        """
        Fit the feature engine to learn pitcher/batter mappings and tendencies.

        Args:
            df: DataFrame with pitcher_id, batter_id, and pitch_type_code columns.

        Returns:
            self for method chaining.
        """
        return self.fit_frames([df])

    def fit_frames(self, frames: Iterable[pl.DataFrame]) -> "PitchFeatureEngine":
        """Fit the feature engine from a stream of season-sized frames."""
        unique_pitchers: set[int] = set()
        unique_batters: set[int] = set()
        pitcher_totals: defaultdict[int, int] = defaultdict(int)
        pitcher_fastballs: defaultdict[int, int] = defaultdict(int)
        pitcher_pitch_types: defaultdict[int, set[str]] = defaultdict(set)

        print("    Computing pitcher tendencies...")
        for df in frames:
            if df.is_empty():
                continue

            unique_pitchers.update(int(pid) for pid in df["pitcher_id"].drop_nulls().unique().to_list())
            unique_batters.update(int(bid) for bid in df["batter_id"].drop_nulls().unique().to_list())

            pitcher_stats = df.group_by("pitcher_id").agg([
                pl.len().alias("total_pitches"),
                pl.col("pitch_type_code").filter(
                    pl.col("pitch_type_code").is_in(self.FASTBALL_TYPES)
                ).len().alias("fastball_count"),
                pl.col("pitch_type_code").drop_nulls().unique().alias("pitch_types"),
            ])

            for row in pitcher_stats.iter_rows(named=True):
                pitcher_id = row["pitcher_id"]
                if pitcher_id is None:
                    continue

                pid = int(pitcher_id)
                pitcher_totals[pid] += int(row["total_pitches"])
                pitcher_fastballs[pid] += int(row["fastball_count"])
                pitch_types = row["pitch_types"] or []
                pitcher_pitch_types[pid].update(
                    str(code) for code in pitch_types if code is not None
                )

        self.pitcher_to_idx = {
            pid: idx for idx, pid in enumerate(sorted(unique_pitchers))
        }
        self.batter_to_idx = {
            bid: idx for idx, bid in enumerate(sorted(unique_batters))
        }
        self.pitcher_ff_pct = {
            pid: (
                pitcher_fastballs[pid] / total if total > 0 else 0.5
            )
            for pid, total in pitcher_totals.items()
        }
        self.pitcher_repertoire_size = {
            pid: min(len(pitch_types), 10)
            for pid, pitch_types in pitcher_pitch_types.items()
        }
        self._fitted = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Transform raw pitch data into model features.

        Args:
            df: Raw pitch DataFrame.

        Returns:
            DataFrame with engineered features.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")

        # Parse count into balls and strikes
        df = df.with_columns([
            pl.col("count_after_pitch")
            .map_elements(lambda x: parse_count(x)[0], return_dtype=pl.Int64)
            .alias("balls"),
            pl.col("count_after_pitch")
            .map_elements(lambda x: parse_count(x)[1], return_dtype=pl.Int64)
            .alias("strikes"),
        ])

        # Count-based features for location prediction
        df = df.with_columns([
            # Two-strike count (0-2, 1-2, 2-2, 3-2) - favors off-plate locations
            (pl.col("strikes") == 2).cast(pl.Int64).alias("two_strike_count"),
            # Hitter's count (2-0, 3-0, 3-1) - favors in-zone locations
            ((pl.col("balls") >= 2) & (pl.col("strikes") <= 1)).cast(pl.Int64).alias("hitters_count"),
            # First pitch - often predictable location patterns
            ((pl.col("balls") == 0) & (pl.col("strikes") == 0)).cast(pl.Int64).alias("first_pitch"),
            # Ahead in count (more strikes than balls)
            (pl.col("strikes") > pl.col("balls")).cast(pl.Int64).alias("pitcher_ahead"),
        ])

        # Encode pitcher and batter IDs
        df = df.with_columns([
            pl.col("pitcher_id")
            .map_elements(
                lambda x: self.pitcher_to_idx.get(x, len(self.pitcher_to_idx)),
                return_dtype=pl.Int64,
            )
            .alias("pitcher_idx"),
            pl.col("batter_id")
            .map_elements(
                lambda x: self.batter_to_idx.get(x, len(self.batter_to_idx)),
                return_dtype=pl.Int64,
            )
            .alias("batter_idx"),
        ])

        # Encode handedness (L=0, R=1)
        df = df.with_columns([
            pl.when(pl.col("throw_side") == "R")
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("throw_side_enc"),
            pl.when(pl.col("bat_side") == "R")
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("bat_side_enc"),
        ])

        # Platoon matchup: 1 = same-side (RHP vs RHB or LHP vs LHB), 0 = opposite
        df = df.with_columns([
            pl.when(pl.col("throw_side") == pl.col("bat_side"))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("platoon_same_side"),
        ])

        # Create runner bitmap (3 bits: 1B, 2B, 3B)
        df = df.with_columns([
            (
                pl.col("is_runner_on_first").cast(pl.Int64) * 1
                + pl.col("is_runner_on_second").cast(pl.Int64) * 2
                + pl.col("is_runner_on_third").cast(pl.Int64) * 4
            ).alias("runners_bitmap"),
        ])

        # Calculate score differential (positive = batting team ahead)
        df = df.with_columns([
            pl.when(pl.col("half_inning") == "top")
            .then(pl.col("away_score") - pl.col("home_score"))
            .otherwise(pl.col("home_score") - pl.col("away_score"))
            .alias("score_diff"),
        ])

        # =====================================================================
        # WEATHER FEATURES
        # =====================================================================
        # Temperature normalized around 70°F (typical game temp)
        df = df.with_columns([
            pl.when(pl.col("weather_temp").is_not_null())
            .then((pl.col("weather_temp") - 70.0) / 30.0)
            .otherwise(pl.lit(0.0))
            .alias("temp_normalized"),
        ])

        # Parse wind speed from strings like "8 mph, R To L"
        df = df.with_columns([
            pl.when(pl.col("weather_wind").is_not_null())
            .then(
                pl.col("weather_wind")
                .str.extract(r"(\d+)")
                .cast(pl.Float64)
                .fill_null(0.0) / 20.0  # Normalize by typical max wind
            )
            .otherwise(pl.lit(0.0))
            .alias("wind_speed"),
            # Wind direction: 1 if blowing out (L to R from pitcher), -1 if in, 0 otherwise
            pl.when(pl.col("weather_wind").str.contains("L To R"))
            .then(pl.lit(1))
            .when(pl.col("weather_wind").str.contains("R To L"))
            .then(pl.lit(-1))
            .when(pl.col("weather_wind").str.contains("Out"))
            .then(pl.lit(1))
            .when(pl.col("weather_wind").str.contains("In"))
            .then(pl.lit(-1))
            .otherwise(pl.lit(0))
            .alias("wind_direction"),
        ])

        # Day/night indicator
        df = df.with_columns([
            pl.when(pl.col("day_night") == "night")
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("is_night_game"),
        ])

        # =====================================================================
        # TEMPORAL FEATURES
        # =====================================================================
        # Extract month from game_date (handles ISO format like "2025-04-15T...")
        df = df.with_columns([
            pl.when(pl.col("game_date").is_not_null())
            .then(
                pl.col("game_date")
                .str.slice(5, 2)
                .cast(pl.Int64)
                .fill_null(7)  # Default to July
            )
            .otherwise(pl.lit(7))
            .alias("month"),
        ])

        # Normalize month (April=4 to October=10 -> 0 to 1 scale)
        df = df.with_columns([
            ((pl.col("month") - 4.0) / 6.0).clip(0.0, 1.0).alias("season_progress"),
        ])

        # =====================================================================
        # ENHANCED GAME SITUATION FEATURES
        # =====================================================================
        # Runners in scoring position (2nd or 3rd base)
        df = df.with_columns([
            (
                pl.col("is_runner_on_second").cast(pl.Int64) |
                pl.col("is_runner_on_third").cast(pl.Int64)
            ).alias("runners_in_scoring_position"),
        ])

        # Simple leverage approximation based on game situation
        # Higher leverage = close game + late inning + runners on
        df = df.with_columns([
            (
                # Close game factor (1 if within 3 runs, scaled down otherwise)
                pl.when(pl.col("score_diff").abs() <= 3)
                .then(pl.lit(1.0))
                .otherwise(pl.lit(0.5))
                *
                # Late inning factor (higher in innings 7+)
                pl.when(pl.col("inning") >= 7)
                .then(pl.lit(1.5))
                .when(pl.col("inning") >= 5)
                .then(pl.lit(1.0))
                .otherwise(pl.lit(0.7))
                *
                # Runners factor
                (1.0 + pl.col("runners_bitmap").cast(pl.Float64) * 0.1)
            ).alias("leverage_approx"),
        ])

        # Encode pitch type to index
        df = df.with_columns([
            pl.col("pitch_type_code")
            .map_elements(
                lambda x: PITCH_TYPE_TO_IDX.get(x, PITCH_TYPE_TO_IDX["OTHER"]),
                return_dtype=pl.Int64,
            )
            .alias("pitch_type_idx"),
        ])

        # =====================================================================
        # PITCHER TENDENCY FEATURES
        # =====================================================================
        # Pitcher fastball percentage (0-1 scale)
        df = df.with_columns([
            pl.col("pitcher_id")
            .map_elements(
                lambda x: self.pitcher_ff_pct.get(x, 0.5),
                return_dtype=pl.Float64,
            )
            .alias("pitcher_ff_pct"),
            # Pitcher repertoire size (normalized: divide by 7 typical max)
            pl.col("pitcher_id")
            .map_elements(
                lambda x: self.pitcher_repertoire_size.get(x, 4) / 7.0,
                return_dtype=pl.Float64,
            )
            .alias("pitcher_repertoire"),
        ])

        # =====================================================================
        # BATTER-SPECIFIC STRIKE ZONE
        # =====================================================================
        # Use personalized strike zone dimensions (normalized around typical values)
        df = df.with_columns([
            # Zone height (top - bottom), normalized around typical 2.0 ft
            (
                (pl.col("pitch_strike_zone_top").fill_null(3.5) -
                 pl.col("pitch_strike_zone_bottom").fill_null(1.5)) / 2.0
            ).alias("batter_zone_height"),
            # Zone midpoint (vertical center of zone)
            (
                (pl.col("pitch_strike_zone_top").fill_null(3.5) +
                 pl.col("pitch_strike_zone_bottom").fill_null(1.5)) / 2.0 / 2.5
            ).alias("batter_zone_mid"),
        ])

        # =====================================================================
        # PITCHER FATIGUE (cumulative pitch count in game)
        # =====================================================================
        # Sort first to ensure correct cumulative count
        df = df.sort(["game_pk", "at_bat_index", "pitch_number"])

        # Add row number within pitcher's game appearance for pitch count
        df = df.with_row_index("_row_idx")

        # Count pitches thrown by each pitcher in the game
        # Use row_number() which is more reliable than cum_sum(1)
        df = df.with_columns([
            (pl.col("_row_idx") - pl.col("_row_idx").first().over(["game_pk", "pitcher_id"]))
            .alias("pitcher_pitch_count_raw"),
        ])

        # Normalize pitcher pitch count (typical starter throws ~100 pitches)
        df = df.with_columns([
            (pl.col("pitcher_pitch_count_raw").cast(pl.Float64) / 100.0).alias("pitcher_pitch_count"),
        ])

        # Clean up temporary column
        df = df.drop("_row_idx")

        # =====================================================================
        # SORT AND COMPUTE SEQUENCE FEATURES
        # =====================================================================
        df = df.sort(["game_pk", "at_bat_index", "pitch_number"])

        # Mark fastballs for cumulative counting
        df = df.with_columns([
            pl.col("pitch_type_code")
            .is_in(self.FASTBALL_TYPES)
            .cast(pl.Int64)
            .alias("is_fastball"),
        ])

        # Previous pitch features within each at-bat
        df = df.with_columns([
            # Previous pitch type
            pl.col("pitch_type_idx")
            .shift(1)
            .over(["game_pk", "at_bat_index"])
            .fill_null(-1)
            .alias("prev_pitch_type_idx"),
            # Previous location
            pl.col("px")
            .shift(1)
            .over(["game_pk", "at_bat_index"])
            .fill_null(0.0)
            .alias("prev_px"),
            pl.col("pz")
            .shift(1)
            .over(["game_pk", "at_bat_index"])
            .fill_null(2.5)
            .alias("prev_pz"),
            # Previous velocity
            pl.col("pitch_start_speed")
            .shift(1)
            .over(["game_pk", "at_bat_index"])
            .fill_null(90.0)
            .alias("prev_velocity"),
            # Previous strike indicator
            pl.col("is_strike")
            .shift(1)
            .over(["game_pk", "at_bat_index"])
            .fill_null(False)
            .cast(pl.Int64)
            .alias("prev_is_strike"),
            # Previous pitch code (for streak calculation)
            pl.col("pitch_type_code")
            .shift(1)
            .over(["game_pk", "at_bat_index"])
            .alias("prev_pitch_code"),
        ])

        # =====================================================================
        # VELOCITY FEATURES
        # =====================================================================
        df = df.with_columns([
            # Velocity delta (current - previous), normalized
            (
                (pl.col("pitch_start_speed").fill_null(90.0) - pl.col("prev_velocity")) / 10.0
            ).alias("velocity_delta"),
        ])

        # =====================================================================
        # SWING/RESULT FEATURES
        # =====================================================================
        # Determine if previous pitch had a swing (using description field)
        # Common swing descriptions: "swinging_strike", "foul", "hit_into_play", etc.
        df = df.with_columns([
            pl.when(
                pl.col("description").is_in([
                    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
                    "foul_bunt", "hit_into_play", "hit_into_play_no_out",
                    "hit_into_play_score", "missed_bunt"
                ])
            ).then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("is_swing"),
            # Encode result type: 0=ball, 1=called_strike, 2=swinging_strike, 3=foul, 4=in_play
            pl.when(pl.col("is_ball") == True).then(pl.lit(0))
            .when(pl.col("description") == "called_strike").then(pl.lit(1))
            .when(pl.col("description").is_in(["swinging_strike", "swinging_strike_blocked"])).then(pl.lit(2))
            .when(pl.col("description").is_in(["foul", "foul_tip", "foul_bunt"])).then(pl.lit(3))
            .when(pl.col("is_in_play") == True).then(pl.lit(4))
            .otherwise(pl.lit(0))
            .alias("result_type"),
        ])

        # Previous swing and result
        df = df.with_columns([
            pl.col("is_swing")
            .shift(1)
            .over(["game_pk", "at_bat_index"])
            .fill_null(0)
            .alias("prev_swing"),
            pl.col("result_type")
            .shift(1)
            .over(["game_pk", "at_bat_index"])
            .fill_null(0)
            .alias("prev_result_type"),
        ])

        # =====================================================================
        # CUMULATIVE AT-BAT FEATURES
        # =====================================================================
        df = df.with_columns([
            # Cumulative fastballs in at-bat (before current pitch)
            pl.col("is_fastball")
            .shift(1)
            .cum_sum()
            .over(["game_pk", "at_bat_index"])
            .fill_null(0)
            .alias("n_fastballs_in_ab"),
            # Cumulative breaking balls (non-fastballs) in at-bat
            (1 - pl.col("is_fastball"))
            .shift(1)
            .cum_sum()
            .over(["game_pk", "at_bat_index"])
            .fill_null(0)
            .alias("n_breaking_in_ab"),
        ])

        # =====================================================================
        # SAME PITCH STREAK
        # =====================================================================
        # Count consecutive same pitch types
        df = df.with_columns([
            pl.when(pl.col("prev_pitch_code") == pl.col("pitch_type_code"))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("same_as_prev"),
        ])

        # Compute streak using cumulative sum with reset
        # This counts how many consecutive same pitches before current
        df = df.with_columns([
            pl.col("same_as_prev")
            .cum_sum()
            .over(["game_pk", "at_bat_index"])
            .alias("same_pitch_cumsum"),
        ])

        # For streak, we need to track when pitch type changes
        # Simplified: use previous match indicator (0 or 1)
        df = df.with_columns([
            pl.col("same_as_prev")
            .shift(1)
            .over(["game_pk", "at_bat_index"])
            .fill_null(0)
            .alias("same_pitch_streak"),
        ])

        # Rename prev_velocity to prev_speed for compatibility
        df = df.with_columns([
            pl.col("prev_velocity").alias("prev_speed"),
        ])

        # =====================================================================
        # PITCH FAMILY × HANDEDNESS INTERACTION FEATURES
        # =====================================================================
        # These capture the strong interaction between pitch type and batter handedness
        # for location prediction (0.4-0.7 ft horizontal shift by handedness)

        # Classify previous pitch into families
        df = df.with_columns([
            pl.col("prev_pitch_code")
            .is_in(self.FASTBALL_TYPES)
            .fill_null(False)
            .cast(pl.Int64)
            .alias("prev_is_fastball"),
            pl.col("prev_pitch_code")
            .is_in(self.OFFSPEED_TYPES)
            .fill_null(False)
            .cast(pl.Int64)
            .alias("prev_is_offspeed"),
            pl.col("prev_pitch_code")
            .is_in(self.BREAKING_TYPES)
            .fill_null(False)
            .cast(pl.Int64)
            .alias("prev_is_breaking"),
        ])

        # Create pitch family × batter handedness interactions
        # These help the model learn that breaking balls to LHB go to different
        # locations than breaking balls to RHB
        df = df.with_columns([
            # Fastball × RHB interaction
            (pl.col("prev_is_fastball") * pl.col("bat_side_enc")).alias("prev_fb_x_rhb"),
            # Offspeed × RHB interaction
            (pl.col("prev_is_offspeed") * pl.col("bat_side_enc")).alias("prev_off_x_rhb"),
            # Breaking × RHB interaction
            (pl.col("prev_is_breaking") * pl.col("bat_side_enc")).alias("prev_brk_x_rhb"),
        ])

        # Count × handedness interactions
        # Pitchers attack differently based on count and batter hand
        df = df.with_columns([
            # Pitcher ahead × RHB (more chase pitches to RHB when ahead)
            (pl.col("pitcher_ahead") * pl.col("bat_side_enc")).alias("ahead_x_rhb"),
            # Two-strike × RHB (more off-plate pitches)
            (pl.col("two_strike_count") * pl.col("bat_side_enc")).alias("two_strike_x_rhb"),
            # Hitter's count × RHB (more in-zone pitches)
            (pl.col("hitters_count") * pl.col("bat_side_enc")).alias("hitters_x_rhb"),
        ])

        # Platoon × pitch family interactions
        # Same-side matchups may favor certain pitch locations
        df = df.with_columns([
            (pl.col("platoon_same_side") * pl.col("prev_is_breaking")).alias("platoon_x_breaking"),
        ])

        if self.movement_profiles_dir is not None:
            df = self._attach_movement_profiles(df)

        return df

    def _attach_movement_profiles(self, df: pl.DataFrame) -> pl.DataFrame:
        """Attach trailing movement profile features (lazy-loads the store)."""
        import json as _json

        from src.ml.movement_profiles import attach_movement_profiles

        assert self.movement_profiles_dir is not None
        if self._movement_wide is None:
            self._movement_wide = pl.read_parquet(
                self.movement_profiles_dir / "pitcher_movement_profiles_wide.parquet"
            )
            defaults_path = (
                self.movement_profiles_dir / "league_default_profiles.json"
            )
            self._movement_defaults = (
                _json.loads(defaults_path.read_text())
                if defaults_path.exists()
                else {}
            )
        assert self._movement_defaults is not None
        return attach_movement_profiles(
            df, self._movement_wide, self._movement_defaults
        )

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def get_feature_columns(self) -> list[str]:
        """Return list of feature column names for model input."""
        columns = [
            # Count features
            "balls",
            "strikes",
            # Count-based location features
            "two_strike_count",
            "hitters_count",
            "first_pitch",
            "pitcher_ahead",
            # Game state
            "inning",
            "outs",
            "runners_bitmap",
            "score_diff",
            # Pitcher/batter (will be embedded)
            "pitcher_idx",
            "batter_idx",
            # Handedness and platoon
            "throw_side_enc",
            "bat_side_enc",
            "platoon_same_side",
            # Pitcher tendencies
            "pitcher_ff_pct",
            "pitcher_repertoire",
            # Batter zone
            "batter_zone_height",
            "batter_zone_mid",
            # Previous pitch (embedded)
            "prev_pitch_type_idx",
            # Previous pitch features
            "prev_px",
            "prev_pz",
            "prev_speed",
            "prev_is_strike",
            # Velocity features
            "velocity_delta",
            # Swing/result features
            "prev_swing",
            "prev_result_type",
            # Cumulative at-bat features
            "n_fastballs_in_ab",
            "n_breaking_in_ab",
            # Sequence features
            "same_pitch_streak",
            "pitch_number",
            # Weather features
            "temp_normalized",
            "wind_speed",
            "wind_direction",
            "is_night_game",
            # Temporal features
            "season_progress",
            # Enhanced game situation
            "runners_in_scoring_position",
            "leverage_approx",
            # Pitcher fatigue
            "pitcher_pitch_count",
            # Pitch family features (for interaction modeling)
            "prev_is_fastball",
            "prev_is_offspeed",
            "prev_is_breaking",
            # Pitch family × handedness interactions
            "prev_fb_x_rhb",
            "prev_off_x_rhb",
            "prev_brk_x_rhb",
            # Count × handedness interactions
            "ahead_x_rhb",
            "two_strike_x_rhb",
            "hitters_x_rhb",
            # Platoon × pitch family interaction
            "platoon_x_breaking",
        ]
        if self.movement_profiles_dir is not None:
            from src.ml.movement_profiles import movement_profile_columns

            columns.extend(movement_profile_columns())
        return columns

    def get_target_columns(self) -> list[str]:
        """Return list of target column names."""
        return ["pitch_type_idx", "px", "pz"]

    def get_feature_indices(self) -> dict:
        """
        Return indices for embedding and continuous features.

        This allows the model to dynamically extract features without
        hardcoding indices. If the feature list changes, the model
        automatically adapts.

        Returns:
            Dict with:
            - embedding_indices: dict mapping embedding name to feature index
            - continuous_indices: list of indices for continuous features
        """
        feature_cols = self.get_feature_columns()

        # Features that get embedded (not used as continuous inputs)
        embedding_features = ["pitcher_idx", "batter_idx", "prev_pitch_type_idx"]

        embedding_indices = {}
        continuous_indices = []

        for i, col in enumerate(feature_cols):
            if col in embedding_features:
                embedding_indices[col] = i
            else:
                continuous_indices.append(i)

        return {
            "embedding_indices": embedding_indices,
            "continuous_indices": continuous_indices,
        }

    @property
    def n_pitchers(self) -> int:
        """Number of unique pitchers (for embedding size)."""
        return len(self.pitcher_to_idx) + 1  # +1 for unknown

    @property
    def n_batters(self) -> int:
        """Number of unique batters (for embedding size)."""
        return len(self.batter_to_idx) + 1  # +1 for unknown

    @property
    def n_pitch_types(self) -> int:
        """Number of pitch type classes."""
        return len(PITCH_TYPE_CODES)

    def save(self, path: Path) -> None:
        """
        Save the fitted feature engine to a JSON file.

        Args:
            path: Path to save the JSON file.
        """
        import json

        if not self._fitted:
            raise RuntimeError("Cannot save unfitted feature engine")

        # Convert integer keys to strings for JSON serialization
        data = {
            "pitcher_to_idx": {str(k): v for k, v in self.pitcher_to_idx.items()},
            "batter_to_idx": {str(k): v for k, v in self.batter_to_idx.items()},
            "pitcher_ff_pct": {str(k): v for k, v in self.pitcher_ff_pct.items()},
            "pitcher_repertoire_size": {str(k): v for k, v in self.pitcher_repertoire_size.items()},
            "movement_profiles_dir": (
                str(self.movement_profiles_dir)
                if self.movement_profiles_dir is not None
                else None
            ),
        }

        path = Path(path)
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "PitchFeatureEngine":
        """
        Load a fitted feature engine from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Fitted PitchFeatureEngine instance.
        """
        import json

        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        engine = cls(
            movement_profiles_dir=data.get("movement_profiles_dir") or None
        )
        # Convert string keys back to integers
        engine.pitcher_to_idx = {int(k): v for k, v in data["pitcher_to_idx"].items()}
        engine.batter_to_idx = {int(k): v for k, v in data["batter_to_idx"].items()}
        engine.pitcher_ff_pct = {int(k): v for k, v in data["pitcher_ff_pct"].items()}
        engine.pitcher_repertoire_size = {int(k): v for k, v in data["pitcher_repertoire_size"].items()}
        engine._fitted = True

        return engine


def compute_class_weights_from_counts(
    count_dict: Mapping[int, int],
    n_classes: int | None = None,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """Compute class weights from pre-aggregated pitch-type counts."""
    n_classes = n_classes or len(PITCH_TYPE_CODES)
    total = sum(count_dict.values())
    if total <= 0:
        return torch.ones(n_classes, dtype=torch.float32)

    weights = []
    for i in range(n_classes):
        count = count_dict.get(i, 1)
        weight = (total / (n_classes * count)) ** smoothing
        weights.append(weight)

    normalized = np.array(weights, dtype=np.float32)
    normalized = normalized / normalized.mean()
    return torch.tensor(normalized, dtype=torch.float32)

def compute_class_weights(
    df: pl.DataFrame,
    pitch_type_col: str = "pitch_type_idx",
    n_classes: int | None = None,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """
    Compute inverse frequency class weights for imbalanced pitch types.

    Uses smoothed inverse frequency: weight_i = (N / (n_classes * count_i))^smoothing

    Args:
        df: DataFrame with pitch type indices.
        pitch_type_col: Column name for pitch type index.
        n_classes: Number of pitch type classes. If None, uses len(PITCH_TYPE_CODES).
        smoothing: Smoothing factor (0=uniform, 1=full inverse frequency).
                   Default 0.1 provides mild correction without over-weighting rare classes.

    Returns:
        Tensor of class weights for use with CrossEntropyLoss.
    """
    counts = df.group_by(pitch_type_col).agg(pl.len().alias("count"))
    count_dict = {
        int(row[pitch_type_col]): int(row["count"])
        for row in counts.to_dicts()
    }
    return compute_class_weights_from_counts(
        count_dict,
        n_classes=n_classes,
        smoothing=smoothing,
    )
