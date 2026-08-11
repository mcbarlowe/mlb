"""Tests for leak-free trailing pitcher movement profiles."""

import polars as pl

from src.ml.movement_profiles import (
    compute_trailing_profiles,
    league_default_profiles,
    pivot_profiles_wide,
    profile_column_name,
)


def _per_game() -> pl.DataFrame:
    """Pitcher 1: three games; throws FF every game, SL only in game 1."""
    return pl.DataFrame(
        {
            "pitcher_id": [1, 1, 1, 1],
            "game_pk": [101, 101, 102, 103],
            "game_date": ["2025-04-01", "2025-04-01", "2025-04-06", "2025-04-11"],
            "pitch_type_code": ["FF", "SL", "FF", "FF"],
            "n": [10, 10, 20, 10],
            "velo": [95.0, 85.0, 96.0, 94.0],
            "pfx_x": [-5.0, 5.0, -6.0, -4.0],
            "pfx_z": [15.0, 2.0, 14.0, 16.0],
            "spin_rate": [2200.0, 2600.0, None, 2300.0],
        }
    )


def _row(frame: pl.DataFrame, game_pk: int, code: str) -> dict:
    return frame.filter(
        (pl.col("game_pk") == game_pk) & (pl.col("pitch_type_code") == code)
    ).to_dicts()[0]


def test_first_appearance_has_no_profile():
    trailing = compute_trailing_profiles(_per_game())
    first = _row(trailing, 101, "FF")
    assert first["trailing_n"] == 0
    assert first["usage"] is None
    assert first["velo"] is None


def test_profile_is_strictly_prior_games():
    trailing = compute_trailing_profiles(_per_game())
    second = _row(trailing, 102, "FF")
    # Game 102 sees exactly game 101: 10 FF at 95 mph, usage 10/20.
    assert second["trailing_n"] == 10
    assert abs(second["usage"] - 0.5) < 1e-9
    assert abs(second["velo"] - 95.0) < 1e-9

    third = _row(trailing, 103, "FF")
    # Game 103 sees games 101+102: 30 FF, weighted velo (10*95 + 20*96)/30.
    assert third["trailing_n"] == 30
    assert abs(third["velo"] - (10 * 95.0 + 20 * 96.0) / 30) < 1e-9
    assert abs(third["usage"] - 30 / 40) < 1e-9


def test_unthrown_type_still_carries_decaying_profile():
    trailing = compute_trailing_profiles(_per_game())
    # SL was thrown only in game 101 but games 102/103 keep an SL row.
    sl_103 = _row(trailing, 103, "SL")
    assert sl_103["trailing_n"] == 10
    assert abs(sl_103["usage"] - 10 / 40) < 1e-9
    assert abs(sl_103["velo"] - 85.0) < 1e-9


def test_null_stats_are_excluded_from_weighted_means():
    trailing = compute_trailing_profiles(_per_game())
    # Game 102 had null spin for FF; game 103's FF spin sees only game 101.
    third = _row(trailing, 103, "FF")
    assert abs(third["spin_rate"] - 2200.0) < 1e-9


def test_window_limits_history():
    trailing = compute_trailing_profiles(_per_game(), window_games=1)
    third = _row(trailing, 103, "FF")
    # Only game 102 is visible: 20 FF at 96.
    assert third["trailing_n"] == 20
    assert abs(third["velo"] - 96.0) < 1e-9


def test_usage_sums_to_one_and_wide_pivot_names():
    trailing = compute_trailing_profiles(_per_game())
    per_game_usage = (
        trailing.filter(pl.col("trailing_n_total") > 0)
        .group_by(["pitcher_id", "game_pk"])
        .agg(pl.col("usage").sum().alias("total"))
    )
    assert all(abs(v - 1.0) < 1e-9 for v in per_game_usage["total"])

    wide = pivot_profiles_wide(trailing)
    assert profile_column_name("usage", "FF") in wide.columns
    assert profile_column_name("velo", "SL") in wide.columns
    assert wide.filter(pl.col("game_pk") == 103).height == 1

    defaults = league_default_profiles(trailing)
    assert profile_column_name("usage", "FF") in defaults
