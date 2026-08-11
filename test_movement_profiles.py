"""Tests for leak-free trailing pitcher movement profiles."""

import json

import polars as pl

from src.ml.movement_profiles import (
    attach_movement_profiles,
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
            # pfx_x is unmeasured in game 102 (FF).
            "pfx_x": [-5.0, 5.0, None, -4.0],
            "pfx_z": [15.0, 2.0, 14.0, 16.0],
            "spin_rate": [2200.0, 2600.0, 2350.0, 2300.0],
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
    # Game 102 had null pfx_x for FF; game 103's FF pfx_x sees only game 101.
    third = _row(trailing, 103, "FF")
    assert abs(third["pfx_x"] - (-5.0)) < 1e-9


def test_nan_stats_are_treated_as_unmeasured():
    import math

    # DB nulls surface as NaN via pandas; they must not poison windows.
    frame = _per_game().with_columns(
        pl.when(pl.col("game_pk") == 102)
        .then(float("nan"))
        .otherwise(pl.col("pfx_x"))
        .alias("pfx_x")
    )
    trailing = compute_trailing_profiles(frame)
    third = _row(trailing, 103, "FF")
    # Game 102's NaN pfx_x is excluded; only game 101's -5.0 remains.
    assert abs(third["pfx_x"] - (-5.0)) < 1e-9
    assert all(
        v is None or not math.isnan(v)
        for v in trailing["pfx_x"].to_list()
    )


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



def _attach_fixtures():
    trailing = compute_trailing_profiles(_per_game())
    wide = pivot_profiles_wide(trailing)
    defaults = league_default_profiles(trailing)
    return wide, defaults


def test_attach_uses_exact_appearance_row_with_scaling():
    wide, defaults = _attach_fixtures()
    frame = pl.DataFrame({"pitcher_id": [1], "game_pk": [102]})
    out = attach_movement_profiles(frame, wide, defaults)
    # Game 102 profile: FF velo 95 scaled by 1/100; usage 0.5 unscaled.
    assert abs(out[profile_column_name("velo", "FF")][0] - 0.95) < 1e-9
    assert abs(out[profile_column_name("usage", "FF")][0] - 0.5) < 1e-9


def test_attach_debut_row_takes_defaults_not_future():
    wide, defaults = _attach_fixtures()
    # Game 101 is the pitcher's debut: store row exists with null stats.
    frame = pl.DataFrame({"pitcher_id": [1], "game_pk": [101]})
    out = attach_movement_profiles(frame, wide, defaults)
    expected = defaults[profile_column_name("velo", "FF")] / 100.0
    got = out[profile_column_name("velo", "FF")][0]
    assert abs(got - expected) < 1e-9
    # Must NOT equal the pitcher's later (future) profile.
    future = wide.filter(pl.col("game_pk") == 103)[
        profile_column_name("velo", "FF")
    ][0]
    assert abs(got - future / 100.0) > 1e-6


def test_attach_unseen_game_falls_back_to_latest_profile():
    wide, defaults = _attach_fixtures()
    # A live game not in the store: use the pitcher's latest profile (103).
    frame = pl.DataFrame({"pitcher_id": [1], "game_pk": [999]})
    out = attach_movement_profiles(frame, wide, defaults)
    latest_velo = wide.filter(pl.col("game_pk") == 103)[
        profile_column_name("velo", "FF")
    ][0]
    assert abs(out[profile_column_name("velo", "FF")][0] - latest_velo / 100.0) < 1e-9


def test_attach_unknown_pitcher_takes_league_defaults():
    wide, defaults = _attach_fixtures()
    frame = pl.DataFrame({"pitcher_id": [42], "game_pk": [999]})
    out = attach_movement_profiles(frame, wide, defaults)
    expected = defaults[profile_column_name("usage", "FF")]
    assert abs(out[profile_column_name("usage", "FF")][0] - expected) < 1e-9


def test_feature_engine_contract_and_save_load_roundtrip(tmp_path):
    trailing = compute_trailing_profiles(_per_game())
    pivot_profiles_wide(trailing).write_parquet(
        tmp_path / "pitcher_movement_profiles_wide.parquet"
    )
    (tmp_path / "league_default_profiles.json").write_text(
        json.dumps(league_default_profiles(trailing))
    )

    from src.ml.features import PitchFeatureEngine

    plain = PitchFeatureEngine()
    enabled = PitchFeatureEngine(movement_profiles_dir=tmp_path)
    assert (
        len(enabled.get_feature_columns())
        == len(plain.get_feature_columns()) + 55
    )

    enabled.pitcher_to_idx = {1: 0}
    enabled._fitted = True
    engine_path = tmp_path / "engine.json"
    enabled.save(engine_path)
    loaded = PitchFeatureEngine.load(engine_path)
    assert loaded.movement_profiles_dir == tmp_path
    plain.pitcher_to_idx = {1: 0}
    plain._fitted = True
    plain.save(tmp_path / "plain.json")
    assert PitchFeatureEngine.load(tmp_path / "plain.json").movement_profiles_dir is None