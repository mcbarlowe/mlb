from __future__ import annotations

import json

import numpy as np
import polars as pl

from src.outcome.dataset import FEATURE_COLUMNS
from src.outcome.inference import (
    OutcomeGameState,
    PitchOutcomePredictor,
    build_feature_frame,
    sample_locations_from_grid,
)


def _profiles() -> tuple[pl.DataFrame, pl.DataFrame]:
    pitcher = pl.DataFrame(
        {
            "pitcher_id": [543037, 543037],
            "pitch_type": ["FF", "SL"],
            "profile_speed_delta": [2.0, 1.0],
            "profile_spin_delta": [100.0, 50.0],
            "profile_ivb_delta": [1.5, -0.2],
            "profile_hb_delta": [0.4, 2.1],
            "profile_release_x": [-1.7, -1.8],
            "profile_release_z": [5.8, 5.7],
            "pitcher_whiff_rate": [0.28, 0.42],
            "pitcher_hr_rate": [0.04, 0.03],
        }
    )
    batter = pl.DataFrame(
        {
            "batter_id": [660271],
            "batter_swing_rate": [0.47],
            "batter_whiff_rate": [0.21],
            "batter_chase_rate": [0.32],
            "batter_hr_rate": [0.07],
            "batter_xbh_rate": [0.14],
            "batter_hit_rate": [0.35],
        }
    )
    return pitcher, batter


def test_location_sampling_returns_plausible_points():
    px = np.array([-1.0, 0.0, 1.0])
    pz = np.array([1.5, 2.5, 3.5])
    density = np.array([[0.0, 0.1, 0.0], [0.2, 1.0, 0.2], [0.0, 0.1, 0.0]])
    samples = sample_locations_from_grid(px, pz, density, n_samples=25, seed=7)
    assert len(samples) == 25
    assert max(abs(x) for x, _ in samples) < 1.6
    assert min(z for _, z in samples) > 0.9
    assert max(z for _, z in samples) < 4.1


def test_build_feature_frame_marginalizes_type_probs_and_joins_profiles():
    state = OutcomeGameState(
        balls=1,
        strikes=2,
        outs=1,
        runner_on_first=True,
        runner_on_second=False,
        runner_on_third=False,
        inning=6,
        is_top_half=False,
        score_diff=1,
        season=2026,
        times_through_order=2,
        pitcher_id=543037,
        batter_id=660271,
        throw_side="R",
        bat_side="L",
        sz_top=3.4,
        sz_bottom=1.6,
    )
    profiles = _profiles()
    features, weights = build_feature_frame(
        state,
        {"SL": 0.75, "FF": 0.25},
        [(0.1, 2.2), (-0.2, 2.5)],
        *profiles,
    )
    assert features.height == 4
    assert np.isclose(weights.sum(), 1.0)
    assert sorted(set(features["pitch_type"].to_list())) == ["FF", "SL"]
    assert features.filter(pl.col("pitch_type") == "SL")["pitcher_whiff_rate"].to_list() == [0.42, 0.42]
    assert features["batter_chase_rate"].to_list()[0] == 0.32
    # State features reinstated after the DB repair carry the state values.
    assert all(v == 1 for v in features["runner_on_first"].to_list())
    assert all(v == 1 for v in features["outs"].to_list())
    sl_weight = weights[features["pitch_type"].to_list().index("SL")]
    ff_weight = weights[features["pitch_type"].to_list().index("FF")]
    assert sl_weight > ff_weight


def test_build_feature_frame_backfills_missing_profile_columns():
    state = OutcomeGameState(
        balls=0,
        strikes=0,
        outs=0,
        runner_on_first=False,
        runner_on_second=False,
        runner_on_third=False,
        inning=1,
        is_top_half=True,
        score_diff=0,
        season=2026,
        times_through_order=1,
        pitcher_id=543037,
        batter_id=660271,
        throw_side="R",
        bat_side="L",
    )
    pitcher, batter = _profiles()
    features, _ = build_feature_frame(
        state,
        {"FF": 1.0},
        [(0.0, 2.5)],
        pitcher.select([column for column in pitcher.columns if column != "pitcher_hr_rate"]),
        batter.select(
            [column for column in batter.columns if column not in {"batter_hr_rate", "batter_xbh_rate", "batter_hit_rate"}]
        ),
    )

    assert "pitcher_hr_rate" in features.columns
    assert "batter_hr_rate" in features.columns
    assert "batter_xbh_rate" in features.columns
    assert "batter_hit_rate" in features.columns
    assert features["pitcher_hr_rate"].to_list() == [None]



def test_load_feature_columns_accepts_older_subset(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    subset = FEATURE_COLUMNS[:-4]
    for stage in ("stage_a", "stage_b"):
        (run_dir / f"{stage}_features.json").write_text(
            json.dumps({"feature_columns": subset})
        )

    stage_a, stage_b = PitchOutcomePredictor._load_feature_columns(run_dir)

    assert stage_a == subset
    assert stage_b == subset


def test_load_feature_columns_rejects_unknown_features(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    bad = FEATURE_COLUMNS[:-1] + ["not_a_real_feature"]
    for stage in ("stage_a", "stage_b"):
        (run_dir / f"{stage}_features.json").write_text(
            json.dumps({"feature_columns": bad})
        )

    try:
        PitchOutcomePredictor._load_feature_columns(run_dir)
    except ValueError as exc:
        assert "unsupported features" in str(exc)
    else:
        raise AssertionError("expected ValueError for unsupported feature metadata")