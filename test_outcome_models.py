from __future__ import annotations

import polars as pl

from src.outcome.dataset import (
    add_state_features,
    build_training_frame,
    stage_a_frame,
    stage_b_frame,
)
from src.outcome.labels import (
    canonicalize_pitch_type,
    map_event_type,
    map_pitch_call,
)
from src.outcome.models import conditional_baseline_log_loss


def _raw_frame(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "game_pk": 1,
        "season": 2023,
        "game_date": "2023-05-01",
        "at_bat_index": 0,
        "pitch_number": 1,
        "half_inning": "top",
        "inning": 1,
        "outs": 0,
        "is_runner_on_first": False,
        "is_runner_on_second": False,
        "is_runner_on_third": False,
        "batter_id": 100,
        "bat_side": "R",
        "pitcher_id": 200,
        "throw_side": "R",
        "pitch_call_description": "Ball",
        "event_type": None,
        "count_after_pitch": "1-0",
        "pitch_type_code": "FF",
        "px": 0.0,
        "pz": 2.5,
        "pitch_strike_zone_top": 3.4,
        "pitch_strike_zone_bottom": 1.6,
        "away_score": 0,
        "home_score": 0,
        "pitch_start_speed": None,
        "spin_rate": None,
        "break_vertical_induced": None,
        "break_horizontal": None,
        "x0": None,
        "z0": None,
    }
    filled = [{**defaults, **row} for row in rows]
    return pl.DataFrame(
        filled,
        schema_overrides={
            "event_type": pl.String,
            "pitch_start_speed": pl.Float64,
            "spin_rate": pl.Float64,
            "break_vertical_induced": pl.Float64,
            "break_horizontal": pl.Float64,
            "x0": pl.Float64,
            "z0": pl.Float64,
        },
    )


def test_pitch_call_mapping_covers_classes_and_exclusions():
    assert map_pitch_call("Ball In Dirt") == "ball"
    assert map_pitch_call("Called Strike") == "called_strike"
    assert map_pitch_call("Foul Tip") == "swinging_strike"
    assert map_pitch_call("Foul") == "foul"
    assert map_pitch_call("In play, run(s)") == "in_play"
    assert map_pitch_call("Hit By Pitch") == "hit_by_pitch"
    assert map_pitch_call("Pickoff Attempt 1B") is None
    assert map_pitch_call("Intent Ball") is None
    assert map_pitch_call(None) is None


def test_event_mapping_groups_outs_and_errors():
    assert map_event_type("grounded_into_double_play") == "out"
    assert map_event_type("sac_fly") == "out"
    assert map_event_type("home_run") == "home_run"
    assert map_event_type("field_error") == "reached_on_error"
    assert map_event_type("game_advisory") is None


def test_pitch_type_canonicalization():
    assert canonicalize_pitch_type("FF") == "FF"
    assert canonicalize_pitch_type("SV") == "OTHER"
    assert canonicalize_pitch_type("None") is None
    assert canonicalize_pitch_type(None) is None


def test_pre_pitch_count_is_shifted_within_at_bat():
    frame = add_state_features(
        _raw_frame(
            [
                {"pitch_number": 1, "count_after_pitch": "1-0"},
                {"pitch_number": 2, "count_after_pitch": "1-1"},
                {"pitch_number": 3, "count_after_pitch": "2-1"},
            ]
        )
    )
    assert frame["balls_before"].to_list() == [0, 1, 1]
    assert frame["strikes_before"].to_list() == [0, 0, 1]


def test_pre_at_bat_score_is_previous_play_state():
    frame = add_state_features(
        _raw_frame(
            [
                # Home run scores on the first at-bat (post-play score 1-0 away).
                {"at_bat_index": 0, "away_score": 1, "home_score": 0},
                {"at_bat_index": 1, "pitch_number": 1, "away_score": 1, "home_score": 0},
            ]
        )
    )
    # First at-bat starts 0-0 even though the stored (post-play) score is 1-0.
    assert frame["score_diff"].to_list()[0] == 0
    # Second at-bat sees the run from the first (away bats in the top half).
    assert frame["score_diff"].to_list()[1] == 1


def test_zone_geometry_center_and_fallback():
    frame = add_state_features(
        _raw_frame(
            [
                {"px": 0.0, "pz": 2.5},
                {
                    "px": 2.0,
                    "pz": 0.5,
                    "pitch_strike_zone_top": None,
                    "pitch_strike_zone_bottom": None,
                },
            ]
        )
    )
    center = frame.row(0, named=True)
    assert center["zone_dist_center"] == 0.0
    assert center["in_zone"] == 1
    far = frame.row(1, named=True)
    assert far["in_zone"] == 0
    assert far["zone_norm_height"] is not None


def test_pitcher_profile_excludes_current_pitch():
    rows = [
        {"pitch_number": 1, "pitch_start_speed": 90.0},
        {"pitch_number": 2, "pitch_start_speed": 94.0},
        {"pitch_number": 3, "pitch_start_speed": 98.0},
    ]
    frame = build_training_frame(_raw_frame(rows))
    deltas = frame["profile_speed_delta"].to_list()
    # First pitch has no history for pitcher or league.
    assert deltas[0] is None
    # Later rows: single pitcher == league, so deltas collapse to zero —
    # but only using PRIOR pitches (a leak would make them nonzero).
    assert deltas[1] == 0.0
    assert deltas[2] == 0.0


def test_pitcher_profile_delta_separates_pitchers():
    rows = [
        {"pitcher_id": 1, "pitch_number": 1, "at_bat_index": 0, "pitch_start_speed": 99.0},
        {"pitcher_id": 2, "pitch_number": 1, "at_bat_index": 1, "pitch_start_speed": 89.0},
        {"pitcher_id": 1, "pitch_number": 1, "at_bat_index": 2, "pitch_start_speed": 99.0},
        {"pitcher_id": 2, "pitch_number": 1, "at_bat_index": 3, "pitch_start_speed": 89.0},
    ]
    frame = build_training_frame(_raw_frame(rows))
    deltas = frame["profile_speed_delta"].to_list()
    # Row 2 (pitcher 1): own profile 99, league profile mean(99, 89) = 94 -> +5.
    assert deltas[2] == 5.0
    # Row 3 (pitcher 2): own profile 89, league profile mean(99, 89, 99) ~ 95.67.
    assert round(deltas[3], 2) == -6.67


def test_batter_rates_are_leak_free_with_correct_denominators():
    rows = [
        # Swing and miss out of the zone (a chase).
        {"pitch_number": 1, "pitch_call_description": "Swinging Strike", "px": 1.5,
         "count_after_pitch": "0-1"},
        # Ball in the zone? No: taken ball out of zone (not a chase).
        {"pitch_number": 2, "pitch_call_description": "Ball", "px": 1.5,
         "count_after_pitch": "1-1"},
        # Third pitch: rates must reflect only the first two pitches.
        {"pitch_number": 3, "pitch_call_description": "Foul", "px": 0.0,
         "count_after_pitch": "1-2"},
    ]
    frame = build_training_frame(_raw_frame(rows))
    third = frame.row(2, named=True)
    assert third["batter_swing_rate"] == 0.5   # 1 swing / 2 pitches
    assert third["batter_whiff_rate"] == 1.0   # 1 whiff / 1 swing
    assert third["batter_chase_rate"] == 0.5   # 1 chase / 2 out-of-zone pitches
    first = frame.row(0, named=True)
    assert first["batter_swing_rate"] is None


def test_stage_frames_filter_correctly():
    rows = [
        {"pitch_number": 1, "pitch_call_description": "Pickoff Attempt 1B",
         "count_after_pitch": "0-0"},
        {"pitch_number": 2, "pitch_call_description": "Ball",
         "count_after_pitch": "1-0"},
        {"pitch_number": 3, "pitch_call_description": "In play, out(s)",
         "event_type": "field_out", "count_after_pitch": "1-0"},
        {"pitch_number": 4, "pitch_call_description": "In play, no out",
         "event_type": "game_advisory", "count_after_pitch": "1-0"},
        {"pitch_number": 5, "pitch_call_description": "Ball",
         "pitch_type_code": "None", "count_after_pitch": "2-0"},
    ]
    frame = build_training_frame(_raw_frame(rows))
    stage_a = stage_a_frame(frame)
    assert stage_a.height == 3  # pickoff excluded, null pitch type excluded
    assert set(stage_a["label_result"].to_list()) == {"ball", "in_play"}

    stage_b = stage_b_frame(frame)
    assert stage_b.height == 1  # only the mapped field_out
    assert stage_b["label_event"].to_list() == ["out"]


def test_conditional_baseline_beats_uniform_on_skewed_data():
    import math

    train = pl.DataFrame(
        {
            "balls_before": [0] * 90 + [3] * 10,
            "strikes_before": [0] * 90 + [0] * 10,
            "label_result": ["called_strike"] * 90 + ["ball"] * 10,
        }
    )
    eval_frame = pl.DataFrame(
        {
            "balls_before": [0, 3],
            "strikes_before": [0, 0],
            "label_result": ["called_strike", "ball"],
        }
    )
    loss = conditional_baseline_log_loss(
        train, eval_frame, "label_result", ["balls_before", "strikes_before"]
    )
    uniform = -math.log(0.5)
    assert loss < uniform
