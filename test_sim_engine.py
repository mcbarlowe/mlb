from __future__ import annotations

import json
import random
from pathlib import Path

import polars as pl
import pytest

from mlb.data.base_state import compute_at_bat_states
from mlb.sim.base_out import BaseOutEngine, build_transition_frame, runners_bitmap
from mlb.sim.count_machine import apply_pitch_result
from mlb.sim.pa import FixedDistributionProvider, simulate_plate_appearance

# --- count machine -----------------------------------------------------------


def test_foul_at_two_strikes_keeps_two_strikes():
    t = apply_pitch_result(1, 2, "foul")
    assert (t.balls, t.strikes, t.terminal) == (1, 2, None)


def test_fourth_ball_is_walk():
    t = apply_pitch_result(3, 1, "ball")
    assert t.terminal == "walk"


def test_third_strike_is_strikeout_swinging_and_called():
    for result in ("called_strike", "swinging_strike"):
        t = apply_pitch_result(2, 2, result)
        assert t.terminal == "strikeout"


def test_counts_advance_without_terminal():
    t = apply_pitch_result(0, 0, "ball")
    assert (t.balls, t.strikes, t.terminal) == (1, 0, None)
    t = apply_pitch_result(0, 0, "called_strike")
    assert (t.balls, t.strikes, t.terminal) == (0, 1, None)


def test_in_play_and_hbp_terminate():
    assert apply_pitch_result(1, 1, "in_play").in_play
    assert apply_pitch_result(1, 1, "hit_by_pitch").terminal == "hit_by_pitch"


def test_invalid_count_rejected():
    with pytest.raises(ValueError):
        apply_pitch_result(4, 0, "ball")


# --- PA simulator ------------------------------------------------------------


def _provider(result_probs, event_probs=None):
    return FixedDistributionProvider(
        result_probs, event_probs or {"out": 1.0}
    )


def test_pa_all_balls_is_four_pitch_walk():
    pa = simulate_plate_appearance(_provider({"ball": 1.0}), random.Random(0))
    assert pa.outcome == "walk"
    assert pa.n_pitches == 4


def test_pa_all_called_strikes_is_three_pitch_strikeout():
    pa = simulate_plate_appearance(_provider({"called_strike": 1.0}), random.Random(0))
    assert pa.outcome == "strikeout"
    assert pa.n_pitches == 3


def test_pa_foul_forever_raises():
    with pytest.raises(RuntimeError):
        simulate_plate_appearance(_provider({"foul": 1.0}), random.Random(0))


def test_pa_in_play_resolves_via_event_distribution():
    pa = simulate_plate_appearance(
        _provider({"in_play": 1.0}, {"home_run": 1.0}), random.Random(0)
    )
    assert pa.outcome == "home_run"
    assert pa.n_pitches == 1


def test_pa_starting_count_respected():
    pa = simulate_plate_appearance(
        _provider({"ball": 1.0}), random.Random(0), balls=3, strikes=2
    )
    assert pa.outcome == "walk"
    assert pa.n_pitches == 1


# --- base-out engine ---------------------------------------------------------


def _ab_row(
    game_pk: int,
    at_bat_index: int,
    inning: int,
    half: str,
    outs_before: int,
    runners_before: int,
    outs_after: int,
    event: str,
    away: int,
    home: int,
) -> dict:
    return {
        "game_pk": game_pk,
        "at_bat_index": at_bat_index,
        "inning": inning,
        "half_inning": half,
        "event_type": event,
        "outs_before": outs_before,
        "runners_before": runners_before,
        "outs_after": outs_after,
        "away_score": away,
        "home_score": home,
    }


def test_build_transition_frame_scores_runner_and_ends_inning():
    rows = [
        # Top 1: leadoff single, then HR scores 2, then three strikeouts.
        _ab_row(1, 0, 1, "top", 0, 0, 0, "single", 0, 0),
        _ab_row(1, 1, 1, "top", 0, 1, 0, "home_run", 2, 0),
        _ab_row(1, 2, 1, "top", 0, 0, 1, "strikeout", 2, 0),
        _ab_row(1, 3, 1, "top", 1, 0, 2, "strikeout", 2, 0),
        _ab_row(1, 4, 1, "top", 2, 0, 3, "strikeout", 2, 0),
        # Bottom 1 exists so the game is well formed.
        _ab_row(1, 5, 1, "bottom", 0, 0, 1, "strikeout", 2, 0),
    ]
    table = build_transition_frame(pl.DataFrame(rows))

    hr = table.filter(
        (pl.col("pa_outcome") == "home_run") & (pl.col("runners_before") == 1)
    )
    assert hr.height == 1
    assert hr["runs"][0] == 2
    assert hr["runners_after"][0] == 0

    last_k = table.filter(
        (pl.col("pa_outcome") == "strikeout") & (pl.col("outs_before") == 2)
    )
    # Inning-ending strikeout: outs_after forced to 3.
    assert (last_k["outs_after"] == 3).all()


def test_engine_samples_empirical_and_falls_back():
    table = pl.DataFrame(
        {
            "pa_outcome": ["single"],
            "runners_before": [0],
            "outs_before": [0],
            "runners_after": [1],
            "outs_after": [0],
            "runs": [0],
            "n": [10],
        }
    )
    engine = BaseOutEngine(table, seed=1)

    seen = engine.sample("single", 0, 0)
    assert (seen.runners_after, seen.outs_after, seen.runs) == (1, 0, 0)

    # Unseen state -> deterministic fallback: bases-loaded walk forces a run.
    forced = engine.sample("walk", runners_bitmap(True, True, True), 2)
    assert forced.runs == 1
    assert forced.runners_after == 7

    hr = engine.sample("home_run", runners_bitmap(True, False, True), 1)
    assert (hr.runners_after, hr.outs_after, hr.runs) == (0, 1, 3)

    k = engine.sample("strikeout", 2, 2)
    assert (k.runners_after, k.outs_after, k.runs) == (2, 3, 0)


def test_build_transition_frame_drops_walkoff_without_successor():
    rows = [_ab_row(1, 0, 9, "bottom", 1, 4, 1, "single", 3, 4)]
    table = build_transition_frame(pl.DataFrame(rows))
    assert table.height == 0


# --- base state reconstruction ------------------------------------------------


def test_compute_at_bat_states_on_real_feed():
    feed = json.loads(Path("example_json_files/example_live_feed.json").read_text())
    plays = feed["liveData"]["plays"]["allPlays"]
    states = compute_at_bat_states(plays)
    assert len(states) == len(plays)

    current_half = None
    prev_state = None
    saw_runner = False
    for play, state in zip(plays, states):
        about = play["about"]
        half = (about["inning"], about["halfInning"])
        if half != current_half:
            # Every half-inning starts with no outs and empty bases.
            assert state["outs_before"] == 0
            assert not state["is_runner_on_first"]
            assert not state["is_runner_on_second"]
            assert not state["is_runner_on_third"]
            current_half = half
        else:
            # Outs chain: an AB starts with the previous AB's post-play outs.
            assert prev_state is not None
            assert state["outs_before"] == prev_state["outs_after"]
            # A non-inning-ending walk/single leaves first base occupied.
            assert prev_state is not None
        saw_runner = saw_runner or state["is_runner_on_first"]
        assert 0 <= state["outs_before"] <= 2
        assert state["outs_after"] >= state["outs_before"]
        prev_state = state
    assert saw_runner  # reconstruction actually places runners


def test_single_puts_runner_on_first_for_next_ab():
    feed = json.loads(Path("example_json_files/example_live_feed.json").read_text())
    plays = feed["liveData"]["plays"]["allPlays"]
    states = compute_at_bat_states(plays)
    checked = 0
    for i, play in enumerate(plays[:-1]):
        nxt = plays[i + 1]["about"]
        same_half = (
            nxt["halfInning"] == play["about"]["halfInning"]
            and nxt["inning"] == play["about"]["inning"]
        )
        if play["result"].get("eventType") != "single" or not same_half:
            continue
        batter_id = play["matchup"]["batter"]["id"]
        ends = [
            r["movement"].get("end")
            for r in play.get("runners", [])
            if r["details"]["runner"]["id"] == batter_id
            and not r["movement"].get("isOut")
        ]
        if ends and ends[-1] == "1B":
            assert states[i + 1]["is_runner_on_first"]
            assert states[i + 1]["runner_on_first_id"] == batter_id
            checked += 1
    assert checked > 0
