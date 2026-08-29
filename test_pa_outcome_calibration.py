from __future__ import annotations

import random

from mlb.sim.calibration import (
    PAOutcomeCalibration,
    apply_multipliers,
)
from mlb.sim.count_machine import PA_OUTCOMES
from mlb.sim.pa import (
    FixedDistributionProvider,
    pa_outcome_distribution,
    simulate_plate_appearance,
)

RESULT = {
    "ball": 0.36,
    "called_strike": 0.17,
    "swinging_strike": 0.10,
    "foul": 0.16,
    "in_play": 0.20,
    "hit_by_pitch": 0.01,
}
EVENT = {
    "out": 0.66,
    "single": 0.20,
    "double": 0.06,
    "triple": 0.005,
    "home_run": 0.05,
    "reached_on_error": 0.025,
}


def _const_by_count(dist: dict) -> dict:
    return {(b, s): dist for b in range(4) for s in range(3)}


def test_closed_form_matches_brute_force():
    closed = pa_outcome_distribution(_const_by_count(RESULT), _const_by_count(EVENT))
    assert abs(sum(closed.values()) - 1.0) < 1e-9

    provider = FixedDistributionProvider(RESULT, EVENT)
    rng = random.Random(0)
    counts = dict.fromkeys(PA_OUTCOMES, 0)
    n = 200_000
    for _ in range(n):
        counts[simulate_plate_appearance(provider, rng).outcome] += 1
    for outcome in PA_OUTCOMES:
        empirical = counts[outcome] / n
        assert abs(closed[outcome] - empirical) < 0.004, (
            outcome, closed[outcome], empirical
        )


def test_all_in_play_reduces_to_event_distribution():
    result = {"ball": 0.0, "called_strike": 0.0, "swinging_strike": 0.0,
              "foul": 0.0, "in_play": 1.0, "hit_by_pitch": 0.0}
    closed = pa_outcome_distribution(_const_by_count(result), _const_by_count(EVENT))
    for cls, p in EVENT.items():
        assert abs(closed[cls] - p) < 1e-9
    assert closed["walk"] == 0.0 and closed["strikeout"] == 0.0


def test_all_balls_is_certain_walk():
    result = {"ball": 1.0, "called_strike": 0.0, "swinging_strike": 0.0,
              "foul": 0.0, "in_play": 0.0, "hit_by_pitch": 0.0}
    closed = pa_outcome_distribution(_const_by_count(result), _const_by_count(EVENT))
    assert abs(closed["walk"] - 1.0) < 1e-9


def test_two_strike_foul_never_terminates_as_strikeout_alone():
    # With only fouls and in-play at 2 strikes, the PA must resolve in play.
    result = {"ball": 0.0, "called_strike": 0.0, "swinging_strike": 0.0,
              "foul": 0.5, "in_play": 0.5, "hit_by_pitch": 0.0}
    closed = pa_outcome_distribution(_const_by_count(result), _const_by_count(EVENT))
    assert abs(closed["strikeout"]) < 1e-9
    assert abs(closed["walk"]) < 1e-9
    # All mass flows through in-play events.
    assert abs(sum(closed[c] for c in EVENT) - 1.0) < 1e-9


def test_pa_outcome_calibration_apply_and_roundtrip(tmp_path):
    cal = PAOutcomeCalibration(
        multipliers={"top": {"walk": 0.5, "single": 1.2}, "bottom": {}}
    )
    base = {"walk": 0.1, "single": 0.15, "out": 0.75}
    adjusted = apply_multipliers(base, cal.for_side(True))
    assert abs(sum(adjusted.values()) - 1.0) < 1e-9
    # Walk scaled down, single up, relative to the renormalized baseline.
    assert adjusted["walk"] < base["walk"]
    assert adjusted["single"] > base["single"]
    # Bottom side has no multipliers -> unchanged (still normalized).
    assert apply_multipliers(base, cal.for_side(False)) == base

    path = tmp_path / "pa_cal.json"
    cal.save(path)
    loaded = PAOutcomeCalibration.load(path)
    assert loaded.multipliers == cal.multipliers
