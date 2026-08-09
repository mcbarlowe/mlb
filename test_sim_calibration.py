from __future__ import annotations

from pathlib import Path

import pytest

from src.sim.calibration import (
    SimCalibration,
    apply_multipliers,
    derive_multipliers,
)
from src.sim.pa import MatchupOutcomeProvider


def test_apply_multipliers_scales_and_renormalizes():
    probs = {"ball": 0.4, "called_strike": 0.4, "in_play": 0.2}
    adjusted = apply_multipliers(probs, {"in_play": 2.0})
    assert adjusted["in_play"] == pytest.approx(0.4 / 1.2)
    assert sum(adjusted.values()) == pytest.approx(1.0)
    # No multipliers -> unchanged object semantics.
    assert apply_multipliers(probs, None) == probs


def test_derive_multipliers_ratio_clip_and_compose():
    actual = {"strikeout": 0.22, "rare": 0.01}
    simulated = {"strikeout": 0.11, "rare": 0.10}
    mults = derive_multipliers(actual, simulated)
    assert mults["strikeout"] == pytest.approx(2.0)
    assert mults["rare"] == pytest.approx(0.25)  # clipped at floor

    # Second pass composes onto the first.
    second = derive_multipliers({"strikeout": 0.22}, {"strikeout": 0.20}, mults)
    assert second["strikeout"] == pytest.approx(2.0 * 1.1)
    # Untouched classes carry through.
    assert second["rare"] == pytest.approx(0.25)


def test_sim_calibration_round_trip_and_sides(tmp_path: Path):
    calibration = SimCalibration(
        result={"top": {"ball": 1.1}, "bottom": {"ball": 0.9}},
        event={"top": {"single": 1.2}, "bottom": {"single": 1.3}},
    )
    path = tmp_path / "cal.json"
    calibration.save(path, meta={"note": "test"})
    loaded = SimCalibration.load(path)
    result_mults, event_mults = loaded.multipliers(is_top_half=True)
    assert result_mults == {"ball": 1.1}
    assert event_mults == {"single": 1.2}
    result_mults, _ = loaded.multipliers(is_top_half=False)
    assert result_mults == {"ball": 0.9}


class _StubPredictor:
    def predict(self, state, type_probabilities, locations):
        return {
            "result": {"ball": 0.5, "called_strike": 0.3, "in_play": 0.2},
            "event_given_in_play": {"out": 0.7, "single": 0.3},
        }


def test_provider_applies_calibration_multipliers():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _State:
        balls: int = 0
        strikes: int = 0

    inputs = {
        (b, s): ({"FF": 1.0}, [(0.0, 2.0)]) for b in range(4) for s in range(3)
    }
    provider = MatchupOutcomeProvider(
        _StubPredictor(),
        _State(),
        inputs,
        result_multipliers={"in_play": 2.0},
        event_multipliers={"single": 2.0},
    )
    result = provider.result_probabilities(0, 0)
    assert result["in_play"] == pytest.approx(0.4 / 1.2)
    assert sum(result.values()) == pytest.approx(1.0)
    event = provider.event_probabilities(1, 2)
    assert event["single"] == pytest.approx(0.6 / 1.3)
    assert sum(event.values()) == pytest.approx(1.0)
