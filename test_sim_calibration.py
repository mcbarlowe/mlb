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


def test_fit_win_calibration_shrinks_overconfident_spread():
    import random

    from src.sim.calibration import WinCalibration, fit_win_calibration

    rng = random.Random(0)
    # True p is 0.5 + 0.5*(raw - 0.5): raw spread is twice as wide as truth.
    probabilities, outcomes = [], []
    for _ in range(4000):
        raw = rng.uniform(0.2, 0.8)
        true_p = 0.5 + 0.5 * (raw - 0.5)
        probabilities.append(raw)
        outcomes.append(1.0 if rng.random() < true_p else 0.0)
    calibration = fit_win_calibration(probabilities, outcomes)
    assert 0.2 < calibration.slope < 0.9  # shrinks
    # Calibrated Brier beats raw on the fit distribution.
    raw_brier = sum((p - y) ** 2 for p, y in zip(probabilities, outcomes)) / len(outcomes)
    cal_brier = sum(
        (calibration.apply(p) - y) ** 2 for p, y in zip(probabilities, outcomes)
    ) / len(outcomes)
    assert cal_brier < raw_brier

    # Round trip.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "win.json"
        calibration.save(path, meta={"fit_season": 2024})
        loaded = WinCalibration.load(path)
        assert loaded == calibration


def test_win_calibration_apply_is_monotone_and_bounded():
    from src.sim.calibration import WinCalibration

    calibration = WinCalibration(intercept=0.1, slope=0.5)
    values = [calibration.apply(p) for p in (0.01, 0.3, 0.5, 0.7, 0.99)]
    assert all(0.0 < v < 1.0 for v in values)
    assert values == sorted(values)


def test_stretch_conditioned_lookups_and_fallbacks():
    from src.sim.calibration import SimCalibration

    calibration = SimCalibration(
        result={"top": {"ball": 1.1}},
        event={"top": {"single": 1.2}},
        result_by_count={
            "top": {
                "windup": {"0-0": {"ball": 1.5}},
                "stretch": {"0-0": {"ball": 0.8}},
            }
        },
        event_by_stretch={"top": {"stretch": {"single": 1.4}, "windup": {}}},
    )
    windup = calibration.result_multipliers_by_count(True, stretch=False)
    stretch = calibration.result_multipliers_by_count(True, stretch=True)
    assert windup[(0, 0)] == {"ball": 1.5}
    assert stretch[(0, 0)] == {"ball": 0.8}
    # Counts missing from the fit fall back to the side aggregate.
    assert windup[(3, 2)] == {"ball": 1.1}
    # Event lookups: stretch cell present, windup empty -> aggregate.
    assert calibration.event_multipliers(True, stretch=True) == {"single": 1.4}
    assert calibration.event_multipliers(True, stretch=False) == {"single": 1.2}
    # Round trip preserves the nested fields.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cal.json"
        calibration.save(path)
        loaded = SimCalibration.load(path)
        assert loaded == calibration
