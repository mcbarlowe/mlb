"""End-to-end simulation calibration multipliers.

The stage models are near-calibrated on real pitch rows, but the simulator
feeds them constructed inputs (count-conditioned type/location pools,
frozen situational state), leaving aggregate biases: K% a few points low,
HBP ~2x, slightly too many outs on contact — and no home-field advantage.

``scripts/calibrate_sim.py`` measures simulated league rates per batting
side (top = away batting, bottom = home batting) against actual league
rates per side, and stores per-class multipliers here. Providers apply
them to each count's distributions and renormalize. Home advantage falls
out naturally: the bottom-half multipliers encode how much better home
offenses do than the simulator's side-blind baseline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CALIBRATION_PATH = Path("models/sim/sim_calibration.json")

MULTIPLIER_FLOOR = 0.25
MULTIPLIER_CEILING = 4.0


def apply_multipliers(
    probabilities: dict[str, float], multipliers: dict[str, float] | None
) -> dict[str, float]:
    """Scale class probabilities by per-class multipliers and renormalize."""
    if not multipliers:
        return probabilities
    adjusted = {
        cls: max(p, 0.0) * multipliers.get(cls, 1.0)
        for cls, p in probabilities.items()
    }
    total = sum(adjusted.values())
    if total <= 0:
        return probabilities
    return {cls: p / total for cls, p in adjusted.items()}


@dataclass(frozen=True)
class SimCalibration:
    """Per-side, per-class multipliers for result and event distributions."""

    result: dict[str, dict[str, float]]  # side ("top" | "bottom") -> class -> x
    event: dict[str, dict[str, float]]

    @classmethod
    def load(cls, path: Path = DEFAULT_CALIBRATION_PATH) -> SimCalibration:
        payload = json.loads(Path(path).read_text())
        return cls(result=payload["result"], event=payload["event"])

    def save(self, path: Path = DEFAULT_CALIBRATION_PATH, meta: dict | None = None) -> None:
        payload: dict = {"result": self.result, "event": self.event}
        if meta:
            payload["meta"] = meta
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def multipliers(
        self, is_top_half: bool
    ) -> tuple[dict[str, float], dict[str, float]]:
        side = "top" if is_top_half else "bottom"
        return self.result.get(side, {}), self.event.get(side, {})


def derive_multipliers(
    actual: dict[str, float],
    simulated: dict[str, float],
    existing: dict[str, float] | None = None,
) -> dict[str, float]:
    """actual/simulated rate ratios, composed onto existing multipliers.

    Classes absent from either distribution keep their existing value.
    Ratios are clipped so a sparse class cannot swing the distribution.
    """
    existing = existing or {}
    out: dict[str, float] = dict(existing)
    for cls, actual_rate in actual.items():
        sim_rate = simulated.get(cls, 0.0)
        if actual_rate <= 0 or sim_rate <= 0:
            continue
        ratio = actual_rate / sim_rate
        composed = existing.get(cls, 1.0) * ratio
        out[cls] = min(max(composed, MULTIPLIER_FLOOR), MULTIPLIER_CEILING)
    return out


DEFAULT_WIN_CALIBRATION_PATH = Path("models/sim/win_calibration.json")


@dataclass(frozen=True)
class WinCalibration:
    """Platt scaling for simulated home-win probabilities.

    Fitted on val-season (2024) simulated games — never on the 2025 test
    season. ``slope < 1`` shrinks an overconfident Monte Carlo spread.
    """

    intercept: float
    slope: float

    def apply(self, p_home: float) -> float:
        import math

        p = min(max(p_home, 1e-3), 1 - 1e-3)
        z = self.intercept + self.slope * math.log(p / (1 - p))
        return 1.0 / (1.0 + math.exp(-z))

    @classmethod
    def load(cls, path: Path = DEFAULT_WIN_CALIBRATION_PATH) -> WinCalibration:
        payload = json.loads(Path(path).read_text())
        return cls(intercept=payload["intercept"], slope=payload["slope"])

    def save(
        self, path: Path = DEFAULT_WIN_CALIBRATION_PATH, meta: dict | None = None
    ) -> None:
        payload: dict = {"intercept": self.intercept, "slope": self.slope}
        if meta:
            payload["meta"] = meta
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))



def load_win_calibration(
    path: Path = DEFAULT_WIN_CALIBRATION_PATH,
) -> WinCalibration | None:
    """Load the fitted win calibration, or None when not fitted yet."""
    if not Path(path).exists():
        return None
    return WinCalibration.load(path)


LEAGUE_HOME_RATE = 0.543


def fit_win_calibration(
    probabilities: list[float],
    outcomes: list[float],
    anchor: float = LEAGUE_HOME_RATE,
) -> WinCalibration:
    """Fit anchored Platt scaling: shrink the spread around the league rate.

    Only the slope is fitted (1-D MLE): ``p = anchor`` maps to ``anchor``,
    so a small fit sample cannot teach the calibrator a bogus home/away
    shift — 150 games carry ~4pp of home-rate sampling noise, which a free
    intercept absorbs and then projects onto every future season.
    """
    import math

    import numpy as np
    from scipy.optimize import minimize_scalar

    def logit(p: float) -> float:
        p = min(max(p, 1e-3), 1 - 1e-3)
        return math.log(p / (1 - p))

    anchor_logit = logit(anchor)
    x = np.array([logit(p) - anchor_logit for p in probabilities])
    y = np.array(outcomes)

    def negative_log_likelihood(slope: float) -> float:
        z = anchor_logit + slope * x
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

    fit = minimize_scalar(negative_log_likelihood, bounds=(0.05, 2.0), method="bounded")
    slope = float(fit.x)
    return WinCalibration(
        intercept=anchor_logit * (1.0 - slope),
        slope=slope,
    )
