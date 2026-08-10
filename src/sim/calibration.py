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


def stretch_key(stretch: bool) -> str:
    return "stretch" if stretch else "windup"


@dataclass(frozen=True)
class SimCalibration:
    """Per-side, per-stretch multipliers for result and event distributions.

    ``result_by_count`` (side -> "windup"/"stretch" -> "balls-strikes" ->
    class -> multiplier) matches each count's per-pitch result distribution
    to league rates CONDITIONAL on the base state. Conditioning on stretch
    matters: the outcome models' runner features partly encode
    pitcher-quality selection effects, and an uncorrected simulator turns
    that correlation into a runner -> uplift -> runner feedback loop that
    inflates run totals. ``event_by_stretch`` does the same for in-play
    event mixes. ``result``/``event`` keep side-aggregate multipliers as
    fallbacks for cells missing from the fit.
    """

    result: dict[str, dict[str, float]]  # side ("top" | "bottom") -> class -> x
    event: dict[str, dict[str, float]]
    result_by_count: dict[str, dict[str, dict[str, dict[str, float]]]] | None = None
    event_by_stretch: dict[str, dict[str, dict[str, float]]] | None = None

    @classmethod
    def load(cls, path: Path = DEFAULT_CALIBRATION_PATH) -> SimCalibration:
        payload = json.loads(Path(path).read_text())
        return cls(
            result=payload["result"],
            event=payload["event"],
            result_by_count=payload.get("result_by_count"),
            event_by_stretch=payload.get("event_by_stretch"),
        )

    def save(self, path: Path = DEFAULT_CALIBRATION_PATH, meta: dict | None = None) -> None:
        payload: dict = {"result": self.result, "event": self.event}
        if self.result_by_count is not None:
            payload["result_by_count"] = self.result_by_count
        if self.event_by_stretch is not None:
            payload["event_by_stretch"] = self.event_by_stretch
        if meta:
            payload["meta"] = meta
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def multipliers(
        self, is_top_half: bool
    ) -> tuple[dict[str, float], dict[str, float]]:
        side = "top" if is_top_half else "bottom"
        return self.result.get(side, {}), self.event.get(side, {})

    def result_multipliers_by_count(
        self, is_top_half: bool, stretch: bool = False
    ) -> dict[tuple[int, int], dict[str, float]]:
        """Per-count result multipliers for one side/stretch state."""
        side = "top" if is_top_half else "bottom"
        aggregate = self.result.get(side, {})
        side_cells = (self.result_by_count or {}).get(side, {})
        by_count = side_cells.get(stretch_key(stretch), {})
        out: dict[tuple[int, int], dict[str, float]] = {}
        for balls in range(4):
            for strikes in range(3):
                key = f"{balls}-{strikes}"
                out[(balls, strikes)] = dict(by_count.get(key, aggregate))
        return out

    def event_multipliers(
        self, is_top_half: bool, stretch: bool = False
    ) -> dict[str, float]:
        """Event multipliers for one side/stretch state; aggregate fallback."""
        side = "top" if is_top_half else "bottom"
        by_stretch = (self.event_by_stretch or {}).get(side, {})
        cell = by_stretch.get(stretch_key(stretch))
        if cell:
            return dict(cell)
        return dict(self.event.get(side, {}))


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
