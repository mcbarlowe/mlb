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
