"""Plate-appearance simulator.

Chains the count machine with per-pitch result sampling and Stage B
in-play events. Pitch-level distributions come from a provider so the
simulator itself has no model dependencies:

- ``FixedDistributionProvider`` — explicit probabilities (tests, baselines)
- ``OutcomeModelProvider`` — precomputes, for every count, the marginal
  Stage A/Stage B distributions from the trained outcome models given one
  matchup's predicted pitch-type distribution and location sample. Sampling
  a PA is then pure table lookups, fast enough for Monte Carlo.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Protocol

from src.sim.count_machine import apply_pitch_result

RESULT_CLASSES = [
    "ball",
    "called_strike",
    "swinging_strike",
    "foul",
    "in_play",
    "hit_by_pitch",
]
EVENT_CLASSES = ["out", "single", "double", "triple", "home_run", "reached_on_error"]
MAX_PITCHES_PER_PA = 25


class PitchDistributionProvider(Protocol):
    """Per-count pitch result and in-play event distributions."""

    def result_probabilities(self, balls: int, strikes: int) -> dict[str, float]: ...

    def event_probabilities(self, balls: int, strikes: int) -> dict[str, float]: ...


@dataclass(frozen=True)
class PAResult:
    outcome: str  # one of count_machine.PA_OUTCOMES
    n_pitches: int
    final_balls: int
    final_strikes: int


class FixedDistributionProvider:
    """Same distributions at every count; useful for tests and baselines."""

    def __init__(self, result_probs: dict[str, float], event_probs: dict[str, float]):
        self._result_probs = result_probs
        self._event_probs = event_probs

    def result_probabilities(self, balls: int, strikes: int) -> dict[str, float]:
        return self._result_probs

    def event_probabilities(self, balls: int, strikes: int) -> dict[str, float]:
        return self._event_probs


Locations = list[tuple[float, float]] | dict[str, list[tuple[float, float]]]
CountInputs = Mapping[tuple[int, int], tuple[dict[str, float], Locations]]


def _precompute_count_tables(
    outcome_predictor, base_state, inputs_by_count: CountInputs
) -> tuple[dict[tuple[int, int], dict[str, float]], dict[tuple[int, int], dict[str, float]]]:
    result: dict[tuple[int, int], dict[str, float]] = {}
    event: dict[tuple[int, int], dict[str, float]] = {}
    for (balls, strikes), (type_probabilities, locations) in inputs_by_count.items():
        state = replace(base_state, balls=balls, strikes=strikes)
        predicted = outcome_predictor.predict(state, type_probabilities, locations)
        result[(balls, strikes)] = predicted["result"]
        event[(balls, strikes)] = predicted["event_given_in_play"]
    return result, event


class OutcomeModelProvider:
    """Precomputed per-count outcome distributions for one matchup.

    ``outcome_predictor`` is a ``src.outcome.inference.PitchOutcomePredictor``;
    ``base_state`` is an ``OutcomeGameState`` describing the matchup/situation;
    ``type_probabilities``/``locations`` come from the upstream pitch models.
    The pitch-type and location distributions are held fixed across counts —
    appropriate for live one-pitch use where they describe the next pitch.
    For full-PA simulation prefer ``MatchupOutcomeProvider`` with
    count-conditioned inputs.
    """

    def __init__(
        self,
        outcome_predictor,
        base_state,
        type_probabilities: dict[str, float],
        locations: list[tuple[float, float]],
    ):
        inputs: CountInputs = {
            (balls, strikes): (type_probabilities, locations)
            for balls in range(4)
            for strikes in range(3)
        }
        self._result, self._event = _precompute_count_tables(
            outcome_predictor, base_state, inputs
        )

    def result_probabilities(self, balls: int, strikes: int) -> dict[str, float]:
        return self._result[(balls, strikes)]

    def event_probabilities(self, balls: int, strikes: int) -> dict[str, float]:
        return self._event[(balls, strikes)]


class MatchupOutcomeProvider:
    """Per-count outcome distributions from count-conditioned pitch inputs.

    ``inputs_by_count`` maps every (balls, strikes) count to that count's
    (type distribution, location sample) — see
    ``src.sim.pitch_mix.PitchMixProfiles.inputs_by_count``. Optional
    per-class ``result_multipliers``/``event_multipliers`` (see
    ``src.sim.calibration``) rescale each count's distributions.
    """

    def __init__(
        self,
        outcome_predictor,
        base_state,
        inputs_by_count: CountInputs,
        result_multipliers: Mapping[str, float]
        | Mapping[tuple[int, int], Mapping[str, float]]
        | None = None,
        event_multipliers: Mapping[str, float] | None = None,
    ):
        from src.sim.calibration import apply_multipliers

        missing = {
            (b, s) for b in range(4) for s in range(3)
        } - set(inputs_by_count)
        if missing:
            raise ValueError(f"inputs_by_count missing counts: {sorted(missing)}")
        self._result, self._event = _precompute_count_tables(
            outcome_predictor, base_state, inputs_by_count
        )
        if result_multipliers:
            per_count = all(
                isinstance(key, tuple) for key in result_multipliers
            )
            self._result = {
                count: apply_multipliers(
                    probs,
                    dict(result_multipliers.get(count, {}))  # type: ignore[arg-type]
                    if per_count
                    else dict(result_multipliers),  # type: ignore[arg-type]
                )
                for count, probs in self._result.items()
            }
        if event_multipliers:
            self._event = {
                count: apply_multipliers(probs, dict(event_multipliers))
                for count, probs in self._event.items()
            }

    def result_probabilities(self, balls: int, strikes: int) -> dict[str, float]:
        return self._result[(balls, strikes)]

    def event_probabilities(self, balls: int, strikes: int) -> dict[str, float]:
        return self._event[(balls, strikes)]


def _sample(probabilities: dict[str, float], classes: list[str], rng: random.Random) -> str:
    weights = [max(probabilities.get(cls, 0.0), 0.0) for cls in classes]
    total = sum(weights)
    if total <= 0:
        raise ValueError("Probabilities sum to zero; cannot sample")
    return rng.choices(classes, weights=weights)[0]


def simulate_plate_appearance(
    provider: PitchDistributionProvider,
    rng: random.Random,
    balls: int = 0,
    strikes: int = 0,
) -> PAResult:
    """Simulate one plate appearance from the given starting count."""
    n_pitches = 0
    while True:
        n_pitches += 1
        if n_pitches > MAX_PITCHES_PER_PA:
            raise RuntimeError("Plate appearance failed to terminate")
        result = _sample(provider.result_probabilities(balls, strikes), RESULT_CLASSES, rng)
        transition = apply_pitch_result(balls, strikes, result)
        if transition.terminal is None:
            balls, strikes = transition.balls, transition.strikes
            continue
        if transition.in_play:
            event = _sample(
                provider.event_probabilities(balls, strikes), EVENT_CLASSES, rng
            )
            return PAResult(event, n_pitches, balls, strikes)
        return PAResult(transition.terminal, n_pitches, balls, strikes)
