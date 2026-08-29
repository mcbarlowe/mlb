"""Matchup provider factory: outcome models × count/stretch pitch mixes.

Builds (and caches) one ``MatchupOutcomeProvider`` per (pitcher, batter,
half, stretch) with situational features frozen at neutral values — one
out, tie score, mid-game inning, second time through the order. The
stretch variant flips the models' runner feature and uses the pitcher's
from-the-stretch pitch mix. Calibration multipliers (per-side, per-count)
are applied to each provider's distributions.
"""

from __future__ import annotations

import random

from mlb.outcome.inference import OutcomeGameState, PitchOutcomePredictor
from mlb.sim.calibration import PAOutcomeCalibration, SimCalibration
from mlb.sim.game import Batter, Pitcher
from mlb.sim.pa import MatchupOutcomeProvider

DEFAULT_SZ_TOP = 3.4
DEFAULT_SZ_BOTTOM = 1.6


def effective_bat_side(bat_side: str, throw_side: str) -> str:
    """Switch hitters bat opposite the pitcher's hand."""
    if bat_side == "S":
        return "L" if throw_side == "R" else "R"
    return bat_side


class MatchupProviderFactory:
    def __init__(
        self,
        outcome_predictor: PitchOutcomePredictor,
        mix_profiles,
        season: int,
        n_locations: int = 12,
        seed: int = 0,
        calibration: SimCalibration | None = None,
        pa_outcome_calibration: PAOutcomeCalibration | None = None,
    ):
        self._predictor = outcome_predictor
        self._mix = mix_profiles
        self._season = season
        self._n_locations = n_locations
        self._seed = seed
        # Retained only for callers that still expect a factory-level stream. Location
        # sampling deliberately does not use it; see __call__.
        self._rng = random.Random(seed)
        self._calibration = calibration
        self._pa_calibration = pa_outcome_calibration
        self._cache: dict[
            tuple[int, int, bool, bool, int], MatchupOutcomeProvider
        ] = {}
        self._env_multipliers: dict[str, float] | None = None

    def set_environment(self, env_multipliers: dict[str, float] | None) -> None:
        """Per-game contact-environment multipliers (park x weather).

        Environment is constant within a game but varies across games that
        share this factory, so setting it clears the provider cache.
        """
        self._env_multipliers = env_multipliers or None
        self._cache.clear()

    def __call__(
        self,
        pitcher: Pitcher,
        batter: Batter,
        is_top_half: bool,
        stretch: bool = False,
        times_through: int = 2,
    ) -> MatchupOutcomeProvider:
        tt = 3 if times_through > 3 else (max(times_through, 1))
        key = (pitcher.player_id, batter.player_id, is_top_half, stretch, tt)
        provider = self._cache.get(key)
        if provider is not None:
            return provider

        state = OutcomeGameState(
            balls=0,
            strikes=0,
            outs=1,
            # The stretch variant represents "any runner on"; runner on first
            # is by far the most common single-runner state.
            runner_on_first=stretch,
            runner_on_second=False,
            runner_on_third=False,
            inning=5,
            is_top_half=is_top_half,
            score_diff=0,
            season=self._season,
            times_through_order=tt,
            pitcher_id=pitcher.player_id,
            batter_id=batter.player_id,
            throw_side=pitcher.throw_side,
            bat_side=effective_bat_side(batter.bat_side, pitcher.throw_side),
            sz_top=DEFAULT_SZ_TOP,
            sz_bottom=DEFAULT_SZ_BOTTOM,
        )
        # Location sampling is seeded from the matchup key rather than drawn from a shared
        # stateful stream. With a shared stream a matchup's locations depend on how many other
        # matchups were built before it, so clearing the cache between games (see
        # set_environment) rebuilt the same matchup with different locations, and identical seeds
        # produced different simulations. Keying the stream makes construction idempotent and
        # order-independent.
        rng = random.Random(f"{self._seed}|{key}")
        matchup_inputs = getattr(self._mix, "inputs_for_matchup", None)
        if matchup_inputs is None:
            inputs = self._mix.inputs_by_count(
                pitcher.player_id,
                n_locations=self._n_locations,
                rng=rng,
                stretch=stretch,
            )
        else:
            inputs = matchup_inputs(
                pitcher_id=pitcher.player_id,
                throw_side=pitcher.throw_side,
                batter_id=batter.player_id,
                bat_side=state.bat_side,
                is_top_half=is_top_half,
                times_through=tt,
                n_locations=self._n_locations,
                rng=rng,
                stretch=stretch,
            )
        result_multipliers = None
        event_multipliers = None
        if self._calibration is not None:
            result_multipliers = self._calibration.result_multipliers_by_count(
                is_top_half, stretch
            )
            event_multipliers = self._calibration.event_multipliers(
                is_top_half, stretch
            )
        pa_outcome_multipliers = (
            self._pa_calibration.for_side(is_top_half)
            if self._pa_calibration is not None
            else None
        )
        if self._env_multipliers:
            combined = dict(pa_outcome_multipliers) if pa_outcome_multipliers else {}
            for cls_, m in self._env_multipliers.items():
                combined[cls_] = combined.get(cls_, 1.0) * m
            pa_outcome_multipliers = combined
        provider = MatchupOutcomeProvider(
            self._predictor,
            state,
            inputs,
            result_multipliers=result_multipliers,
            event_multipliers=event_multipliers,
            pa_outcome_multipliers=pa_outcome_multipliers,
        )
        self._cache[key] = provider
        return provider
