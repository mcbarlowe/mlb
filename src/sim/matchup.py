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

from src.outcome.inference import OutcomeGameState, PitchOutcomePredictor
from src.sim.calibration import PAOutcomeCalibration, SimCalibration
from src.sim.game import Batter, Pitcher
from src.sim.pa import MatchupOutcomeProvider
from src.sim.pitch_mix import PitchMixProfiles

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
        mix_profiles: PitchMixProfiles,
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
        self._rng = random.Random(seed)
        self._calibration = calibration
        self._pa_calibration = pa_outcome_calibration
        self._cache: dict[tuple[int, int, bool, bool], MatchupOutcomeProvider] = {}

    def __call__(
        self,
        pitcher: Pitcher,
        batter: Batter,
        is_top_half: bool,
        stretch: bool = False,
    ) -> MatchupOutcomeProvider:
        key = (pitcher.player_id, batter.player_id, is_top_half, stretch)
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
            times_through_order=2,
            pitcher_id=pitcher.player_id,
            batter_id=batter.player_id,
            throw_side=pitcher.throw_side,
            bat_side=effective_bat_side(batter.bat_side, pitcher.throw_side),
            sz_top=DEFAULT_SZ_TOP,
            sz_bottom=DEFAULT_SZ_BOTTOM,
        )
        inputs = self._mix.inputs_by_count(
            pitcher.player_id,
            n_locations=self._n_locations,
            rng=self._rng,
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
