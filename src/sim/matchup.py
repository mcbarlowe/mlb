"""Matchup provider factory: outcome models × count-conditioned pitch mixes.

Builds (and caches) one ``MatchupOutcomeProvider`` per (pitcher, batter,
half) with situational features frozen at neutral values — bases empty,
one out, tie score, mid-game inning, second time through the order. Count,
matchup identity, and profile features carry the signal.
"""

from __future__ import annotations

import random

from src.outcome.inference import OutcomeGameState, PitchOutcomePredictor
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
    ):
        self._predictor = outcome_predictor
        self._mix = mix_profiles
        self._season = season
        self._n_locations = n_locations
        self._rng = random.Random(seed)
        self._cache: dict[tuple[int, int, bool], MatchupOutcomeProvider] = {}

    def __call__(
        self, pitcher: Pitcher, batter: Batter, is_top_half: bool
    ) -> MatchupOutcomeProvider:
        key = (pitcher.player_id, batter.player_id, is_top_half)
        provider = self._cache.get(key)
        if provider is not None:
            return provider

        state = OutcomeGameState(
            balls=0,
            strikes=0,
            outs=1,
            runner_on_first=False,
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
            pitcher.player_id, n_locations=self._n_locations, rng=self._rng
        )
        provider = MatchupOutcomeProvider(self._predictor, state, inputs)
        self._cache[key] = provider
        return provider
