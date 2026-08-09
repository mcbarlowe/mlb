from __future__ import annotations

import json
import random
from pathlib import Path

import polars as pl
import pytest

from src.sim.base_out import BaseOutEngine
from src.sim.game import (
    Batter,
    GameConfig,
    GameSimulator,
    Lineup,
    Pitcher,
    summarize,
)
from src.sim.matchup import effective_bat_side
from src.sim.pa import FixedDistributionProvider, MatchupOutcomeProvider
from src.sim.pitch_mix import PitchMixProfiles, build_pitch_mix_tables

# --- pitch mix ----------------------------------------------------------------


def _mix_frames():
    # Pitcher 1: heavy slider usage at 0-2 (60 pitches). Pitcher 2: two pitches.
    mix = pl.DataFrame(
        {
            "pitcher_id": [1, 1, 2, 2],
            "balls": [0, 0, 0, 0],
            "strikes": [2, 2, 2, 2],
            "pitch_type": ["SL", "FF", "FF", "SI"],
            "n": [45, 15, 1, 1],
        }
    )
    locations = pl.DataFrame(
        {
            "pitcher_id": [1] * 5 + [2],
            "balls": [0] * 6,
            "strikes": [2] * 6,
            "pitch_type": ["SL", "SL", "SL", "FF", "FF", "FF"],
            "px": [0.1, 0.2, -0.1, 0.4, -0.3, 0.0],
            "pz": [1.8, 2.0, 2.2, 1.5, 2.6, 2.4],
        }
    )
    return mix, locations


def test_type_distribution_shrinks_toward_league():
    mix, locations = _mix_frames()
    profiles = PitchMixProfiles(mix, locations, shrinkage=60.0, seed=0)

    heavy = profiles.type_distribution(1, 0, 2)
    league = profiles._league_mix[(0, 2)]
    # Pitcher 1 throws 75% sliders; league is ~74% FF-ish. Blend sits between.
    own_sl = 45 / 60
    assert league["SL"] < heavy["SL"] < own_sl

    thin = profiles.type_distribution(2, 0, 2)
    # Two observed pitches barely move the league prior.
    assert abs(thin["SL"] - league["SL"]) < 0.03

    unknown = profiles.type_distribution(999, 0, 2)
    assert unknown == pytest.approx(league)


def test_sample_locations_prefers_own_then_league():
    mix, locations = _mix_frames()
    profiles = PitchMixProfiles(mix, locations, seed=0)
    rng = random.Random(0)
    # Pitcher 2 has no own FF locations stored; league FF pool tops up.
    picked = profiles.sample_locations(2, 0, 2, "FF", n=5, rng=rng)
    assert len(picked) == 5
    # Type never seen at this count anywhere: falls back to any-type pool.
    picked = profiles.sample_locations(1, 0, 2, "CU", n=4, rng=rng)
    assert len(picked) == 4

    with pytest.raises(KeyError):
        profiles.sample_locations(1, 3, 0, "FF", n=5, rng=rng)  # count unseen


def test_inputs_by_count_covers_types_with_locations():
    mix, locations = _mix_frames()
    profiles = PitchMixProfiles(mix, locations, seed=0)
    with pytest.raises(KeyError):
        # Only count 0-2 exists in the fixture, so full coverage must fail.
        profiles.inputs_by_count(1)
    types = profiles.type_distribution(1, 0, 2)
    locations_by_type = {
        t: profiles.sample_locations(1, 0, 2, t, 3, random.Random(1)) for t in types
    }
    assert set(locations_by_type) == set(types)
    assert all(len(v) == 3 for v in locations_by_type.values())


def test_build_pitch_mix_tables_counts_and_canonicalizes():
    raw = pl.DataFrame(
        {
            "pitcher_id": [1, 1, 1],
            "game_pk": [10, 10, 10],
            "at_bat_index": [0, 0, 0],
            "pitch_number": [1, 2, 3],
            "count_after_pitch": ["0-1", "1-1", "1-2"],
            "pitch_type_code": ["FF", "XX", "SL"],
            "px": [0.0, 0.1, 0.2],
            "pz": [2.0, 2.1, 2.2],
        }
    )
    mix, locations = build_pitch_mix_tables(raw)
    # First pitch of the AB is at 0-0; unknown code XX maps to OTHER.
    first = mix.filter((pl.col("balls") == 0) & (pl.col("strikes") == 0))
    assert first["pitch_type"].to_list() == ["FF"]
    assert set(mix["pitch_type"].to_list()) == {"FF", "OTHER", "SL"}
    assert locations.height == 3


# --- matchup provider wiring ----------------------------------------------


class _StubPredictor:
    """Echoes which inputs were used so per-count wiring is observable."""

    def predict(self, state, type_probabilities, locations):
        ball_prob = 0.9 if "SL" in type_probabilities else 0.1
        return {
            "result": {"ball": ball_prob, "called_strike": 1 - ball_prob},
            "event_given_in_play": {"out": 1.0},
        }


def test_matchup_provider_uses_per_count_inputs():
    inputs = {}
    for balls in range(4):
        for strikes in range(3):
            types = {"SL": 1.0} if (balls, strikes) == (0, 2) else {"FF": 1.0}
            inputs[(balls, strikes)] = (types, [(0.0, 2.0)])

    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _StubState:
        balls: int = 0
        strikes: int = 0

    provider = MatchupOutcomeProvider(_StubPredictor(), _StubState(), inputs)
    assert provider.result_probabilities(0, 2)["ball"] == pytest.approx(0.9)
    assert provider.result_probabilities(3, 0)["ball"] == pytest.approx(0.1)


def test_matchup_provider_requires_all_counts():
    with pytest.raises(ValueError):
        MatchupOutcomeProvider(_StubPredictor(), None, {(0, 0): ({"FF": 1.0}, [])})


def test_effective_bat_side_for_switch_hitters():
    assert effective_bat_side("S", "R") == "L"
    assert effective_bat_side("S", "L") == "R"
    assert effective_bat_side("L", "L") == "L"


# --- game loop -----------------------------------------------------------


def _lineup(prefix: int) -> Lineup:
    return Lineup(
        batters=[Batter(prefix * 100 + i, "R") for i in range(9)],
        starter=Pitcher(prefix, "R"),
    )


def _engine() -> BaseOutEngine:
    # Minimal empirical rows: strikeout adds an out; home run scores 1+runners.
    table = pl.DataFrame(
        {
            "pa_outcome": ["strikeout"] * 3 + ["home_run"],
            "runners_before": [0, 0, 0, 0],
            "outs_before": [0, 1, 2, 0],
            "runners_after": [0, 0, 0, 0],
            "outs_after": [1, 2, 3, 0],
            "runs": [0, 0, 0, 1],
            "n": [1, 1, 1, 1],
        }
    )
    return BaseOutEngine(table, seed=0)


def test_all_strikeout_game_reaches_cap_and_ties():
    provider = FixedDistributionProvider({"called_strike": 1.0}, {"out": 1.0})
    sim = GameSimulator(
        lambda p, b, top: provider,
        _engine(),
        rng=random.Random(0),
        config=GameConfig(max_innings=12),
    )
    result = sim.simulate(_lineup(1), _lineup(2))
    assert result.tie
    assert result.innings == 12
    assert (result.away_runs, result.home_runs) == (0, 0)
    stats = summarize([result])
    assert stats["home_win_probability"] == 0.5


def test_home_dominant_game_ends_without_bottom_nine():
    # Away team never reaches; home team homers 10% of PAs.
    away_provider = FixedDistributionProvider({"called_strike": 1.0}, {"out": 1.0})
    home_provider = FixedDistributionProvider(
        {"called_strike": 0.9, "in_play": 0.1}, {"home_run": 1.0}
    )

    def factory(pitcher: Pitcher, batter: Batter, is_top: bool):
        return away_provider if is_top else home_provider

    sim = GameSimulator(factory, _engine(), rng=random.Random(1))
    results = sim.simulate_many(_lineup(1), _lineup(2), 50)
    for result in results:
        assert result.away_runs == 0
        assert result.home_won
        assert result.innings == 9  # home leads after top 9; bottom 9 skipped


def test_extra_innings_walkoff_with_ghost_runner():
    # Both sides strike out for 9 innings; in extras the home side singles
    # until the ghost runner scores (single from 2nd scores 100% here).
    table = pl.DataFrame(
        {
            "pa_outcome": ["strikeout"] * 3 + ["single"],
            "runners_before": [0, 0, 0, 2],
            "outs_before": [0, 1, 2, 0],
            "runners_after": [0, 0, 0, 1],
            "outs_after": [1, 2, 3, 0],
            "runs": [0, 0, 0, 1],
            "n": [1, 1, 1, 1],
        }
    )
    engine = BaseOutEngine(table, seed=0)
    k_provider = FixedDistributionProvider({"called_strike": 1.0}, {"out": 1.0})
    single_provider = FixedDistributionProvider({"in_play": 1.0}, {"single": 1.0})
    innings_seen: list[int] = []

    def factory(pitcher: Pitcher, batter: Batter, is_top: bool):
        # Home team strikes out through regulation, singles in extras.
        if is_top:
            return k_provider
        return k_provider if len(innings_seen) <= 9 else single_provider

    class _TrackingSim(GameSimulator):
        def _play_half_inning(self, inning, is_top, batting, staff, opponent):
            if is_top:
                innings_seen.append(inning)
            return super()._play_half_inning(inning, is_top, batting, staff, opponent)

    sim = _TrackingSim(factory, engine, rng=random.Random(2))
    result = sim.simulate(_lineup(1), _lineup(2))
    assert result.home_won
    assert result.innings == 10
    assert result.home_runs == 1  # ghost runner walked off
    assert result.away_runs == 0


def test_starter_pitch_limit_hands_off_to_bullpen():
    pitchers_seen: list[int] = []
    provider = FixedDistributionProvider({"called_strike": 1.0}, {"out": 1.0})

    def factory(pitcher: Pitcher, batter: Batter, is_top: bool):
        if is_top:
            pitchers_seen.append(pitcher.player_id)
        return provider

    sim = GameSimulator(
        factory,
        _engine(),
        rng=random.Random(0),
        config=GameConfig(max_innings=9, starter_pitch_limit=10),
    )
    sim.simulate(_lineup(1), _lineup(2))
    # Home starter (id 2) faces away hitters until 10 pitches, then id 0.
    assert pitchers_seen[0] == 2
    assert 0 in pitchers_seen
    assert set(pitchers_seen) == {2, 0}


def test_lineup_requires_nine_batters():
    with pytest.raises(ValueError):
        Lineup(batters=[Batter(1, "R")], starter=Pitcher(2, "R"))


# --- CLI lineup extraction --------------------------------------------------

def test_lineup_from_feed_extracts_nine_with_handedness():
    from src.sim.lineups import lineup_from_feed

    feed = json.loads(Path("example_json_files/example_live_feed.json").read_text())
    away = lineup_from_feed(feed, "away")
    home = lineup_from_feed(feed, "home")
    assert len(away.batters) == 9
    assert len(home.batters) == 9
    assert all(b.bat_side in {"L", "R", "S"} for b in away.batters)
    assert home.starter.throw_side in {"L", "R"}
