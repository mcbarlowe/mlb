from __future__ import annotations

import math

import pytest

from src.sim.team_strength import (
    CompletedGame,
    StarterLine,
    StrengthFeatureBuilder,
    TeamStrengthPredictor,
)


def starter(
    player_id: int,
    *,
    outs: int = 18,
    earned_runs: int = 2,
    strikeouts: int = 6,
    walks: int = 2,
    home_runs: int = 1,
) -> StarterLine:
    return StarterLine(
        player_id=player_id,
        outs=outs,
        earned_runs=earned_runs,
        strikeouts=strikeouts,
        walks=walks,
        home_runs=home_runs,
        hit_batters=0,
    )


def game(
    *,
    game_pk: int = 1,
    season: int = 2024,
    away_runs: int = 2,
    home_runs: int = 5,
    away_starter: StarterLine | None = None,
    home_starter: StarterLine | None = None,
) -> CompletedGame:
    return CompletedGame(
        game_pk=game_pk,
        season=season,
        game_datetime=f"{season}-04-01T17:00:00Z",
        away_team_id=10,
        home_team_id=20,
        away_runs=away_runs,
        home_runs=home_runs,
        away_starter=away_starter or starter(100),
        home_starter=home_starter or starter(200),
    )


def test_current_game_result_does_not_leak_into_pregame_features() -> None:
    home_win_builder = StrengthFeatureBuilder()
    away_win_builder = StrengthFeatureBuilder()

    home_win_features = home_win_builder.observe(game(home_runs=5, away_runs=2))
    away_win_features = away_win_builder.observe(game(home_runs=2, away_runs=5))

    assert home_win_features == away_win_features
    assert home_win_builder.matchup_features(
        season=2024,
        away_team_id=10,
        home_team_id=20,
        away_starter_id=100,
        home_starter_id=200,
    ) != away_win_builder.matchup_features(
        season=2024,
        away_team_id=10,
        home_team_id=20,
        away_starter_id=100,
        home_starter_id=200,
    )


def test_starter_features_use_only_prior_appearances() -> None:
    builder = StrengthFeatureBuilder()
    first = builder.observe(
        game(
            away_starter=starter(
                100, outs=6, earned_runs=7, strikeouts=0, walks=5, home_runs=3
            ),
            home_starter=starter(
                200, outs=24, earned_runs=0, strikeouts=12, walks=0, home_runs=0
            ),
        )
    )
    second = builder.matchup_features(
        season=2024,
        away_team_id=10,
        home_team_id=20,
        away_starter_id=100,
        home_starter_id=200,
    )

    assert first.starter_era_edge == pytest.approx(0.0)
    assert first.starter_fip_edge == pytest.approx(0.0)
    assert second.starter_era_edge > 0.0
    assert second.starter_fip_edge > 0.0
    assert second.starter_length_edge > 0.0


def test_new_season_regresses_team_state_toward_neutral() -> None:
    builder = StrengthFeatureBuilder()
    builder.observe(game())
    before = builder.matchup_features(
        season=2024,
        away_team_id=10,
        home_team_id=20,
        away_starter_id=100,
        home_starter_id=200,
    )
    after = builder.matchup_features(
        season=2025,
        away_team_id=10,
        home_team_id=20,
        away_starter_id=100,
        home_starter_id=200,
    )

    assert after.elo_diff == pytest.approx(before.elo_diff * 0.75)
    assert after.run_edge == pytest.approx(before.run_edge * 0.75)


def test_predictor_applies_fitted_logistic_probability() -> None:
    predictor = TeamStrengthPredictor(
        coefficients=(1.0, 0.0, 0.0, 0.0, 0.0),
        intercept=0.2,
        feature_builder=StrengthFeatureBuilder(),
    )

    probability = predictor.predict_home_probability(
        season=2025,
        away_team_id=10,
        home_team_id=20,
        away_starter_id=100,
        home_starter_id=200,
    )

    assert probability == pytest.approx(1.0 / (1.0 + math.exp(-0.2)))
