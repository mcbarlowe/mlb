from __future__ import annotations

from datetime import date

import pytest

from src.sim.season import (
    SeasonProjection,
    SeasonScheduleGame,
    TeamInfo,
    TeamProjection,
    _logistic,
    _probability_to_logit,
    actual_outcomes,
    build_baseline_projection,
    evaluate_projection,
    first_regular_season_date,
    schedule_strength_offsets_from_games,
    simulate_season,
)


class PairPredictor:
    def __init__(self, probabilities: dict[tuple[int, int], float]):
        self.probabilities = probabilities
        self.calls = []

    def predict_home_probability(self, **kwargs):
        self.calls.append(kwargs)
        key = (kwargs["away_team_id"], kwargs["home_team_id"])
        return self.probabilities[key]


def _team(team_id: int, league: str, division: str) -> TeamInfo:
    return TeamInfo(
        team_id=team_id,
        abbreviation=f"T{team_id}",
        team_name=f"Team {team_id}",
        league_name=league,
        division_name=division,
    )


def _game(
    game_pk: int,
    game_date: date,
    away_team_id: int,
    home_team_id: int,
    *,
    status: str = "Preview",
    away_runs: int | None = None,
    home_runs: int | None = None,
    away_starter: int | None = None,
    home_starter: int | None = None,
) -> SeasonScheduleGame:
    return SeasonScheduleGame(
        game_pk=game_pk,
        season=2026,
        game_date=game_date,
        game_datetime=f"{game_date.isoformat()}T17:00:00Z",
        status=status,
        away_team_id=away_team_id,
        home_team_id=home_team_id,
        away_probable_pitcher_id=away_starter,
        home_probable_pitcher_id=home_starter,
        away_runs=away_runs,
        home_runs=home_runs,
    )


def test_simulate_season_uses_actual_pre_as_of_wins_and_future_probabilities():
    teams = {
        1: _team(1, "AL", "East"),
        2: _team(2, "AL", "East"),
        3: _team(3, "AL", "West"),
        4: _team(4, "AL", "West"),
    }
    games = [
        _game(
            1,
            date(2026, 3, 28),
            1,
            2,
            status="Final",
            away_runs=5,
            home_runs=3,
        ),
        _game(2, date(2026, 3, 29), 2, 1, away_starter=20, home_starter=10),
        _game(3, date(2026, 3, 29), 3, 4),
    ]
    predictor = PairPredictor({(2, 1): 1.0, (3, 4): 0.0})

    projection = simulate_season(
        games=games,
        teams=teams,
        as_of_date=date(2026, 3, 29),
        trials=10,
        predictor=predictor,
        wild_cards_per_league=0,
    )

    by_team = projection.by_team_id()
    assert by_team[1].actual_wins_as_of == 1
    assert by_team[1].expected_wins == 2.0
    assert by_team[1].division_win_prob == 1.0
    assert by_team[1].playoff_prob == 1.0
    assert by_team[1].division_series_prob == 1.0
    assert by_team[1].league_championship_prob == 1.0
    assert by_team[2].expected_wins == 0.0
    assert by_team[3].expected_wins == 1.0
    assert by_team[4].playoff_prob == 0.0
    assert predictor.calls[0]["away_starter_id"] == 20
    assert predictor.calls[0]["home_starter_id"] == 10
    assert predictor.calls[1]["away_starter_id"] == 0
    assert predictor.calls[1]["home_starter_id"] == 0
    assert projection.probability_logit_scale == 1.0
    assert projection.team_strength_sd == 0.0
    assert projection.team_prior_scale == 0.0
    assert projection.market_prior_scale == 0.0
    assert projection.schedule_strength_scale == 0.0


def test_probability_adjustments_are_logit_based():
    base_probability = 0.80

    assert _logistic(_probability_to_logit(base_probability)) == pytest.approx(
        base_probability
    )
    shrunk = _logistic(_probability_to_logit(base_probability) * 0.5)

    assert 0.5 < shrunk < base_probability
    assert _logistic(_probability_to_logit(0.5) + 0.3) > 0.5


def test_simulate_season_applies_team_prior_offsets_to_future_logits():
    teams = {1: _team(1, "AL", "East"), 2: _team(2, "AL", "East")}
    games = [_game(1, date(2026, 3, 29), 1, 2)]
    predictor = PairPredictor({(1, 2): 0.5})

    projection = simulate_season(
        games=games,
        teams=teams,
        as_of_date=date(2026, 3, 29),
        trials=3,
        predictor=predictor,
        wild_cards_per_league=0,
        team_prior_offsets={1: 100.0, 2: 0.0},
        team_prior_scale=1.0,
    )

    by_team = projection.by_team_id()
    assert by_team[1].expected_wins == 1.0
    assert by_team[1].division_win_prob == 1.0
    assert by_team[2].expected_wins == 0.0
    assert projection.team_prior_scale == 1.0


def test_simulate_season_applies_market_prior_offsets_to_future_logits():
    teams = {1: _team(1, "AL", "East"), 2: _team(2, "AL", "East")}
    games = [_game(1, date(2026, 3, 29), 1, 2)]
    predictor = PairPredictor({(1, 2): 0.5})

    projection = simulate_season(
        games=games,
        teams=teams,
        as_of_date=date(2026, 3, 29),
        trials=3,
        predictor=predictor,
        wild_cards_per_league=0,
        market_prior_offsets={1: 100.0, 2: 0.0},
        market_prior_scale=1.0,
    )

    by_team = projection.by_team_id()
    assert by_team[1].expected_wins == 1.0
    assert by_team[1].division_win_prob == 1.0
    assert by_team[2].expected_wins == 0.0
    assert projection.market_prior_scale == 1.0


def test_simulate_season_applies_schedule_strength_offsets_to_future_logits():
    teams = {1: _team(1, "AL", "East"), 2: _team(2, "AL", "East")}
    games = [_game(1, date(2026, 3, 29), 1, 2)]
    predictor = PairPredictor({(1, 2): 0.5})

    projection = simulate_season(
        games=games,
        teams=teams,
        as_of_date=date(2026, 3, 29),
        trials=3,
        predictor=predictor,
        wild_cards_per_league=0,
        schedule_strength_offsets={1: 0.0, 2: 100.0},
        schedule_strength_scale=1.0,
    )

    by_team = projection.by_team_id()
    assert by_team[1].expected_wins == 0.0
    assert by_team[2].expected_wins == 1.0
    assert by_team[2].division_win_prob == 1.0
    assert projection.schedule_strength_scale == 1.0


def test_schedule_strength_offsets_use_future_opponent_priors_only():
    games = [
        _game(1, date(2026, 3, 28), 1, 2),
        _game(2, date(2026, 3, 29), 1, 3),
    ]

    offsets = schedule_strength_offsets_from_games(
        games,
        as_of_date=date(2026, 3, 29),
        team_prior_offsets={1: 0.5, 2: -1.0, 3: 1.0},
    )

    assert offsets[1] < 0.0
    assert offsets[2] == 0.0
    assert offsets[3] > 0.0


def test_simulate_season_rejects_invalid_probability_adjustments():
    teams = {1: _team(1, "AL", "East"), 2: _team(2, "AL", "East")}
    games = [_game(1, date(2026, 3, 29), 1, 2)]
    predictor = PairPredictor({(1, 2): 0.5})

    with pytest.raises(ValueError, match="probability_logit_scale"):
        simulate_season(
            games=games,
            teams=teams,
            as_of_date=date(2026, 3, 29),
            trials=1,
            predictor=predictor,
            probability_logit_scale=0.0,
        )

    with pytest.raises(ValueError, match="team_strength_sd"):
        simulate_season(
            games=games,
            teams=teams,
            as_of_date=date(2026, 3, 29),
            trials=1,
            predictor=predictor,
            team_strength_sd=-0.1,
        )

    with pytest.raises(ValueError, match="team_prior_scale"):
        simulate_season(
            games=games,
            teams=teams,
            as_of_date=date(2026, 3, 29),
            trials=1,
            predictor=predictor,
            team_prior_scale=-0.1,
        )
    with pytest.raises(ValueError, match="market_prior_scale"):
        simulate_season(
            games=games,
            teams=teams,
            as_of_date=date(2026, 3, 29),
            trials=1,
            predictor=predictor,
            market_prior_scale=-0.1,
        )

    with pytest.raises(ValueError, match="schedule_strength_scale"):
        simulate_season(
            games=games,
            teams=teams,
            as_of_date=date(2026, 3, 29),
            trials=1,
            predictor=predictor,
            schedule_strength_scale=-0.1,
        )


def test_evaluate_projection_scores_against_final_standings():
    teams = {
        1: _team(1, "AL", "East"),
        2: _team(2, "AL", "East"),
        3: _team(3, "AL", "West"),
        4: _team(4, "AL", "West"),
    }
    games = [
        _game(1, date(2026, 3, 28), 1, 2, status="Final", away_runs=5, home_runs=3),
        _game(2, date(2026, 3, 29), 2, 1, status="Final", away_runs=2, home_runs=6),
        _game(3, date(2026, 3, 29), 3, 4, status="Final", away_runs=4, home_runs=1),
    ]
    projection = SeasonProjection(
        season=2026,
        as_of_date=date(2026, 3, 28),
        trials=100,
        wild_cards_per_league=0,
        teams=(
            TeamProjection(1, 0, 1.5, 0.75, 0.75),
            TeamProjection(2, 0, 0.5, 0.25, 0.25),
            TeamProjection(3, 0, 1.0, 0.80, 0.80),
            TeamProjection(4, 0, 0.0, 0.20, 0.20),
        ),
    )

    evaluation = evaluate_projection(projection, games, teams)

    assert evaluation.teams == 4
    assert evaluation.actual_wins_mae == pytest.approx(0.25)
    assert evaluation.actual_wins_rmse == pytest.approx(0.125**0.5)
    assert evaluation.division_brier == pytest.approx(
        ((0.75 - 1.0) ** 2 + 0.25**2 + (0.80 - 1.0) ** 2 + 0.20**2) / 4
    )
    assert evaluation.playoff_brier == evaluation.division_brier


def test_baseline_projection_uses_flat_berths_and_coin_flip_schedule():
    teams = {
        1: _team(1, "AL", "East"),
        2: _team(2, "AL", "East"),
        3: _team(3, "AL", "West"),
        4: _team(4, "AL", "West"),
    }
    games = [
        _game(
            1,
            date(2026, 3, 28),
            1,
            2,
            status="Final",
            away_runs=5,
            home_runs=3,
        ),
        _game(2, date(2026, 3, 29), 2, 1),
        _game(3, date(2026, 3, 29), 3, 4),
    ]

    baseline = build_baseline_projection(
        games=games,
        teams=teams,
        as_of_date=date(2026, 3, 29),
        wild_cards_per_league=0,
    )

    by_team = baseline.by_team_id()
    assert baseline.trials == 0
    assert by_team[1].actual_wins_as_of == 1
    assert by_team[1].expected_wins == 1.5
    assert by_team[1].division_win_prob == 0.5
    assert by_team[1].playoff_prob == 0.5
    assert by_team[1].division_series_prob == 0.5
    assert by_team[1].league_championship_prob == 0.5
    assert by_team[1].championship_prob == 0.25
    assert by_team[2].expected_wins == 0.5
    assert by_team[3].expected_wins == 0.5
    assert by_team[4].expected_wins == 0.5


def test_simulate_season_records_playoff_stage_probabilities():
    teams = {
        1: _team(1, "AL", "East"),
        2: _team(2, "AL", "East"),
        3: _team(3, "AL", "Central"),
        4: _team(4, "AL", "Central"),
        5: _team(5, "AL", "West"),
        6: _team(6, "AL", "West"),
        7: _team(7, "NL", "East"),
        8: _team(8, "NL", "East"),
        9: _team(9, "NL", "Central"),
        10: _team(10, "NL", "Central"),
        11: _team(11, "NL", "West"),
        12: _team(12, "NL", "West"),
    }
    games = [
        _game(
            index,
            date(2026, 3, 28),
            index * 2 - 1,
            index * 2,
            status="Final",
            away_runs=5,
            home_runs=3,
        )
        for index in range(1, 7)
    ]

    projection = simulate_season(
        games=games,
        teams=teams,
        as_of_date=date(2026, 3, 29),
        trials=50,
        predictor=PairPredictor({}),
        wild_cards_per_league=3,
    )

    rows = projection.teams
    assert sum(row.playoff_prob for row in rows) == pytest.approx(12.0)
    assert sum(row.division_series_prob for row in rows) == pytest.approx(8.0)
    assert sum(row.league_championship_prob for row in rows) == pytest.approx(4.0)
    assert sum(row.world_series_prob for row in rows) == pytest.approx(2.0)
    assert sum(row.championship_prob for row in rows) == pytest.approx(1.0)
    for row in rows:
        assert row.playoff_prob >= row.division_series_prob
        assert row.division_series_prob >= row.league_championship_prob
        assert row.league_championship_prob >= row.world_series_prob
        assert row.world_series_prob >= row.championship_prob


def test_actual_outcomes_rejects_incomplete_season():
    teams = {1: _team(1, "AL", "East"), 2: _team(2, "AL", "East")}
    games = [_game(1, date(2026, 3, 29), 1, 2)]

    with pytest.raises(ValueError, match="not final"):
        actual_outcomes(games, teams)


def test_first_regular_season_date_rejects_empty_schedule():
    with pytest.raises(ValueError, match="empty schedule"):
        first_regular_season_date([])
