from __future__ import annotations

import pytest

from mlb.sim.team_priors import TeamSeasonResult, team_prior_offsets_from_results


def test_team_prior_offsets_use_prior_seasons_only_and_order_teams():
    results = [
        TeamSeasonResult(2022, 1, 162, 90, 820, 700),
        TeamSeasonResult(2023, 1, 162, 100, 900, 650),
        TeamSeasonResult(2024, 1, 162, 40, 520, 850),
        TeamSeasonResult(2022, 2, 162, 70, 650, 720),
        TeamSeasonResult(2023, 2, 162, 62, 600, 780),
        TeamSeasonResult(2024, 2, 162, 110, 880, 620),
    ]

    offsets = team_prior_offsets_from_results(
        results,
        prediction_season=2024,
        lookback=2,
        weights=(0.70, 0.30),
    )

    assert offsets[1] > 0.0
    assert offsets[2] < 0.0
    assert offsets[1] > offsets[2]


def test_team_prior_offsets_reject_invalid_configuration():
    results = [TeamSeasonResult(2023, 1, 162, 90, 800, 700)]

    with pytest.raises(ValueError, match="lookback"):
        team_prior_offsets_from_results(results, prediction_season=2024, lookback=0)

    with pytest.raises(ValueError, match="weights"):
        team_prior_offsets_from_results(
            results,
            prediction_season=2024,
            weights=(0.55, 0.0),
        )

    with pytest.raises(ValueError, match="win_pct_weight"):
        team_prior_offsets_from_results(
            results,
            prediction_season=2024,
            win_pct_weight=1.5,
        )
