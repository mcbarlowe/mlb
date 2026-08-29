from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from scripts.backtest_season_projections import (
    ProbabilityCalibration,
    _apply_playoff_calibration,
    _candidate_params,
    _fit_probability_calibration,
    _market_tuning_season_count,
    _season_summary_rows,
)
from mlb.sim.season import SeasonEvaluation, SeasonProjection, TeamProjection


def test_probability_calibration_shrinks_overconfident_playoff_probs():
    calibration = ProbabilityCalibration(anchor=0.40, slope=0.50, samples=90)

    assert 0.40 < calibration.apply(0.80) < 0.80
    assert 0.10 < calibration.apply(0.10) < 0.40


def test_fit_probability_calibration_can_improve_brier_on_fit_sample():
    probabilities = [0.10] * 20 + [0.90] * 20
    outcomes = [0.0] * 15 + [1.0] * 5 + [0.0] * 5 + [1.0] * 15

    calibration = _fit_probability_calibration(probabilities, outcomes)

    assert calibration is not None
    raw_brier = sum((p - y) ** 2 for p, y in zip(probabilities, outcomes, strict=True))
    calibrated_brier = sum(
        (calibration.apply(p) - y) ** 2
        for p, y in zip(probabilities, outcomes, strict=True)
    )
    assert calibrated_brier < raw_brier


def test_apply_playoff_calibration_only_updates_playoff_probabilities():
    projection = SeasonProjection(
        season=2026,
        as_of_date=date(2026, 3, 29),
        trials=100,
        wild_cards_per_league=3,
        teams=(
            TeamProjection(1, 0, 90.0, 0.70, 0.80, 0.60, 0.30, 0.15, 0.08),
            TeamProjection(2, 0, 70.0, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01),
        ),
        probability_logit_scale=0.85,
        team_strength_sd=0.20,
        team_prior_scale=0.25,
        schedule_strength_scale=0.50,
        market_prior_scale=0.75,
    )

    calibrated = _apply_playoff_calibration(
        projection,
        ProbabilityCalibration(anchor=0.40, slope=0.50, samples=90),
    )

    assert calibrated.teams[0].expected_wins == 90.0
    assert calibrated.teams[0].division_win_prob == 0.70
    assert calibrated.teams[0].playoff_prob < 0.80
    assert calibrated.teams[1].playoff_prob > 0.20
    assert calibrated.teams[0].division_series_prob / calibrated.teams[
        0
    ].playoff_prob == pytest.approx(0.60 / 0.80)
    assert calibrated.schedule_strength_scale == 0.50
    assert calibrated.market_prior_scale == 0.75
    assert calibrated.playoff_calibration_slope == 0.50


def test_candidate_params_enable_market_grid_only_with_market_input():
    base = {
        "probability_logit_scale_grid": [1.0],
        "team_strength_sd_grid": [0.0],
        "team_prior_scale_grid": [0.0],
        "market_prior_scale_grid": [0.0, 0.5],
        "schedule_strength_scale_grid": [0.0],
    }

    without_market = _candidate_params(SimpleNamespace(**base, market_win_totals=None))
    with_market = _candidate_params(
        SimpleNamespace(**base, market_win_totals="win_totals.csv")
    )

    assert [params.market_prior_scale for params in without_market] == [0.0]
    assert [params.market_prior_scale for params in with_market] == [0.0, 0.5]


def test_market_tuning_season_count_requires_prior_market_data():
    contexts = (
        SimpleNamespace(market_prior_offsets={}),
        SimpleNamespace(market_prior_offsets={1: 0.2}),
        SimpleNamespace(market_prior_offsets={2: -0.1}),
    )

    assert _market_tuning_season_count(contexts) == 2


def test_season_summary_rows_include_baseline_improvement():
    model = SeasonEvaluation(
        season=2026,
        teams=30,
        actual_wins_mae=6.0,
        actual_wins_rmse=8.0,
        division_brier=0.12,
        division_log_loss=0.40,
        playoff_brier=0.18,
        playoff_log_loss=0.50,
    )
    baseline = SeasonEvaluation(
        season=2026,
        teams=30,
        actual_wins_mae=9.0,
        actual_wins_rmse=11.0,
        division_brier=0.16,
        division_log_loss=0.55,
        playoff_brier=0.24,
        playoff_log_loss=0.70,
    )

    rows = _season_summary_rows([model], [baseline])

    assert [row["projection_type"] for row in rows] == [
        "model",
        "baseline",
        "improvement_vs_baseline",
    ]
    assert rows[2]["actual_wins_mae"] == pytest.approx(3.0)
    assert rows[2]["division_brier"] == pytest.approx(0.04)
    assert rows[2]["playoff_brier"] == pytest.approx(0.06)
