"""Chronological posterior-predictive evaluation for pitcher prop models."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from mlb.pitcher_props.model import (
    SUPPORTED_MARKETS,
    PitcherPropModel,
    PitcherStart,
)


@dataclass(frozen=True)
class ComponentMetrics:
    observations: int
    mean_actual: float
    mean_prediction: float
    mae: float
    rmse: float
    count_log_loss: float
    interval_90_coverage: float
    league_mae: float
    league_rmse: float
    league_count_log_loss: float
    mae_improvement: float
    log_loss_improvement: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationReport:
    validation_season: int
    starts: int
    draws_per_start: int
    components: Mapping[str, ComponentMetrics]

    def to_dict(self) -> dict[str, object]:
        return {
            "validation_season": self.validation_season,
            "starts": self.starts,
            "draws_per_start": self.draws_per_start,
            "components": {
                market: metrics.to_dict()
                for market, metrics in self.components.items()
            },
        }


def _summary(
    actual: list[float],
    predicted: list[float],
    log_probabilities: list[float],
    covered: list[float],
) -> tuple[float, float, float, float]:
    observed = np.asarray(actual, dtype=float)
    estimates = np.asarray(predicted, dtype=float)
    errors = estimates - observed
    return (
        float(np.mean(np.abs(errors))),
        float(math.sqrt(np.mean(np.square(errors)))),
        float(-np.mean(log_probabilities)),
        float(np.mean(covered)),
    )


def evaluate_walk_forward(
    model: PitcherPropModel,
    starts: Sequence[PitcherStart],
    *,
    draws_per_start: int = 1_000,
    seed: int = 2026,
) -> EvaluationReport:
    """Predict each start before observing it, then update posterior state."""

    if not starts:
        raise ValueError("validation starts cannot be empty")
    if draws_per_start < 100:
        raise ValueError("draws_per_start must be at least 100")
    ordered = sorted(starts, key=lambda row: (row.game_time, row.game_pk, row.pitcher_id))
    if model.trained_through is not None and ordered[0].game_time <= model.trained_through:
        raise ValueError("validation must begin after the fitted training cutoff")
    seasons = {start.season for start in ordered}
    if len(seasons) != 1:
        raise ValueError("walk-forward evaluation requires one validation season")

    working = model.clone()
    observations: dict[str, dict[str, list[float]]] = {
        market: {
            "actual": [],
            "model_mean": [],
            "model_log_probability": [],
            "model_covered": [],
            "league_mean": [],
            "league_log_probability": [],
            "league_covered": [],
        }
        for market in SUPPORTED_MARKETS
    }
    for index, start in enumerate(ordered):
        row_seed = seed + index * 2
        posterior = working.simulate_components(
            pitcher_id=start.pitcher_id,
            opponent_team_id=start.opponent_team_id,
            as_of=start.game_time,
            is_home=start.is_home,
            rest_days=start.rest_days,
            samples=draws_per_start,
            seed=row_seed,
        )
        league = model.simulate_components(
            pitcher_id=-1,
            opponent_team_id=None,
            as_of=start.game_time,
            is_home=start.is_home,
            rest_days=start.rest_days,
            samples=draws_per_start,
            seed=row_seed + 1,
        )
        for market in SUPPORTED_MARKETS:
            actual = start.actual(market)
            model_draws = posterior[market]
            league_draws = league[market]
            model_probability = (
                float(np.count_nonzero(model_draws == actual)) + 0.5
            ) / (draws_per_start + 1.0)
            league_probability = (
                float(np.count_nonzero(league_draws == actual)) + 0.5
            ) / (draws_per_start + 1.0)
            model_low, model_high = np.quantile(model_draws, (0.05, 0.95))
            league_low, league_high = np.quantile(league_draws, (0.05, 0.95))
            values = observations[market]
            values["actual"].append(float(actual))
            values["model_mean"].append(float(np.mean(model_draws)))
            values["model_log_probability"].append(math.log(model_probability))
            values["model_covered"].append(float(model_low <= actual <= model_high))
            values["league_mean"].append(float(np.mean(league_draws)))
            values["league_log_probability"].append(math.log(league_probability))
            values["league_covered"].append(float(league_low <= actual <= league_high))
        working.observe(start)

    metrics: dict[str, ComponentMetrics] = {}
    for market, values in observations.items():
        model_mae, model_rmse, model_log_loss, model_coverage = _summary(
            values["actual"],
            values["model_mean"],
            values["model_log_probability"],
            values["model_covered"],
        )
        league_mae, league_rmse, league_log_loss, _league_coverage = _summary(
            values["actual"],
            values["league_mean"],
            values["league_log_probability"],
            values["league_covered"],
        )
        metrics[market] = ComponentMetrics(
            observations=len(values["actual"]),
            mean_actual=float(np.mean(values["actual"])),
            mean_prediction=float(np.mean(values["model_mean"])),
            mae=model_mae,
            rmse=model_rmse,
            count_log_loss=model_log_loss,
            interval_90_coverage=model_coverage,
            league_mae=league_mae,
            league_rmse=league_rmse,
            league_count_log_loss=league_log_loss,
            mae_improvement=league_mae - model_mae,
            log_loss_improvement=league_log_loss - model_log_loss,
        )
    return EvaluationReport(
        validation_season=next(iter(seasons)),
        starts=len(ordered),
        draws_per_start=draws_per_start,
        components=metrics,
    )


__all__ = ["ComponentMetrics", "EvaluationReport", "evaluate_walk_forward"]
