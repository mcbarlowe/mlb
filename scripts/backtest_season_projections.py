#!/usr/bin/env python3
"""Backtest season standings projections with optional preseason market priors."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.sim.market_priors import load_market_prior_offsets
from src.sim.projection_charts import write_projection_graphics
from src.sim.season import (
    HomeWinPredictor,
    SeasonEvaluation,
    SeasonProjection,
    SeasonScheduleGame,
    TeamInfo,
    TeamProjection,
    actual_outcomes,
    build_baseline_projection,
    evaluate_projection,
    first_regular_season_date,
    load_season_schedule,
    load_team_info,
    schedule_strength_offsets_from_games,
    simulate_season,
)
from src.sim.team_priors import load_team_prior_offsets
from src.sim.team_strength import (
    load_completed_games,
    train_strength_model,
)


@dataclass(frozen=True)
class SimulationParams:
    probability_logit_scale: float
    team_strength_sd: float
    team_prior_scale: float
    schedule_strength_scale: float
    market_prior_scale: float


@dataclass(frozen=True)
class ProjectionContext:
    season: int
    schedule: tuple[SeasonScheduleGame, ...]
    as_of_date: date
    predictor: HomeWinPredictor
    train_seasons: tuple[int, ...]
    observed_games: int
    skipped_games: int
    team_prior_offsets: dict[int, float]
    schedule_strength_offsets: dict[int, float]
    market_prior_offsets: dict[int, float]


@dataclass(frozen=True)
class LabeledProjection:
    projection_type: str
    projection: SeasonProjection


@dataclass(frozen=True)
class ProbabilityCalibration:
    anchor: float
    slope: float
    samples: int

    def apply(self, probability: float) -> float:
        anchor_logit = _probability_to_logit(self.anchor)
        feature = _probability_to_logit(probability) - anchor_logit
        return _logistic(anchor_logit + self.slope * feature)


@dataclass(frozen=True)
class ModelAdjustments:
    params: SimulationParams
    tune_seasons: tuple[int, ...]
    playoff_calibration: ProbabilityCalibration | None
    calibration_seasons: tuple[int, ...]


ContextCache = dict[tuple[int, str | None], ProjectionContext]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest preseason division/playoff projections."
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[2022, 2023, 2024, 2025],
        help="Completed seasons to evaluate. Defaults to the current playoff format.",
    )
    parser.add_argument("--start-season", type=int, default=2015)
    parser.add_argument(
        "--train-window",
        type=int,
        default=4,
        help="Prior seasons used to fit each walk-forward model.",
    )
    parser.add_argument("--trials", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--wild-cards-per-league",
        type=int,
        default=3,
        help="Use 3 for MLB seasons 2022 and later.",
    )
    parser.add_argument(
        "--probability-logit-scale",
        type=float,
        default=1.0,
        help="Fixed game-probability logit scale when tuning is disabled or unavailable.",
    )
    parser.add_argument(
        "--team-strength-sd",
        type=float,
        default=0.20,
        help="Fixed per-trial team latent strength standard deviation when tuning is disabled or unavailable.",
    )
    parser.add_argument(
        "--tune-simulation-params",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tune logit scale/team uncertainty on prior seasons only before evaluating each target season.",
    )
    parser.add_argument(
        "--tune-window",
        type=int,
        default=3,
        help="Number of prior seasons used to tune simulation parameters.",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=1_000,
        help="Trials per candidate during prior-season simulation-parameter tuning.",
    )
    parser.add_argument(
        "--probability-logit-scale-grid",
        type=float,
        nargs="+",
        default=[0.70, 0.85, 1.0],
        help="Candidate logit scales for prior-season tuning.",
    )
    parser.add_argument(
        "--team-strength-sd-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.10, 0.20, 0.30],
        help="Candidate team-strength standard deviations for prior-season tuning.",
    )
    parser.add_argument(
        "--team-prior-lookback",
        type=int,
        default=3,
        help="Completed prior seasons used to build preseason team prior offsets.",
    )
    parser.add_argument(
        "--team-prior-scale",
        type=float,
        default=0.0,
        help="Fixed preseason team-prior logit scale when tuning is disabled or unavailable.",
    )
    parser.add_argument(
        "--team-prior-scale-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.25],
        help="Candidate preseason team-prior scales for prior-season tuning.",
    )
    parser.add_argument(
        "--market-win-totals",
        type=Path,
        default=None,
        help="Optional CSV of preseason market win totals with season, win_total, and team_id/abbreviation/team_name.",
    )
    parser.add_argument(
        "--market-prior-scale",
        type=float,
        default=0.0,
        help="Fixed preseason market win-total logit scale when tuning is disabled or unavailable.",
    )
    parser.add_argument(
        "--market-prior-scale-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.50, 0.75, 1.0],
        help="Candidate preseason market win-total scales for prior-season tuning.",
    )
    parser.add_argument(
        "--market-prior-min-tune-seasons",
        type=int,
        default=2,
        help="Minimum prior seasons with market win totals required before tuning nonzero market-prior scales.",
    )
    parser.add_argument(
        "--schedule-strength-scale",
        type=float,
        default=0.0,
        help="Fixed remaining-schedule-strength logit scale when tuning is disabled or unavailable.",
    )
    parser.add_argument(
        "--schedule-strength-scale-grid",
        type=float,
        nargs="+",
        default=[0.0],
        help="Candidate remaining-schedule-strength scales for prior-season tuning.",
    )
    parser.add_argument(
        "--calibrate-playoff-probs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optionally fit anchored playoff-probability calibration on prior seasons before evaluating each target season.",
    )
    parser.add_argument(
        "--playoff-calibration-min-teams",
        type=int,
        default=30,
        help="Minimum prior-season team outcomes required to fit playoff-probability calibration.",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Override projection date for a single --seasons value; defaults to first regular-season date.",
    )
    parser.add_argument(
        "--no-rosters",
        action="store_true",
        help="Fit the faster team/starter-only variant by omitting batter and bullpen history.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional CSV path for team-level model and baseline projection rows.",
    )
    parser.add_argument(
        "--calibration-out",
        type=Path,
        default=None,
        help="Optional CSV path for aggregate probability calibration buckets.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional CSV path for per-season model, baseline, and improvement metric rows.",
    )
    parser.add_argument(
        "--graphics-out-dir",
        type=Path,
        default=None,
        help="Optional directory for model playoff-probability and playoff-stage PNG graphics.",
    )
    return parser.parse_args()


def _fit_as_of_predictor(
    *,
    season: int,
    as_of_date: date,
    start_season: int,
    train_window: int,
    include_rosters: bool,
) -> tuple[HomeWinPredictor, tuple[int, ...], int]:
    train_seasons = tuple(range(season - train_window, season))
    if not train_seasons:
        raise ValueError("--train-window must be positive")
    if train_seasons[0] < start_season:
        raise ValueError(
            f"--start-season {start_season} is after first training season {train_seasons[0]}"
        )

    completed = load_completed_games(
        start_season=start_season,
        end_season=season,
        include_rosters=include_rosters,
    )
    cutoff = as_of_date.isoformat()
    as_of_games = [game for game in completed if game.game_datetime[:10] < cutoff]
    fitted = train_strength_model(
        as_of_games,
        prediction_season=season,
        train_seasons=train_seasons,
    )
    return fitted.predictor, train_seasons, len(as_of_games)


def _completed_played_schedule(
    season: int,
) -> tuple[tuple[SeasonScheduleGame, ...], int]:
    schedule = load_season_schedule(season)
    if not schedule:
        raise ValueError(f"No regular-season schedule found for {season}")
    played = tuple(game for game in schedule if game.is_final)
    if not played:
        raise ValueError(f"No scored regular-season finals found for {season}")
    return played, len(schedule) - len(played)


def _projection_context(
    season: int,
    *,
    args: argparse.Namespace,
    teams: dict[int, TeamInfo],
    cache: ContextCache,
    as_of_override: str | None,
) -> ProjectionContext:
    key = (season, as_of_override)
    if key in cache:
        return cache[key]

    schedule, skipped_games = _completed_played_schedule(season)
    as_of_date = (
        date.fromisoformat(as_of_override)
        if as_of_override is not None
        else first_regular_season_date(schedule)
    )
    predictor, train_seasons, observed_games = _fit_as_of_predictor(
        season=season,
        as_of_date=as_of_date,
        start_season=args.start_season,
        train_window=args.train_window,
        include_rosters=not args.no_rosters,
    )
    team_prior_offsets = load_team_prior_offsets(
        prediction_season=season,
        lookback=args.team_prior_lookback,
    )
    market_prior_offsets = (
        load_market_prior_offsets(
            args.market_win_totals,
            prediction_season=season,
            teams=teams,
        )
        if args.market_win_totals is not None
        else {}
    )
    schedule_strength_offsets = schedule_strength_offsets_from_games(
        schedule,
        as_of_date=as_of_date,
        team_prior_offsets=team_prior_offsets,
    )
    context = ProjectionContext(
        season=season,
        schedule=schedule,
        as_of_date=as_of_date,
        predictor=predictor,
        train_seasons=train_seasons,
        observed_games=observed_games,
        skipped_games=skipped_games,
        team_prior_offsets=team_prior_offsets,
        schedule_strength_offsets=schedule_strength_offsets,
        market_prior_offsets=market_prior_offsets,
    )
    cache[key] = context
    return context


def _fixed_params(args: argparse.Namespace) -> SimulationParams:
    return SimulationParams(
        probability_logit_scale=args.probability_logit_scale,
        team_strength_sd=args.team_strength_sd,
        team_prior_scale=args.team_prior_scale,
        market_prior_scale=args.market_prior_scale,
        schedule_strength_scale=args.schedule_strength_scale,
    )


def _candidate_params(args: argparse.Namespace) -> tuple[SimulationParams, ...]:
    market_scale_grid = (
        args.market_prior_scale_grid if args.market_win_totals is not None else [0.0]
    )
    candidates = tuple(
        SimulationParams(
            probability_logit_scale=scale,
            team_strength_sd=team_sd,
            team_prior_scale=prior_scale,
            market_prior_scale=market_scale,
            schedule_strength_scale=schedule_scale,
        )
        for scale in args.probability_logit_scale_grid
        for team_sd in args.team_strength_sd_grid
        for prior_scale in args.team_prior_scale_grid
        for market_scale in market_scale_grid
        for schedule_scale in args.schedule_strength_scale_grid
    )
    if not candidates:
        raise ValueError("Simulation parameter grid must not be empty")
    for candidate in candidates:
        if (
            not math.isfinite(candidate.probability_logit_scale)
            or candidate.probability_logit_scale <= 0.0
        ):
            raise ValueError("All probability logit scale candidates must be positive")
        if (
            not math.isfinite(candidate.team_strength_sd)
            or candidate.team_strength_sd < 0.0
        ):
            raise ValueError("All team-strength sd candidates must be non-negative")
        if (
            not math.isfinite(candidate.team_prior_scale)
            or candidate.team_prior_scale < 0.0
        ):
            raise ValueError("All team-prior scale candidates must be non-negative")
        if (
            not math.isfinite(candidate.market_prior_scale)
            or candidate.market_prior_scale < 0.0
        ):
            raise ValueError("All market-prior scale candidates must be non-negative")
        if (
            not math.isfinite(candidate.schedule_strength_scale)
            or candidate.schedule_strength_scale < 0.0
        ):
            raise ValueError(
                "All schedule-strength scale candidates must be non-negative"
            )
    return candidates


def _market_tuning_season_count(contexts: Sequence[ProjectionContext]) -> int:
    return sum(1 for context in contexts if context.market_prior_offsets)


def _tuning_seasons(season: int, args: argparse.Namespace) -> tuple[int, ...]:
    first_tunable = args.start_season + args.train_window
    start = max(first_tunable, season - args.tune_window)
    return tuple(range(start, season))


def _simulate_context(
    context: ProjectionContext,
    *,
    teams: dict[int, TeamInfo],
    args: argparse.Namespace,
    params: SimulationParams,
    trials: int,
) -> SeasonProjection:
    return simulate_season(
        games=context.schedule,
        teams=teams,
        as_of_date=context.as_of_date,
        trials=trials,
        predictor=context.predictor,
        seed=args.seed + context.season,
        wild_cards_per_league=args.wild_cards_per_league,
        probability_logit_scale=params.probability_logit_scale,
        team_strength_sd=params.team_strength_sd,
        team_prior_offsets=context.team_prior_offsets,
        team_prior_scale=params.team_prior_scale,
        market_prior_offsets=context.market_prior_offsets,
        market_prior_scale=params.market_prior_scale,
        schedule_strength_offsets=context.schedule_strength_offsets,
        schedule_strength_scale=params.schedule_strength_scale,
    )


def _evaluation_objective(evaluation: SeasonEvaluation) -> float:
    return evaluation.division_brier + evaluation.playoff_brier


def _fit_playoff_calibration(
    pairs: Sequence[tuple[SeasonProjection, ProjectionContext]],
    *,
    teams: dict[int, TeamInfo],
    min_teams: int,
) -> ProbabilityCalibration | None:
    probabilities: list[float] = []
    outcomes: list[float] = []
    for projection, context in pairs:
        actual = actual_outcomes(
            context.schedule,
            teams,
            wild_cards_per_league=projection.wild_cards_per_league,
        )
        for row in projection.teams:
            probabilities.append(row.playoff_prob)
            outcomes.append(1.0 if actual[row.team_id].playoff_team else 0.0)

    if len(probabilities) < min_teams or len(set(outcomes)) < 2:
        return None
    return _fit_probability_calibration(probabilities, outcomes)


def _fit_probability_calibration(
    probabilities: Sequence[float],
    outcomes: Sequence[float],
) -> ProbabilityCalibration | None:
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have the same length")
    if not probabilities:
        raise ValueError("probabilities must not be empty")

    parsed_outcomes = tuple(
        _bounded_probability(outcome, "outcome") for outcome in outcomes
    )
    anchor = sum(parsed_outcomes) / len(parsed_outcomes)
    if anchor <= 0.0 or anchor >= 1.0:
        return None

    anchor_logit = _probability_to_logit(anchor)
    features = tuple(
        _probability_to_logit(probability) - anchor_logit
        for probability in probabilities
    )

    from scipy.optimize import minimize_scalar

    parsed_probabilities = tuple(
        _bounded_probability(probability, "probability")
        for probability in probabilities
    )

    def brier_loss(slope: float) -> float:
        loss = 0.0
        for feature, outcome in zip(features, parsed_outcomes, strict=True):
            probability = _logistic(anchor_logit + slope * feature)
            loss += (probability - outcome) ** 2
        return loss / len(parsed_outcomes)

    raw_brier = sum(
        (probability - outcome) ** 2
        for probability, outcome in zip(
            parsed_probabilities, parsed_outcomes, strict=True
        )
    ) / len(parsed_outcomes)
    fit = minimize_scalar(brier_loss, bounds=(0.05, 2.0), method="bounded")
    if brier_loss(float(fit.x)) >= raw_brier:
        return None
    return ProbabilityCalibration(
        anchor=anchor,
        slope=float(fit.x),
        samples=len(probabilities),
    )


def _scale_stage_probability(
    stage_probability: float,
    *,
    raw_playoff_probability: float,
    calibrated_playoff_probability: float,
) -> float:
    if raw_playoff_probability <= 0.0:
        return 0.0
    return min(
        stage_probability * calibrated_playoff_probability / raw_playoff_probability,
        calibrated_playoff_probability,
    )


def _apply_playoff_calibration(
    projection: SeasonProjection,
    calibration: ProbabilityCalibration | None,
) -> SeasonProjection:
    if calibration is None:
        return projection
    calibrated_rows = []
    for row in projection.teams:
        playoff_prob = calibration.apply(row.playoff_prob)
        calibrated_rows.append(
            TeamProjection(
                team_id=row.team_id,
                actual_wins_as_of=row.actual_wins_as_of,
                expected_wins=row.expected_wins,
                division_win_prob=row.division_win_prob,
                playoff_prob=playoff_prob,
                division_series_prob=_scale_stage_probability(
                    row.division_series_prob,
                    raw_playoff_probability=row.playoff_prob,
                    calibrated_playoff_probability=playoff_prob,
                ),
                league_championship_prob=_scale_stage_probability(
                    row.league_championship_prob,
                    raw_playoff_probability=row.playoff_prob,
                    calibrated_playoff_probability=playoff_prob,
                ),
                world_series_prob=_scale_stage_probability(
                    row.world_series_prob,
                    raw_playoff_probability=row.playoff_prob,
                    calibrated_playoff_probability=playoff_prob,
                ),
                championship_prob=_scale_stage_probability(
                    row.championship_prob,
                    raw_playoff_probability=row.playoff_prob,
                    calibrated_playoff_probability=playoff_prob,
                ),
            )
        )
    calibrated_teams = tuple(calibrated_rows)
    return SeasonProjection(
        season=projection.season,
        as_of_date=projection.as_of_date,
        trials=projection.trials,
        wild_cards_per_league=projection.wild_cards_per_league,
        teams=calibrated_teams,
        probability_logit_scale=projection.probability_logit_scale,
        team_strength_sd=projection.team_strength_sd,
        team_prior_scale=projection.team_prior_scale,
        market_prior_scale=projection.market_prior_scale,
        schedule_strength_scale=projection.schedule_strength_scale,
        playoff_calibration_slope=calibration.slope,
    )


def _probability_to_logit(probability: float) -> float:
    p = min(max(_finite_float(probability, "probability"), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _logistic(logit: float) -> float:
    if logit >= 0.0:
        denominator = 1.0 + math.exp(-logit)
        return 1.0 / denominator
    exp_value = math.exp(logit)
    return exp_value / (1.0 + exp_value)


def _bounded_probability(value: float, label: str) -> float:
    probability = _finite_float(value, label)
    if probability < 0.0 or probability > 1.0:
        raise ValueError(f"{label} must be between zero and one")
    return probability


def _finite_float(value: float, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _resolve_simulation_params(
    context: ProjectionContext,
    *,
    teams: dict[int, TeamInfo],
    args: argparse.Namespace,
    cache: ContextCache,
) -> ModelAdjustments:
    fixed = _fixed_params(args)
    tune_seasons = _tuning_seasons(context.season, args)
    selected_params = fixed
    selected_pairs: tuple[tuple[SeasonProjection, ProjectionContext], ...] = ()

    if args.tune_simulation_params and tune_seasons:
        tune_contexts = tuple(
            _projection_context(
                tune_season,
                teams=teams,
                args=args,
                cache=cache,
                as_of_override=None,
            )
            for tune_season in tune_seasons
        )
        market_tune_seasons = _market_tuning_season_count(tune_contexts)
        best: (
            tuple[
                float,
                float,
                SimulationParams,
                tuple[tuple[SeasonProjection, ProjectionContext], ...],
            ]
            | None
        ) = None
        for candidate in _candidate_params(args):
            if (
                candidate.market_prior_scale > 0.0
                and market_tune_seasons < args.market_prior_min_tune_seasons
            ):
                continue
            pairs: list[tuple[SeasonProjection, ProjectionContext]] = []
            evaluations: list[SeasonEvaluation] = []
            for tune_context in tune_contexts:
                projection = _simulate_context(
                    tune_context,
                    teams=teams,
                    args=args,
                    params=candidate,
                    trials=args.tune_trials,
                )
                pairs.append((projection, tune_context))
                evaluations.append(
                    evaluate_projection(projection, tune_context.schedule, teams)
                )
            objective = sum(_evaluation_objective(item) for item in evaluations) / len(
                evaluations
            )
            playoff_brier = sum(item.playoff_brier for item in evaluations) / len(
                evaluations
            )
            score = (objective, playoff_brier, candidate, tuple(pairs))
            if best is None or score[:2] < best[:2]:
                best = score

        if best is not None:
            selected_params = best[2]
            selected_pairs = best[3]

    calibration_seasons: tuple[int, ...] = ()
    playoff_calibration = None
    if args.calibrate_playoff_probs and tune_seasons:
        if not selected_pairs:
            selected_pairs = tuple(
                (
                    _simulate_context(
                        tune_context,
                        teams=teams,
                        args=args,
                        params=selected_params,
                        trials=args.tune_trials,
                    ),
                    tune_context,
                )
                for tune_context in (
                    _projection_context(
                        tune_season,
                        teams=teams,
                        args=args,
                        cache=cache,
                        as_of_override=None,
                    )
                    for tune_season in tune_seasons
                )
            )
        playoff_calibration = _fit_playoff_calibration(
            selected_pairs,
            teams=teams,
            min_teams=args.playoff_calibration_min_teams,
        )
        calibration_seasons = tune_seasons if playoff_calibration is not None else ()

    return ModelAdjustments(
        params=selected_params,
        tune_seasons=tune_seasons if args.tune_simulation_params else (),
        playoff_calibration=playoff_calibration,
        calibration_seasons=calibration_seasons,
    )


def _baseline_for_context(
    context: ProjectionContext,
    *,
    teams: dict[int, TeamInfo],
    args: argparse.Namespace,
) -> SeasonProjection:
    return build_baseline_projection(
        games=context.schedule,
        teams=teams,
        as_of_date=context.as_of_date,
        wild_cards_per_league=args.wild_cards_per_league,
    )


def _evaluate_one_season(
    *,
    season: int,
    teams: dict[int, TeamInfo],
    args: argparse.Namespace,
    cache: ContextCache,
) -> tuple[SeasonProjection, SeasonEvaluation, SeasonProjection, SeasonEvaluation]:
    context = _projection_context(
        season,
        args=args,
        cache=cache,
        teams=teams,
        as_of_override=args.as_of,
    )
    adjustments = _resolve_simulation_params(
        context,
        teams=teams,
        args=args,
        cache=cache,
    )
    params = adjustments.params
    projection = _simulate_context(
        context,
        teams=teams,
        args=args,
        params=params,
        trials=args.trials,
    )
    projection = _apply_playoff_calibration(
        projection,
        adjustments.playoff_calibration,
    )
    baseline = _baseline_for_context(context, teams=teams, args=args)
    evaluation = evaluate_projection(projection, context.schedule, teams)
    baseline_evaluation = evaluate_projection(baseline, context.schedule, teams)
    skipped_label = (
        f" skipped_unscored={context.skipped_games}" if context.skipped_games else ""
    )
    tuning_label = (
        f" tuned_on={min(adjustments.tune_seasons)}-{max(adjustments.tune_seasons)}"
        if adjustments.tune_seasons
        else " fixed_params"
    )
    calibration_label = (
        f" playoff_cal_slope={adjustments.playoff_calibration.slope:.2f}"
        f" calibrated_on={min(adjustments.calibration_seasons)}-{max(adjustments.calibration_seasons)}"
        if adjustments.playoff_calibration is not None
        else ""
    )
    print(
        f"{season}: as_of={context.as_of_date} "
        f"train={min(context.train_seasons)}-{max(context.train_seasons)} "
        f"observed_games={context.observed_games:,} "
        f"played_games={len(context.schedule):,} trials={args.trials:,} "
        f"logit_scale={params.probability_logit_scale:.2f} "
        f"team_sd={params.team_strength_sd:.2f} "
        f"prior_scale={params.team_prior_scale:.2f} "
        f"market_scale={params.market_prior_scale:.2f} "
        f"schedule_scale={params.schedule_strength_scale:.2f}"
        f"{tuning_label}{calibration_label}{skipped_label}"
    )
    _print_evaluation("model", evaluation)
    _print_evaluation("baseline", baseline_evaluation)
    _print_top_playoff_probs(projection, teams)
    return projection, evaluation, baseline, baseline_evaluation


def _print_evaluation(label: str, evaluation: SeasonEvaluation) -> None:
    print(
        f"  {label:<8} wins_mae={evaluation.actual_wins_mae:.2f} "
        f"wins_rmse={evaluation.actual_wins_rmse:.2f} "
        f"division_brier={evaluation.division_brier:.4f} "
        f"playoff_brier={evaluation.playoff_brier:.4f}"
    )


def _print_top_playoff_probs(
    projection: SeasonProjection,
    teams: dict[int, TeamInfo],
    limit: int = 6,
) -> None:
    ranked = sorted(projection.teams, key=lambda row: row.playoff_prob, reverse=True)
    leaders = ", ".join(
        f"{teams[row.team_id].abbreviation} {row.playoff_prob:.0%}"
        for row in ranked[:limit]
    )
    print(f"  top playoff probs: {leaders}")


def _metric_average(
    evaluations: Sequence[SeasonEvaluation],
    attribute: str,
) -> float:
    return sum(float(getattr(item, attribute)) for item in evaluations) / len(
        evaluations
    )


def _print_aggregate(
    model_evaluations: Sequence[SeasonEvaluation],
    baseline_evaluations: Sequence[SeasonEvaluation],
) -> None:
    if not model_evaluations:
        return
    print("Aggregate:")
    for label, evaluations in (
        ("model", model_evaluations),
        ("baseline", baseline_evaluations),
    ):
        print(
            f"  {label:<8} wins_mae={_metric_average(evaluations, 'actual_wins_mae'):.2f} "
            f"wins_rmse={_metric_average(evaluations, 'actual_wins_rmse'):.2f} "
            f"division_brier={_metric_average(evaluations, 'division_brier'):.4f} "
            f"playoff_brier={_metric_average(evaluations, 'playoff_brier'):.4f}"
        )
    improvement = {
        "actual_wins_mae": _metric_average(baseline_evaluations, "actual_wins_mae")
        - _metric_average(model_evaluations, "actual_wins_mae"),
        "actual_wins_rmse": _metric_average(baseline_evaluations, "actual_wins_rmse")
        - _metric_average(model_evaluations, "actual_wins_rmse"),
        "division_brier": _metric_average(baseline_evaluations, "division_brier")
        - _metric_average(model_evaluations, "division_brier"),
        "playoff_brier": _metric_average(baseline_evaluations, "playoff_brier")
        - _metric_average(model_evaluations, "playoff_brier"),
    }
    print(
        f"  {'improve':<8} wins_mae={improvement['actual_wins_mae']:.2f} "
        f"wins_rmse={improvement['actual_wins_rmse']:.2f} "
        f"division_brier={improvement['division_brier']:.4f} "
        f"playoff_brier={improvement['playoff_brier']:.4f}"
    )


def _write_projection_rows(
    path: Path,
    labeled_projections: Sequence[LabeledProjection],
    schedules: dict[int, Sequence[SeasonScheduleGame]],
    teams: dict[int, TeamInfo],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "projection_type",
        "season",
        "as_of_date",
        "team_id",
        "abbreviation",
        "team_name",
        "league_name",
        "division_name",
        "actual_wins_as_of",
        "expected_wins",
        "division_win_prob",
        "playoff_prob",
        "division_series_prob",
        "league_championship_prob",
        "world_series_prob",
        "championship_prob",
        "probability_logit_scale",
        "team_strength_sd",
        "team_prior_scale",
        "market_prior_scale",
        "schedule_strength_scale",
        "playoff_calibration_slope",
        "actual_wins",
        "actual_division_winner",
        "actual_playoff_team",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for labeled in labeled_projections:
            projection = labeled.projection
            actual = actual_outcomes(
                schedules[projection.season],
                teams,
                wild_cards_per_league=projection.wild_cards_per_league,
            )
            for row in projection.teams:
                team = teams[row.team_id]
                outcome = actual[row.team_id]
                writer.writerow(
                    {
                        **asdict(row),
                        "projection_type": labeled.projection_type,
                        "season": projection.season,
                        "as_of_date": projection.as_of_date.isoformat(),
                        "abbreviation": team.abbreviation,
                        "team_name": team.team_name,
                        "league_name": team.league_name,
                        "division_name": team.division_name,
                        "probability_logit_scale": (
                            projection.probability_logit_scale
                            if labeled.projection_type == "model"
                            else ""
                        ),
                        "team_strength_sd": (
                            projection.team_strength_sd
                            if labeled.projection_type == "model"
                            else ""
                        ),
                        "team_prior_scale": (
                            projection.team_prior_scale
                            if labeled.projection_type == "model"
                            else ""
                        ),
                        "market_prior_scale": (
                            projection.market_prior_scale
                            if labeled.projection_type == "model"
                            else ""
                        ),
                        "schedule_strength_scale": (
                            projection.schedule_strength_scale
                            if labeled.projection_type == "model"
                            else ""
                        ),
                        "playoff_calibration_slope": (
                            projection.playoff_calibration_slope
                            if labeled.projection_type == "model"
                            else ""
                        ),
                        "actual_wins": outcome.wins,
                        "actual_division_winner": int(outcome.division_winner),
                        "actual_playoff_team": int(outcome.playoff_team),
                    }
                )
    print(f"Wrote {path}")


def _write_graphics(
    output_dir: Path,
    labeled_projections: Sequence[LabeledProjection],
    teams: dict[int, TeamInfo],
) -> None:
    for labeled in labeled_projections:
        if labeled.projection_type != "model":
            continue
        for path in write_projection_graphics(
            labeled.projection,
            teams,
            output_dir,
            projection_type=labeled.projection_type,
        ):
            print(f"Wrote {path}")


def _season_summary_rows(
    model_evaluations: Sequence[SeasonEvaluation],
    baseline_evaluations: Sequence[SeasonEvaluation],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model, baseline in zip(model_evaluations, baseline_evaluations, strict=True):
        rows.append(_evaluation_summary_row("model", model))
        rows.append(_evaluation_summary_row("baseline", baseline))
        rows.append(
            {
                "season": model.season,
                "projection_type": "improvement_vs_baseline",
                "teams": model.teams,
                "actual_wins_mae": baseline.actual_wins_mae - model.actual_wins_mae,
                "actual_wins_rmse": baseline.actual_wins_rmse - model.actual_wins_rmse,
                "division_brier": baseline.division_brier - model.division_brier,
                "division_log_loss": baseline.division_log_loss
                - model.division_log_loss,
                "playoff_brier": baseline.playoff_brier - model.playoff_brier,
                "playoff_log_loss": baseline.playoff_log_loss - model.playoff_log_loss,
            }
        )
    return rows


def _evaluation_summary_row(
    projection_type: str,
    evaluation: SeasonEvaluation,
) -> dict[str, object]:
    return {
        "season": evaluation.season,
        "projection_type": projection_type,
        "teams": evaluation.teams,
        "actual_wins_mae": evaluation.actual_wins_mae,
        "actual_wins_rmse": evaluation.actual_wins_rmse,
        "division_brier": evaluation.division_brier,
        "division_log_loss": evaluation.division_log_loss,
        "playoff_brier": evaluation.playoff_brier,
        "playoff_log_loss": evaluation.playoff_log_loss,
    }


def _write_summary_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "season",
        "projection_type",
        "teams",
        "actual_wins_mae",
        "actual_wins_rmse",
        "division_brier",
        "division_log_loss",
        "playoff_brier",
        "playoff_log_loss",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def _calibration_bucket(probability: float) -> tuple[float, float, str]:
    bucket_index = min(int(max(probability, 0.0) * 10), 9)
    lower = bucket_index / 10.0
    upper = (bucket_index + 1) / 10.0
    return lower, upper, f"{lower:.1f}-{upper:.1f}"


def _calibration_rows(
    labeled_projections: Sequence[LabeledProjection],
    schedules: dict[int, Sequence[SeasonScheduleGame]],
    teams: dict[int, TeamInfo],
) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str, float, float, str], list[tuple[float, float]]] = {}
    for labeled in labeled_projections:
        projection = labeled.projection
        actual = actual_outcomes(
            schedules[projection.season],
            teams,
            wild_cards_per_league=projection.wild_cards_per_league,
        )
        for row in projection.teams:
            outcome = actual[row.team_id]
            for market, probability, target in (
                (
                    "division",
                    row.division_win_prob,
                    1.0 if outcome.division_winner else 0.0,
                ),
                ("playoff", row.playoff_prob, 1.0 if outcome.playoff_team else 0.0),
            ):
                lower, upper, label = _calibration_bucket(probability)
                key = (labeled.projection_type, market, lower, upper, label)
                buckets.setdefault(key, []).append((probability, target))

    rows: list[dict[str, object]] = []
    for key, values in sorted(buckets.items()):
        projection_type, market, lower, upper, label = key
        count = len(values)
        mean_probability = sum(probability for probability, _target in values) / count
        observed_rate = sum(target for _probability, target in values) / count
        brier = (
            sum((probability - target) ** 2 for probability, target in values) / count
        )
        rows.append(
            {
                "projection_type": projection_type,
                "market": market,
                "bucket": label,
                "bucket_lower": lower,
                "bucket_upper": upper,
                "count": count,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
                "brier": brier,
            }
        )
    return rows


def _print_calibration(rows: Sequence[dict[str, object]]) -> None:
    print("Calibration buckets (model):")
    for market in ("division", "playoff"):
        print(f"  {market}:")
        for row in rows:
            if row["projection_type"] != "model" or row["market"] != market:
                continue
            print(
                f"    {row['bucket']} n={row['count']} "
                f"p={float(row['mean_probability']):.3f} "
                f"obs={float(row['observed_rate']):.3f} "
                f"brier={float(row['brier']):.4f}"
            )


def _write_calibration_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "projection_type",
        "market",
        "bucket",
        "bucket_lower",
        "bucket_upper",
        "count",
        "mean_probability",
        "observed_rate",
        "brier",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def _default_calibration_path(args: argparse.Namespace) -> Path | None:
    if args.calibration_out is not None:
        return args.calibration_out
    if args.out is None:
        return None
    return args.out.with_name(f"{args.out.stem}_calibration{args.out.suffix}")


def _default_summary_path(args: argparse.Namespace) -> Path | None:
    if args.summary_out is not None:
        return args.summary_out
    if args.out is None:
        return None
    return args.out.with_name(f"{args.out.stem}_summary{args.out.suffix}")


def main() -> None:
    args = parse_args()
    if args.as_of is not None and len(args.seasons) != 1:
        raise SystemExit("--as-of can only be used with one season")
    if args.tune_window < 1:
        raise SystemExit("--tune-window must be positive")
    if args.tune_trials < 1:
        raise SystemExit("--tune-trials must be positive")
    if args.playoff_calibration_min_teams < 1:
        raise SystemExit("--playoff-calibration-min-teams must be positive")
    if args.market_prior_min_tune_seasons < 0:
        raise SystemExit("--market-prior-min-tune-seasons must be non-negative")
    if args.market_win_totals is not None and not args.market_win_totals.exists():
        raise SystemExit(f"--market-win-totals not found: {args.market_win_totals}")

    teams = load_team_info()
    context_cache: ContextCache = {}
    labeled_projections: list[LabeledProjection] = []
    model_evaluations: list[SeasonEvaluation] = []
    baseline_evaluations: list[SeasonEvaluation] = []
    schedules: dict[int, Sequence[SeasonScheduleGame]] = {}
    for season in args.seasons:
        projection, evaluation, baseline, baseline_evaluation = _evaluate_one_season(
            season=season,
            teams=teams,
            args=args,
            cache=context_cache,
        )
        labeled_projections.append(LabeledProjection("model", projection))
        labeled_projections.append(LabeledProjection("baseline", baseline))
        model_evaluations.append(evaluation)
        baseline_evaluations.append(baseline_evaluation)
        schedules[season] = context_cache[(season, args.as_of)].schedule

    _print_aggregate(model_evaluations, baseline_evaluations)
    calibration_rows = _calibration_rows(labeled_projections, schedules, teams)
    _print_calibration(calibration_rows)
    if args.out is not None:
        _write_projection_rows(args.out, labeled_projections, schedules, teams)
    calibration_path = _default_calibration_path(args)
    if calibration_path is not None:
        _write_calibration_rows(calibration_path, calibration_rows)
    summary_path = _default_summary_path(args)
    if summary_path is not None:
        _write_summary_rows(
            summary_path,
            _season_summary_rows(model_evaluations, baseline_evaluations),
        )
    if args.graphics_out_dir is not None:
        _write_graphics(args.graphics_out_dir, labeled_projections, teams)


if __name__ == "__main__":
    main()
