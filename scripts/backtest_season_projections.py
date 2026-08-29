#!/usr/bin/env python3
"""Backtest season standings projections with optional preseason market priors."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mlb.sim.market_priors import load_market_prior_offsets
from mlb.sim.projection_charts import write_projection_graphics
from mlb.sim.season import (
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
from mlb.sim.team_priors import load_team_prior_offsets
from mlb.sim.team_strength import (
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
    team_prior_decay_games: float | None
    market_prior_decay_games: float | None
    roster_prior_scale: float
    roster_prior_decay_games: float | None


@dataclass(frozen=True)
class ProjectionContext:
    season: int
    as_of_label: str
    schedule: tuple[SeasonScheduleGame, ...]
    as_of_date: date
    predictor: HomeWinPredictor
    train_seasons: tuple[int, ...]
    observed_games: int
    skipped_games: int
    team_prior_offsets: dict[int, float]
    schedule_strength_offsets: dict[int, float]
    market_prior_offsets: dict[int, float]
    roster_prior_offsets: dict[int, float]
    input_market_sources: str


@dataclass(frozen=True)
class LabeledProjection:
    projection_type: str
    projection: SeasonProjection
    as_of_label: str = ""

@dataclass(frozen=True)
class PostseasonActualStages:
    division_series_teams: frozenset[int] = frozenset()
    league_championship_teams: frozenset[int] = frozenset()
    world_series_teams: frozenset[int] = frozenset()
    champion: int | None = None


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
    calibrations: dict[str, ProbabilityCalibration]
    calibration_seasons: tuple[int, ...]
    tuning_objective: str


ContextCache = dict[tuple[int, str], ProjectionContext]


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
        "--team-prior-decay-games",
        type=float,
        default=0.0,
        help="Fixed per-team games-played decay constant for team priors; 0 disables decay.",
    )
    parser.add_argument(
        "--team-prior-decay-games-grid",
        type=float,
        nargs="+",
        default=[0.0, 30.0, 60.0],
        help="Candidate per-team games-played decay constants for team priors.",
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
        "--market-prior-decay-games",
        type=float,
        default=0.0,
        help="Fixed per-team games-played decay constant for market priors; 0 disables decay.",
    )
    parser.add_argument(
        "--market-prior-decay-games-grid",
        type=float,
        nargs="+",
        default=[0.0, 30.0, 60.0],
        help="Candidate per-team games-played decay constants for market priors.",
    )
    parser.add_argument(
        "--roster-priors",
        type=Path,
        default=None,
        help="Optional CSV of preseason roster/depth-chart priors.",
    )
    parser.add_argument(
        "--roster-prior-scale",
        type=float,
        default=0.0,
        help="Fixed roster-prior logit scale when tuning is disabled or unavailable.",
    )
    parser.add_argument(
        "--roster-prior-scale-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.50],
        help="Candidate roster-prior scales for prior-season tuning.",
    )
    parser.add_argument(
        "--roster-prior-decay-games",
        type=float,
        default=0.0,
        help="Fixed per-team games-played decay constant for roster priors; 0 disables decay.",
    )
    parser.add_argument(
        "--roster-prior-decay-games-grid",
        type=float,
        nargs="+",
        default=[0.0, 30.0, 60.0],
        help="Candidate per-team games-played decay constants for roster priors.",
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
        "--calibration-targets",
        nargs="+",
        default=["playoff"],
        choices=[
            "all",
            "division",
            "playoff",
            "division_series",
            "league_championship",
            "world_series",
            "championship",
        ],
        help="Targets calibrated when --calibrate-playoff-probs is enabled.",
    )
    parser.add_argument(
        "--postseason-results",
        type=Path,
        default=None,
        help="Optional CSV with actual postseason stage outcomes by season/team.",
    )
    parser.add_argument(
        "--tuning-objective",
        choices=["combined", "wins", "division", "playoff", "championship"],
        default="combined",
        help="Metric used to choose simulation parameters on prior seasons.",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Override projection date for a single --seasons value; defaults to first regular-season date.",
    )
    parser.add_argument(
        "--as-of-buckets",
        nargs="+",
        default=["opening_day"],
        help="As-of buckets: opening_day, day7, day14, team30, team60, playoff_start.",
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


def _as_of_labels(args: argparse.Namespace) -> tuple[str, ...]:
    if args.as_of is not None:
        return (f"custom:{args.as_of}",)
    labels = tuple(str(label) for label in args.as_of_buckets)
    if not labels:
        raise ValueError("--as-of-buckets must not be empty")
    return labels


def _resolve_as_of_date(
    schedule: Sequence[SeasonScheduleGame],
    *,
    label: str,
) -> date:
    if label.startswith("custom:"):
        return date.fromisoformat(label.split(":", 1)[1])
    first_date = first_regular_season_date(schedule)
    if label == "opening_day":
        return first_date
    if label.startswith("day"):
        return first_date + timedelta(days=_suffix_int(label, "day"))
    if label.endswith("_days"):
        return first_date + timedelta(days=_prefix_int(label, "_days"))
    if label.startswith("team"):
        return _as_of_date_for_team_games(schedule, _suffix_int(label, "team"))
    if label.endswith(("_games", "_games_per_team")):
        suffix = "_games_per_team" if label.endswith("_games_per_team") else "_games"
        return _as_of_date_for_team_games(schedule, _prefix_int(label, suffix))
    if label == "playoff_start":
        return max(game.game_date for game in schedule) + timedelta(days=1)
    raise ValueError(
        "Unsupported as-of bucket "
        f"{label!r}; use opening_day, dayN, N_days, teamN, N_games, or playoff_start"
    )


def _suffix_int(label: str, prefix: str) -> int:
    text = label[len(prefix) :]
    if not text:
        raise ValueError(f"{label!r} must include a numeric suffix")
    value = int(text)
    if value < 0:
        raise ValueError(f"{label!r} must be non-negative")
    return value


def _prefix_int(label: str, suffix: str) -> int:
    text = label[: -len(suffix)]
    if not text:
        raise ValueError(f"{label!r} must include a numeric prefix")
    value = int(text)
    if value < 0:
        raise ValueError(f"{label!r} must be non-negative")
    return value


def _as_of_date_for_team_games(
    schedule: Sequence[SeasonScheduleGame],
    target_games: int,
) -> date:
    if target_games == 0:
        return first_regular_season_date(schedule)
    team_ids = sorted({team for game in schedule for team in (game.away_team_id, game.home_team_id)})
    counts = {team_id: 0 for team_id in team_ids}
    for game in sorted(schedule, key=lambda item: (item.game_date, item.game_pk)):
        if not game.is_final:
            continue
        counts[game.away_team_id] += 1
        counts[game.home_team_id] += 1
        if all(count >= target_games for count in counts.values()):
            return game.game_date + timedelta(days=1)
    return max(game.game_date for game in schedule) + timedelta(days=1)


def _load_roster_offsets(
    path: Path | None,
    *,
    season: int,
    teams: dict[int, TeamInfo],
) -> dict[int, float]:
    if path is None:
        return {}
    from mlb.sim.roster_priors import load_roster_prior_offsets

    return load_roster_prior_offsets(
        path,
        prediction_season=season,
        teams=teams,
    )


def _input_market_sources(args: argparse.Namespace) -> str:
    sources: list[str] = []
    if args.market_win_totals is not None:
        sources.append("win_totals")
    if args.roster_priors is not None:
        sources.append("roster_priors")
    return ",".join(sources)


def _projection_context(
    season: int,
    *,
    args: argparse.Namespace,
    teams: dict[int, TeamInfo],
    cache: ContextCache,
    as_of_label: str,
) -> ProjectionContext:
    key = (season, as_of_label)
    if key in cache:
        return cache[key]

    schedule, skipped_games = _completed_played_schedule(season)
    as_of_date = _resolve_as_of_date(schedule, label=as_of_label)
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
    roster_prior_offsets = _load_roster_offsets(
        args.roster_priors,
        season=season,
        teams=teams,
    )
    schedule_strength_offsets = schedule_strength_offsets_from_games(
        schedule,
        as_of_date=as_of_date,
        team_prior_offsets=team_prior_offsets,
    )
    context = ProjectionContext(
        season=season,
        as_of_label=as_of_label,
        schedule=schedule,
        as_of_date=as_of_date,
        predictor=predictor,
        train_seasons=train_seasons,
        observed_games=observed_games,
        skipped_games=skipped_games,
        team_prior_offsets=team_prior_offsets,
        schedule_strength_offsets=schedule_strength_offsets,
        market_prior_offsets=market_prior_offsets,
        roster_prior_offsets=roster_prior_offsets,
        input_market_sources=_input_market_sources(args),
    )
    cache[key] = context
    return context


def _decay_arg(value: float) -> float | None:
    parsed = float(value)
    return None if parsed == 0.0 else parsed


def _fixed_params(args: argparse.Namespace) -> SimulationParams:
    return SimulationParams(
        probability_logit_scale=args.probability_logit_scale,
        team_strength_sd=args.team_strength_sd,
        team_prior_scale=args.team_prior_scale,
        market_prior_scale=args.market_prior_scale,
        schedule_strength_scale=args.schedule_strength_scale,
        team_prior_decay_games=_decay_arg(args.team_prior_decay_games),
        market_prior_decay_games=_decay_arg(args.market_prior_decay_games),
        roster_prior_scale=args.roster_prior_scale,
        roster_prior_decay_games=_decay_arg(args.roster_prior_decay_games),
    )


def _arg_values(args: argparse.Namespace, name: str, default: Sequence[float]) -> Sequence[float]:
    return getattr(args, name, default)


def _candidate_params(args: argparse.Namespace) -> tuple[SimulationParams, ...]:
    market_scale_grid = (
        args.market_prior_scale_grid if args.market_win_totals is not None else [0.0]
    )
    roster_scale_grid = (
        _arg_values(args, "roster_prior_scale_grid", [0.0])
        if getattr(args, "roster_priors", None) is not None
        else [0.0]
    )
    candidates = tuple(
        SimulationParams(
            probability_logit_scale=scale,
            team_strength_sd=team_sd,
            team_prior_scale=prior_scale,
            market_prior_scale=market_scale,
            schedule_strength_scale=schedule_scale,
            team_prior_decay_games=_decay_arg(team_decay),
            market_prior_decay_games=_decay_arg(market_decay),
            roster_prior_scale=roster_scale,
            roster_prior_decay_games=_decay_arg(roster_decay),
        )
        for scale in args.probability_logit_scale_grid
        for team_sd in args.team_strength_sd_grid
        for prior_scale in args.team_prior_scale_grid
        for team_decay in _arg_values(args, "team_prior_decay_games_grid", [0.0])
        for market_scale in market_scale_grid
        for market_decay in _arg_values(args, "market_prior_decay_games_grid", [0.0])
        for roster_scale in roster_scale_grid
        for roster_decay in _arg_values(args, "roster_prior_decay_games_grid", [0.0])
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
        for label, value in (
            ("team-prior scale", candidate.team_prior_scale),
            ("market-prior scale", candidate.market_prior_scale),
            ("roster-prior scale", candidate.roster_prior_scale),
            ("schedule-strength scale", candidate.schedule_strength_scale),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"All {label} candidates must be non-negative")
        for label, value in (
            ("team-prior decay", candidate.team_prior_decay_games),
            ("market-prior decay", candidate.market_prior_decay_games),
            ("roster-prior decay", candidate.roster_prior_decay_games),
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"All {label} candidates must be non-negative")
    return candidates


def _market_tuning_season_count(contexts: Sequence[ProjectionContext]) -> int:
    return sum(1 for context in contexts if context.market_prior_offsets)


def _tuning_seasons(season: int, args: argparse.Namespace) -> tuple[int, ...]:
    first_tunable = args.start_season + args.train_window
    start = max(first_tunable, season - args.tune_window)
    return tuple(range(start, season))


CALIBRATION_TARGETS = (
    "division",
    "playoff",
    "division_series",
    "league_championship",
    "world_series",
    "championship",
)


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
        team_prior_decay_games=params.team_prior_decay_games,
        market_prior_offsets=context.market_prior_offsets,
        market_prior_scale=params.market_prior_scale,
        market_prior_decay_games=params.market_prior_decay_games,
        roster_prior_offsets=context.roster_prior_offsets,
        roster_prior_scale=params.roster_prior_scale,
        roster_prior_decay_games=params.roster_prior_decay_games,
        schedule_strength_offsets=context.schedule_strength_offsets,
        schedule_strength_scale=params.schedule_strength_scale,
        input_market_sources=context.input_market_sources,
    )


def _projection_probability(row: TeamProjection, target: str) -> float:
    if target == "division":
        return row.division_win_prob
    if target == "playoff":
        return row.playoff_prob
    if target == "division_series":
        return row.division_series_prob
    if target == "league_championship":
        return row.league_championship_prob
    if target == "world_series":
        return row.world_series_prob
    if target == "championship":
        return row.championship_prob
    raise KeyError(target)


def _actual_target(
    *,
    target: str,
    team_id: int,
    outcome,
    stages: PostseasonActualStages | None,
) -> float | None:
    if target == "division":
        return 1.0 if outcome.division_winner else 0.0
    if target == "playoff":
        return 1.0 if outcome.playoff_team else 0.0
    if stages is None:
        return None
    if target == "division_series":
        return 1.0 if team_id in stages.division_series_teams else 0.0
    if target == "league_championship":
        return 1.0 if team_id in stages.league_championship_teams else 0.0
    if target == "world_series":
        return 1.0 if team_id in stages.world_series_teams else 0.0
    if target == "championship":
        return 1.0 if team_id == stages.champion else 0.0
    raise KeyError(target)


def _evaluation_objective(
    evaluation: SeasonEvaluation,
    projection: SeasonProjection,
    context: ProjectionContext,
    *,
    teams: dict[int, TeamInfo],
    objective: str,
    postseason_results: dict[int, PostseasonActualStages],
) -> float:
    if objective == "combined":
        return evaluation.division_brier + evaluation.playoff_brier
    if objective == "wins":
        return evaluation.actual_wins_rmse
    if objective == "division":
        return evaluation.division_brier
    if objective == "playoff":
        return evaluation.playoff_brier
    if objective == "championship":
        stages = postseason_results.get(context.season)
        if stages is None or stages.champion is None:
            return evaluation.playoff_brier
        return _target_brier(
            projection,
            context,
            teams=teams,
            target="championship",
            stages=stages,
        )
    raise KeyError(objective)


def _target_brier(
    projection: SeasonProjection,
    context: ProjectionContext,
    *,
    teams: dict[int, TeamInfo],
    target: str,
    stages: PostseasonActualStages | None,
) -> float:
    actual = actual_outcomes(
        context.schedule,
        teams,
        wild_cards_per_league=projection.wild_cards_per_league,
    )
    losses: list[float] = []
    for row in projection.teams:
        target_value = _actual_target(
            target=target,
            team_id=row.team_id,
            outcome=actual[row.team_id],
            stages=stages,
        )
        if target_value is None:
            continue
        losses.append((_projection_probability(row, target) - target_value) ** 2)
    if not losses:
        return math.inf
    return sum(losses) / len(losses)


def _fit_playoff_calibration(
    pairs: Sequence[tuple[SeasonProjection, ProjectionContext]],
    *,
    teams: dict[int, TeamInfo],
    min_teams: int,
) -> ProbabilityCalibration | None:
    calibrations = _fit_target_calibrations(
        pairs,
        teams=teams,
        min_teams=min_teams,
        targets=("playoff",),
        postseason_results={},
    )
    return calibrations.get("playoff")


def _fit_target_calibrations(
    pairs: Sequence[tuple[SeasonProjection, ProjectionContext]],
    *,
    teams: dict[int, TeamInfo],
    min_teams: int,
    targets: Sequence[str],
    postseason_results: dict[int, PostseasonActualStages],
) -> dict[str, ProbabilityCalibration]:
    calibrations: dict[str, ProbabilityCalibration] = {}
    for target in targets:
        probabilities: list[float] = []
        outcomes: list[float] = []
        for projection, context in pairs:
            actual = actual_outcomes(
                context.schedule,
                teams,
                wild_cards_per_league=projection.wild_cards_per_league,
            )
            stages = postseason_results.get(context.season)
            for row in projection.teams:
                target_value = _actual_target(
                    target=target,
                    team_id=row.team_id,
                    outcome=actual[row.team_id],
                    stages=stages,
                )
                if target_value is None:
                    continue
                probabilities.append(_projection_probability(row, target))
                outcomes.append(target_value)
        if len(probabilities) < min_teams or len(set(outcomes)) < 2:
            continue
        calibration = _fit_probability_calibration(probabilities, outcomes)
        if calibration is not None:
            calibrations[target] = calibration
    return calibrations


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


def _clamp_stage_order(
    *,
    division: float,
    playoff: float,
    division_series: float,
    league_championship: float,
    world_series: float,
    championship: float,
) -> tuple[float, float, float, float, float, float]:
    division = min(max(division, 0.0), playoff)
    division_series = min(max(division_series, 0.0), playoff)
    league_championship = min(max(league_championship, 0.0), division_series)
    world_series = min(max(world_series, 0.0), league_championship)
    championship = min(max(championship, 0.0), world_series)
    return (
        division,
        playoff,
        division_series,
        league_championship,
        world_series,
        championship,
    )


def _apply_target_calibrations(
    projection: SeasonProjection,
    calibrations: dict[str, ProbabilityCalibration],
    *,
    enforce_stage_order: bool = True,
) -> SeasonProjection:
    if not calibrations:
        return projection
    calibrated_rows = []
    for row in projection.teams:
        raw_playoff_prob = row.playoff_prob
        division_prob = calibrations.get("division", _IdentityCalibration()).apply(
            row.division_win_prob
        )
        playoff_prob = calibrations.get("playoff", _IdentityCalibration()).apply(
            row.playoff_prob
        )
        if "division_series" in calibrations:
            division_series_prob = calibrations["division_series"].apply(
                row.division_series_prob
            )
        elif "playoff" in calibrations:
            division_series_prob = _scale_stage_probability(
                row.division_series_prob,
                raw_playoff_probability=raw_playoff_prob,
                calibrated_playoff_probability=playoff_prob,
            )
        else:
            division_series_prob = row.division_series_prob
        league_championship_prob = (
            calibrations["league_championship"].apply(row.league_championship_prob)
            if "league_championship" in calibrations
            else row.league_championship_prob
        )
        world_series_prob = (
            calibrations["world_series"].apply(row.world_series_prob)
            if "world_series" in calibrations
            else row.world_series_prob
        )
        championship_prob = (
            calibrations["championship"].apply(row.championship_prob)
            if "championship" in calibrations
            else row.championship_prob
        )
        if enforce_stage_order:
            (
                division_prob,
                playoff_prob,
                division_series_prob,
                league_championship_prob,
                world_series_prob,
                championship_prob,
            ) = _clamp_stage_order(
                division=division_prob,
                playoff=playoff_prob,
                division_series=division_series_prob,
                league_championship=league_championship_prob,
                world_series=world_series_prob,
                championship=championship_prob,
            )
        calibrated_rows.append(
            TeamProjection(
                team_id=row.team_id,
                actual_wins_as_of=row.actual_wins_as_of,
                expected_wins=row.expected_wins,
                division_win_prob=division_prob,
                playoff_prob=playoff_prob,
                division_series_prob=division_series_prob,
                league_championship_prob=league_championship_prob,
                world_series_prob=world_series_prob,
                championship_prob=championship_prob,
                team_prior_offset=row.team_prior_offset,
                team_prior_weight=row.team_prior_weight,
                market_prior_offset=row.market_prior_offset,
                market_prior_weight=row.market_prior_weight,
                roster_prior_offset=row.roster_prior_offset,
                roster_prior_weight=row.roster_prior_weight,
                combined_prior_offset=row.combined_prior_offset,
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
        playoff_calibration_slope=(
            calibrations["playoff"].slope if "playoff" in calibrations else None
        ),
        division_calibration_slope=(
            calibrations["division"].slope if "division" in calibrations else None
        ),
        division_series_calibration_slope=(
            calibrations["division_series"].slope
            if "division_series" in calibrations
            else None
        ),
        league_championship_calibration_slope=(
            calibrations["league_championship"].slope
            if "league_championship" in calibrations
            else None
        ),
        world_series_calibration_slope=(
            calibrations["world_series"].slope
            if "world_series" in calibrations
            else None
        ),
        championship_calibration_slope=(
            calibrations["championship"].slope if "championship" in calibrations else None
        ),
        team_prior_decay_games=projection.team_prior_decay_games,
        market_prior_decay_games=projection.market_prior_decay_games,
        roster_prior_scale=projection.roster_prior_scale,
        roster_prior_decay_games=projection.roster_prior_decay_games,
        input_market_sources=projection.input_market_sources,
    )


class _IdentityCalibration:
    def apply(self, probability: float) -> float:
        return probability


def _apply_playoff_calibration(
    projection: SeasonProjection,
    calibration: ProbabilityCalibration | None,
) -> SeasonProjection:
    return _apply_target_calibrations(
        projection,
        {"playoff": calibration} if calibration is not None else {},
        enforce_stage_order=False,
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


def _selected_calibration_targets(args: argparse.Namespace) -> tuple[str, ...]:
    targets = tuple(str(target) for target in args.calibration_targets)
    if "all" in targets:
        return CALIBRATION_TARGETS
    return targets


def _load_postseason_results(path: Path | None) -> dict[int, PostseasonActualStages]:
    if path is None:
        return {}
    results: dict[int, dict[str, set[int] | int | None]] = {}
    with path.open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=2):
            season = int(row["season"])
            team_id = int(row["team_id"])
            entry = results.setdefault(
                season,
                {
                    "division_series": set(),
                    "league_championship": set(),
                    "world_series": set(),
                    "champion": None,
                },
            )
            for field in ("division_series", "league_championship", "world_series"):
                if _truthy(row.get(field, "")):
                    cast_set = entry[field]
                    if not isinstance(cast_set, set):
                        raise ValueError(f"Invalid postseason field state at row {index}")
                    cast_set.add(team_id)
            if _truthy(row.get("championship", "")):
                entry["champion"] = team_id
    return {
        season: PostseasonActualStages(
            division_series_teams=frozenset(value["division_series"]),
            league_championship_teams=frozenset(value["league_championship"]),
            world_series_teams=frozenset(value["world_series"]),
            champion=value["champion"] if isinstance(value["champion"], int) else None,
        )
        for season, value in results.items()
    }


def _truthy(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _resolve_simulation_params(
    context: ProjectionContext,
    *,
    teams: dict[int, TeamInfo],
    args: argparse.Namespace,
    cache: ContextCache,
    postseason_results: dict[int, PostseasonActualStages],
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
                as_of_label=context.as_of_label,
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
            objective_values: list[float] = []
            playoff_briers: list[float] = []
            for tune_context in tune_contexts:
                projection = _simulate_context(
                    tune_context,
                    teams=teams,
                    args=args,
                    params=candidate,
                    trials=args.tune_trials,
                )
                evaluation = evaluate_projection(projection, tune_context.schedule, teams)
                pairs.append((projection, tune_context))
                evaluations.append(evaluation)
                objective_values.append(
                    _evaluation_objective(
                        evaluation,
                        projection,
                        tune_context,
                        teams=teams,
                        objective=args.tuning_objective,
                        postseason_results=postseason_results,
                    )
                )
                playoff_briers.append(evaluation.playoff_brier)
            objective = sum(objective_values) / len(objective_values)
            playoff_brier = sum(playoff_briers) / len(playoff_briers)
            score = (objective, playoff_brier, candidate, tuple(pairs))
            if best is None or score[:2] < best[:2]:
                best = score

        if best is not None:
            selected_params = best[2]
            selected_pairs = best[3]

    calibration_seasons: tuple[int, ...] = ()
    calibrations: dict[str, ProbabilityCalibration] = {}
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
                        as_of_label=context.as_of_label,
                    )
                    for tune_season in tune_seasons
                )
            )
        calibrations = _fit_target_calibrations(
            selected_pairs,
            teams=teams,
            min_teams=args.playoff_calibration_min_teams,
            targets=_selected_calibration_targets(args),
            postseason_results=postseason_results,
        )
        calibration_seasons = tune_seasons if calibrations else ()

    return ModelAdjustments(
        params=selected_params,
        tune_seasons=tune_seasons if args.tune_simulation_params else (),
        calibrations=calibrations,
        calibration_seasons=calibration_seasons,
        tuning_objective=args.tuning_objective,
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
    as_of_label: str,
    teams: dict[int, TeamInfo],
    args: argparse.Namespace,
    cache: ContextCache,
    postseason_results: dict[int, PostseasonActualStages],
) -> tuple[SeasonProjection, SeasonEvaluation, SeasonProjection, SeasonEvaluation]:
    context = _projection_context(
        season,
        args=args,
        cache=cache,
        teams=teams,
        as_of_label=as_of_label,
    )
    adjustments = _resolve_simulation_params(
        context,
        teams=teams,
        args=args,
        cache=cache,
        postseason_results=postseason_results,
    )
    params = adjustments.params
    projection = _simulate_context(
        context,
        teams=teams,
        args=args,
        params=params,
        trials=args.trials,
    )
    projection = _apply_target_calibrations(
        projection,
        adjustments.calibrations,
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
    if adjustments.calibrations:
        slopes = ",".join(
            f"{target}:{calibration.slope:.2f}"
            for target, calibration in sorted(adjustments.calibrations.items())
        )
        calibration_label = (
            f" cal_slopes={slopes}"
            f" calibrated_on={min(adjustments.calibration_seasons)}-{max(adjustments.calibration_seasons)}"
        )
    else:
        calibration_label = ""
    print(
        f"{season}: bucket={context.as_of_label} as_of={context.as_of_date} "
        f"train={min(context.train_seasons)}-{max(context.train_seasons)} "
        f"observed_games={context.observed_games:,} "
        f"played_games={len(context.schedule):,} trials={args.trials:,} "
        f"objective={adjustments.tuning_objective} "
        f"logit_scale={params.probability_logit_scale:.2f} "
        f"team_sd={params.team_strength_sd:.2f} "
        f"prior_scale={params.team_prior_scale:.2f} "
        f"team_decay={params.team_prior_decay_games or 0.0:.1f} "
        f"market_scale={params.market_prior_scale:.2f} "
        f"market_decay={params.market_prior_decay_games or 0.0:.1f} "
        f"roster_scale={params.roster_prior_scale:.2f} "
        f"roster_decay={params.roster_prior_decay_games or 0.0:.1f} "
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
        "as_of_bucket",
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
        "team_prior_offset",
        "team_prior_weight",
        "market_prior_offset",
        "market_prior_weight",
        "roster_prior_offset",
        "roster_prior_weight",
        "combined_prior_offset",
        "probability_logit_scale",
        "team_strength_sd",
        "team_prior_scale",
        "market_prior_scale",
        "team_prior_decay_games",
        "market_prior_decay_games",
        "roster_prior_scale",
        "roster_prior_decay_games",
        "schedule_strength_scale",
        "input_market_sources",
        "playoff_calibration_slope",
        "division_calibration_slope",
        "division_series_calibration_slope",
        "league_championship_calibration_slope",
        "world_series_calibration_slope",
        "championship_calibration_slope",
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
                        "as_of_bucket": labeled.as_of_label,
                        "abbreviation": team.abbreviation,
                        "team_name": team.team_name,
                        "league_name": team.league_name,
                        "division_name": team.division_name,
                        "probability_logit_scale": (
                            projection.probability_logit_scale
                            if labeled.projection_type.startswith("model")
                            else ""
                        ),
                        "team_strength_sd": (
                            projection.team_strength_sd
                            if labeled.projection_type.startswith("model")
                            else ""
                        ),
                        "team_prior_scale": (
                            projection.team_prior_scale
                            if labeled.projection_type.startswith("model")
                            else ""
                        ),
                        "market_prior_scale": (
                            projection.market_prior_scale
                            if labeled.projection_type.startswith("model")
                            else ""
                        ),
                        "team_prior_decay_games": (
                            projection.team_prior_decay_games
                            if labeled.projection_type.startswith("model")
                            else ""
                        ),
                        "market_prior_decay_games": (
                            projection.market_prior_decay_games
                            if labeled.projection_type.startswith("model")
                            else ""
                        ),
                        "roster_prior_scale": (
                            projection.roster_prior_scale
                            if labeled.projection_type.startswith("model")
                            else ""
                        ),
                        "roster_prior_decay_games": (
                            projection.roster_prior_decay_games
                            if labeled.projection_type.startswith("model")
                            else ""
                        ),
                        "schedule_strength_scale": (
                            projection.schedule_strength_scale
                            if labeled.projection_type.startswith("model")
                            else ""
                        ),
                        "input_market_sources": (
                            projection.input_market_sources
                            if labeled.projection_type.startswith("model")
                            else ""
                        ),
                        "playoff_calibration_slope": (
                            projection.playoff_calibration_slope
                            if labeled.projection_type.startswith("model")
                            else ""
                        ),
                        "division_calibration_slope": (
                            projection.division_calibration_slope
                            if labeled.projection_type.startswith("model")
                            else ""
                        ),
                        "division_series_calibration_slope": (
                            projection.division_series_calibration_slope
                            if labeled.projection_type.startswith("model")
                            else ""
                        ),
                        "league_championship_calibration_slope": (
                            projection.league_championship_calibration_slope
                            if labeled.projection_type.startswith("model")
                            else ""
                        ),
                        "world_series_calibration_slope": (
                            projection.world_series_calibration_slope
                            if labeled.projection_type.startswith("model")
                            else ""
                        ),
                        "championship_calibration_slope": (
                            projection.championship_calibration_slope
                            if labeled.projection_type.startswith("model")
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
    as_of_labels: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    labels = (
        tuple(as_of_labels)
        if as_of_labels is not None
        else tuple("" for _ in model_evaluations)
    )
    for label, model, baseline in zip(
        labels,
        model_evaluations,
        baseline_evaluations,
        strict=True,
    ):
        rows.append(_evaluation_summary_row("model", model, label))
        rows.append(_evaluation_summary_row("baseline", baseline, label))
        rows.append(
            {
                "season": model.season,
                "as_of_bucket": label,
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
    as_of_label: str = "",
) -> dict[str, object]:
    return {
        "season": evaluation.season,
        "as_of_bucket": as_of_label,
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
        "as_of_bucket",
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
    postseason_results: dict[int, PostseasonActualStages] | None = None,
) -> list[dict[str, object]]:
    buckets: dict[
        tuple[str, str, str, float, float, str],
        list[tuple[float, float]],
    ] = {}
    stage_results = postseason_results or {}
    for labeled in labeled_projections:
        projection = labeled.projection
        actual = actual_outcomes(
            schedules[projection.season],
            teams,
            wild_cards_per_league=projection.wild_cards_per_league,
        )
        stages = stage_results.get(projection.season)
        for row in projection.teams:
            outcome = actual[row.team_id]
            for market in CALIBRATION_TARGETS:
                target = _actual_target(
                    target=market,
                    team_id=row.team_id,
                    outcome=outcome,
                    stages=stages,
                )
                if target is None:
                    continue
                probability = _projection_probability(row, market)
                lower, upper, label = _calibration_bucket(probability)
                key = (
                    labeled.projection_type,
                    labeled.as_of_label,
                    market,
                    lower,
                    upper,
                    label,
                )
                buckets.setdefault(key, []).append((probability, target))

    rows: list[dict[str, object]] = []
    for key, values in sorted(buckets.items()):
        projection_type, as_of_label, market, lower, upper, label = key
        count = len(values)
        mean_probability = sum(probability for probability, _target in values) / count
        observed_rate = sum(target for _probability, target in values) / count
        brier = (
            sum((probability - target) ** 2 for probability, target in values) / count
        )
        rows.append(
            {
                "projection_type": projection_type,
                "as_of_bucket": as_of_label,
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
    markets = sorted({str(row["market"]) for row in rows if row["projection_type"] == "model"})
    for market in markets:
        print(f"  {market}:")
        for row in rows:
            if row["projection_type"] != "model" or row["market"] != market:
                continue
            print(
                f"    {row['as_of_bucket']} {row['bucket']} n={row['count']} "
                f"p={float(row['mean_probability']):.3f} "
                f"obs={float(row['observed_rate']):.3f} "
                f"brier={float(row['brier']):.4f}"
            )


def _write_calibration_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "projection_type",
        "as_of_bucket",
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
    if args.roster_priors is not None and not args.roster_priors.exists():
        raise SystemExit(f"--roster-priors not found: {args.roster_priors}")
    if args.postseason_results is not None and not args.postseason_results.exists():
        raise SystemExit(f"--postseason-results not found: {args.postseason_results}")

    as_of_labels = _as_of_labels(args)
    if args.graphics_out_dir is not None and len(as_of_labels) > 1:
        raise SystemExit("--graphics-out-dir requires a single as-of bucket")
    teams = load_team_info()
    postseason_results = _load_postseason_results(args.postseason_results)
    context_cache: ContextCache = {}
    labeled_projections: list[LabeledProjection] = []
    model_evaluations: list[SeasonEvaluation] = []
    baseline_evaluations: list[SeasonEvaluation] = []
    evaluation_labels: list[str] = []
    schedules: dict[int, Sequence[SeasonScheduleGame]] = {}
    for season in args.seasons:
        for as_of_label in as_of_labels:
            projection, evaluation, baseline, baseline_evaluation = _evaluate_one_season(
                season=season,
                as_of_label=as_of_label,
                teams=teams,
                args=args,
                cache=context_cache,
                postseason_results=postseason_results,
            )
            labeled_projections.append(LabeledProjection("model", projection, as_of_label))
            labeled_projections.append(
                LabeledProjection("baseline", baseline, as_of_label)
            )
            model_evaluations.append(evaluation)
            baseline_evaluations.append(baseline_evaluation)
            evaluation_labels.append(as_of_label)
            schedules[season] = context_cache[(season, as_of_label)].schedule

    _print_aggregate(model_evaluations, baseline_evaluations)
    calibration_rows = _calibration_rows(
        labeled_projections,
        schedules,
        teams,
        postseason_results,
    )
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
            _season_summary_rows(
                model_evaluations,
                baseline_evaluations,
                evaluation_labels,
            ),
        )
    if args.graphics_out_dir is not None:
        _write_graphics(args.graphics_out_dir, labeled_projections, teams)


if __name__ == "__main__":
    main()
