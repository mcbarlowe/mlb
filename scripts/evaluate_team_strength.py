"""Evaluate win models with leak-free rolling seasons and uncertainty gates."""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mlb.ml.mlflow_utils import (
    DEFAULT_MLFLOW_EXPERIMENT,
    build_param_dict,
    configure_mlflow,
)
from mlb.sim.team_strength import (
    DEFAULT_REGISTERED_STRENGTH_MODEL as DEFAULT_REGISTERED_MODEL,
)
from mlb.sim.team_strength import (
    FEATURE_NAMES,
    LEGACY_FEATURE_NAMES,
    WIN_PROBABILITY_MODEL_COLLECTION,
    WIN_PROBABILITY_MODEL_TYPE,
    StrengthModelFit,
    TeamStrengthPredictor,
    load_completed_games,
    train_strength_model,
)
from mlb.sim.team_strength import (
    STRENGTH_MODEL_CONTRACT_VERSION as MODEL_CONTRACT_VERSION,
)
from mlb.sim.team_strength import (
    STRENGTH_MODEL_FAMILY as MODEL_FAMILY,
)


@dataclass(frozen=True)
class ProbabilityMetrics:
    brier: float
    log_loss: float
    pick_accuracy: float
    mean_probability: float


@dataclass(frozen=True)
class ImprovementInterval:
    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True)
class RollingFold:
    season: int
    train_seasons: tuple[int, ...]
    games: int
    candidate: ProbabilityMetrics
    legacy_v1: ProbabilityMetrics
    home_rate: ProbabilityMetrics


@dataclass(frozen=True)
class RollingEvaluation:
    folds: tuple[RollingFold, ...]
    aggregate_candidate: ProbabilityMetrics
    aggregate_legacy_v1: ProbabilityMetrics
    aggregate_home_rate: ProbabilityMetrics
    intervals: dict[str, ImprovementInterval]
    bootstrap_seed: int
    bootstrap_resamples: int
    block_unit: str
    max_season_regression: float
    gate_checks: dict[str, bool]
    gate_passed: bool


@dataclass(frozen=True)
class LoggedModelVersion:
    run_id: str
    registered_model_name: str
    version: str
    tracking_uri: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pregame home-win probabilities on a held-out season."
    )
    parser.add_argument("--test-season", type=int, default=2025)
    parser.add_argument("--start-season", type=int, default=2015)
    parser.add_argument(
        "--train-seasons",
        type=int,
        nargs="*",
        default=None,
        help="Defaults to the --rolling-window seasons before --test-season.",
    )
    parser.add_argument(
        "--home-rate-baseline",
        type=float,
        default=0.543,
        help="Pregame constant home-win probability baseline.",
    )
    parser.add_argument(
        "--rolling-seasons",
        type=int,
        nargs="*",
        default=None,
        help="Walk-forward folds; defaults to the four seasons through --test-season.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=4,
        help="Number of preceding seasons used to train each rolling fold.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--max-season-regression",
        type=float,
        default=0.001,
        help="Maximum candidate loss increase versus v1 in any rolling fold.",
    )
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="Shared tracking URI; defaults through src.ml.mlflow_utils.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=DEFAULT_MLFLOW_EXPERIMENT,
    )
    parser.add_argument(
        "--registered-model-name",
        type=str,
        default=DEFAULT_REGISTERED_MODEL,
    )
    parser.add_argument(
        "--set-champion",
        action="store_true",
        help="Mark this passing registered version as the comparison champion.",
    )
    return parser.parse_args()


def score(probabilities: np.ndarray, outcomes: np.ndarray) -> ProbabilityMetrics:
    clipped = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    return ProbabilityMetrics(
        brier=float(np.mean((clipped - outcomes) ** 2)),
        log_loss=float(
            np.mean(
                -(outcomes * np.log(clipped) + (1.0 - outcomes) * np.log(1.0 - clipped))
            )
        ),
        pick_accuracy=float(np.mean((clipped >= 0.5) == outcomes.astype(bool))),
        mean_probability=float(np.mean(clipped)),
    )


def predict_frame(predictor: TeamStrengthPredictor, frame) -> np.ndarray:
    values = frame[list(FEATURE_NAMES)].to_numpy(dtype=float)
    coefficients = np.asarray(predictor.coefficients, dtype=float)
    log_odds = predictor.intercept + values @ coefficients
    return 1.0 / (1.0 + np.exp(-log_odds))


def _losses(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    metric: str,
) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    if metric == "brier":
        return (clipped - outcomes) ** 2
    if metric == "log_loss":
        return -(outcomes * np.log(clipped) + (1.0 - outcomes) * np.log(1.0 - clipped))
    raise ValueError(f"Unknown probability metric {metric!r}")


def paired_block_improvement_interval(
    candidate_probabilities: np.ndarray,
    comparator_probabilities: np.ndarray,
    outcomes: np.ndarray,
    blocks: np.ndarray,
    *,
    metric: str,
    resamples: int,
    seed: int,
) -> ImprovementInterval:
    """Bootstrap paired comparator-minus-candidate loss by game-date block."""
    if resamples < 100:
        raise ValueError("bootstrap resamples must be at least 100")
    if not (
        len(candidate_probabilities)
        == len(comparator_probabilities)
        == len(outcomes)
        == len(blocks)
    ):
        raise ValueError("Bootstrap inputs must have equal lengths")
    if len(outcomes) == 0:
        raise ValueError("Bootstrap inputs must not be empty")
    improvement = _losses(comparator_probabilities, outcomes, metric) - _losses(
        candidate_probabilities, outcomes, metric
    )
    _, inverse = np.unique(np.asarray(blocks, dtype=str), return_inverse=True)
    block_sums = np.bincount(inverse, weights=improvement)
    block_counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0,
        len(block_sums),
        size=(resamples, len(block_sums)),
    )
    sampled = block_sums[draws].sum(axis=1) / block_counts[draws].sum(axis=1)
    lower, upper = np.quantile(sampled, (0.025, 0.975))
    interval = ImprovementInterval(
        estimate=float(np.mean(improvement)),
        lower=float(lower),
        upper=float(upper),
    )
    if not all(math.isfinite(value) for value in asdict(interval).values()):
        raise RuntimeError("Non-finite bootstrap improvement interval")
    return interval


def evaluate_rolling_seasons(
    feature_frame: pd.DataFrame,
    *,
    seasons: Sequence[int],
    train_window: int,
    home_rate_baseline: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    max_season_regression: float,
) -> RollingEvaluation:
    """Compare v2 against walk-forward v1 and home-rate baselines."""
    from sklearn.linear_model import LogisticRegression

    resolved_seasons = tuple(sorted({int(season) for season in seasons}))
    if not resolved_seasons:
        raise ValueError("At least one rolling season is required")
    if train_window < 1:
        raise ValueError("rolling window must be positive")
    if max_season_regression < 0.0:
        raise ValueError("max season regression must be non-negative")

    folds: list[RollingFold] = []
    candidate_parts: list[np.ndarray] = []
    legacy_parts: list[np.ndarray] = []
    home_parts: list[np.ndarray] = []
    outcome_parts: list[np.ndarray] = []
    block_parts: list[np.ndarray] = []
    for season in resolved_seasons:
        train_seasons = tuple(range(season - train_window, season))
        train = feature_frame[feature_frame["season"].isin(train_seasons)]
        test = feature_frame[feature_frame["season"] == season]
        if train.empty or test.empty:
            raise ValueError(
                f"Rolling fold {season} requires non-empty seasons "
                f"{list(train_seasons)} and holdout data"
            )
        candidate = LogisticRegression(C=1.0, max_iter=1000)
        legacy = LogisticRegression(C=1.0, max_iter=1000)
        candidate.fit(train[list(FEATURE_NAMES)], train["home_won"])
        legacy.fit(train[list(LEGACY_FEATURE_NAMES)], train["home_won"])
        outcomes = test["home_won"].to_numpy(dtype=float)
        candidate_probabilities = candidate.predict_proba(test[list(FEATURE_NAMES)])[
            :, 1
        ]
        legacy_probabilities = legacy.predict_proba(test[list(LEGACY_FEATURE_NAMES)])[
            :, 1
        ]
        home_probabilities = np.full(len(test), home_rate_baseline, dtype=float)
        folds.append(
            RollingFold(
                season=season,
                train_seasons=train_seasons,
                games=len(test),
                candidate=score(candidate_probabilities, outcomes),
                legacy_v1=score(legacy_probabilities, outcomes),
                home_rate=score(home_probabilities, outcomes),
            )
        )
        candidate_parts.append(candidate_probabilities)
        legacy_parts.append(legacy_probabilities)
        home_parts.append(home_probabilities)
        outcome_parts.append(outcomes)
        block_parts.append(
            (str(season) + ":" + test["game_date"].astype(str)).to_numpy()
        )

    candidate_probabilities = np.concatenate(candidate_parts)
    legacy_probabilities = np.concatenate(legacy_parts)
    home_probabilities = np.concatenate(home_parts)
    outcomes = np.concatenate(outcome_parts)
    blocks = np.concatenate(block_parts)
    comparators = {
        "legacy_v1": legacy_probabilities,
        "home_rate": home_probabilities,
    }
    intervals = {
        f"{metric}_vs_{name}": paired_block_improvement_interval(
            candidate_probabilities,
            probabilities,
            outcomes,
            blocks,
            metric=metric,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + offset,
        )
        for offset, (name, probabilities, metric) in enumerate(
            (
                (name, probabilities, metric)
                for name, probabilities in comparators.items()
                for metric in ("brier", "log_loss")
            )
        )
    }
    gate_checks = {
        f"{name}_lower_bound_positive": interval.lower > 0.0
        for name, interval in intervals.items()
    }
    gate_checks["no_material_season_regression"] = all(
        fold.candidate.brier <= fold.legacy_v1.brier + max_season_regression
        and fold.candidate.log_loss <= fold.legacy_v1.log_loss + max_season_regression
        for fold in folds
    )
    return RollingEvaluation(
        folds=tuple(folds),
        aggregate_candidate=score(candidate_probabilities, outcomes),
        aggregate_legacy_v1=score(legacy_probabilities, outcomes),
        aggregate_home_rate=score(home_probabilities, outcomes),
        intervals=intervals,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
        block_unit="season_game_date",
        max_season_regression=max_season_regression,
        gate_checks=gate_checks,
        gate_passed=all(gate_checks.values()),
    )


def print_metrics(label: str, metrics: ProbabilityMetrics) -> None:
    print(
        f"{label:<22} "
        f"Brier={metrics.brier:.4f}  "
        f"log_loss={metrics.log_loss:.4f}  "
        f"accuracy={metrics.pick_accuracy:.1%}  "
        f"mean_p(home)={metrics.mean_probability:.3f}"
    )


def _metric_payload(
    model: ProbabilityMetrics,
    home_rate: ProbabilityMetrics,
    coin_flip: ProbabilityMetrics,
) -> dict[str, float]:
    return {
        "holdout_brier": model.brier,
        "holdout_log_loss": model.log_loss,
        "holdout_pick_accuracy": model.pick_accuracy,
        "holdout_mean_probability": model.mean_probability,
        "baseline_home_brier": home_rate.brier,
        "baseline_home_log_loss": home_rate.log_loss,
        "baseline_home_pick_accuracy": home_rate.pick_accuracy,
        "baseline_coin_brier": coin_flip.brier,
        "baseline_coin_log_loss": coin_flip.log_loss,
        "brier_improvement": home_rate.brier - model.brier,
        "log_loss_improvement": home_rate.log_loss - model.log_loss,
        "pick_accuracy_improvement": (model.pick_accuracy - home_rate.pick_accuracy),
    }


def log_model_version(
    *,
    fitted: StrengthModelFit,
    train: pd.DataFrame,
    test: pd.DataFrame,
    test_season: int,
    start_season: int,
    home_rate_baseline: float,
    model_metrics: ProbabilityMetrics,
    home_rate_metrics: ProbabilityMetrics,
    coin_flip_metrics: ProbabilityMetrics,
    gate_passed: bool,
    tracking_uri: str | None,
    experiment_name: str,
    registered_model_name: str,
    set_champion: bool,
    rolling_evidence: RollingEvaluation | None = None,
) -> LoggedModelVersion:
    """Log a comparable run and immutable registered sklearn model version."""
    import mlflow
    import sklearn
    from mlflow.data.pandas_dataset import from_pandas
    from mlflow.models import infer_signature
    from mlflow.sklearn import log_model
    from mlflow.tracking import MlflowClient

    resolved_uri = configure_mlflow(
        experiment_name,
        tracking_uri,
        require_tracking_uri=True,
    )
    feature_columns = list(FEATURE_NAMES)
    input_example = test[feature_columns].head(5)
    signature = infer_signature(
        input_example,
        fitted.estimator.predict_proba(input_example),
    )
    metrics = _metric_payload(
        model_metrics,
        home_rate_metrics,
        coin_flip_metrics,
    )
    if rolling_evidence is not None:
        metrics["rolling_games"] = float(
            sum(fold.games for fold in rolling_evidence.folds)
        )
        metrics["rolling_fold_count"] = float(len(rolling_evidence.folds))
        for name, interval in rolling_evidence.intervals.items():
            metrics[f"rolling_{name}_estimate"] = interval.estimate
            metrics[f"rolling_{name}_lower"] = interval.lower
            metrics[f"rolling_{name}_upper"] = interval.upper
    contract = {
        "contract_version": MODEL_CONTRACT_VERSION,
        "model_family": MODEL_FAMILY,
        "model_type": WIN_PROBABILITY_MODEL_TYPE,
        "model_collection": WIN_PROBABILITY_MODEL_COLLECTION,
        "registered_model_name": registered_model_name,
        "estimator": "sklearn.linear_model.LogisticRegression",
        "features": feature_columns,
        "feature_units": {
            "elo_diff": "100_elo_points",
            "run_edge": "runs_per_game",
            "starter_era_edge": "earned_runs_per_nine",
            "starter_fip_edge": "fip_runs_per_nine",
            "starter_length_edge": "innings",
            "lineup_woba_edge": "10_woba_points",
            "bullpen_fip_edge": "fip_runs_per_nine",
            "bullpen_availability_edge": "workload_adjusted_fip_runs",
        },
        "target": "home_won",
        "probability_output": {
            "method": "predict_proba",
            "home_win_column": 1,
        },
        "chronology": {
            "features_emitted_before_game_update": True,
            "training_seasons_precede_holdout": True,
        },
        "training": {
            "start_season": start_season,
            "train_seasons": list(fitted.train_seasons),
            "test_season": test_season,
            "training_games": len(train),
            "holdout_games": len(test),
        },
        "rolling_evaluation": (
            asdict(rolling_evidence) if rolling_evidence is not None else None
        ),
        "strength_config": asdict(fitted.config),
        "coefficients": dict(zip(FEATURE_NAMES, fitted.predictor.coefficients)),
        "intercept": fitted.predictor.intercept,
    }
    with mlflow.start_run(run_name=f"team-strength-win-holdout-{test_season}") as run:
        mlflow.set_tags(
            {
                "model_family": MODEL_FAMILY,
                "model_type": WIN_PROBABILITY_MODEL_TYPE,
                "model_collection": WIN_PROBABILITY_MODEL_COLLECTION,
                "model_contract_version": str(MODEL_CONTRACT_VERSION),
                "promotion_gate": "passed" if gate_passed else "failed",
                "production_model": str(set_champion and gate_passed).lower(),
                "registered_model_name": registered_model_name,
            }
        )
        mlflow.log_params(
            build_param_dict(
                {
                    "start_season": start_season,
                    "train_seasons": list(fitted.train_seasons),
                    "test_season": test_season,
                    "training_games": len(train),
                    "holdout_games": len(test),
                    "home_rate_baseline": home_rate_baseline,
                    "feature_names": feature_columns,
                    "estimator": "logistic_regression",
                    "estimator_c": fitted.estimator.C,
                    "estimator_max_iter": fitted.estimator.max_iter,
                    "strength_config": asdict(fitted.config),
                    "rolling_seasons": (
                        [fold.season for fold in rolling_evidence.folds]
                        if rolling_evidence is not None
                        else []
                    ),
                    "bootstrap_seed": (
                        rolling_evidence.bootstrap_seed
                        if rolling_evidence is not None
                        else None
                    ),
                    "bootstrap_resamples": (
                        rolling_evidence.bootstrap_resamples
                        if rolling_evidence is not None
                        else None
                    ),
                }
            )
        )
        mlflow.log_metrics(metrics)
        mlflow.log_dict(contract, "model_contract.json")
        if rolling_evidence is not None:
            mlflow.log_dict(
                asdict(rolling_evidence),
                "rolling_evaluation.json",
            )
        mlflow.log_input(
            from_pandas(
                train[feature_columns + ["home_won"]].astype(float),
                name=f"team-strength-training-through-{max(fitted.train_seasons)}",
                targets="home_won",
            ),
            context="training",
        )
        mlflow.log_input(
            from_pandas(
                test[feature_columns + ["home_won"]].astype(float),
                name=f"team-strength-holdout-{test_season}",
                targets="home_won",
            ),
            context="validation",
        )
        env_key = "MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING"
        previous_env_setting = os.environ.get(env_key)
        os.environ[env_key] = "false"
        try:
            model_info = log_model(
                fitted.estimator,
                name=WIN_PROBABILITY_MODEL_TYPE,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example,
                pyfunc_predict_fn="predict_proba",
                metadata={
                    "contract_version": MODEL_CONTRACT_VERSION,
                    "model_type": WIN_PROBABILITY_MODEL_TYPE,
                    "model_collection": WIN_PROBABILITY_MODEL_COLLECTION,
                    "feature_names": feature_columns,
                    "home_win_probability_column": 1,
                },
                pip_requirements=[
                    f"mlflow=={mlflow.__version__}",
                    f"scikit-learn=={sklearn.__version__}",
                    f"pandas=={pd.__version__}",
                ],
                await_registration_for=120,
            )
        finally:
            if previous_env_setting is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = previous_env_setting
        run_id = run.info.run_id

    if model_info.registered_model_version is None:
        raise RuntimeError("MLflow did not return a registered model version")
    version = str(model_info.registered_model_version)
    client = MlflowClient(tracking_uri=resolved_uri)
    client.update_registered_model(
        name=registered_model_name,
        description=(
            "MLB win probability model (`win_probability_model`) using "
            "chronological team, starter, projected-lineup, and individual "
            "bullpen strength with recent workload availability."
        ),
    )
    classification_tags = {
        "model_type": WIN_PROBABILITY_MODEL_TYPE,
        "model_collection": WIN_PROBABILITY_MODEL_COLLECTION,
        "model_family": MODEL_FAMILY,
    }
    for key, value in classification_tags.items():
        client.set_registered_model_tag(registered_model_name, key, value)
    version_tags = {
        "model_family": MODEL_FAMILY,
        "model_type": WIN_PROBABILITY_MODEL_TYPE,
        "model_collection": WIN_PROBABILITY_MODEL_COLLECTION,
        "model_contract_version": str(MODEL_CONTRACT_VERSION),
        "holdout_season": str(test_season),
        "promotion_gate": "passed" if gate_passed else "failed",
        "holdout_brier": f"{model_metrics.brier:.12f}",
        "holdout_log_loss": f"{model_metrics.log_loss:.12f}",
        "holdout_pick_accuracy": f"{model_metrics.pick_accuracy:.12f}",
    }
    if rolling_evidence is not None:
        version_tags.update(
            {
                f"{name}_ci_lower": f"{interval.lower:.12f}"
                for name, interval in rolling_evidence.intervals.items()
            }
        )
    for key, value in version_tags.items():
        client.set_model_version_tag(
            name=registered_model_name,
            version=version,
            key=key,
            value=value,
        )
    client.set_registered_model_tag(
        registered_model_name,
        "latest_logged_version",
        version,
    )
    client.set_registered_model_tag(
        registered_model_name,
        "latest_run_id",
        run_id,
    )
    if set_champion and gate_passed:
        champion_tags = {
            "champion_version": version,
            "champion_run_id": run_id,
            "champion_holdout_season": str(test_season),
            "champion_holdout_brier": f"{model_metrics.brier:.12f}",
            "champion_holdout_log_loss": f"{model_metrics.log_loss:.12f}",
            "champion_holdout_pick_accuracy": (f"{model_metrics.pick_accuracy:.12f}"),
        }
        for key, value in champion_tags.items():
            client.set_registered_model_tag(
                registered_model_name,
                key,
                value,
            )
        client.set_registered_model_alias(
            name=registered_model_name,
            alias="champion",
            version=version,
        )
    return LoggedModelVersion(
        run_id=run_id,
        registered_model_name=registered_model_name,
        version=version,
        tracking_uri=resolved_uri,
    )


def _resolve_season_windows(
    *,
    test_season: int,
    train_seasons: Sequence[int] | None,
    rolling_seasons: Sequence[int] | None,
    rolling_window: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if rolling_window < 1:
        raise ValueError("--rolling-window must be positive")
    resolved_train = tuple(
        sorted(
            {
                int(season)
                for season in (
                    train_seasons
                    if train_seasons is not None
                    else range(test_season - rolling_window, test_season)
                )
            }
        )
    )
    resolved_rolling = tuple(
        sorted(
            {
                int(season)
                for season in (
                    rolling_seasons
                    if rolling_seasons is not None
                    else range(test_season - 3, test_season + 1)
                )
            }
        )
    )
    if len(resolved_rolling) < 3:
        raise ValueError("Promotion requires at least three rolling-season folds")
    if not resolved_rolling or resolved_rolling[-1] != test_season:
        raise ValueError("Rolling seasons must end with --test-season")
    expected_train = tuple(range(test_season - rolling_window, test_season))
    if resolved_train != expected_train:
        raise ValueError(
            "--train-seasons must match the terminal rolling fold's training window"
        )
    return resolved_train, resolved_rolling


def main() -> None:
    args = parse_args()
    train_seasons, rolling_seasons = _resolve_season_windows(
        test_season=args.test_season,
        train_seasons=args.train_seasons,
        rolling_seasons=args.rolling_seasons,
        rolling_window=args.rolling_window,
    )
    if not 0.0 < args.home_rate_baseline < 1.0:
        raise ValueError("--home-rate-baseline must be between zero and one")
    if args.set_champion and not args.log_mlflow:
        raise ValueError("--set-champion requires --log-mlflow")

    games = load_completed_games(
        start_season=args.start_season,
        end_season=args.test_season,
    )
    fitted = train_strength_model(
        games,
        prediction_season=args.test_season,
        train_seasons=train_seasons,
    )
    predictor = fitted.predictor
    feature_frame = fitted.feature_frame
    test = feature_frame[feature_frame["season"] == args.test_season]
    train = feature_frame[feature_frame["season"].isin(fitted.train_seasons)]
    if test.empty:
        raise ValueError(f"No held-out games found for {args.test_season}")
    outcomes = test["home_won"].to_numpy(dtype=float)
    model = score(predict_frame(predictor, test), outcomes)
    home_rate = score(
        np.full(len(test), args.home_rate_baseline, dtype=float), outcomes
    )
    coin_flip = score(np.full(len(test), 0.5, dtype=float), outcomes)
    rolling = evaluate_rolling_seasons(
        feature_frame,
        seasons=rolling_seasons,
        train_window=args.rolling_window,
        home_rate_baseline=args.home_rate_baseline,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        max_season_regression=args.max_season_regression,
    )

    print(f"Held-out season: {args.test_season} ({len(test):,} games)")
    print(f"Training seasons: {', '.join(map(str, train_seasons))}")
    print(f"Actual home-win rate: {float(np.mean(outcomes)):.3f}")
    print_metrics("Team-strength model", model)
    print_metrics("League home rate", home_rate)
    print_metrics("Coin flip", coin_flip)
    print("Coefficients:")
    for name, coefficient in zip(FEATURE_NAMES, predictor.coefficients, strict=True):
        print(f"  {name:<28} {coefficient:+.6f}")
    print(f"  {'intercept':<28} {predictor.intercept:+.6f}")
    print("Rolling walk-forward folds:")
    for fold in rolling.folds:
        print(
            f"  {fold.season}: {fold.games:,} games, "
            f"Brier candidate={fold.candidate.brier:.6f} "
            f"v1={fold.legacy_v1.brier:.6f}, "
            f"log_loss candidate={fold.candidate.log_loss:.6f} "
            f"v1={fold.legacy_v1.log_loss:.6f}"
        )
    print("Paired date-block 95% improvement intervals:")
    for name, interval in rolling.intervals.items():
        print(
            f"  {name:<26} {interval.estimate:+.6f} "
            f"[{interval.lower:+.6f}, {interval.upper:+.6f}]"
        )
    print(
        "Gate: "
        + ", ".join(
            f"{name}={'PASS' if passed else 'FAIL'}"
            for name, passed in rolling.gate_checks.items()
        )
    )
    if args.log_mlflow:
        logged = log_model_version(
            fitted=fitted,
            train=train,
            test=test,
            test_season=args.test_season,
            start_season=args.start_season,
            home_rate_baseline=args.home_rate_baseline,
            model_metrics=model,
            home_rate_metrics=home_rate,
            coin_flip_metrics=coin_flip,
            gate_passed=rolling.gate_passed,
            tracking_uri=args.mlflow_tracking_uri,
            experiment_name=args.mlflow_experiment,
            registered_model_name=args.registered_model_name,
            set_champion=args.set_champion,
            rolling_evidence=rolling,
        )
        print(
            f"MLflow: run={logged.run_id} "
            f"registered_model={logged.registered_model_name} "
            f"version={logged.version}"
        )
    if not rolling.gate_passed:
        raise SystemExit(1)
    print(
        "Promotion gate passed: all Brier/log-loss lower bounds are positive "
        "versus walk-forward v1 and league home-rate baselines, with no "
        "material season regression."
    )


if __name__ == "__main__":
    main()
