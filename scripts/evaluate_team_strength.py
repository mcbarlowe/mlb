"""Evaluate the leak-free team-strength win model on a held-out season."""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.mlflow_utils import (
    DEFAULT_MLFLOW_EXPERIMENT,
    build_param_dict,
    configure_mlflow,
)
from src.sim.team_strength import (
    DEFAULT_REGISTERED_STRENGTH_MODEL as DEFAULT_REGISTERED_MODEL,
)
from src.sim.team_strength import (
    FEATURE_NAMES,
    WIN_PROBABILITY_MODEL_COLLECTION,
    WIN_PROBABILITY_MODEL_TYPE,
    StrengthModelFit,
    TeamStrengthPredictor,
    load_completed_games,
    train_strength_model,
)
from src.sim.team_strength import (
    STRENGTH_MODEL_CONTRACT_VERSION as MODEL_CONTRACT_VERSION,
)
from src.sim.team_strength import (
    STRENGTH_MODEL_FAMILY as MODEL_FAMILY,
)


@dataclass(frozen=True)
class ProbabilityMetrics:
    brier: float
    log_loss: float
    pick_accuracy: float
    mean_probability: float


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
        help="Defaults to the four seasons immediately before --test-season.",
    )
    parser.add_argument(
        "--home-rate-baseline",
        type=float,
        default=0.543,
        help="Pregame constant home-win probability baseline.",
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
                -(
                    outcomes * np.log(clipped)
                    + (1.0 - outcomes) * np.log(1.0 - clipped)
                )
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
        "pick_accuracy_improvement": (
            model.pick_accuracy - home_rate.pick_accuracy
        ),
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
    contract = {
        "contract_version": MODEL_CONTRACT_VERSION,
        "model_family": MODEL_FAMILY,
        "model_type": WIN_PROBABILITY_MODEL_TYPE,
        "model_collection": WIN_PROBABILITY_MODEL_COLLECTION,
        "registered_model_name": registered_model_name,
        "estimator": "sklearn.linear_model.LogisticRegression",
        "features": feature_columns,
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
        "strength_config": asdict(fitted.config),
        "coefficients": dict(
            zip(FEATURE_NAMES, fitted.predictor.coefficients)
        ),
        "intercept": fitted.predictor.intercept,
    }
    with mlflow.start_run(
        run_name=f"team-strength-win-holdout-{test_season}"
    ) as run:
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
                }
            )
        )
        mlflow.log_metrics(metrics)
        mlflow.log_dict(contract, "model_contract.json")
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
            "MLB win probability model (`win_probability_model`) using the "
            "`team_strength_win` implementation: chronological team Elo, run "
            "form, and Bayesian-shrunk starting-pitcher quality."
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
            "champion_holdout_pick_accuracy": (
                f"{model_metrics.pick_accuracy:.12f}"
            ),
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


def main() -> None:
    args = parse_args()
    train_seasons = args.train_seasons or list(
        range(args.test_season - 4, args.test_season)
    )
    if any(season >= args.test_season for season in train_seasons):
        raise ValueError("Training seasons must precede the held-out test season")
    if not 0.0 < args.home_rate_baseline < 1.0:
        raise ValueError("--home-rate-baseline must be between zero and one")

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

    print(f"Held-out season: {args.test_season} ({len(test):,} games)")
    print(f"Training seasons: {', '.join(map(str, train_seasons))}")
    print(f"Actual home-win rate: {float(np.mean(outcomes)):.3f}")
    print_metrics("Team-strength model", model)
    print_metrics("League home rate", home_rate)
    print_metrics("Coin flip", coin_flip)
    print("Coefficients:")
    for name, coefficient in zip(FEATURE_NAMES, predictor.coefficients):
        print(f"  {name:<22} {coefficient:+.6f}")
    print(f"  {'intercept':<22} {predictor.intercept:+.6f}")

    checks = {
        "Brier": model.brier < home_rate.brier,
        "log loss": model.log_loss < home_rate.log_loss,
        "pick accuracy": model.pick_accuracy > home_rate.pick_accuracy,
    }
    print(
        "Gate: "
        + ", ".join(
            f"{name}={'PASS' if passed else 'FAIL'}"
            for name, passed in checks.items()
        )
    )
    gate_passed = all(checks.values())
    brier_gain = home_rate.brier - model.brier
    log_loss_gain = home_rate.log_loss - model.log_loss
    accuracy_gain = model.pick_accuracy - home_rate.pick_accuracy
    if not all(
        math.isfinite(value)
        for value in (brier_gain, log_loss_gain, accuracy_gain)
    ):
        raise RuntimeError("Non-finite evaluation improvement")
    if args.set_champion and not args.log_mlflow:
        raise ValueError("--set-champion requires --log-mlflow")
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
            gate_passed=gate_passed,
            tracking_uri=args.mlflow_tracking_uri,
            experiment_name=args.mlflow_experiment,
            registered_model_name=args.registered_model_name,
            set_champion=args.set_champion,
        )
        print(
            f"MLflow: run={logged.run_id} "
            f"registered_model={logged.registered_model_name} "
            f"version={logged.version}"
        )
    if not gate_passed:
        raise SystemExit(1)
    print(
        "Promotion gate passed: "
        f"Brier {brier_gain:+.4f}, log loss {log_loss_gain:+.4f}, "
        f"accuracy {accuracy_gain:+.1%} versus league home rate."
    )


if __name__ == "__main__":
    main()
