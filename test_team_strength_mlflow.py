from __future__ import annotations

from datetime import date
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression

import src.sim.team_strength as team_strength_module
from scripts.evaluate_team_strength import log_model_version, score
from src.sim.team_strength import (
    DEFAULT_STRENGTH_CONFIG,
    FEATURE_NAMES,
    StrengthFeatureBuilder,
    StrengthModelFit,
    TeamStrengthPredictor,
    build_live_strength_predictor,
)


def _fitted_model() -> tuple[StrengthModelFit, pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(
        [
            [2021, -0.8, -0.4, -0.2, -0.3, -0.1, 0],
            [2021, 0.7, 0.2, 0.3, 0.4, 0.1, 1],
            [2022, -0.5, -0.3, 0.1, -0.2, -0.2, 0],
            [2022, 0.9, 0.5, -0.1, 0.3, 0.2, 1],
            [2023, -0.4, -0.1, 0.2, -0.1, -0.1, 0],
            [2023, 0.6, 0.4, -0.2, 0.2, 0.3, 1],
        ],
        columns=["season", *FEATURE_NAMES, "home_won"],
    )
    train = frame[frame["season"].isin([2021, 2022])]
    test = frame[frame["season"] == 2023]
    estimator = LogisticRegression(C=1.0, max_iter=1000)
    estimator.fit(train[list(FEATURE_NAMES)], train["home_won"])
    predictor = TeamStrengthPredictor(
        coefficients=tuple(
            float(value) for value in estimator.coef_.reshape(-1).tolist()
        ),
        intercept=float(np.asarray(estimator.intercept_).reshape(-1)[0]),
        feature_builder=StrengthFeatureBuilder(),
    )
    return (
        StrengthModelFit(
            estimator=estimator,
            predictor=predictor,
            feature_frame=frame,
            train_seasons=(2021, 2022),
            config=DEFAULT_STRENGTH_CONFIG,
        ),
        train,
        test,
    )


def test_logs_comparable_registered_model_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    experiment_name = "team-strength-test"
    registered_model_name = "test-team-strength-win"
    original_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.create_experiment(
        experiment_name,
        artifact_location=artifact_root.as_uri(),
    )
    fitted, train, test = _fitted_model()
    outcomes = test["home_won"].to_numpy(dtype=float)
    probabilities = fitted.estimator.predict_proba(
        test[list(FEATURE_NAMES)]
    )[:, 1]
    model_metrics = score(probabilities, outcomes)
    home_metrics = score(np.full(len(test), 0.543), outcomes)
    coin_metrics = score(np.full(len(test), 0.5), outcomes)

    try:
        logged = log_model_version(
            fitted=fitted,
            train=train,
            test=test,
            test_season=2023,
            start_season=2021,
            home_rate_baseline=0.543,
            model_metrics=model_metrics,
            home_rate_metrics=home_metrics,
            coin_flip_metrics=coin_metrics,
            gate_passed=True,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            registered_model_name=registered_model_name,
            set_champion=True,
        )

        client = MlflowClient(tracking_uri=tracking_uri)
        versions = client.search_model_versions(
            f"name = '{registered_model_name}'"
        )
        registered_model = client.get_registered_model(registered_model_name)
        run = client.get_run(logged.run_id)
        contract_path = client.download_artifacts(
            logged.run_id,
            "model_contract.json",
            dst_path=str(tmp_path / "download"),
        )
        loaded = mlflow.pyfunc.load_model(f"runs:/{logged.run_id}/model")
        loaded_probabilities = np.asarray(
            loaded.predict(test[list(FEATURE_NAMES)])
        )

        assert logged.version == "1"
        assert str(versions[0].version) == logged.version
        assert registered_model.tags["champion_version"] == logged.version
        assert registered_model.tags["champion_run_id"] == logged.run_id
        assert run.data.metrics["holdout_brier"] == model_metrics.brier
        assert Path(contract_path).exists()
        np.testing.assert_allclose(
            loaded_probabilities,
            fitted.estimator.predict_proba(test[list(FEATURE_NAMES)]),
        )

        challenger = log_model_version(
            fitted=fitted,
            train=train,
            test=test,
            test_season=2023,
            start_season=2021,
            home_rate_baseline=0.543,
            model_metrics=model_metrics,
            home_rate_metrics=home_metrics,
            coin_flip_metrics=coin_metrics,
            gate_passed=False,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            registered_model_name=registered_model_name,
            set_champion=True,
        )
        updated_model = client.get_registered_model(registered_model_name)
        challenger_run = client.get_run(challenger.run_id)

        assert challenger.version == "2"
        assert updated_model.tags["latest_logged_version"] == challenger.version
        assert updated_model.tags["champion_version"] == logged.version
        assert (
            challenger_run.data.tags["promotion_gate"]
            == "failed"
        )

        def no_completed_games(
            **_kwargs: object,
        ) -> list[team_strength_module.CompletedGame]:
            return []

        monkeypatch.setattr(
            team_strength_module,
            "load_completed_games",
            no_completed_games,
        )
        live_predictor = build_live_strength_predictor(
            date(2024, 4, 1),
            tracking_uri=tracking_uri,
            registered_model_name=registered_model_name,
        )

        assert live_predictor.source is not None
        assert live_predictor.source.version == logged.version
        assert live_predictor.source.run_id == logged.run_id
        assert live_predictor.feature_builder.config == DEFAULT_STRENGTH_CONFIG
        np.testing.assert_allclose(
            live_predictor.coefficients,
            fitted.predictor.coefficients,
        )
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)
