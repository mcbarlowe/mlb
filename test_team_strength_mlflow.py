from __future__ import annotations

from dataclasses import asdict
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from mlflow.sklearn import log_model as log_sklearn_model
from sklearn.linear_model import LogisticRegression

import mlb.sim.team_strength as team_strength_module
from scripts.evaluate_team_strength import log_model_version, score
from mlb.sim.team_strength import (
    DEFAULT_STRENGTH_CONFIG,
    FEATURE_NAMES,
    LEGACY_FEATURE_NAMES,
    LEGACY_STRENGTH_MODEL_CONTRACT_VERSION,
    STRENGTH_MODEL_FAMILY,
    WIN_PROBABILITY_MODEL_COLLECTION,
    WIN_PROBABILITY_MODEL_TYPE,
    CompletedGame,
    StarterLine,
    StrengthFeatureBuilder,
    StrengthModelFit,
    TeamStrengthPredictor,
    build_live_strength_predictor,
)


def _fitted_model() -> tuple[StrengthModelFit, pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(
        [
            [2021, -0.8, -0.4, -0.2, -0.3, -0.1, -0.02, -0.1, -0.1, 0],
            [2021, 0.7, 0.2, 0.3, 0.4, 0.1, 0.02, 0.1, 0.1, 1],
            [2022, -0.5, -0.3, 0.1, -0.2, -0.2, -0.01, -0.1, 0.0, 0],
            [2022, 0.9, 0.5, -0.1, 0.3, 0.2, 0.03, 0.2, 0.1, 1],
            [2023, -0.4, -0.1, 0.2, -0.1, -0.1, -0.02, 0.0, -0.1, 0],
            [2023, 0.6, 0.4, -0.2, 0.2, 0.3, 0.01, 0.1, 0.1, 1],
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
    probabilities = fitted.estimator.predict_proba(test[list(FEATURE_NAMES)])[:, 1]
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
        versions = client.search_model_versions(f"name = '{registered_model_name}'")
        registered_model = client.get_registered_model(registered_model_name)
        run = client.get_run(logged.run_id)
        contract_path = client.download_artifacts(
            logged.run_id,
            "model_contract.json",
            dst_path=str(tmp_path / "download"),
        )
        loaded = mlflow.pyfunc.load_model(
            f"models:/{registered_model_name}/{logged.version}"
        )
        loaded_probabilities = np.asarray(loaded.predict(test[list(FEATURE_NAMES)]))

        assert logged.version == "1"
        assert str(versions[0].version) == logged.version
        assert versions[0].tags["promotion_gate"] == "passed"
        assert versions[0].tags["model_collection"] == WIN_PROBABILITY_MODEL_COLLECTION
        assert versions[0].tags["model_type"] == WIN_PROBABILITY_MODEL_TYPE
        assert registered_model.tags["champion_version"] == logged.version
        assert registered_model.tags["champion_run_id"] == logged.run_id
        assert (
            registered_model.tags["model_collection"]
            == WIN_PROBABILITY_MODEL_COLLECTION
        )
        assert registered_model.tags["model_type"] == WIN_PROBABILITY_MODEL_TYPE
        assert run.data.tags["model_collection"] == WIN_PROBABILITY_MODEL_COLLECTION
        assert run.data.tags["model_type"] == WIN_PROBABILITY_MODEL_TYPE
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
        champion = client.get_model_version_by_alias(
            registered_model_name,
            "champion",
        )
        assert str(champion.version) == logged.version
        assert challenger_run.data.tags["production_model"] == "false"
        assert challenger_run.data.tags["promotion_gate"] == "failed"

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


def test_literal_v1_champion_remains_loadable_with_v2_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'legacy-tracking.db'}"
    artifact_root = tmp_path / "legacy-artifacts"
    artifact_root.mkdir()
    experiment_name = "legacy-team-strength-test"
    registered_model_name = "legacy-team-strength-win"
    original_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment(
        experiment_name,
        artifact_location=artifact_root.as_uri(),
    )
    frame = pd.DataFrame(
        [
            [-1.0, -0.5, 0.2, -0.3, -0.2, 0],
            [-0.5, -0.2, 0.1, -0.1, -0.1, 0],
            [0.5, 0.2, -0.1, 0.1, 0.1, 1],
            [1.0, 0.5, -0.2, 0.3, 0.2, 1],
        ],
        columns=[*LEGACY_FEATURE_NAMES, "home_won"],
    )
    estimator = LogisticRegression(C=1.0, max_iter=1000)
    estimator.fit(frame[list(LEGACY_FEATURE_NAMES)], frame["home_won"])
    config = asdict(DEFAULT_STRENGTH_CONFIG)
    config.pop("roster")

    try:
        with mlflow.start_run(experiment_id=experiment_id) as run:
            mlflow.set_tag("promotion_gate", "passed")
            mlflow.log_dict(
                {
                    "contract_version": LEGACY_STRENGTH_MODEL_CONTRACT_VERSION,
                    "model_family": STRENGTH_MODEL_FAMILY,
                    "features": list(LEGACY_FEATURE_NAMES),
                    "training": {"start_season": 2015},
                    "strength_config": config,
                },
                "model_contract.json",
            )
            model_info = log_sklearn_model(
                estimator,
                name=WIN_PROBABILITY_MODEL_TYPE,
                registered_model_name=registered_model_name,
                await_registration_for=120,
            )
            run_id = run.info.run_id
        assert model_info.registered_model_version is not None
        version = str(model_info.registered_model_version)
        client = MlflowClient(tracking_uri=tracking_uri)
        client.set_model_version_tag(
            registered_model_name,
            version,
            "promotion_gate",
            "passed",
        )
        client.set_model_version_tag(
            registered_model_name,
            version,
            "model_type",
            WIN_PROBABILITY_MODEL_TYPE,
        )
        client.set_registered_model_tag(
            registered_model_name,
            "champion_version",
            version,
        )
        client.set_registered_model_tag(
            registered_model_name,
            "champion_run_id",
            run_id,
        )
        client.set_registered_model_alias(
            registered_model_name,
            "champion",
            version,
        )
        monkeypatch.setattr(
            team_strength_module,
            "load_completed_games",
            lambda **_kwargs: [],
        )

        predictor = build_live_strength_predictor(
            date(2024, 4, 1),
            tracking_uri=tracking_uri,
            registered_model_name=registered_model_name,
        )
        probability = predictor.predict_home_probability(
            season=2024,
            away_team_id=1,
            home_team_id=2,
            away_starter_id=10,
            home_starter_id=20,
            prediction_date=date(2024, 4, 1),
            away_batter_ids=(101,),
            home_batter_ids=(201,),
            away_reliever_ids=(301,),
            home_reliever_ids=(401,),
        )

        assert predictor.feature_names == LEGACY_FEATURE_NAMES
        assert len(predictor.coefficients) == len(LEGACY_FEATURE_NAMES)
        assert 0.0 <= probability <= 1.0
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)


def test_live_reconstruction_excludes_slate_day_games(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def completed(
        game_pk: int,
        game_datetime: str,
        *,
        home_runs: int,
        away_runs: int,
    ) -> CompletedGame:
        starter = StarterLine(
            player_id=100,
            outs=18,
            earned_runs=2,
            strikeouts=6,
            walks=1,
            home_runs=1,
            hit_batters=0,
        )
        return CompletedGame(
            game_pk=game_pk,
            season=2024,
            game_datetime=game_datetime,
            away_team_id=1,
            home_team_id=2,
            away_runs=away_runs,
            home_runs=home_runs,
            away_starter=starter,
            home_starter=StarterLine(
                player_id=200,
                outs=starter.outs,
                earned_runs=starter.earned_runs,
                strikeouts=starter.strikeouts,
                walks=starter.walks,
                home_runs=starter.home_runs,
                hit_batters=starter.hit_batters,
            ),
        )

    prior = completed(
        1,
        "2024-04-01T17:00:00Z",
        home_runs=5,
        away_runs=2,
    )
    slate_day = completed(
        2,
        "2024-04-02T17:00:00Z",
        home_runs=2,
        away_runs=5,
    )
    estimator = SimpleNamespace(
        coef_=np.array([[1.0, 0.0, 0.0, 0.0, 0.0]]),
        intercept_=np.array([0.0]),
    )
    monkeypatch.setattr(
        team_strength_module,
        "_load_champion_strength_model",
        lambda **_kwargs: SimpleNamespace(
            estimator=estimator,
            config=DEFAULT_STRENGTH_CONFIG,
            feature_names=LEGACY_FEATURE_NAMES,
            start_season=2024,
            source=None,
        ),
    )
    monkeypatch.setattr(
        team_strength_module,
        "load_completed_games",
        lambda **_kwargs: [prior, slate_day],
    )

    predictor = build_live_strength_predictor(date(2024, 4, 2))
    live_features = predictor.feature_builder.matchup_features(
        season=2024,
        away_team_id=1,
        home_team_id=2,
        away_starter_id=100,
        home_starter_id=200,
    )
    expected_builder = StrengthFeatureBuilder()
    expected_builder.observe(prior)
    expected_features = expected_builder.matchup_features(
        season=2024,
        away_team_id=1,
        home_team_id=2,
        away_starter_id=100,
        home_starter_id=200,
    )

    assert live_features == expected_features
