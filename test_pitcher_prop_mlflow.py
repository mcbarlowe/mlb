from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow import MlflowClient

from mlb.pitcher_props.evaluation import evaluate_walk_forward
from mlb.pitcher_props.mlflow_registry import (
    REGISTERED_MODEL_NAMES,
    register_pitcher_prop_models,
)
from mlb.pitcher_props.model import PitcherPropConfig, PitcherPropModel, PitcherStart


def _start(index: int, *, season: int) -> PitcherStart:
    return PitcherStart(
        game_pk=900_000 + index,
        game_time=datetime(season, 4, 1, 17, tzinfo=UTC) + timedelta(days=index),
        season=season,
        pitcher_id=20 + index % 3,
        opponent_team_id=200 + index % 2,
        is_home=index % 2 == 0,
        outs=15 + index % 6,
        batters_faced=21 + index % 7,
        strikeouts=4 + index % 5,
        hits_allowed=4 + index % 4,
        walks=1 + index % 3,
        pitch_count=82 + index % 20,
        rest_days=5.0,
    )


def test_registers_every_component_as_loadable_idempotent_pyfunc(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    experiment_name = "pitcher-prop-registry-test"
    original_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.create_experiment(
        experiment_name,
        artifact_location=artifact_root.as_uri(),
    )
    config = PitcherPropConfig(
        min_draws=200,
        max_draws=200,
        draw_batch=200,
        mc_tolerance=0.02,
        seed=5,
    )
    training = [_start(index, season=2023) for index in range(30)]
    validation = [_start(100 + index, season=2024) for index in range(6)]
    model = PitcherPropModel.fit(training, config=config)
    report = evaluate_walk_forward(model, validation, draws_per_start=100, seed=11)

    try:
        registered = register_pitcher_prop_models(
            model,
            report,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            await_registration_for=30,
        )
        client = MlflowClient(tracking_uri=tracking_uri)

        assert {item.market for item in registered} == set(REGISTERED_MODEL_NAMES)
        assert all(not item.skipped_existing for item in registered)
        for item in registered:
            versions = client.search_model_versions(
                f"name = '{item.registered_model_name}'"
            )
            assert len(versions) == 1
            assert versions[0].tags["model_contract_version"] == "pitcher-prop-bayes-v1"
            assert versions[0].tags["model_type"] == item.market

        strikeouts = next(item for item in registered if item.market == "pitcher_strikeouts")
        loaded = mlflow.pyfunc.load_model(
            f"models:/{strikeouts.registered_model_name}/{strikeouts.version}"
        )
        prediction = loaded.predict(
            pd.DataFrame(
                {
                    "pitcher_id": [20],
                    "opponent_team_id": [200],
                    "point": [5.5],
                    "prediction_time": ["2024-05-01T12:00:00Z"],
                    "is_home": [True],
                    "rest_days": [5.0],
                    "seed": [99],
                }
            )
        )

        assert prediction.loc[0, "market"] == "pitcher_strikeouts"
        assert prediction.loc[0, "probability_over"] >= 0.0
        assert prediction.loc[0, "probability_under"] >= 0.0

        repeated = register_pitcher_prop_models(
            model,
            report,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            await_registration_for=30,
        )
        assert all(item.skipped_existing for item in repeated)
        assert all(
            len(
                client.search_model_versions(
                    f"name = '{item.registered_model_name}'"
                )
            )
            == 1
            for item in repeated
        )
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)
