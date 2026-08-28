"""Reproduce leak-free moneyline model inputs for historical evaluation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

from src.sim.team_strength import (
    StrengthConfig,
    build_feature_frame,
    load_completed_games,
)

MLFLOW_HTTP_URI = "http://10.0.0.171:5001"
CHAMPION_MODEL = "mlb-team-strength-win"
_STRENGTH_CONFIG_FIELDS = (
    "initial_elo",
    "elo_k",
    "elo_home_advantage",
    "elo_season_regression",
    "initial_runs_per_game",
    "run_alpha",
    "run_season_regression",
    "starter_prior_ip",
    "starter_season_decay",
)

__all__ = ["walkforward_home_probs"]


def walkforward_home_probs(
    test_season: int, train_seasons: Sequence[int]
) -> pd.DataFrame:
    """Return leak-free home-win probabilities from a pre-test-season refit.

    The champion artifact supplies the feature and strength-state contract. The
    logistic model is refit only on ``train_seasons`` and then evaluated on
    ``test_season``.
    """
    test_season = int(test_season)
    train_seasons = tuple(int(season) for season in train_seasons)
    if test_season in train_seasons:
        raise ValueError("test season must not be in train seasons")

    client = MlflowClient(tracking_uri=MLFLOW_HTTP_URI)
    version = client.get_model_version_by_alias(CHAMPION_MODEL, "champion")
    contract_path = download_artifacts(
        run_id=version.run_id,
        artifact_path="model_contract.json",
        tracking_uri=MLFLOW_HTTP_URI,
    )
    contract = json.loads(Path(contract_path).read_text())
    features = list(contract["features"])
    strength = contract["strength_config"]
    config = StrengthConfig(
        **{field: strength[field] for field in _STRENGTH_CONFIG_FIELDS}
    )

    games = load_completed_games(start_season=2015, end_season=test_season)
    frame, _ = build_feature_frame(games, config)
    train = frame[frame["season"].isin(train_seasons)]
    test = frame[frame["season"] == test_season]

    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(train[features], train["home_won"])
    probabilities = model.predict_proba(test[features])[:, 1]
    return pd.DataFrame(
        {
            "game_pk": test["game_pk"].to_numpy(),
            "model_prob_home": probabilities,
        }
    )
