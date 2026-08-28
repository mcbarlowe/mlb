"""Reproduce registered MLB model probabilities for predictive evaluation."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

MLFLOW_HTTP_URI = "http://10.0.0.171:5001"
CHAMPION_MODEL = "mlb-team-strength-win"


def load_finals(seasons: Sequence[int]) -> pd.DataFrame:
    """Return regular-season final outcomes from the model's game universe."""
    from src.sim.team_strength import load_completed_games

    wanted = {int(season) for season in seasons}
    games = load_completed_games(
        start_season=min(wanted),
        end_season=max(wanted),
        include_rosters=False,
    )
    rows = [
        {
            "game_pk": game.game_pk,
            "game_date": game.game_datetime[:10],
            "season": game.season,
            "away_team_id": game.away_team_id,
            "home_team_id": game.home_team_id,
            "home_won": game.home_won,
        }
        for game in games
        if game.season in wanted and game.home_runs != game.away_runs
    ]
    return pd.DataFrame(rows)


def champion_home_probs(seasons: Sequence[int]) -> pd.DataFrame:
    """Reproduce champion team-strength home-win probabilities."""
    from mlflow.artifacts import download_artifacts
    from mlflow.tracking import MlflowClient

    from src.sim.team_strength import (
        LEGACY_FEATURE_NAMES,
        StrengthConfig,
        build_feature_frame,
        load_completed_games,
    )

    client = MlflowClient(tracking_uri=MLFLOW_HTTP_URI)
    version = client.get_model_version_by_alias(CHAMPION_MODEL, "champion")
    contract_path = download_artifacts(
        run_id=version.run_id,
        artifact_path="model_contract.json",
        tracking_uri=MLFLOW_HTTP_URI,
    )
    contract = json.loads(Path(contract_path).read_text())
    features = tuple(contract["features"])
    coefficients = contract["coefficients"]
    intercept = float(contract["intercept"])
    strength = contract["strength_config"]
    config = StrengthConfig(
        initial_elo=strength["initial_elo"],
        elo_k=strength["elo_k"],
        elo_home_advantage=strength["elo_home_advantage"],
        elo_season_regression=strength["elo_season_regression"],
        initial_runs_per_game=strength["initial_runs_per_game"],
        run_alpha=strength["run_alpha"],
        run_season_regression=strength["run_season_regression"],
        starter_prior_ip=strength["starter_prior_ip"],
        starter_season_decay=strength["starter_season_decay"],
    )
    if set(features) - set(LEGACY_FEATURE_NAMES) and features != LEGACY_FEATURE_NAMES:
        pass
    start_season = int(contract["training"]["start_season"])
    games = load_completed_games(
        start_season=start_season,
        end_season=max(seasons),
    )
    frame, _ = build_feature_frame(games, config)
    log_odds = intercept + sum(
        float(coefficients[name]) * frame[name] for name in features
    )
    frame = frame.assign(model_prob_home=1.0 / (1.0 + (-log_odds).map(math.exp)))
    keep = frame[frame["season"].isin([int(season) for season in seasons])]
    return keep[["game_pk", "model_prob_home"]].reset_index(drop=True)
