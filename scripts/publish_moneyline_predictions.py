#!/usr/bin/env python3
"""Publish model-only MLB moneyline predictions as a versioned JSON artifact."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_contracts.moneyline_predictions import (
    MoneylineGamePrediction,
    MoneylinePredictionBatch,
    write_prediction_batch,
)
from src.ml.mlflow_utils import DEFAULT_MLFLOW_TRACKING_URI
from src.sim.slate import SlateGame, active_roster_ids, fetch_slate_games
from src.sim.team_strength import (
    DEFAULT_REGISTERED_STRENGTH_MODEL,
    TeamStrengthPredictor,
    build_live_strength_predictor,
)

STATS_API = "https://statsapi.mlb.com/api/v1"
TBD_PITCHER = "TBD (league-average arm)"


class PredictorBuilder(Protocol):
    def __call__(
        self,
        prediction_date: date,
        *,
        tracking_uri: str | None,
        registered_model_name: str,
    ) -> TeamStrengthPredictor: ...


TeamLabels = Mapping[int, tuple[str, str]]




def resolve_registered_model_version(
    tracking_uri: str | None,
    model_name: str,
) -> str:
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    version = client.get_model_version_by_alias(model_name, "champion")
    return f"v{version.version}"
def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def _fetch_team_labels(season: int) -> dict[int, tuple[str, str]]:
    response = requests.get(
        f"{STATS_API}/teams",
        params={"sportId": 1, "season": season},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    teams = payload.get("teams") if isinstance(payload, Mapping) else None
    if not isinstance(teams, list):
        raise ValueError("MLB teams response is missing teams")
    labels: dict[int, tuple[str, str]] = {}
    for item in teams:
        if not isinstance(item, Mapping):
            continue
        team_id = item.get("id")
        name = str(item.get("name") or "").strip()
        abbreviation = str(item.get("abbreviation") or "").strip()
        if team_id is None or not name or not abbreviation:
            continue
        labels[int(team_id)] = (name, abbreviation)
    return labels


def _active_rosters(
    game: SlateGame,
    *,
    enabled: bool,
    roster_loader: Callable[[int, str], tuple[tuple[int, ...], tuple[int, ...]]],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if not enabled:
        return (), (), (), ()
    away_batters, away_pitchers = roster_loader(game.away_team_id, game.slate_date)
    home_batters, home_pitchers = roster_loader(game.home_team_id, game.slate_date)
    return away_batters, home_batters, away_pitchers, home_pitchers


def _predict_home_probability(
    *,
    game: SlateGame,
    predictor: TeamStrengthPredictor,
    target_date: date,
    include_active_rosters: bool,
    roster_loader: Callable[[int, str], tuple[tuple[int, ...], tuple[int, ...]]],
) -> float:
    away_batters, home_batters, away_pitchers, home_pitchers = _active_rosters(
        game,
        enabled=include_active_rosters,
        roster_loader=roster_loader,
    )
    return predictor.predict_home_probability(
        season=target_date.year,
        away_team_id=game.away_team_id,
        home_team_id=game.home_team_id,
        away_starter_id=game.away_probable.player_id or 0,
        home_starter_id=game.home_probable.player_id or 0,
        prediction_date=target_date,
        away_active_batter_ids=away_batters,
        home_active_batter_ids=home_batters,
        away_reliever_ids=away_pitchers,
        home_reliever_ids=home_pitchers,
    )


def _label_for(team_id: int, labels: TeamLabels) -> tuple[str, str]:
    label = labels.get(team_id)
    if label is None:
        raise ValueError(f"Missing team identity for MLB team {team_id}")
    name, abbreviation = label
    if not name.strip() or not abbreviation.strip():
        raise ValueError(f"Incomplete team identity for MLB team {team_id}")
    return name, abbreviation


def build_prediction_batch(
    target_date: date,
    *,
    model_name: str = DEFAULT_REGISTERED_STRENGTH_MODEL,
    tracking_uri: str | None = DEFAULT_MLFLOW_TRACKING_URI,
    all_games: bool = False,
    model_version: str | None = None,
    include_active_rosters: bool = True,
    slate_loader: Callable[..., Sequence[SlateGame]] = fetch_slate_games,
    team_label_loader: Callable[[int], TeamLabels] = _fetch_team_labels,
    predictor_builder: PredictorBuilder = build_live_strength_predictor,
    roster_loader: Callable[
        [int, str], tuple[tuple[int, ...], tuple[int, ...]]
    ] = active_roster_ids,
    clock: Callable[[], datetime] | None = None,
    model_version_resolver: Callable[[str | None, str], str] = (
        resolve_registered_model_version
    ),
) -> MoneylinePredictionBatch:
    """Score a slate without loading odds or applying any betting policy."""
    states = None if all_games else {"Preview"}
    games = tuple(slate_loader(target_date, abstract_states=states))
    if not games:
        raise ValueError(f"No slate games found for {target_date.isoformat()}")

    labels = team_label_loader(target_date.year)
    resolved_model_version = model_version or model_version_resolver(
        tracking_uri,
        model_name,
    )
    predictor = predictor_builder(
        target_date,
        tracking_uri=tracking_uri,
        registered_model_name=model_name,
    )
    predictions: list[MoneylineGamePrediction] = []
    for game in games:
        if not game.game_datetime:
            raise ValueError(f"Game {game.game_pk} is missing game_datetime")
        home_name, home_abbrev = _label_for(game.home_team_id, labels)
        away_name, away_abbrev = _label_for(game.away_team_id, labels)
        predictions.append(
            MoneylineGamePrediction(
                game_pk=game.game_pk,
                game_time=game.game_datetime,
                home_team_id=game.home_team_id,
                home_team_name=home_name,
                home_team_abbrev=home_abbrev,
                away_team_id=game.away_team_id,
                away_team_name=away_name,
                away_team_abbrev=away_abbrev,
                home_probable_pitcher_id=game.home_probable.player_id,
                home_probable_pitcher_name=(
                    game.home_probable.full_name or TBD_PITCHER
                ),
                away_probable_pitcher_id=game.away_probable.player_id,
                away_probable_pitcher_name=(
                    game.away_probable.full_name or TBD_PITCHER
                ),
                model_prob_home=_predict_home_probability(
                    game=game,
                    predictor=predictor,
                    target_date=target_date,
                    include_active_rosters=include_active_rosters,
                    roster_loader=roster_loader,
                ),
            )
        )
    predicted_at = (clock or (lambda: datetime.now(UTC)))()
    if predicted_at.tzinfo is None:
        raise ValueError("prediction clock must return a timezone-aware datetime")
    return MoneylinePredictionBatch(
        prediction_date=target_date.isoformat(),
        predicted_at=predicted_at.astimezone(UTC).isoformat(),
        model_name=model_name,
        model_version=resolved_model_version,
        games=tuple(predictions),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=_parse_date, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--all-games", action="store_true")
    parser.add_argument("--skip-active-rosters", action="store_true")
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=DEFAULT_MLFLOW_TRACKING_URI,
        help="Tracking server used to resolve the registered model.",
    )
    parser.add_argument(
        "--win-model-name",
        default=DEFAULT_REGISTERED_STRENGTH_MODEL,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    batch = build_prediction_batch(
        args.date,
        model_name=args.win_model_name,
        tracking_uri=args.mlflow_tracking_uri,
        all_games=args.all_games,
        include_active_rosters=not args.skip_active_rosters,
    )
    write_prediction_batch(args.output_json, batch)
    print(
        f"published {len(batch.games)} moneyline predictions for "
        f"{batch.prediction_date} to {args.output_json}"
    )


if __name__ == "__main__":
    main()
