from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from mlb.cli.publish_moneyline_predictions import build_prediction_batch
from mlb.data_contracts.moneyline_predictions import (
    MoneylineGamePrediction,
    MoneylinePredictionBatch,
    read_prediction_batch,
    write_prediction_batch,
)
from mlb.sim.slate import ProbablePitcher, SlateGame

PREDICTED_AT = datetime(2026, 8, 27, 14, 0, tzinfo=UTC)


def game_prediction(*, game_pk: int = 777) -> MoneylineGamePrediction:
    return MoneylineGamePrediction(
        game_pk=game_pk,
        game_time="2026-08-28T02:10:00+00:00",
        home_team_id=147,
        home_team_name="New York Yankees",
        home_team_abbrev="NYY",
        away_team_id=111,
        away_team_name="Boston Red Sox",
        away_team_abbrev="BOS",
        home_probable_pitcher_id=10,
        home_probable_pitcher_name="Home Starter",
        away_probable_pitcher_id=20,
        away_probable_pitcher_name="Away Starter",
        model_prob_home=0.61,
    )


def prediction_batch(
    *,
    games: tuple[MoneylineGamePrediction, ...] | None = None,
) -> MoneylinePredictionBatch:
    return MoneylinePredictionBatch(
        prediction_date="2026-08-27",
        predicted_at=PREDICTED_AT.isoformat(),
        model_name="mlb-team-strength-win",
        model_version="v42",
        games=games if games is not None else (game_prediction(),),
    )


def test_contract_round_trip_and_atomic_replace(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "moneyline.json"
    path.parent.mkdir()
    path.write_text("incomplete", encoding="utf-8")

    write_prediction_batch(path, prediction_batch())

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "contract_version",
        "prediction_date",
        "predicted_at",
        "model_name",
        "model_version",
        "games",
    }
    assert payload["contract_version"] == "v1"
    assert payload["games"][0]["game_pk"] == 777
    assert read_prediction_batch(path) == prediction_batch()
    assert list(path.parent.iterdir()) == [path]


def test_contract_rejects_invalid_probability_duplicate_key_and_version() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        MoneylineGamePrediction(
            **{
                **game_prediction().__dict__,
                "model_prob_home": 1.01,
            }
        )

    duplicated = game_prediction()
    with pytest.raises(ValueError, match="duplicate game_pk"):
        prediction_batch(games=(duplicated, duplicated))

    with pytest.raises(ValueError, match="Unsupported contract_version"):
        MoneylinePredictionBatch(
            prediction_date="2026-08-27",
            predicted_at=PREDICTED_AT.isoformat(),
            model_name="model",
            model_version="v1",
            games=(),
            contract_version="v2",
        )


def test_build_batch_reuses_live_model_inference_without_betting_inputs() -> None:
    slate = SlateGame(
        game_pk=777,
        slate_date="2026-08-27",
        game_datetime="2026-08-28T02:10:00Z",
        status="Preview",
        away_team_id=111,
        home_team_id=147,
        away_abbrev="BOS",
        home_abbrev="NYY",
        venue="Yankee Stadium",
        away_probable=ProbablePitcher(player_id=20, full_name="Away Starter"),
        home_probable=ProbablePitcher(player_id=10, full_name="Home Starter"),
    )
    slate_calls: list[set[str] | None] = []

    def slate_loader(
        target_date: date, *, abstract_states: set[str] | None
    ) -> list[SlateGame]:
        assert target_date == date(2026, 8, 27)
        slate_calls.append(abstract_states)
        return [slate]

    predictor_calls: list[dict[str, object]] = []

    class Predictor:
        def predict_home_probability(self, **kwargs: object) -> float:
            predictor_calls.append(dict(kwargs))
            return 0.61

    def predictor_builder(
        prediction_date: date,
        *,
        tracking_uri: str | None,
        registered_model_name: str,
    ) -> Predictor:
        assert prediction_date == date(2026, 8, 27)
        assert tracking_uri == "file:mlruns"
        assert registered_model_name == "registered-win-model"
        return Predictor()

    roster_calls: list[tuple[int, str]] = []

    def roster_loader(
        team_id: int, slate_date: str
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        roster_calls.append((team_id, slate_date))
        if team_id == 111:
            return (1, 2), (20, 21)
        return (3, 4), (10, 11)

    batch = build_prediction_batch(
        date(2026, 8, 27),
        model_name="registered-win-model",
        tracking_uri="file:mlruns",
        model_version="v42",
        slate_loader=slate_loader,
        team_label_loader=lambda season: {
            111: ("Boston Red Sox", "BOS"),
            147: ("New York Yankees", "NYY"),
        },
        predictor_builder=predictor_builder,
        roster_loader=roster_loader,
        clock=lambda: PREDICTED_AT,
    )

    assert slate_calls == [{"Preview"}]
    assert roster_calls == [(111, "2026-08-27"), (147, "2026-08-27")]
    assert predictor_calls == [
        {
            "season": 2026,
            "away_team_id": 111,
            "home_team_id": 147,
            "away_starter_id": 20,
            "home_starter_id": 10,
            "prediction_date": date(2026, 8, 27),
            "away_active_batter_ids": (1, 2),
            "home_active_batter_ids": (3, 4),
            "away_reliever_ids": (20, 21),
            "home_reliever_ids": (10, 11),
        }
    ]
    assert batch.model_name == "registered-win-model"
    assert batch.model_version == "v42"
    assert batch.games == (game_prediction(),)


def _slate_game(game_pk: int, game_datetime: str) -> SlateGame:
    return SlateGame(
        game_pk=game_pk,
        slate_date="2026-08-27",
        game_datetime=game_datetime,
        status="Preview",
        away_team_id=111,
        home_team_id=147,
        away_abbrev="BOS",
        home_abbrev="NYY",
        venue="Yankee Stadium",
        away_probable=ProbablePitcher(player_id=20, full_name="Away Starter"),
        home_probable=ProbablePitcher(player_id=10, full_name="Home Starter"),
    )


def _build(games: list[SlateGame], *, clock_at: datetime) -> MoneylinePredictionBatch:
    class Predictor:
        def predict_home_probability(self, **kwargs: object) -> float:
            return 0.61

    return build_prediction_batch(
        date(2026, 8, 27),
        model_name="registered-win-model",
        tracking_uri="file:mlruns",
        model_version="v42",
        slate_loader=lambda target_date, *, abstract_states: list(games),
        team_label_loader=lambda season: {
            111: ("Boston Red Sox", "BOS"),
            147: ("New York Yankees", "NYY"),
        },
        predictor_builder=lambda d, *, tracking_uri, registered_model_name: Predictor(),
        roster_loader=lambda team_id, slate_date: ((1, 2), (20, 21)),
        clock=lambda: clock_at,
    )


def test_started_games_are_dropped_instead_of_invalidating_the_slate() -> None:
    # Regression: the MLB API's abstract state lags first pitch, so a game that
    # had already begun was scored and then rejected by the contract, discarding
    # every other game in the batch.
    already_started = _slate_game(1, "2026-08-27T13:35:00Z")
    upcoming = _slate_game(2, "2026-08-28T02:10:00Z")

    batch = _build([already_started, upcoming], clock_at=PREDICTED_AT)

    assert [game.game_pk for game in batch.games] == [2]


def test_a_fully_started_slate_reports_that_plainly() -> None:
    started = _slate_game(1, "2026-08-27T13:35:00Z")

    with pytest.raises(ValueError, match="had already started"):
        _build([started], clock_at=PREDICTED_AT)


def test_predicted_at_precedes_scoring_so_the_contract_holds_by_construction() -> None:
    # The clock must be read before inference, not after: scoring is slow enough
    # that a game could start mid-run.
    reads: list[int] = []

    def clock() -> datetime:
        reads.append(len(reads))
        return PREDICTED_AT

    class Predictor:
        def predict_home_probability(self, **kwargs: object) -> float:
            assert reads, "clock must be read before any game is scored"
            return 0.61

    batch = build_prediction_batch(
        date(2026, 8, 27),
        model_name="registered-win-model",
        tracking_uri="file:mlruns",
        model_version="v42",
        slate_loader=lambda target_date, *, abstract_states: [
            _slate_game(2, "2026-08-28T02:10:00Z")
        ],
        team_label_loader=lambda season: {
            111: ("Boston Red Sox", "BOS"),
            147: ("New York Yankees", "NYY"),
        },
        predictor_builder=lambda d, *, tracking_uri, registered_model_name: Predictor(),
        roster_loader=lambda team_id, slate_date: ((1, 2), (20, 21)),
        clock=clock,
    )

    assert len(reads) == 1
    assert batch.predicted_at == PREDICTED_AT.isoformat()
