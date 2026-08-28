from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from src.data_contracts.prop_predictions import (
    MODEL_NAME,
    MODEL_VERSION,
    PlayerLines,
    build_prop_prediction_artifact,
    validate_request,
)


def _stats(*, hits: int = 0, home_runs: int = 0) -> dict[str, int]:
    return {
        "hits": hits,
        "doubles": 0,
        "triples": 0,
        "homeruns": home_runs,
        "totalbases": hits + 3 * home_runs,
        "rbi": 0,
        "runs": 0,
        "baseonballs": 0,
        "stolenbases": 0,
        "strikeouts": 0,
    }


def test_contract_scores_over_once_without_betting_inputs() -> None:
    requests = {
        "contract_version": "v1",
        "requests": [
            {
                "request_id": "event-1|batter_home_runs|Jane Doe|0.5",
                "event_id": "event-1",
                "game_time": "2026-08-27T23:10:00Z",
                "home_team": "Boston Red Sox",
                "away_team": "New York Yankees",
                "home_team_id": 111,
                "away_team_id": 147,
                "player": "Jane Doe",
                "market": "batter_home_runs",
                "point": 0.5,
            }
        ],
    }
    history = PlayerLines(
        lines=[
            (2026, 28.0, 4, _stats(home_runs=index % 2))
            for index in range(200)
        ],
        age_now=28.5,
    )

    artifact = build_prop_prediction_artifact(
        requests,
        prediction_date=date(2026, 8, 27),
        predicted_at=datetime(2026, 8, 27, 16, tzinfo=UTC),
        lines_by_name={"jane doe": history},
        aging_curves={},
        park_factors={},
        team_ids={},
        game_ids={"event-1|batter_home_runs|Jane Doe|0.5": 123456},
        recency_half_life=0,
    )

    assert artifact["contract_version"] == "v1"
    assert artifact["prediction_date"] == "2026-08-27"
    assert artifact["model_name"] == MODEL_NAME
    assert artifact["model_version"] == MODEL_VERSION
    prediction = artifact["predictions"][0]
    assert prediction["probability_over"] == pytest.approx(0.5)
    assert prediction["sample_games"] == 200
    assert prediction["predicted_at"] == "2026-08-27T16:00:00Z"
    assert prediction["game_pk"] == 123456
    assert not ({"book", "price", "odds", "ev", "side"} & prediction.keys())


def test_conditioned_market_excludes_bench_appearances_from_sample() -> None:
    history = PlayerLines(
        lines=[
            (2026, 27.0, 1, _stats(hits=1)),
            (2026, 27.0, 4, _stats(hits=1)),
            (2026, 27.0, 4, _stats(hits=0)),
        ],
        age_now=27.0,
    )
    artifact = build_prop_prediction_artifact(
        {
            "requests": [
                {
                    "player": "Jane Doe",
                    "market": "batter_hits",
                    "point": 0.5,
                    "home_team_id": 111,
                }
            ]
        },
        prediction_date=date(2026, 8, 27),
        lines_by_name={"jane doe": history},
        aging_curves={},
        park_factors={},
        team_ids={},
        recency_half_life=0,
    )

    assert artifact["predictions"][0]["sample_games"] == 2


@pytest.mark.parametrize("forbidden", ["book", "price", "decimal_odds", "ev", "side"])
def test_request_rejects_betting_owned_fields(forbidden: str) -> None:
    request = {
        "player": "Jane Doe",
        "market": "batter_hits",
        "point": 0.5,
        forbidden: 1,
    }

    with pytest.raises(ValueError, match="non-model fields"):
        validate_request({"requests": [request]})
