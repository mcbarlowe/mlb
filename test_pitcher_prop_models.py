from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from mlb.data_contracts.pitcher_prop_predictions import (
    build_pitcher_prop_prediction_artifact,
    validate_request,
)
from mlb.pitcher_props.model import (
    BATTERS_FACED_MARKET,
    HITS_ALLOWED_MARKET,
    OUTS_MARKET,
    STRIKEOUT_MARKET,
    WALKS_MARKET,
    PitcherPropConfig,
    PitcherPropModel,
    PitcherStart,
)


def _start(
    index: int,
    *,
    pitcher_id: int = 10,
    opponent_team_id: int = 100,
    season: int = 2023,
    outs: int = 18,
    batters_faced: int = 24,
    strikeouts: int = 6,
    hits: int = 5,
    walks: int = 2,
) -> PitcherStart:
    return PitcherStart(
        game_pk=700_000 + index,
        game_time=datetime(season, 4, 1, 17, tzinfo=UTC) + timedelta(days=index),
        season=season,
        pitcher_id=pitcher_id,
        opponent_team_id=opponent_team_id,
        is_home=index % 2 == 0,
        outs=outs,
        batters_faced=batters_faced,
        strikeouts=strikeouts,
        hits_allowed=hits,
        walks=walks,
        pitch_count=90,
        rest_days=5.0,
    )


def _config() -> PitcherPropConfig:
    return PitcherPropConfig(
        min_draws=2_000,
        max_draws=2_000,
        draw_batch=1_000,
        mc_tolerance=0.001,
        seed=9,
    )


def _model() -> PitcherPropModel:
    starts = [
        _start(
            index,
            pitcher_id=10 + index % 4,
            opponent_team_id=100 + index % 2,
            outs=15 + index % 7,
            batters_faced=21 + index % 8,
            strikeouts=(9 if index % 2 == 0 else 3),
            hits=(4 if index % 2 == 0 else 8),
            walks=(1 if index % 2 == 0 else 4),
        )
        for index in range(80)
    ]
    return PitcherPropModel.fit(starts, config=_config())


def test_posterior_predictions_are_normalized_monotone_and_reproducible() -> None:
    model = _model()
    as_of = datetime(2024, 4, 1, tzinfo=UTC)
    low_line = model.predict(
        pitcher_id=10,
        opponent_team_id=100,
        market=STRIKEOUT_MARKET,
        point=4.5,
        as_of=as_of,
        rest_days=5.0,
        seed=123,
    )
    repeated = model.predict(
        pitcher_id=10,
        opponent_team_id=100,
        market=STRIKEOUT_MARKET,
        point=4.5,
        as_of=as_of,
        rest_days=5.0,
        seed=123,
    )
    high_line = model.predict(
        pitcher_id=10,
        opponent_team_id=100,
        market=STRIKEOUT_MARKET,
        point=7.5,
        as_of=as_of,
        rest_days=5.0,
        seed=123,
    )

    assert low_line == repeated
    assert low_line.probability_over >= high_line.probability_over
    assert (
        low_line.probability_over
        + low_line.probability_push
        + low_line.probability_under
    ) == pytest.approx(1.0)
    assert low_line.samples == 2_000


def test_shared_opportunity_draws_respect_count_bounds() -> None:
    model = _model()
    draws = model.simulate_components(
        pitcher_id=10,
        opponent_team_id=100,
        as_of=datetime(2024, 4, 1, tzinfo=UTC),
        samples=5_000,
        seed=123,
    )

    assert np.all(draws[BATTERS_FACED_MARKET] >= draws[OUTS_MARKET])
    assert np.all(draws[STRIKEOUT_MARKET] <= draws[BATTERS_FACED_MARKET])
    assert np.all(draws[HITS_ALLOWED_MARKET] <= draws[BATTERS_FACED_MARKET])
    assert np.all(draws[WALKS_MARKET] <= draws[BATTERS_FACED_MARKET])


def test_more_pitcher_history_increases_rate_posterior_concentration() -> None:
    starts = [_start(index, strikeouts=6) for index in range(30)]
    small = PitcherPropModel.fit(starts[:3], config=_config())
    large = PitcherPropModel.fit(starts, config=_config())
    as_of = datetime(2024, 4, 1, tzinfo=UTC)

    small_alpha, small_beta = small.rate_posterior(
        market=STRIKEOUT_MARKET,
        pitcher_id=10,
        opponent_team_id=None,
        as_of=as_of,
    )
    large_alpha, large_beta = large.rate_posterior(
        market=STRIKEOUT_MARKET,
        pitcher_id=10,
        opponent_team_id=None,
        as_of=as_of,
    )

    assert large_alpha + large_beta > small_alpha + small_beta


def test_high_strikeout_opponent_raises_same_pitcher_rate() -> None:
    model = _model()
    as_of = datetime(2024, 4, 1, tzinfo=UTC)
    high_alpha, high_beta = model.rate_posterior(
        market=STRIKEOUT_MARKET,
        pitcher_id=999,
        opponent_team_id=100,
        as_of=as_of,
    )
    low_alpha, low_beta = model.rate_posterior(
        market=STRIKEOUT_MARKET,
        pitcher_id=999,
        opponent_team_id=101,
        as_of=as_of,
    )

    assert high_alpha / (high_alpha + high_beta) > low_alpha / (low_alpha + low_beta)


def test_artifact_round_trip_and_price_free_prediction_contract(tmp_path) -> None:
    model = _model()
    path = model.save(tmp_path / "pitcher-props.json")
    loaded = PitcherPropModel.load(path)
    request = {
        "contract_version": "v1",
        "requests": [
            {
                "request_id": "event-1|pitcher_strikeouts|10|5.5",
                "event_id": "event-1",
                "game_pk": 800_001,
                "game_time": "2024-04-02T17:00:00Z",
                "pitcher": "Test Pitcher",
                "pitcher_id": 10,
                "opponent_team_id": 100,
                "market": STRIKEOUT_MARKET,
                "point": 5.5,
                "is_home": True,
                "rest_days": 5.0,
                "seed": 77,
            }
        ],
    }

    artifact = build_pitcher_prop_prediction_artifact(
        loaded,
        request,
        prediction_date=date(2024, 4, 2),
        predicted_at=datetime(2024, 4, 2, 12, tzinfo=UTC),
    )

    assert loaded.fingerprint() == model.fingerprint()
    assert artifact["model_contract_version"] == "pitcher-prop-bayes-v1"
    predictions = artifact["predictions"]
    assert isinstance(predictions, list)
    prediction = predictions[0]
    assert isinstance(prediction, dict)
    assert prediction["probability_over"] + prediction["probability_under"] == pytest.approx(1.0)
    assert not ({"book", "price", "decimal_odds", "ev", "side", "stake"} & prediction.keys())


def test_request_rejects_betting_owned_fields() -> None:
    with pytest.raises(ValueError, match="non-model fields"):
        validate_request(
            {
                "requests": [
                    {
                        "pitcher_id": 10,
                        "market": STRIKEOUT_MARKET,
                        "point": 5.5,
                        "price": -110,
                    }
                ]
            }
        )
