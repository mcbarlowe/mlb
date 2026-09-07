"""Versioned, price-free posterior predictions for starting-pitcher props."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, date, datetime

from mlb.pitcher_props.model import (
    MODEL_CONTRACT_VERSION,
    MODEL_FAMILY,
    SUPPORTED_MARKETS,
    PitcherPropModel,
)

CONTRACT_VERSION = "v1"
REQUEST_FIELDS = frozenset(
    {
        "request_id",
        "event_id",
        "game_pk",
        "game_time",
        "pitcher",
        "pitcher_id",
        "opponent_team_id",
        "market",
        "point",
        "is_home",
        "rest_days",
        "seed",
    }
)
BETTING_FIELDS = frozenset(
    {"book", "bookmaker", "price", "decimal_odds", "ev", "side", "stake"}
)


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("predicted_at must include a timezone")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{label} must be numeric")
    return int(value)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{label} must be numeric")
    return float(value)




def validate_request(payload: Mapping[str, object]) -> list[dict[str, object]]:
    version = payload.get("contract_version", CONTRACT_VERSION)
    if version != CONTRACT_VERSION:
        raise ValueError(f"unsupported pitcher prop prediction contract {version!r}")
    raw_requests = payload.get("requests")
    if not isinstance(raw_requests, list):
        raise ValueError("pitcher prop prediction request must contain a requests list")
    requests: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_requests):
        if not isinstance(raw, Mapping):
            raise ValueError(f"requests[{index}] must be an object")
        unknown = set(raw) - REQUEST_FIELDS
        if unknown:
            raise ValueError(
                f"requests[{index}] contains non-model fields: {sorted(unknown)!r}"
            )
        forbidden = set(raw) & BETTING_FIELDS
        if forbidden:
            raise ValueError(
                f"requests[{index}] contains betting-owned fields: {sorted(forbidden)!r}"
            )
        request = dict(raw)
        for field in ("pitcher_id", "market", "point"):
            if request.get(field) in (None, ""):
                raise ValueError(f"requests[{index}] is missing {field!r}")
        pitcher_id = _integer(request["pitcher_id"], f"requests[{index}].pitcher_id")
        point = _number(request["point"], f"requests[{index}].point")
        market = str(request["market"])
        if pitcher_id <= 0:
            raise ValueError(f"requests[{index}].pitcher_id must be positive")
        if market not in SUPPORTED_MARKETS:
            raise ValueError(f"unsupported pitcher prop market {market!r}")
        if not math.isfinite(point) or point < 0.0:
            raise ValueError(f"requests[{index}].point must be non-negative and finite")
        request_id = str(request.get("request_id") or index)
        if request_id in seen:
            raise ValueError(f"duplicate request_id {request_id!r}")
        seen.add(request_id)
        request["request_id"] = request_id
        request["pitcher_id"] = pitcher_id
        request["market"] = market
        request["point"] = point
        if request.get("opponent_team_id") not in (None, ""):
            request["opponent_team_id"] = _integer(
                request["opponent_team_id"],
                f"requests[{index}].opponent_team_id",
            )
        if request.get("rest_days") not in (None, ""):
            request["rest_days"] = _number(
                request["rest_days"],
                f"requests[{index}].rest_days",
            )
        requests.append(request)
    return requests


def build_pitcher_prop_prediction_artifact(
    model: PitcherPropModel,
    payload: Mapping[str, object],
    *,
    prediction_date: date,
    predicted_at: datetime | None = None,
) -> dict[str, object]:
    """Score a price-free request batch with one fitted posterior artifact."""

    timestamp = (predicted_at or datetime.now(UTC)).astimezone(UTC)
    requests = validate_request(payload)
    predictions: list[dict[str, object]] = []
    for request in requests:
        prediction = model.predict(
            pitcher_id=_integer(request["pitcher_id"], "pitcher_id"),
            opponent_team_id=(
                _integer(request["opponent_team_id"], "opponent_team_id")
                if request.get("opponent_team_id") not in (None, "")
                else None
            ),
            market=str(request["market"]),
            point=_number(request["point"], "point"),
            as_of=timestamp,
            is_home=bool(request.get("is_home", False)),
            rest_days=(
                _number(request["rest_days"], "rest_days")
                if request.get("rest_days") not in (None, "")
                else None
            ),
            seed=(
                _integer(request["seed"], "seed")
                if request.get("seed") not in (None, "")
                else None
            ),
        )
        row = {
            key: request[key]
            for key in (
                "request_id",
                "event_id",
                "game_pk",
                "game_time",
                "pitcher",
                "pitcher_id",
                "opponent_team_id",
                "market",
                "point",
            )
            if key in request
        }
        row.update(prediction.to_dict())
        row["predicted_at"] = _utc_text(timestamp)
        predictions.append(row)
    return {
        "contract_version": CONTRACT_VERSION,
        "model_family": MODEL_FAMILY,
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "model_fingerprint": model.fingerprint(),
        "prediction_date": prediction_date.isoformat(),
        "predicted_at": _utc_text(timestamp),
        "predictions": predictions,
    }


__all__ = [
    "CONTRACT_VERSION",
    "build_pitcher_prop_prediction_artifact",
    "validate_request",
]
