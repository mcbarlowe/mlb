"""Versioned JSON contract for model-only MLB moneyline predictions."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

CONTRACT_VERSION = "v1"


def _required_text(value: object, field: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _positive_int(value: object, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def _optional_positive_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field)


def _utc_timestamp(value: object, field: str) -> str:
    text = _required_text(value, field)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(UTC).isoformat()


@dataclass(frozen=True)
class MoneylineGamePrediction:
    game_pk: int
    game_time: str
    home_team_id: int
    home_team_name: str
    home_team_abbrev: str
    away_team_id: int
    away_team_name: str
    away_team_abbrev: str
    home_probable_pitcher_id: int | None
    home_probable_pitcher_name: str
    away_probable_pitcher_id: int | None
    away_probable_pitcher_name: str
    model_prob_home: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "game_pk", _positive_int(self.game_pk, "game_pk")
        )
        object.__setattr__(
            self, "game_time", _utc_timestamp(self.game_time, "game_time")
        )
        for side in ("home", "away"):
            object.__setattr__(
                self,
                f"{side}_team_id",
                _positive_int(getattr(self, f"{side}_team_id"), f"{side}_team_id"),
            )
            for suffix in ("team_name", "team_abbrev", "probable_pitcher_name"):
                field = f"{side}_{suffix}"
                object.__setattr__(
                    self,
                    field,
                    _required_text(getattr(self, field), field),
                )
            pitcher_field = f"{side}_probable_pitcher_id"
            object.__setattr__(
                self,
                pitcher_field,
                _optional_positive_int(getattr(self, pitcher_field), pitcher_field),
            )
        try:
            probability = float(self.model_prob_home)
        except (TypeError, ValueError) as exc:
            raise ValueError("model_prob_home must be numeric") from exc
        if not 0.0 <= probability <= 1.0:
            raise ValueError("model_prob_home must be between 0 and 1")
        object.__setattr__(self, "model_prob_home", probability)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> MoneylineGamePrediction:
        raw: Any = payload
        return cls(
            game_pk=raw.get("game_pk"),
            game_time=raw.get("game_time"),
            home_team_id=raw.get("home_team_id"),
            home_team_name=raw.get("home_team_name"),
            home_team_abbrev=raw.get("home_team_abbrev"),
            away_team_id=raw.get("away_team_id"),
            away_team_name=raw.get("away_team_name"),
            away_team_abbrev=raw.get("away_team_abbrev"),
            home_probable_pitcher_id=raw.get("home_probable_pitcher_id"),
            home_probable_pitcher_name=raw.get("home_probable_pitcher_name"),
            away_probable_pitcher_id=raw.get("away_probable_pitcher_id"),
            away_probable_pitcher_name=raw.get("away_probable_pitcher_name"),
            model_prob_home=raw.get("model_prob_home"),
        )


@dataclass(frozen=True)
class MoneylinePredictionBatch:
    prediction_date: str
    predicted_at: str
    model_name: str
    games: tuple[MoneylineGamePrediction, ...]
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError(
                f"Unsupported contract_version {self.contract_version!r}; "
                f"expected {CONTRACT_VERSION!r}"
            )
        try:
            parsed_date = date.fromisoformat(str(self.prediction_date))
        except ValueError as exc:
            raise ValueError("prediction_date must be an ISO date") from exc
        object.__setattr__(self, "prediction_date", parsed_date.isoformat())
        predicted_at = _utc_timestamp(self.predicted_at, "predicted_at")
        object.__setattr__(self, "predicted_at", predicted_at)
        object.__setattr__(
            self,
            "model_name",
            _required_text(self.model_name, "model_name"),
        )
        games = tuple(self.games)
        game_pks = [game.game_pk for game in games]
        if len(game_pks) != len(set(game_pks)):
            raise ValueError("games contains duplicate game_pk values")
        prediction_time = datetime.fromisoformat(predicted_at)
        if any(
            prediction_time >= datetime.fromisoformat(game.game_time)
            for game in games
        ):
            raise ValueError("predictions must be produced before every game_time")
        object.__setattr__(self, "games", games)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> MoneylinePredictionBatch:
        raw_games = payload.get("games")
        if not isinstance(raw_games, Sequence) or isinstance(raw_games, (str, bytes)):
            raise ValueError("games must be a list")
        games: list[MoneylineGamePrediction] = []
        for index, item in enumerate(raw_games):
            if not isinstance(item, Mapping):
                raise ValueError(f"games[{index}] must be an object")
            games.append(MoneylineGamePrediction.from_dict(item))
        return cls(
            contract_version=str(payload.get("contract_version") or ""),
            prediction_date=str(payload.get("prediction_date") or ""),
            predicted_at=str(payload.get("predicted_at") or ""),
            model_name=str(payload.get("model_name") or ""),
            games=tuple(games),
        )


def write_prediction_batch(path: Path, batch: MoneylinePredictionBatch) -> None:
    """Atomically replace ``path`` with a fully serialized prediction batch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(batch.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def read_prediction_batch(path: Path) -> MoneylinePredictionBatch:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("moneyline prediction document must be an object")
    return MoneylinePredictionBatch.from_dict(payload)
