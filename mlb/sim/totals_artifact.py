"""Versioned JSON contract for model-only MLB totals simulations."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CONTRACT_VERSION = "v1"
ARTIFACT_TYPE = "mlb_totals_simulation"

_TOP_LEVEL_KEYS = {
    "contract_version",
    "artifact_type",
    "generated_at",
    "season",
    "sims_per_game",
    "model_name",
    "model_version",
    "games",
}
_GAME_KEYS = {
    "game_pk",
    "total_counts",
    "sim_mean_total",
    "sim_total_stdev",
}


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _strict_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _positive_int(value: object, field: str) -> int:
    parsed = _strict_int(value, field)
    if parsed <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def _nonnegative_int(value: object, field: str) -> int:
    parsed = _strict_int(value, field)
    if parsed < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return parsed


def _finite_nonnegative_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{field} must be a finite non-negative float")
    return parsed


def _timestamp(value: object, field: str) -> str:
    text = _required_text(value, field)
    parseable = f"{text[:-1]}+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(parseable)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(UTC).isoformat()


def _reject_unknown_keys(
    payload: Mapping[str, object], allowed: set[str], context: str
) -> None:
    missing = allowed - set(payload)
    if missing:
        raise ValueError(f"{context} is missing required fields {sorted(missing)!r}")
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"{context} contains unknown fields {sorted(unknown)!r}")


def _total_counts(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError("total_counts must be an object")
    if not value:
        raise ValueError("total_counts must not be empty")

    counts: dict[str, int] = {}
    for raw_total, raw_count in value.items():
        if not isinstance(raw_total, str) or not raw_total.isdecimal():
            raise ValueError(
                "total_counts keys must be non-negative integer strings"
            )
        total = int(raw_total)
        if raw_total != str(total):
            raise ValueError(
                "total_counts keys must be canonical non-negative integer strings"
            )
        count = _positive_int(raw_count, "total_counts values")
        counts[str(total)] = count
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _stats_from_counts(counts: Mapping[str, int]) -> tuple[float, float]:
    n = sum(counts.values())
    if n <= 0:
        raise ValueError("total_counts must contain at least one simulation")
    total_sum = sum(int(total) * count for total, count in counts.items())
    mean = total_sum / n
    second_moment = sum((int(total) ** 2) * count for total, count in counts.items()) / n
    variance = max(0.0, second_moment - mean**2)
    return mean, math.sqrt(variance)


@dataclass(frozen=True)
class TotalsSimulationGame:
    game_pk: int
    total_counts: Mapping[str, int]
    sim_mean_total: float
    sim_total_stdev: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "game_pk", _positive_int(self.game_pk, "game_pk"))
        counts = _total_counts(self.total_counts)
        mean = _finite_nonnegative_float(self.sim_mean_total, "sim_mean_total")
        stdev = _finite_nonnegative_float(
            self.sim_total_stdev,
            "sim_total_stdev",
        )
        expected_mean, expected_stdev = _stats_from_counts(counts)
        if not math.isclose(mean, expected_mean, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("sim_mean_total must match total_counts")
        if not math.isclose(stdev, expected_stdev, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("sim_total_stdev must match total_counts")
        object.__setattr__(self, "total_counts", counts)
        object.__setattr__(self, "sim_mean_total", mean)
        object.__setattr__(self, "sim_total_stdev", stdev)

    @property
    def simulation_count(self) -> int:
        return sum(self.total_counts.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "game_pk": self.game_pk,
            "total_counts": dict(self.total_counts),
            "sim_mean_total": self.sim_mean_total,
            "sim_total_stdev": self.sim_total_stdev,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TotalsSimulationGame:
        _reject_unknown_keys(payload, _GAME_KEYS, "games[]")
        return cls(
            game_pk=payload["game_pk"],
            total_counts=payload["total_counts"],
            sim_mean_total=payload["sim_mean_total"],
            sim_total_stdev=payload["sim_total_stdev"],
        )


@dataclass(frozen=True)
class TotalsSimulationArtifact:
    generated_at: str
    season: int
    sims_per_game: int
    model_name: str
    model_version: str
    games: Sequence[TotalsSimulationGame]
    contract_version: str = CONTRACT_VERSION
    artifact_type: str = ARTIFACT_TYPE

    def __post_init__(self) -> None:
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError(
                f"Unsupported contract_version {self.contract_version!r}; "
                f"expected {CONTRACT_VERSION!r}"
            )
        if self.artifact_type != ARTIFACT_TYPE:
            raise ValueError(
                f"Unsupported artifact_type {self.artifact_type!r}; "
                f"expected {ARTIFACT_TYPE!r}"
            )
        object.__setattr__(self, "generated_at", _timestamp(self.generated_at, "generated_at"))
        object.__setattr__(self, "season", _positive_int(self.season, "season"))
        sims_per_game = _positive_int(self.sims_per_game, "sims_per_game")
        object.__setattr__(self, "sims_per_game", sims_per_game)
        object.__setattr__(self, "model_name", _required_text(self.model_name, "model_name"))
        object.__setattr__(self, "model_version", _required_text(self.model_version, "model_version"))
        if isinstance(self.games, (str, bytes)) or not isinstance(self.games, Sequence):
            raise ValueError("games must be a list")
        games = tuple(self.games)
        for index, game in enumerate(games):
            if not isinstance(game, TotalsSimulationGame):
                raise ValueError(f"games[{index}] must be a TotalsSimulationGame")
            if game.simulation_count != sims_per_game:
                raise ValueError(
                    f"games[{index}] total_counts sum must equal sims_per_game"
                )
        game_pks = [game.game_pk for game in games]
        if len(game_pks) != len(set(game_pks)):
            raise ValueError("games contains duplicate game_pk values")
        object.__setattr__(self, "games", games)

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "artifact_type": self.artifact_type,
            "generated_at": self.generated_at,
            "season": self.season,
            "sims_per_game": self.sims_per_game,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "games": [game.to_dict() for game in self.games],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TotalsSimulationArtifact:
        _reject_unknown_keys(payload, _TOP_LEVEL_KEYS, "totals simulation artifact")
        raw_games = payload["games"]
        if not isinstance(raw_games, Sequence) or isinstance(raw_games, (str, bytes)):
            raise ValueError("games must be a list")
        games: list[TotalsSimulationGame] = []
        for index, raw_game in enumerate(raw_games):
            if not isinstance(raw_game, Mapping):
                raise ValueError(f"games[{index}] must be an object")
            games.append(TotalsSimulationGame.from_dict(raw_game))
        return cls(
            contract_version=payload["contract_version"],
            artifact_type=payload["artifact_type"],
            generated_at=payload["generated_at"],
            season=payload["season"],
            sims_per_game=payload["sims_per_game"],
            model_name=payload["model_name"],
            model_version=payload["model_version"],
            games=tuple(games),
        )


def game_totals_from_simulations(
    game_pk: int, simulated_totals: Iterable[int]
) -> TotalsSimulationGame:
    if isinstance(simulated_totals, (str, bytes)):
        raise ValueError("simulated_totals must be an iterable of integers")

    counts: Counter[str] = Counter()
    n = 0
    total_sum = 0
    total_square_sum = 0
    for raw_total in simulated_totals:
        total = _nonnegative_int(raw_total, "simulated_totals values")
        counts[str(total)] += 1
        n += 1
        total_sum += total
        total_square_sum += total * total
    if n == 0:
        raise ValueError("simulated_totals must not be empty")

    mean = total_sum / n
    variance = max(0.0, total_square_sum / n - mean**2)
    return TotalsSimulationGame(
        game_pk=game_pk,
        total_counts=dict(counts),
        sim_mean_total=mean,
        sim_total_stdev=math.sqrt(variance),
    )


def write_totals_artifact(path: Path, artifact: TotalsSimulationArtifact) -> None:
    """Atomically replace ``path`` with a serialized totals simulation artifact."""
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
            json.dump(artifact.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def read_totals_artifact(path: Path) -> TotalsSimulationArtifact:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("totals simulation artifact must be an object")
    return TotalsSimulationArtifact.from_dict(payload)
