"""Preseason market prior offsets from regular-season win totals.

The input is a static, pre-Opening-Day market expectation. The output is on
the same game-logit scale as the season simulator's other team offsets, so a
home matchup adjustment is ``home_offset - away_offset``.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

__all__ = [
    "MarketWinTotal",
    "load_market_prior_offsets",
    "load_market_win_totals_csv",
    "market_prior_offsets_from_win_totals",
]


class TeamLabel(Protocol):
    abbreviation: str
    team_name: str


@dataclass(frozen=True)
class MarketWinTotal:
    season: int
    team_id: int
    win_total: float
    total_games: int = 162
    source: str = ""

    @property
    def win_probability(self) -> float:
        if self.total_games <= 0:
            raise ValueError("total_games must be positive")
        if not math.isfinite(self.win_total):
            raise ValueError("win_total must be finite")
        if not 0.0 < self.win_total < self.total_games:
            raise ValueError("win_total must be between zero and total_games")
        return self.win_total / self.total_games


def load_market_prior_offsets(
    path: str | Path,
    *,
    prediction_season: int,
    teams: Mapping[int, TeamLabel],
) -> dict[int, float]:
    records = load_market_win_totals_csv(path, teams=teams)
    return market_prior_offsets_from_win_totals(
        records,
        prediction_season=prediction_season,
    )


def load_market_win_totals_csv(
    path: str | Path,
    *,
    teams: Mapping[int, TeamLabel],
) -> tuple[MarketWinTotal, ...]:
    labels = _team_label_lookup(teams)
    records: list[MarketWinTotal] = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=2):
            team_id = _row_team_id(row, labels, index)
            total_games_text = row.get("total_games") or "162"
            records.append(
                MarketWinTotal(
                    season=_required_int(row, "season", index),
                    team_id=team_id,
                    win_total=_required_float(row, "win_total", index),
                    total_games=_parse_int(total_games_text, "total_games", index),
                    source=str(row.get("source") or ""),
                )
            )
    return tuple(records)


def market_prior_offsets_from_win_totals(
    records: Iterable[MarketWinTotal],
    *,
    prediction_season: int,
) -> dict[int, float]:
    offsets: dict[int, float] = {}
    for record in records:
        if record.season != prediction_season:
            continue
        offsets[record.team_id] = _probability_to_logit(record.win_probability)
    return offsets


def _team_label_lookup(teams: Mapping[int, TeamLabel]) -> dict[str, int]:
    labels: dict[str, int] = {}
    for team_id, team in teams.items():
        for label in (team.abbreviation, team.team_name):
            normalized = _normalize_label(label)
            if normalized:
                labels[normalized] = team_id
    return labels


def _row_team_id(row: Mapping[str, str], labels: Mapping[str, int], index: int) -> int:
    team_id_text = row.get("team_id")
    if team_id_text:
        return _parse_int(team_id_text, "team_id", index)
    for field in ("abbreviation", "team_name"):
        value = row.get(field)
        if not value:
            continue
        normalized = _normalize_label(value)
        if normalized in labels:
            return labels[normalized]
    raise ValueError(
        f"Market win totals row {index} must include a known team_id, abbreviation, or team_name"
    )


def _required_int(row: Mapping[str, str], key: str, index: int) -> int:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"Market win totals row {index} missing {key}")
    return _parse_int(value, key, index)


def _parse_int(value: str, key: str, index: int) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Market win totals row {index} has invalid {key}") from exc


def _required_float(row: Mapping[str, str], key: str, index: int) -> float:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"Market win totals row {index} missing {key}")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"Market win totals row {index} has invalid {key}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Market win totals row {index} has invalid {key}")
    return parsed


def _normalize_label(value: str) -> str:
    return "".join(character for character in value.upper() if character.isalnum())


def _probability_to_logit(probability: float) -> float:
    p = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))
