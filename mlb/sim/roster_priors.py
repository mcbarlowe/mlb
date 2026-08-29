"""Roster/depth-chart prior offsets for season projections.

The priors are static, preseason roster inputs expressed as per-team offsets on
season simulator game-logit scale. Team labels in CSV inputs intentionally match
``market_priors`` behavior: ``team_id`` is accepted directly, while
``abbreviation`` or ``team_name`` may be resolved from a caller-supplied teams
mapping.

Scoring is deterministic and intentionally conservative:

1. ``win_probability`` wins when supplied.
2. Else ``projected_wins / total_games`` wins when supplied.
3. Else available roster components are centered, weighted, averaged, and mapped
   to ``0.5 + 0.14 * tanh(weighted_score)`` before conversion to logit.

The component centers/scales are rough MLB preseason anchors, not fitted state:
``projected_war`` 32/18, ``returning_pa_share`` 0.75/0.15,
``returning_ip_share`` 0.70/0.15, ``lineup_woba`` .315/.030,
``rotation_fip`` 4.20/.60, and ``bullpen_fip`` 4.20/.60. Lower FIP is better,
so FIP z-scores use ``center - value``.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

__all__ = [
    "RosterPrior",
    "load_roster_prior_offsets",
    "load_roster_priors_csv",
    "roster_prior_offsets_from_rows",
]

_DEFAULT_TOTAL_GAMES = 162

_COMPONENT_WEIGHTS = {
    "projected_war": 0.40,
    "returning_pa_share": 0.10,
    "returning_ip_share": 0.10,
    "lineup_woba": 0.15,
    "rotation_fip": 0.15,
    "bullpen_fip": 0.10,
}


class TeamLabel(Protocol):
    abbreviation: str
    team_name: str


@dataclass(frozen=True)
class RosterPrior:
    """One team's preseason roster prior inputs for a prediction season."""

    season: int
    team_id: int
    win_probability: float | None = None
    projected_wins: float | None = None
    total_games: int = _DEFAULT_TOTAL_GAMES
    projected_war: float | None = None
    returning_pa_share: float | None = None
    returning_ip_share: float | None = None
    lineup_woba: float | None = None
    rotation_fip: float | None = None
    bullpen_fip: float | None = None
    source: str = ""

    @property
    def score_probability(self) -> float:
        """Return the bounded roster prior probability before logit conversion."""

        _validate_roster_prior(self)
        if self.win_probability is not None:
            return self.win_probability
        if self.projected_wins is not None:
            return self.projected_wins / self.total_games
        return _component_probability(self)


def load_roster_prior_offsets(
    path: str | Path,
    prediction_season: int,
    teams: Mapping[int, TeamLabel] | None = None,
) -> dict[int, float]:
    """Load roster priors from CSV and return per-team game-logit offsets."""

    rows = load_roster_priors_csv(path, teams=teams)
    return roster_prior_offsets_from_rows(rows, prediction_season)


def load_roster_priors_csv(
    path: str | Path,
    *,
    teams: Mapping[int, TeamLabel] | None = None,
) -> tuple[RosterPrior, ...]:
    """Load roster priors from CSV without touching databases or model state."""

    labels = _team_label_lookup(teams or {})
    rows: list[RosterPrior] = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=2):
            rows.append(_row_to_roster_prior(row, labels, index))
    return tuple(rows)


def roster_prior_offsets_from_rows(
    rows: Iterable[RosterPrior | Mapping[str, object]],
    prediction_season: int,
) -> dict[int, float]:
    """Convert target-season roster-prior rows to per-team game-logit offsets."""

    offsets: dict[int, float] = {}
    for index, row in enumerate(rows, start=1):
        prior = (
            row
            if isinstance(row, RosterPrior)
            else _row_to_roster_prior(row, labels={}, index=index)
        )
        if prior.season != prediction_season:
            continue
        offsets[prior.team_id] = _probability_to_logit(prior.score_probability)
    return offsets


def _row_to_roster_prior(
    row: Mapping[str, object],
    labels: Mapping[str, int],
    index: int,
) -> RosterPrior:
    total_games = _optional_int(row, "total_games", index)
    return RosterPrior(
        season=_required_int(row, "season", index),
        team_id=_row_team_id(row, labels, index),
        win_probability=_optional_float(row, "win_probability", index),
        projected_wins=_optional_float(row, "projected_wins", index),
        total_games=total_games if total_games is not None else _DEFAULT_TOTAL_GAMES,
        projected_war=_optional_float(row, "projected_war", index),
        returning_pa_share=_optional_float(row, "returning_pa_share", index),
        returning_ip_share=_optional_float(row, "returning_ip_share", index),
        lineup_woba=_optional_float(row, "lineup_woba", index),
        rotation_fip=_optional_float(row, "rotation_fip", index),
        bullpen_fip=_optional_float(row, "bullpen_fip", index),
        source=_text_value(row.get("source")),
    )


def _validate_roster_prior(prior: RosterPrior) -> None:
    if prior.total_games <= 0:
        raise ValueError("total_games must be positive")
    _validate_probability(prior.win_probability, "win_probability")
    if prior.projected_wins is not None:
        _validate_finite(prior.projected_wins, "projected_wins")
        if not 0.0 < prior.projected_wins < prior.total_games:
            raise ValueError("projected_wins must be between zero and total_games")
    _validate_range(prior.projected_war, "projected_war", -20.0, 80.0)
    _validate_range(prior.returning_pa_share, "returning_pa_share", 0.0, 1.0)
    _validate_range(prior.returning_ip_share, "returning_ip_share", 0.0, 1.0)
    _validate_range(prior.lineup_woba, "lineup_woba", 0.200, 0.500)
    _validate_range(prior.rotation_fip, "rotation_fip", 1.0, 8.0)
    _validate_range(prior.bullpen_fip, "bullpen_fip", 1.0, 8.0)
    if not _has_score_field(prior):
        raise ValueError("roster prior must include a supported scoring field")


def _has_score_field(prior: RosterPrior) -> bool:
    return any(
        value is not None
        for value in (
            prior.win_probability,
            prior.projected_wins,
            prior.projected_war,
            prior.returning_pa_share,
            prior.returning_ip_share,
            prior.lineup_woba,
            prior.rotation_fip,
            prior.bullpen_fip,
        )
    )


def _component_probability(prior: RosterPrior) -> float:
    weighted_score = 0.0
    total_weight = 0.0

    if prior.projected_war is not None:
        weight = _COMPONENT_WEIGHTS["projected_war"]
        weighted_score += weight * ((prior.projected_war - 32.0) / 18.0)
        total_weight += weight
    if prior.returning_pa_share is not None:
        weight = _COMPONENT_WEIGHTS["returning_pa_share"]
        weighted_score += weight * ((prior.returning_pa_share - 0.75) / 0.15)
        total_weight += weight
    if prior.returning_ip_share is not None:
        weight = _COMPONENT_WEIGHTS["returning_ip_share"]
        weighted_score += weight * ((prior.returning_ip_share - 0.70) / 0.15)
        total_weight += weight
    if prior.lineup_woba is not None:
        weight = _COMPONENT_WEIGHTS["lineup_woba"]
        weighted_score += weight * ((prior.lineup_woba - 0.315) / 0.030)
        total_weight += weight
    if prior.rotation_fip is not None:
        weight = _COMPONENT_WEIGHTS["rotation_fip"]
        weighted_score += weight * ((4.20 - prior.rotation_fip) / 0.60)
        total_weight += weight
    if prior.bullpen_fip is not None:
        weight = _COMPONENT_WEIGHTS["bullpen_fip"]
        weighted_score += weight * ((4.20 - prior.bullpen_fip) / 0.60)
        total_weight += weight

    if total_weight <= 0.0:
        raise ValueError("roster prior must include a supported component field")
    return 0.5 + 0.14 * math.tanh(weighted_score / total_weight)


def _team_label_lookup(teams: Mapping[int, TeamLabel]) -> dict[str, int]:
    labels: dict[str, int] = {}
    for team_id, team in teams.items():
        for label in (team.abbreviation, team.team_name):
            normalized = _normalize_label(label)
            if normalized:
                labels[normalized] = team_id
    return labels


def _row_team_id(row: Mapping[str, object], labels: Mapping[str, int], index: int) -> int:
    team_id_text = _text_value(row.get("team_id"))
    if team_id_text:
        return _parse_int(team_id_text, "team_id", index)
    for field in (
        "abbreviation",
        "team_abbreviation",
        "team",
        "team_name",
        "name",
        "team_label",
    ):
        value = _text_value(row.get(field))
        if not value:
            continue
        normalized = _normalize_label(value)
        if normalized in labels:
            return labels[normalized]
    raise ValueError(
        f"Roster prior row {index} must include a known team_id, abbreviation, or team_name"
    )


def _required_int(row: Mapping[str, object], key: str, index: int) -> int:
    value = _text_value(row.get(key))
    if not value:
        raise ValueError(f"Roster prior row {index} missing {key}")
    return _parse_int(value, key, index)


def _optional_int(row: Mapping[str, object], key: str, index: int) -> int | None:
    value = _text_value(row.get(key))
    if not value:
        return None
    return _parse_int(value, key, index)


def _parse_int(value: str, key: str, index: int) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Roster prior row {index} has invalid {key}") from exc


def _optional_float(row: Mapping[str, object], key: str, index: int) -> float | None:
    value = _text_value(row.get(key))
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"Roster prior row {index} has invalid {key}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Roster prior row {index} has invalid {key}")
    return parsed


def _validate_probability(value: float | None, key: str) -> None:
    if value is None:
        return
    _validate_finite(value, key)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{key} must be between zero and one")


def _validate_range(
    value: float | None,
    key: str,
    lower_bound: float,
    upper_bound: float,
) -> None:
    if value is None:
        return
    _validate_finite(value, key)
    if not lower_bound <= value <= upper_bound:
        raise ValueError(f"{key} must be between {lower_bound:g} and {upper_bound:g}")


def _validate_finite(value: float, key: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")


def _text_value(value: object) -> str:
    return "" if value is None else str(value).strip()


def _normalize_label(value: str) -> str:
    return "".join(character for character in value.upper() if character.isalnum())


def _probability_to_logit(probability: float) -> float:
    p = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))
