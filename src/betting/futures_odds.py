"""CSV ingestion for season futures odds.

The parser is deliberately file-only: no database, network, or sportsbook API
assumptions live here. Rows can identify teams by MLB team id or by a label
that is later resolved against a projection CSV.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from src.betting.odds import american_to_decimal, american_to_prob, prob_to_decimal

__all__ = [
    "FUTURES_MARKETS",
    "FuturesOddsRow",
    "load_futures_odds_csv",
    "normalize_market_type",
    "normalize_team_label",
    "parse_futures_odds_row",
]

FUTURES_MARKETS = (
    "division",
    "playoff",
    "division_series",
    "league_championship",
    "world_series",
    "championship",
)

_MARKET_ALIASES = {
    "division": "division",
    "division_win": "division",
    "division_winner": "division",
    "win_division": "division",
    "playoff": "playoff",
    "playoffs": "playoff",
    "make_playoffs": "playoff",
    "make_playoff": "playoff",
    "division_series": "division_series",
    "ds": "division_series",
    "reach_division_series": "division_series",
    "league_championship": "league_championship",
    "lcs": "league_championship",
    "reach_league_championship": "league_championship",
    "world_series": "world_series",
    "pennant": "world_series",
    "reach_world_series": "world_series",
    "championship": "championship",
    "world_series_winner": "championship",
    "champion": "championship",
}

TEAM_ID_COLUMNS = ("team_id", "mlb_team_id")
TEAM_LABEL_COLUMNS = ("team_label", "team", "team_name", "abbreviation")
AMERICAN_ODDS_COLUMNS = ("american_odds", "odds", "price", "american")
IMPLIED_PROBABILITY_COLUMNS = (
    "implied_probability",
    "implied_prob",
    "probability",
    "implied",
)
TARGET_TOTAL_COLUMNS = ("target_total", "slots", "winners")


@dataclass(frozen=True)
class FuturesOddsRow:
    market_type: str
    implied_probability: float
    team_id: int | None = None
    team_label: str | None = None
    american_odds: float | None = None
    bookmaker: str = ""
    source: str = ""
    market_scope: str = ""
    season: str = ""
    as_of_date: str = ""
    last_update: str = ""
    target_total: float | None = None
    raw: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)

    @property
    def decimal_payout(self) -> float:
        """Offered decimal payout implied by the provided odds/probability."""
        if self.american_odds is not None:
            return american_to_decimal(self.american_odds)
        return prob_to_decimal(self.implied_probability)


def normalize_market_type(value: str) -> str:
    key = _normalize_token(value)
    try:
        return _MARKET_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(FUTURES_MARKETS)
        raise ValueError(f"Unknown futures market {value!r}; expected one of {allowed}") from exc


def normalize_team_label(value: str) -> str:
    """Normalize team labels for CSV joins without requiring exact punctuation."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def load_futures_odds_csv(path: str | Path) -> list[FuturesOddsRow]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [parse_futures_odds_row(row, row_number=i) for i, row in enumerate(reader, start=2)]


def parse_futures_odds_row(
    row: Mapping[str, str], *, row_number: int | None = None
) -> FuturesOddsRow:
    prefix = f"row {row_number}: " if row_number is not None else ""
    market_value = _first_nonblank(row, ("market_type", "market", "market_key"))
    if market_value is None:
        raise ValueError(f"{prefix}missing market_type/market")
    market_type = normalize_market_type(market_value)

    team_id = _parse_optional_int(_first_nonblank(row, TEAM_ID_COLUMNS), prefix=prefix)
    team_label = _first_nonblank(row, TEAM_LABEL_COLUMNS)
    if team_id is None and team_label is None:
        raise ValueError(f"{prefix}missing team_id or team label")

    american_odds = _parse_optional_float(
        _first_nonblank(row, AMERICAN_ODDS_COLUMNS), prefix=prefix, field="american odds"
    )
    implied_value = _first_nonblank(row, IMPLIED_PROBABILITY_COLUMNS)
    implied_probability = (
        _parse_probability(implied_value, prefix=prefix)
        if implied_value is not None
        else None
    )
    if implied_probability is None:
        if american_odds is None:
            raise ValueError(f"{prefix}missing american odds or implied probability")
        implied_probability = american_to_prob(american_odds)

    return FuturesOddsRow(
        market_type=market_type,
        team_id=team_id,
        team_label=team_label.strip() if team_label is not None else None,
        american_odds=american_odds,
        implied_probability=implied_probability,
        bookmaker=_clean_optional(_first_nonblank(row, ("bookmaker", "book", "sportsbook"))),
        source=_clean_optional(_first_nonblank(row, ("source", "market_source"))),
        market_scope=_clean_optional(_first_nonblank(row, ("market_scope", "group", "division", "league"))),
        season=_clean_optional(_first_nonblank(row, ("season",))),
        as_of_date=_clean_optional(_first_nonblank(row, ("as_of_date", "snapshot_date"))),
        last_update=_clean_optional(_first_nonblank(row, ("last_update", "updated_at"))),
        target_total=_parse_optional_float(
            _first_nonblank(row, TARGET_TOTAL_COLUMNS), prefix=prefix, field="target_total"
        ),
        raw=dict(row),
    )


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _first_nonblank(row: Mapping[str, str], columns: Sequence[str]) -> str | None:
    for column in columns:
        value = row.get(column)
        if value is not None and str(value).strip() != "":
            return str(value)
    return None


def _clean_optional(value: str | None) -> str:
    return value.strip() if value is not None else ""


def _parse_optional_int(value: str | None, *, prefix: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{prefix}invalid team_id {value!r}") from exc


def _parse_optional_float(
    value: str | None, *, prefix: str, field: str
) -> float | None:
    if value is None:
        return None
    try:
        return float(value.replace("+", ""))
    except ValueError as exc:
        raise ValueError(f"{prefix}invalid {field} {value!r}") from exc


def _parse_probability(value: str, *, prefix: str) -> float:
    raw = value.strip()
    is_percent = raw.endswith("%")
    if is_percent:
        raw = raw[:-1]
    try:
        probability = float(raw)
    except ValueError as exc:
        raise ValueError(f"{prefix}invalid implied probability {value!r}") from exc
    if is_percent or probability > 1.0:
        probability /= 100.0
    if not 0.0 < probability < 1.0:
        raise ValueError(f"{prefix}implied probability must be in (0, 1)")
    return probability
