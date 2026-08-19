"""Futures edge reporting from projection CSVs and odds CSVs.

This module keeps futures betting analysis pure and file-backed. It joins model
season projection rows to market futures rows, removes multi-runner vig within
market groups, and reports model-vs-market edge plus EV/stake sizing fields.
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.betting.backtest import kelly_fraction
from src.betting.futures_odds import (
    FuturesOddsRow,
    load_futures_odds_csv,
    normalize_market_type,
    normalize_team_label,
)
from src.betting.odds import devig_proportional_many, prob_to_american, prob_to_decimal

__all__ = [
    "DEFAULT_MARKET_TARGET_TOTALS",
    "FUTURES_REPORT_COLUMNS",
    "MARKET_PROBABILITY_COLUMNS",
    "FuturesEdgeRow",
    "build_futures_edge_report",
    "generate_futures_edge_report_csv",
    "load_projection_csv",
    "write_futures_edge_report_csv",
]

MARKET_PROBABILITY_COLUMNS = {
    "division": "division_win_prob",
    "playoff": "playoff_prob",
    "division_series": "division_series_prob",
    "league_championship": "league_championship_prob",
    "world_series": "world_series_prob",
    "championship": "championship_prob",
}

DEFAULT_MARKET_TARGET_TOTALS = {
    "division": 1.0,
    "playoff": 12.0,
    "division_series": 8.0,
    "league_championship": 4.0,
    "world_series": 2.0,
    "championship": 1.0,
}

FUTURES_REPORT_COLUMNS = [
    "market_type",
    "market_scope",
    "bookmaker",
    "source",
    "season",
    "as_of_date",
    "last_update",
    "team_id",
    "abbreviation",
    "team_name",
    "projection_type",
    "offered_american_odds",
    "offered_decimal_payout",
    "market_implied_probability",
    "no_vig_market_probability",
    "model_probability",
    "edge",
    "fair_american_odds",
    "fair_decimal_payout",
    "unit_ev",
    "kelly_fraction",
    "stake_units",
    "target_total",
    "market_overround",
]


@dataclass(frozen=True)
class ProjectionTeamRow:
    team_id: int
    abbreviation: str
    team_name: str
    league_name: str
    division_name: str
    projection_type: str
    input_market_sources: str
    raw: Mapping[str, str]


@dataclass(frozen=True)
class FuturesEdgeRow:
    market_type: str
    market_scope: str
    bookmaker: str
    source: str
    season: str
    as_of_date: str
    last_update: str
    team_id: int
    abbreviation: str
    team_name: str
    projection_type: str
    offered_american_odds: float | None
    offered_decimal_payout: float
    market_implied_probability: float
    no_vig_market_probability: float
    model_probability: float
    edge: float
    fair_american_odds: float | None
    fair_decimal_payout: float | None
    unit_ev: float
    kelly_fraction: float
    stake_units: float | None
    target_total: float
    market_overround: float

    def to_csv_row(self) -> dict[str, object]:
        return {
            "market_type": self.market_type,
            "market_scope": self.market_scope,
            "bookmaker": self.bookmaker,
            "source": self.source,
            "season": self.season,
            "as_of_date": self.as_of_date,
            "last_update": self.last_update,
            "team_id": self.team_id,
            "abbreviation": self.abbreviation,
            "team_name": self.team_name,
            "projection_type": self.projection_type,
            "offered_american_odds": _blank_none(self.offered_american_odds),
            "offered_decimal_payout": self.offered_decimal_payout,
            "market_implied_probability": self.market_implied_probability,
            "no_vig_market_probability": self.no_vig_market_probability,
            "model_probability": self.model_probability,
            "edge": self.edge,
            "fair_american_odds": _blank_none(self.fair_american_odds),
            "fair_decimal_payout": _blank_none(self.fair_decimal_payout),
            "unit_ev": self.unit_ev,
            "kelly_fraction": self.kelly_fraction,
            "stake_units": _blank_none(self.stake_units),
            "target_total": self.target_total,
            "market_overround": self.market_overround,
        }


@dataclass(frozen=True)
class _Candidate:
    odds: FuturesOddsRow
    projection: ProjectionTeamRow
    market_scope: str
    model_probability: float
    target_total: float


def load_projection_csv(
    path: str | Path,
    *,
    projection_type: str = "model",
    as_of_bucket: str | None = None,
) -> list[ProjectionTeamRow]:
    with Path(path).open(newline="") as handle:
        return load_projection_rows(
            csv.DictReader(handle),
            projection_type=projection_type,
            as_of_bucket=as_of_bucket,
        )


def load_projection_rows(
    rows: Iterable[Mapping[str, str]],
    *,
    projection_type: str = "model",
    as_of_bucket: str | None = None,
) -> list[ProjectionTeamRow]:
    projections: list[ProjectionTeamRow] = []
    for row in rows:
        row_projection_type = str(row.get("projection_type", projection_type) or projection_type)
        if row_projection_type != projection_type:
            continue
        row_as_of_bucket = row.get("as_of_bucket")
        if as_of_bucket is not None and row_as_of_bucket != as_of_bucket:
            continue
        team_id_value = row.get("team_id")
        if team_id_value is None or str(team_id_value).strip() == "":
            raise ValueError("projection row missing team_id")
        projections.append(
            ProjectionTeamRow(
                team_id=int(team_id_value),
                abbreviation=str(row.get("abbreviation", "") or ""),
                team_name=str(row.get("team_name", "") or ""),
                league_name=str(row.get("league_name", "") or ""),
                division_name=str(row.get("division_name", "") or ""),
                projection_type=row_projection_type,
                input_market_sources=str(
                    row.get("input_market_sources", row.get("market_sources", "")) or ""
                ),
                raw=dict(row),
            )
        )
    if not projections:
        suffix = (
            f" and as_of_bucket={as_of_bucket!r}"
            if as_of_bucket is not None
            else ""
        )
        raise ValueError(
            f"No projection rows found for projection_type={projection_type!r}{suffix}"
        )
    return projections


def build_futures_edge_report(
    projection_rows: Sequence[Mapping[str, str]] | Sequence[ProjectionTeamRow],
    odds_rows: Sequence[FuturesOddsRow],
    *,
    projection_type: str = "model",
    markets: Sequence[str] | None = None,
    edge_threshold: float | None = None,
    kelly_multiplier: float | None = None,
    kelly_cap: float = 0.05,
    target_total_overrides: Mapping[str, float] | None = None,
    as_of_bucket: str | None = None,
    allow_market_source_leakage: bool = False,
) -> list[FuturesEdgeRow]:
    """Build model-vs-market futures rows.

    ``target_total_overrides`` can override de-vig totals by market, e.g.
    ``{"world_series": 1.0}`` for separate AL/NL pennant groups instead of
    the MLB-wide default of 2 World Series teams.
    """
    if not odds_rows:
        return []
    selected_markets = _selected_markets(markets, odds_rows)
    normalized_overrides = {
        normalize_market_type(market): total
        for market, total in (target_total_overrides or {}).items()
    }
    projections = _projection_team_rows(
        projection_rows,
        projection_type=projection_type,
        as_of_bucket=as_of_bucket,
    )
    if not allow_market_source_leakage:
        _reject_market_source_leakage(projections, selected_markets)

    by_id, by_label = _projection_indexes(projections)
    groups: dict[tuple[str, str, str, str, str, str], list[_Candidate]] = {}
    for odds in odds_rows:
        if odds.market_type not in selected_markets:
            continue
        projection = _match_projection(odds, by_id=by_id, by_label=by_label)
        market_scope = odds.market_scope or _default_market_scope(
            odds.market_type, projection
        )
        model_probability = _projection_probability(projection, odds.market_type)
        target_total = _target_total(
            odds,
            overrides=normalized_overrides,
            market_scope=market_scope,
        )
        key = (
            odds.market_type,
            odds.bookmaker,
            odds.source,
            odds.season,
            odds.as_of_date,
            market_scope,
        )
        groups.setdefault(key, []).append(
            _Candidate(
                odds=odds,
                projection=projection,
                market_scope=market_scope,
                model_probability=model_probability,
                target_total=target_total,
            )
        )

    report: list[FuturesEdgeRow] = []
    for group_key in sorted(groups):
        candidates = sorted(
            groups[group_key],
            key=lambda item: (item.projection.team_id, item.odds.team_label or ""),
        )
        target_total = _consistent_target_total(candidates)
        no_vig_probs = devig_proportional_many(
            [candidate.odds.implied_probability for candidate in candidates],
            target_total=target_total,
        )
        market_overround = (
            sum(candidate.odds.implied_probability for candidate in candidates)
            - target_total
        )
        for candidate, no_vig_probability in zip(candidates, no_vig_probs, strict=True):
            edge = candidate.model_probability - no_vig_probability
            if edge_threshold is not None and edge < edge_threshold:
                continue
            offered_decimal = candidate.odds.decimal_payout
            unit_ev = candidate.model_probability * offered_decimal - 1.0
            full_kelly = kelly_fraction(candidate.model_probability, offered_decimal)
            stake_units = (
                min(full_kelly * kelly_multiplier, kelly_cap)
                if kelly_multiplier is not None
                else None
            )
            report.append(
                FuturesEdgeRow(
                    market_type=candidate.odds.market_type,
                    market_scope=candidate.market_scope,
                    bookmaker=candidate.odds.bookmaker,
                    source=candidate.odds.source,
                    season=candidate.odds.season,
                    as_of_date=candidate.odds.as_of_date,
                    last_update=candidate.odds.last_update,
                    team_id=candidate.projection.team_id,
                    abbreviation=candidate.projection.abbreviation,
                    team_name=candidate.projection.team_name,
                    projection_type=candidate.projection.projection_type,
                    offered_american_odds=candidate.odds.american_odds,
                    offered_decimal_payout=offered_decimal,
                    market_implied_probability=candidate.odds.implied_probability,
                    no_vig_market_probability=no_vig_probability,
                    model_probability=candidate.model_probability,
                    edge=edge,
                    fair_american_odds=_fair_american(candidate.model_probability),
                    fair_decimal_payout=_fair_decimal(candidate.model_probability),
                    unit_ev=unit_ev,
                    kelly_fraction=full_kelly,
                    stake_units=stake_units,
                    target_total=target_total,
                    market_overround=market_overround,
                )
            )
    return sorted(
        report,
        key=lambda row: (
            row.market_type,
            row.market_scope,
            row.bookmaker,
            row.source,
            row.edge,
            row.team_id,
        ),
        reverse=True,
    )


def write_futures_edge_report_csv(path: str | Path, rows: Sequence[FuturesEdgeRow]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FUTURES_REPORT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


def generate_futures_edge_report_csv(
    *,
    projection_csv: str | Path,
    odds_csv: str | Path,
    out_csv: str | Path,
    projection_type: str = "model",
    markets: Sequence[str] | None = None,
    edge_threshold: float | None = None,
    kelly_multiplier: float | None = None,
    kelly_cap: float = 0.05,
    target_total_overrides: Mapping[str, float] | None = None,
    as_of_bucket: str | None = None,
    allow_market_source_leakage: bool = False,
) -> list[FuturesEdgeRow]:
    projection_rows = load_projection_csv(
        projection_csv,
        projection_type=projection_type,
        as_of_bucket=as_of_bucket,
    )
    odds_rows = load_futures_odds_csv(odds_csv)
    report = build_futures_edge_report(
        projection_rows,
        odds_rows,
        projection_type=projection_type,
        markets=markets,
        edge_threshold=edge_threshold,
        kelly_multiplier=kelly_multiplier,
        kelly_cap=kelly_cap,
        target_total_overrides=target_total_overrides,
        allow_market_source_leakage=allow_market_source_leakage,
        as_of_bucket=as_of_bucket,
    )
    write_futures_edge_report_csv(out_csv, report)
    return report


def _projection_team_rows(
    rows: Sequence[Mapping[str, str]] | Sequence[ProjectionTeamRow],
    *,
    projection_type: str,
    as_of_bucket: str | None = None,
) -> list[ProjectionTeamRow]:
    if not rows:
        raise ValueError("No projection rows supplied")
    first = rows[0]
    if isinstance(first, ProjectionTeamRow):
        typed_rows = [row for row in rows if isinstance(row, ProjectionTeamRow)]
        if as_of_bucket is None:
            return typed_rows
        return [
            row for row in typed_rows if row.raw.get("as_of_bucket") == as_of_bucket
        ]
    return load_projection_rows(
        rows,
        projection_type=projection_type,
        as_of_bucket=as_of_bucket,
    )  # type: ignore[arg-type]


def _selected_markets(
    markets: Sequence[str] | None, odds_rows: Sequence[FuturesOddsRow]
) -> set[str]:
    if markets is not None:
        return {normalize_market_type(market) for market in markets}
    return {odds.market_type for odds in odds_rows}


def _projection_indexes(
    projections: Sequence[ProjectionTeamRow],
) -> tuple[dict[int, ProjectionTeamRow], dict[str, ProjectionTeamRow]]:
    by_id: dict[int, ProjectionTeamRow] = {}
    by_label: dict[str, ProjectionTeamRow] = {}
    for projection in projections:
        if projection.team_id in by_id:
            raise ValueError(f"Duplicate projection team_id {projection.team_id}")
        by_id[projection.team_id] = projection
        labels = [str(projection.team_id), projection.abbreviation, projection.team_name]
        for label in labels:
            if label.strip() == "":
                continue
            normalized = normalize_team_label(label)
            existing = by_label.get(normalized)
            if existing is not None and existing.team_id != projection.team_id:
                raise ValueError(f"Ambiguous projection team label {label!r}")
            by_label[normalized] = projection
    return by_id, by_label


def _match_projection(
    odds: FuturesOddsRow,
    *,
    by_id: Mapping[int, ProjectionTeamRow],
    by_label: Mapping[str, ProjectionTeamRow],
) -> ProjectionTeamRow:
    if odds.team_id is not None:
        projection = by_id.get(odds.team_id)
        if projection is None:
            raise ValueError(f"No projection row for team_id={odds.team_id}")
        return projection
    if odds.team_label is not None:
        projection = by_label.get(normalize_team_label(odds.team_label))
        if projection is not None:
            return projection
    raise ValueError(f"No projection row for team label {odds.team_label!r}")


def _projection_probability(projection: ProjectionTeamRow, market_type: str) -> float:
    column = MARKET_PROBABILITY_COLUMNS[market_type]
    value = projection.raw.get(column)
    if value is None or str(value).strip() == "":
        raise ValueError(f"projection row for team_id={projection.team_id} missing {column}")
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{column} for team_id={projection.team_id} must be in [0, 1]")
    return probability


def _default_market_scope(market_type: str, projection: ProjectionTeamRow) -> str:
    if market_type == "division":
        if projection.division_name == "":
            raise ValueError("division odds require market_scope or projection division_name")
        return projection.division_name
    return "MLB"


def _target_total(
    odds: FuturesOddsRow,
    *,
    overrides: Mapping[str, float],
    market_scope: str,
) -> float:
    if odds.market_type in overrides:
        total = overrides[odds.market_type]
    elif odds.target_total is not None:
        total = odds.target_total
    else:
        total = DEFAULT_MARKET_TARGET_TOTALS[odds.market_type]
    if total <= 0.0:
        raise ValueError(
            f"target_total must be positive for {odds.market_type}/{market_scope}"
        )
    return total


def _consistent_target_total(candidates: Sequence[_Candidate]) -> float:
    target_total = candidates[0].target_total
    for candidate in candidates[1:]:
        if not math.isclose(candidate.target_total, target_total, abs_tol=1e-12):
            raise ValueError(
                f"Inconsistent target_total values for {candidate.odds.market_type} "
                f"{candidate.market_scope!r}"
            )
    return target_total


def _reject_market_source_leakage(
    projections: Sequence[ProjectionTeamRow], selected_markets: set[str]
) -> None:
    for projection in projections:
        sources = projection.input_market_sources
        if not sources:
            continue
        for market in selected_markets:
            if _sources_include_market(sources, market):
                raise ValueError(
                    "Projection input_market_sources includes target market "
                    f"{market!r}; pass allow_market_source_leakage=True only for "
                    "explicit leakage audits"
                )


def _sources_include_market(sources: str, market: str) -> bool:
    normalized_market = normalize_market_type(market)
    for chunk in re.split(r"[,;|]+", sources.lower()):
        normalized_chunk = re.sub(r"[^a-z0-9]+", "_", chunk).strip("_")
        if not normalized_chunk:
            continue
        if normalized_chunk == normalized_market:
            return True
        if normalized_chunk.startswith(f"{normalized_market}_"):
            return True
        if normalized_chunk.endswith(f"_{normalized_market}"):
            return True
        if f"_{normalized_market}_" in normalized_chunk:
            return True
    return False


def _fair_american(probability: float) -> float | None:
    if not 0.0 < probability < 1.0:
        return None
    return prob_to_american(probability)


def _fair_decimal(probability: float) -> float | None:
    if not 0.0 < probability < 1.0:
        return None
    return prob_to_decimal(probability)


def _blank_none(value: object | None) -> object:
    return "" if value is None else value
