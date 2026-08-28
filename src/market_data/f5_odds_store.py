"""Postgres storage for first-five-innings odds snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

from psycopg import sql

from src.database import PostgresConfig, PostgresHandler

__all__ = [
    "F5_ODDS_DDL",
    "ensure_f5_odds_table",
    "normalize_f5_odds_row",
    "upsert_f5_odds_rows",
]

F5_ODDS_DDL = """
CREATE TABLE IF NOT EXISTS f5_odds (
    game_pk integer NOT NULL,
    game_date date,
    game_time timestamptz,
    away_team text,
    home_team text,
    away_team_id integer,
    home_team_id integer,
    bookmaker text NOT NULL,
    line_type text NOT NULL DEFAULT 'current',
    snapshot_time timestamptz NOT NULL,
    h2h_last_update timestamptz,
    spreads_last_update timestamptz,
    totals_last_update timestamptz,
    home_ml integer,
    away_ml integer,
    home_spread double precision,
    home_spread_ml integer,
    away_spread double precision,
    away_spread_ml integer,
    total_point double precision,
    over_ml integer,
    under_ml integer,
    source text NOT NULL DEFAULT 'the-odds-api',
    ingested_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (game_pk, bookmaker, line_type)
);

CREATE INDEX IF NOT EXISTS idx_f5_odds_game_time
    ON f5_odds (game_time);
CREATE INDEX IF NOT EXISTS idx_f5_odds_line_type
    ON f5_odds (line_type);
"""

INSERT_COLUMNS = (
    "game_pk",
    "game_date",
    "game_time",
    "away_team",
    "home_team",
    "away_team_id",
    "home_team_id",
    "bookmaker",
    "line_type",
    "snapshot_time",
    "h2h_last_update",
    "spreads_last_update",
    "totals_last_update",
    "home_ml",
    "away_ml",
    "home_spread",
    "home_spread_ml",
    "away_spread",
    "away_spread_ml",
    "total_point",
    "over_ml",
    "under_ml",
    "source",
)


def _blank_to_none(value: object) -> object | None:
    if value is None:
        return None
    if isinstance(value, str) and value == "":
        return None
    return value


def _text(row: Mapping[str, Any], key: str) -> str | None:
    value = _blank_to_none(row.get(key))
    return None if value is None else str(value)


def _required_text(row: Mapping[str, Any], key: str) -> str:
    value = _text(row, key)
    if value is None:
        raise ValueError(f"Missing required field {key!r}")
    return value


def _float(row: Mapping[str, Any], key: str) -> float | None:
    value = _blank_to_none(row.get(key))
    return None if value is None else float(str(value))


def _int(row: Mapping[str, Any], key: str) -> int | None:
    value = _blank_to_none(row.get(key))
    return None if value is None else int(float(str(value)))


def _required_int(row: Mapping[str, Any], key: str) -> int:
    value = _int(row, key)
    if value is None:
        raise ValueError(f"Missing required field {key!r}")
    return value


def _date_value(value: object | None) -> object | None:
    value = _blank_to_none(value)
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _datetime_value(value: object | None) -> object | None:
    value = _blank_to_none(value)
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def normalize_f5_odds_row(row: Mapping[str, Any]) -> dict[str, object | None]:
    return {
        "game_pk": _required_int(row, "game_pk"),
        "game_date": _date_value(row.get("game_date")),
        "game_time": _datetime_value(row.get("game_time")),
        "away_team": _text(row, "away_team"),
        "home_team": _text(row, "home_team"),
        "away_team_id": _int(row, "away_team_id"),
        "home_team_id": _int(row, "home_team_id"),
        "bookmaker": _required_text(row, "bookmaker"),
        "line_type": _text(row, "line_type") or "current",
        "snapshot_time": _required_text(row, "snapshot_time"),
        "h2h_last_update": _datetime_value(row.get("h2h_last_update")),
        "spreads_last_update": _datetime_value(row.get("spreads_last_update")),
        "totals_last_update": _datetime_value(row.get("totals_last_update")),
        "home_ml": _int(row, "home_ml"),
        "away_ml": _int(row, "away_ml"),
        "home_spread": _float(row, "home_spread"),
        "home_spread_ml": _int(row, "home_spread_ml"),
        "away_spread": _float(row, "away_spread"),
        "away_spread_ml": _int(row, "away_spread_ml"),
        "total_point": _float(row, "total_point"),
        "over_ml": _int(row, "over_ml"),
        "under_ml": _int(row, "under_ml"),
        "source": _text(row, "source") or "the-odds-api",
    }


def ensure_f5_odds_table(db_config: PostgresConfig | None = None) -> None:
    with PostgresHandler(db_config) as db:
        db.connection.execute(F5_ODDS_DDL)


def upsert_f5_odds_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    db_config: PostgresConfig | None = None,
) -> int:
    if not rows:
        return 0
    normalized_rows = [normalize_f5_odds_row(row) for row in rows]
    columns = sql.SQL(", ").join(sql.Identifier(column) for column in INSERT_COLUMNS)
    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in INSERT_COLUMNS)
    updates = sql.SQL(", ").join(
        sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(column), sql.Identifier(column))
        for column in INSERT_COLUMNS
        if column not in {"game_pk", "bookmaker", "line_type"}
    )
    query = sql.SQL(
        """
        INSERT INTO f5_odds ({columns})
        VALUES ({placeholders})
        ON CONFLICT (game_pk, bookmaker, line_type) DO UPDATE SET
            {updates},
            ingested_at = now()
        """
    ).format(columns=columns, placeholders=placeholders, updates=updates)
    values = [tuple(row[column] for column in INSERT_COLUMNS) for row in normalized_rows]
    inserted = 0
    with PostgresHandler(db_config) as db:
        db.connection.execute(F5_ODDS_DDL)
        with db.connection.cursor() as cursor:
            for value in values:
                cursor.execute(query, value)
                inserted += max(0, cursor.rowcount)
    return inserted
