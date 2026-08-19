"""Postgres storage for MLB full-game moneyline odds snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

from psycopg import sql

from src.database import PostgresConfig, PostgresHandler

__all__ = [
    "H2H_ODDS_DDL",
    "ensure_h2h_odds_table",
    "normalize_h2h_odds_row",
    "upsert_h2h_odds_rows",
]

H2H_ODDS_DDL = """
CREATE TABLE IF NOT EXISTS odds (
    game_pk       integer NOT NULL,
    game_date     date,
    away_team_id  integer,
    home_team_id  integer,
    bookmaker     varchar NOT NULL,
    market        varchar NOT NULL DEFAULT 'h2h',
    line_type     varchar NOT NULL DEFAULT 'open',
    home_ml       integer,
    away_ml       integer,
    snapshot_time timestamptz,
    source        varchar DEFAULT 'the-odds-api',
    ingested_at   timestamptz DEFAULT now(),
    PRIMARY KEY (game_pk, bookmaker, market, line_type)
);
"""

INSERT_COLUMNS = (
    "game_pk",
    "game_date",
    "away_team_id",
    "home_team_id",
    "bookmaker",
    "market",
    "line_type",
    "home_ml",
    "away_ml",
    "snapshot_time",
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


def normalize_h2h_odds_row(row: Mapping[str, Any]) -> dict[str, object | None]:
    """Coerce a full-game moneyline odds row to DB-ready typed values."""
    return {
        "game_pk": _required_int(row, "game_pk"),
        "game_date": _date_value(row.get("game_date")),
        "away_team_id": _int(row, "away_team_id"),
        "home_team_id": _int(row, "home_team_id"),
        "bookmaker": _required_text(row, "bookmaker"),
        "market": _text(row, "market") or "h2h",
        "line_type": _text(row, "line_type") or "open",
        "home_ml": _int(row, "home_ml"),
        "away_ml": _int(row, "away_ml"),
        "snapshot_time": _datetime_value(_required_text(row, "snapshot_time")),
        "source": _text(row, "source") or "the-odds-api",
    }


def ensure_h2h_odds_table(db_config: PostgresConfig | None = None) -> None:
    with PostgresHandler(db_config) as db:
        db.connection.execute(H2H_ODDS_DDL)


def upsert_h2h_odds_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    db_config: PostgresConfig | None = None,
) -> int:
    """Upsert h2h odds rows, preserving the earliest saved open snapshot."""
    if not rows:
        return 0
    normalized_rows = [normalize_h2h_odds_row(row) for row in rows]
    columns = sql.SQL(", ").join(sql.Identifier(column) for column in INSERT_COLUMNS)
    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in INSERT_COLUMNS)
    updates = sql.SQL(", ").join(
        sql.SQL("{} = EXCLUDED.{}").format(
            sql.Identifier(column), sql.Identifier(column)
        )
        for column in INSERT_COLUMNS
        if column not in {"game_pk", "bookmaker", "market", "line_type"}
    )
    query = sql.SQL(
        """
        INSERT INTO odds ({columns})
        VALUES ({placeholders})
        ON CONFLICT (game_pk, bookmaker, market, line_type) DO UPDATE SET
            {updates},
            ingested_at = now()
        WHERE odds.line_type <> 'open'
           OR odds.snapshot_time IS NULL
           OR EXCLUDED.snapshot_time < odds.snapshot_time
        """
    ).format(columns=columns, placeholders=placeholders, updates=updates)
    values = [
        tuple(row[column] for column in INSERT_COLUMNS) for row in normalized_rows
    ]
    upserted = 0
    with PostgresHandler(db_config) as db:
        db.connection.execute(H2H_ODDS_DDL)
        with db.connection.cursor() as cursor:
            for value in values:
                cursor.execute(query, value)
                upserted += max(0, cursor.rowcount)
    return upserted
