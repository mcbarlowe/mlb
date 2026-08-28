"""Postgres storage for NRFI/YRFI odds snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

from psycopg import sql

from src.database import PostgresConfig, PostgresHandler

__all__ = [
    "NRFI_YRFI_ODDS_DDL",
    "ensure_nrfi_yrfi_odds_table",
    "normalize_nrfi_yrfi_odds_row",
    "upsert_nrfi_yrfi_odds_rows",
]

NRFI_YRFI_ODDS_DDL = """
CREATE TABLE IF NOT EXISTS nrfi_yrfi_odds (
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
    market_key text NOT NULL DEFAULT 'totals_1st_1_innings',
    market_last_update timestamptz,
    total_point double precision,
    yrfi_ml integer,
    nrfi_ml integer,
    source text NOT NULL DEFAULT 'the-odds-api',
    ingested_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (game_pk, bookmaker, line_type)
);

CREATE INDEX IF NOT EXISTS idx_nrfi_yrfi_odds_game_time
    ON nrfi_yrfi_odds (game_time);
CREATE INDEX IF NOT EXISTS idx_nrfi_yrfi_odds_line_type
    ON nrfi_yrfi_odds (line_type);
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
    "market_key",
    "market_last_update",
    "total_point",
    "yrfi_ml",
    "nrfi_ml",
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


def normalize_nrfi_yrfi_odds_row(row: Mapping[str, Any]) -> dict[str, object | None]:
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
        "market_key": _text(row, "market_key") or "totals_1st_1_innings",
        "market_last_update": _datetime_value(row.get("market_last_update")),
        "total_point": _float(row, "total_point"),
        "yrfi_ml": _int(row, "yrfi_ml"),
        "nrfi_ml": _int(row, "nrfi_ml"),
        "source": _text(row, "source") or "the-odds-api",
    }


def ensure_nrfi_yrfi_odds_table(db_config: PostgresConfig | None = None) -> None:
    with PostgresHandler(db_config) as db:
        db.connection.execute(NRFI_YRFI_ODDS_DDL)


def upsert_nrfi_yrfi_odds_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    db_config: PostgresConfig | None = None,
) -> int:
    if not rows:
        return 0
    normalized_rows = [normalize_nrfi_yrfi_odds_row(row) for row in rows]
    columns = sql.SQL(", ").join(sql.Identifier(column) for column in INSERT_COLUMNS)
    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in INSERT_COLUMNS)
    updates = sql.SQL(", ").join(
        sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(column), sql.Identifier(column))
        for column in INSERT_COLUMNS
        if column not in {"game_pk", "bookmaker", "line_type"}
    )
    query = sql.SQL(
        """
        INSERT INTO nrfi_yrfi_odds ({columns})
        VALUES ({placeholders})
        ON CONFLICT (game_pk, bookmaker, line_type) DO UPDATE SET
            {updates},
            ingested_at = now()
        """
    ).format(columns=columns, placeholders=placeholders, updates=updates)
    values = [tuple(row[column] for column in INSERT_COLUMNS) for row in normalized_rows]
    upserted = 0
    with PostgresHandler(db_config) as db:
        db.connection.execute(NRFI_YRFI_ODDS_DDL)
        with db.connection.cursor() as cursor:
            for value in values:
                cursor.execute(query, value)
                upserted += max(0, cursor.rowcount)
    return upserted
