"""Postgres storage for paper-trade ledgers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

from psycopg import sql

from src.database import PostgresConfig, PostgresHandler

__all__ = [
    "ensure_paper_trades_table",
    "load_paper_trade_rows",
    "normalize_paper_trade_row",
    "update_paper_trade_settlement_rows",
    "upsert_paper_trade_rows",
]

PAPER_TRADES_DDL = """
CREATE TABLE IF NOT EXISTS paper_trades (
    strategy_version text NOT NULL,
    paper_date date NOT NULL,
    snapshot_time timestamptz NOT NULL,
    game_pk integer NOT NULL,
    game_time timestamptz,
    away_team text,
    home_team text,
    away_team_id integer,
    home_team_id integer,
    away_probable text,
    home_probable text,
    side text NOT NULL CHECK (side IN ('home', 'away')),
    model_prob_home double precision,
    selected_model_prob double precision,
    consensus_market_prob double precision,
    edge double precision,
    consensus_home_prob double precision,
    consensus_away_prob double precision,
    consensus_home_ml double precision,
    consensus_away_ml double precision,
    best_books text[],
    best_ml double precision,
    best_decimal double precision,
    best_fair_prob double precision,
    staking text,
    stake_fraction double precision,
    stake_units double precision,
    status text NOT NULL DEFAULT 'open',
    close_ml double precision,
    close_fair_prob double precision,
    clv double precision,
    result text,
    profit_units double precision,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (strategy_version, game_pk)
);
"""

INSERT_COLUMNS = (
    "strategy_version",
    "paper_date",
    "snapshot_time",
    "game_pk",
    "game_time",
    "away_team",
    "home_team",
    "away_team_id",
    "home_team_id",
    "away_probable",
    "home_probable",
    "side",
    "model_prob_home",
    "selected_model_prob",
    "consensus_market_prob",
    "edge",
    "consensus_home_prob",
    "consensus_away_prob",
    "consensus_home_ml",
    "consensus_away_ml",
    "best_books",
    "best_ml",
    "best_decimal",
    "best_fair_prob",
    "staking",
    "stake_fraction",
    "stake_units",
    "status",
    "close_ml",
    "close_fair_prob",
    "clv",
    "result",
    "profit_units",
)

ROW_TO_DB = {
    "snapshot_time": "snapshot_time_utc",
}


def _blank_to_none(value: object) -> object | None:
    if value is None:
        return None
    if isinstance(value, str) and value == "":
        return None
    return value


def _text(row: Mapping[str, str], key: str) -> str | None:
    value = _blank_to_none(row.get(key, ""))
    return None if value is None else str(value)


def _required_text(row: Mapping[str, str], key: str) -> str:
    value = _text(row, key)
    if value is None:
        raise ValueError(f"Missing required field {key!r}")
    return value


def _float(row: Mapping[str, str], key: str) -> float | None:
    value = _blank_to_none(row.get(key, ""))
    return None if value is None else float(str(value))


def _int(row: Mapping[str, str], key: str) -> int | None:
    value = _blank_to_none(row.get(key, ""))
    return None if value is None else int(str(value))


def _required_int(row: Mapping[str, str], key: str) -> int:
    value = _int(row, key)
    if value is None:
        raise ValueError(f"Missing required field {key!r}")
    return value


def _books(row: Mapping[str, str]) -> list[str]:
    text = _text(row, "best_books")
    if not text:
        return []
    return [book for book in text.split("|") if book]


def normalize_paper_trade_row(row: Mapping[str, str]) -> dict[str, Any]:
    """Coerce a paper-trade CSV row to DB-ready typed values."""
    return {
        "strategy_version": _required_text(row, "strategy_version"),
        "paper_date": _required_text(row, "paper_date"),
        "snapshot_time": _required_text(row, "snapshot_time_utc"),
        "game_pk": _required_int(row, "game_pk"),
        "game_time": _text(row, "game_time"),
        "away_team": _text(row, "away_team"),
        "home_team": _text(row, "home_team"),
        "away_team_id": _int(row, "away_team_id"),
        "home_team_id": _int(row, "home_team_id"),
        "away_probable": _text(row, "away_probable"),
        "home_probable": _text(row, "home_probable"),
        "side": _required_text(row, "side"),
        "model_prob_home": _float(row, "model_prob_home"),
        "selected_model_prob": _float(row, "selected_model_prob"),
        "consensus_market_prob": _float(row, "consensus_market_prob"),
        "edge": _float(row, "edge"),
        "consensus_home_prob": _float(row, "consensus_home_prob"),
        "consensus_away_prob": _float(row, "consensus_away_prob"),
        "consensus_home_ml": _float(row, "consensus_home_ml"),
        "consensus_away_ml": _float(row, "consensus_away_ml"),
        "best_books": _books(row),
        "best_ml": _float(row, "best_ml"),
        "best_decimal": _float(row, "best_decimal"),
        "best_fair_prob": _float(row, "best_fair_prob"),
        "staking": _text(row, "staking"),
        "stake_fraction": _float(row, "stake_fraction"),
        "stake_units": _float(row, "stake_units"),
        "status": _text(row, "status") or "open",
        "close_ml": _float(row, "close_ml"),
        "close_fair_prob": _float(row, "close_fair_prob"),
        "clv": _float(row, "clv"),
        "result": _text(row, "result"),
        "profit_units": _float(row, "profit_units"),
    }


def ensure_paper_trades_table(db: PostgresHandler) -> None:
    db.connection.execute(PAPER_TRADES_DDL)


def upsert_paper_trade_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    db_config: PostgresConfig | None = None,
) -> int:
    """Insert new paper-trade rows; existing strategy/game rows are preserved."""
    if not rows:
        return 0
    query = sql.SQL(
        """
        INSERT INTO paper_trades ({columns})
        VALUES ({placeholders})
        ON CONFLICT (strategy_version, game_pk) DO NOTHING
        """
    ).format(
        columns=sql.SQL(", ").join(sql.Identifier(column) for column in INSERT_COLUMNS),
        placeholders=sql.SQL(", ").join(sql.Placeholder() for _ in INSERT_COLUMNS),
    )
    inserted = 0
    with PostgresHandler(db_config) as db:
        ensure_paper_trades_table(db)
        with db.connection.cursor() as cursor:
            for row in rows:
                normalized = normalize_paper_trade_row(row)
                values = tuple(normalized[column] for column in INSERT_COLUMNS)
                cursor.execute(query, values)
                inserted += max(0, cursor.rowcount)
    return inserted


DB_TO_ROW = {
    "snapshot_time": "snapshot_time_utc",
}


def _db_value_to_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def load_paper_trade_rows(
    *,
    db_config: PostgresConfig | None = None,
) -> list[dict[str, str]]:
    """Load DB paper trades as CSV-shaped rows for shared settlement helpers."""
    query = sql.SQL("SELECT {columns} FROM paper_trades ORDER BY snapshot_time, game_pk").format(
        columns=sql.SQL(", ").join(sql.Identifier(column) for column in INSERT_COLUMNS)
    )
    with PostgresHandler(db_config) as db:
        ensure_paper_trades_table(db)
        with db.connection.cursor() as cursor:
            cursor.execute(query)
            names = [
                DB_TO_ROW.get(description.name, description.name)
                for description in cursor.description or ()
            ]
            return [
                {
                    key: _db_value_to_text(value)
                    for key, value in zip(names, values, strict=True)
                }
                for values in cursor.fetchall()
            ]


def update_paper_trade_settlement_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    db_config: PostgresConfig | None = None,
) -> int:
    """Update settlement fields for existing DB paper-trade rows."""
    if not rows:
        return 0
    query = sql.SQL(
        """
        UPDATE paper_trades
        SET status = %s,
            close_ml = %s,
            close_fair_prob = %s,
            clv = %s,
            result = %s,
            profit_units = %s,
            updated_at = now()
        WHERE strategy_version = %s
          AND game_pk = %s
        """
    )
    updated = 0
    with PostgresHandler(db_config) as db:
        ensure_paper_trades_table(db)
        with db.connection.cursor() as cursor:
            for row in rows:
                normalized = normalize_paper_trade_row(row)
                cursor.execute(
                    query,
                    (
                        normalized["status"],
                        normalized["close_ml"],
                        normalized["close_fair_prob"],
                        normalized["clv"],
                        normalized["result"],
                        normalized["profit_units"],
                        normalized["strategy_version"],
                        normalized["game_pk"],
                    ),
                )
                updated += max(0, cursor.rowcount)
    return updated
