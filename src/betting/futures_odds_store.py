"""Database storage for season futures odds.

Stores odds for championship, division, and playoff futures markets from
sportsbooks. Unlike game-by-game odds, futures are season-level and keyed
by (season, market_type, team_id, bookmaker, snapshot_time).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.database import PostgresHandler

FUTURES_ODDS_DDL = """
CREATE TABLE IF NOT EXISTS futures_odds (
    season integer NOT NULL,
    market_type text NOT NULL CHECK (market_type IN (
        'championship', 'division', 'playoff', 'make_playoffs', 'miss_playoffs',
        'al_pennant', 'nl_pennant',
        'league_championship', 'division_series', 'world_series'
    )),
    team_id integer NOT NULL,
    team_name text,
    bookmaker text NOT NULL,
    american_odds integer NOT NULL,
    implied_probability double precision,
    snapshot_time timestamptz NOT NULL,
    source text NOT NULL DEFAULT 'the-odds-api',
    ingested_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (season, market_type, team_id, bookmaker, snapshot_time)
);

CREATE INDEX IF NOT EXISTS idx_futures_odds_season_market
    ON futures_odds (season, market_type, snapshot_time DESC);
CREATE INDEX IF NOT EXISTS idx_futures_odds_team
    ON futures_odds (team_id, market_type, season);
"""

INSERT_COLUMNS = (
    "season",
    "market_type",
    "team_id",
    "team_name",
    "bookmaker",
    "american_odds",
    "implied_probability",
    "snapshot_time",
    "source",
)


def ensure_futures_odds_table(pg: PostgresHandler) -> None:
    """Create futures_odds table if it doesn't exist."""
    pg.connection.execute(FUTURES_ODDS_DDL)


def insert_futures_odds(pg: PostgresHandler, rows: Sequence[Mapping[str, object]]) -> int:
    """Insert futures odds rows, ignoring conflicts on PK.
    
    Returns the number of rows inserted.
    """
    if not rows:
        return 0
    
    placeholders = ", ".join(["%s"] * len(INSERT_COLUMNS))
    insert_sql = f"""
        INSERT INTO futures_odds ({", ".join(INSERT_COLUMNS)})
        VALUES ({placeholders})
        ON CONFLICT (season, market_type, team_id, bookmaker, snapshot_time)
        DO UPDATE SET
            team_name = EXCLUDED.team_name,
            american_odds = EXCLUDED.american_odds,
            implied_probability = EXCLUDED.implied_probability,
            source = EXCLUDED.source,
            ingested_at = now()
    """
    
    values = [
        tuple(_coerce_value(row, col) for col in INSERT_COLUMNS)
        for row in rows
    ]
    
    with pg.connection.cursor() as cursor:
        cursor.executemany(insert_sql, values)
        return cursor.rowcount


def load_latest_futures_odds(
    pg: PostgresHandler,
    *,
    season: int,
    market_type: str | None = None,
    as_of: datetime | None = None,
) -> list[dict[str, object]]:
    """Load the most recent futures odds snapshot for a season.
    
    Args:
        season: Season year
        market_type: Optional market filter (championship, division, playoff)
        as_of: Optional cutoff timestamp (default: latest available)
    
    Returns:
        List of odds rows with all columns
    """
    conditions = ["season = %s"]
    params: list[object] = [season]
    
    if market_type is not None:
        conditions.append("market_type = %s")
        params.append(market_type)
    
    if as_of is not None:
        conditions.append("snapshot_time <= %s")
        params.append(as_of)
    
    where_clause = " AND ".join(conditions)
    
    # Get latest snapshot time first
    snapshot_sql = f"""
        SELECT MAX(snapshot_time) as latest
        FROM futures_odds
        WHERE {where_clause}
    """
    
    result = pg.connection.execute(snapshot_sql, params).fetchone()
    
    if not result:
        return []
    
    latest_snapshot = result[0]
    
    # Get all odds from that snapshot
    conditions.append("snapshot_time = %s")
    params.append(latest_snapshot)
    where_clause = " AND ".join(conditions)
    
    odds_sql = f"""
        SELECT season, market_type, team_id, team_name, bookmaker,
               american_odds, implied_probability, snapshot_time, source
        FROM futures_odds
        WHERE {where_clause}
        ORDER BY team_id, bookmaker
    """
    
    cursor = pg.connection.execute(odds_sql, params)
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _coerce_value(row: Mapping[str, object], column: str) -> object:
    """Extract and coerce a column value for insertion."""
    value = row.get(column)
    if value is None:
        return None
    if column in ("season", "team_id", "american_odds"):
        return int(value)
    if column == "implied_probability":
        return float(value) if value is not None else None
    if column == "snapshot_time":
        if isinstance(value, str):
            # Parse ISO format
            return value
        return value
    return str(value)
