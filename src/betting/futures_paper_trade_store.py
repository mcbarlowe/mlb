"""Database storage for futures paper trades.

Tracks hypothetical futures bets on season-level markets (championship,
division, playoff). Unlike game-by-game paper trades, these settle at
season end based on actual playoff/championship outcomes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.database import PostgresHandler

FUTURES_PAPER_TRADES_DDL = """
CREATE TABLE IF NOT EXISTS futures_paper_trades (
    strategy_version text NOT NULL,
    season integer NOT NULL,
    market_type text NOT NULL CHECK (market_type IN (
        'championship', 'division', 'playoff',
        'league_championship', 'division_series', 'world_series'
    )),
    team_id integer NOT NULL,
    team_name text,
    paper_date date NOT NULL,
    snapshot_time timestamptz NOT NULL,
    model_probability double precision NOT NULL,
    consensus_market_prob double precision,
    edge double precision,
    best_bookmaker text,
    best_american_odds integer,
    best_decimal_odds double precision,
    best_fair_prob double precision,
    staking text NOT NULL,
    stake_fraction double precision,
    stake_units double precision NOT NULL,
    status text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'won', 'lost', 'push', 'cancelled')),
    result text CHECK (result IN ('won', 'lost', 'push', 'cancelled')),
    profit_units double precision,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (strategy_version, season, market_type, team_id)
);

CREATE INDEX IF NOT EXISTS idx_futures_paper_trades_season
    ON futures_paper_trades (season, market_type, status);
CREATE INDEX IF NOT EXISTS idx_futures_paper_trades_strategy
    ON futures_paper_trades (strategy_version, paper_date DESC);
"""

INSERT_COLUMNS = (
    "strategy_version",
    "season",
    "market_type",
    "team_id",
    "team_name",
    "paper_date",
    "snapshot_time",
    "model_probability",
    "consensus_market_prob",
    "edge",
    "best_bookmaker",
    "best_american_odds",
    "best_decimal_odds",
    "best_fair_prob",
    "staking",
    "stake_fraction",
    "stake_units",
    "status",
)


def ensure_futures_paper_trades_table(pg: PostgresHandler) -> None:
    """Create futures_paper_trades table if it doesn't exist."""
    pg.connection.execute(FUTURES_PAPER_TRADES_DDL)


def insert_futures_paper_trades(
    pg: PostgresHandler, rows: Sequence[Mapping[str, object]]
) -> int:
    """Insert futures paper trade rows.
    
    On conflict (same strategy/season/market/team), updates the bet details
    if still open, otherwise ignores.
    
    Returns the number of rows inserted.
    """
    if not rows:
        return 0
    
    placeholders = ", ".join([f"%s"] * len(INSERT_COLUMNS))
    insert_sql = f"""
        INSERT INTO futures_paper_trades ({", ".join(INSERT_COLUMNS)})
        VALUES ({placeholders})
        ON CONFLICT (strategy_version, season, market_type, team_id)
        DO UPDATE SET
            paper_date = EXCLUDED.paper_date,
            snapshot_time = EXCLUDED.snapshot_time,
            model_probability = EXCLUDED.model_probability,
            consensus_market_prob = EXCLUDED.consensus_market_prob,
            edge = EXCLUDED.edge,
            best_bookmaker = EXCLUDED.best_bookmaker,
            best_american_odds = EXCLUDED.best_american_odds,
            best_decimal_odds = EXCLUDED.best_decimal_odds,
            best_fair_prob = EXCLUDED.best_fair_prob,
            staking = EXCLUDED.staking,
            stake_fraction = EXCLUDED.stake_fraction,
            stake_units = EXCLUDED.stake_units,
            updated_at = now()
        WHERE futures_paper_trades.status = 'open'
    """
    
    values = [
        tuple(_coerce_value(row, col) for col in INSERT_COLUMNS)
        for row in rows
    ]
    
    with pg.connection.cursor() as cursor:
        cursor.executemany(insert_sql, values)
        return cursor.rowcount

def settle_futures_paper_trade(
    pg: PostgresHandler,
    *,
    strategy_version: str,
    season: int,
    market_type: str,
    team_id: int,
    result: str,
    profit_units: float,
) -> int:
    """Settle a single futures paper trade.
    
    Updates status, result, and profit_units for a completed bet.
    
    Returns 1 if updated, 0 if not found or already settled.
    """
    status_map = {"won": "won", "lost": "lost", "push": "push", "cancelled": "cancelled"}
    if result not in status_map:
        raise ValueError(f"Invalid result {result!r}; expected one of {list(status_map)}")
    
    update_sql = """
        UPDATE futures_paper_trades
        SET status = %s,
            result = %s,
            profit_units = %s,
            updated_at = now()
        WHERE strategy_version = %s
          AND season = %s
          AND market_type = %s
          AND team_id = %s
          AND status = 'open'
    """
    
    result = pg.connection.execute(
        update_sql,
        (status_map[result], result, profit_units, strategy_version, season, market_type, team_id),
    )
    return result.rowcount if hasattr(result, 'rowcount') else 0


def load_open_futures_paper_trades(
    pg: PostgresHandler,
    *,
    strategy_version: str,
    season: int | None = None,
) -> list[dict[str, object]]:
    """Load all open futures paper trades for a strategy.
    
    Args:
        strategy_version: Strategy identifier
        season: Optional season filter
    
    Returns:
        List of open trade rows
    """
    conditions = ["strategy_version = %s", "status = 'open'"]
    params: list[object] = [strategy_version]
    
    if season is not None:
        conditions.append("season = %s")
        params.append(season)
    
    where_clause = " AND ".join(conditions)
    
    sql = f"""
        SELECT strategy_version, season, market_type, team_id, team_name,
               paper_date, snapshot_time, model_probability,
               consensus_market_prob, edge, best_bookmaker,
               best_american_odds, best_decimal_odds, best_fair_prob,
               staking, stake_fraction, stake_units, status
        FROM futures_paper_trades
        WHERE {where_clause}
        ORDER BY season, market_type, team_id
    """
    
    cursor = pg.connection.execute(sql, params)
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _coerce_value(row: Mapping[str, object], column: str) -> object:
    """Extract and coerce a column value for insertion."""
    value = row.get(column)
    if value is None:
        return None
    if column in ("season", "team_id", "best_american_odds"):
        return int(value)
    if column in (
        "model_probability",
        "consensus_market_prob",
        "edge",
        "best_decimal_odds",
        "best_fair_prob",
        "stake_fraction",
        "stake_units",
    ):
        return float(value) if value is not None else None
    return str(value)
