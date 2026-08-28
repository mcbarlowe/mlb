"""Market consensus and realized F5 totals for MLB model calibration."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import psycopg
from psycopg import sql

from src.database import PostgresConfig
from src.market_data.pricing import (
    american_to_decimal,
    decimal_to_american,
    no_vig_two_way,
)


@dataclass(frozen=True)
class F5BookLine:
    bookmaker: str
    point: float
    over_ml: float
    under_ml: float


@dataclass(frozen=True)
class F5ConsensusLine:
    point: float
    over_ml: float
    under_ml: float
    prob_over: float
    prob_under: float


def consensus_f5_totals_line(
    lines: Sequence[F5BookLine], *, devig_method: str = "proportional"
) -> F5ConsensusLine | None:
    if not lines:
        return None
    point = statistics.median(line.point for line in lines)
    over_decimal = statistics.median(
        american_to_decimal(line.over_ml) for line in lines
    )
    under_decimal = statistics.median(
        american_to_decimal(line.under_ml) for line in lines
    )
    over_ml = decimal_to_american(over_decimal)
    under_ml = decimal_to_american(under_decimal)
    prob_over, prob_under = no_vig_two_way(
        over_ml,
        under_ml,
        method=devig_method,
    )
    return F5ConsensusLine(
        point=point,
        over_ml=over_ml,
        under_ml=under_ml,
        prob_over=prob_over,
        prob_under=prob_under,
    )


def load_f5_actual_totals(
    game_pks: Sequence[int], *, prefix_innings: int
) -> dict[int, float]:
    if not game_pks:
        return {}
    config = PostgresConfig.from_env()
    query = sql.SQL(
        """
        SELECT game_pk,
               SUM(runs) FILTER (WHERE team_type = 'away')::float AS away_runs,
               SUM(runs) FILTER (WHERE team_type = 'home')::float AS home_runs,
               COUNT(*) FILTER (
                   WHERE team_type = 'away' AND runs IS NOT NULL
               ) AS away_rows,
               COUNT(*) FILTER (
                   WHERE team_type = 'home' AND runs IS NOT NULL
               ) AS home_rows
        FROM {}.linescore
        WHERE game_pk = ANY(%s)
          AND inning BETWEEN 1 AND %s
        GROUP BY game_pk
        """
    ).format(sql.Identifier(config.schema))
    totals: dict[int, float] = {}
    with psycopg.connect(
        dbname=config.dbname,
        user=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
        connect_timeout=15,
    ) as connection, connection.cursor() as cursor:
        cursor.execute(query, (list(game_pks), prefix_innings))
        for game_pk, away_runs, home_runs, away_rows, home_rows in cursor.fetchall():
            if away_rows < prefix_innings or home_rows < prefix_innings:
                continue
            if away_runs is None or home_runs is None:
                continue
            totals[int(game_pk)] = float(away_runs) + float(home_runs)
    return totals
