"""PostgreSQL training data for chronological starting-pitcher prop models."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import psycopg
from psycopg import sql

from mlb.database import PostgresConfig
from mlb.pitcher_props.model import PitcherStart


def load_pitcher_starts(
    *,
    start_season: int,
    end_season: int,
    through: datetime | None = None,
    config: PostgresConfig | None = None,
) -> list[PitcherStart]:
    """Return complete regular-season starts in strict chronological order."""

    if start_season > end_season:
        raise ValueError("start_season cannot be after end_season")
    database = config or PostgresConfig.from_env()
    query = sql.SQL(
        """
        WITH completed_starts AS (
            SELECT
                p.game_pk,
                g.game_datetime,
                g.season,
                p.player_id AS pitcher_id,
                CASE
                    WHEN p.team_type = 'home' THEN g.away_team_id
                    ELSE g.home_team_id
                END AS opponent_team_id,
                p.team_type = 'home' AS is_home,
                p.outs,
                p.battersfaced,
                p.strikeouts,
                p.hits,
                p.baseonballs,
                COALESCE(p.numberofpitches, p.pitchesthrown, 0) AS pitch_count,
                LAG(g.game_datetime) OVER (
                    PARTITION BY p.player_id
                    ORDER BY g.game_datetime, p.game_pk
                ) AS previous_start_time
            FROM {schema}.pitching AS p
            JOIN {schema}.games AS g USING (game_pk)
            WHERE g.game_type = 'R'
              AND g.season BETWEEN %(start_season)s AND %(end_season)s
              AND p.gamesstarted = 1
              AND p.player_id IS NOT NULL
              AND p.team_type IN ('home', 'away')
              AND g.game_datetime IS NOT NULL
              AND p.outs IS NOT NULL
              AND p.battersfaced IS NOT NULL
              AND p.strikeouts IS NOT NULL
              AND p.hits IS NOT NULL
              AND p.baseonballs IS NOT NULL
        )
        SELECT
            game_pk,
            game_datetime,
            season,
            pitcher_id,
            opponent_team_id,
            is_home,
            outs,
            battersfaced,
            strikeouts,
            hits,
            baseonballs,
            pitch_count,
            EXTRACT(EPOCH FROM (game_datetime - previous_start_time)) / 86400.0
                AS rest_days
        FROM completed_starts
        WHERE (%(through)s::timestamptz IS NULL OR game_datetime < %(through)s)
        ORDER BY game_datetime, game_pk, pitcher_id
        """
    ).format(schema=sql.Identifier(database.schema))
    params = {
        "start_season": start_season,
        "end_season": end_season,
        "through": through,
    }
    with psycopg.connect(
        dbname=database.dbname,
        user=database.user,
        password=database.password,
        host=database.host,
        port=database.port,
        connect_timeout=30,
    ) as connection, connection.cursor() as cursor:
        cursor.execute(query, params)
        rows = cursor.fetchall()
    return [
        PitcherStart(
            game_pk=int(row[0]),
            game_time=row[1],
            season=int(row[2]),
            pitcher_id=int(row[3]),
            opponent_team_id=int(row[4]),
            is_home=bool(row[5]),
            outs=int(row[6]),
            batters_faced=int(row[7]),
            strikeouts=int(row[8]),
            hits_allowed=int(row[9]),
            walks=int(row[10]),
            pitch_count=int(row[11]),
            rest_days=float(row[12]) if row[12] is not None else None,
        )
        for row in rows
    ]


def seasons(
    starts: Sequence[PitcherStart],
    selected: Sequence[int],
) -> list[PitcherStart]:
    wanted = frozenset(int(season) for season in selected)
    return [start for start in starts if start.season in wanted]


__all__ = ["load_pitcher_starts", "seasons"]
