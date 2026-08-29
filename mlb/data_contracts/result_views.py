"""Install the read-only Postgres contracts consumed by the betting service."""

from __future__ import annotations

import os

from mlb.database import PostgresConfig

RESULT_VIEW_SQL = """
CREATE OR REPLACE VIEW mlb.betting_game_results_v1 AS
SELECT
    g.game_pk,
    g.game_date,
    g.abstract_game_state,
    g.home_team_id,
    g.away_team_id,
    scores.home_score,
    scores.away_score
FROM mlb.games AS g
LEFT JOIN (
    SELECT
        game_pk,
        SUM(runs) FILTER (WHERE team_type = 'home')::integer AS home_score,
        SUM(runs) FILTER (WHERE team_type = 'away')::integer AS away_score
    FROM mlb.linescore
    GROUP BY game_pk
) AS scores USING (game_pk);

COMMENT ON VIEW mlb.betting_game_results_v1 IS
    'Read-only v1 betting result contract. Settle only Final rows with non-null scores.';

CREATE OR REPLACE VIEW mlb.betting_player_results_v1 AS
SELECT
    b.game_pk,
    g.game_date,
    g.abstract_game_state,
    b.player_id,
    b.player_name,
    CASE
        WHEN b.gamesplayed = 0 THEN FALSE
        WHEN b.gamesplayed > 0 THEN TRUE
        ELSE NULL
    END AS appeared,
    b.hits,
    b.doubles,
    b.triples,
    b.homeruns AS home_runs,
    b.totalbases AS total_bases,
    b.rbi,
    b.runs,
    b.baseonballs AS walks,
    b.stolenbases AS stolen_bases,
    b.strikeouts
FROM mlb.batting AS b
JOIN mlb.games AS g USING (game_pk);

COMMENT ON VIEW mlb.betting_player_results_v1 IS
    'Read-only v1 player result contract. Missing rows and null appeared remain pending.';
COMMENT ON COLUMN mlb.betting_player_results_v1.appeared IS
    'False only when batting.gamesplayed explicitly equals zero, while null is unknown.';
"""


def install_result_views(config: PostgresConfig | None = None) -> None:
    """Create or replace both additive contracts in the existing MLB schema."""
    import psycopg
    from psycopg import sql

    selected = config or PostgresConfig.from_env()
    if selected.schema != "mlb":
        raise ValueError("result contracts must be installed in the mlb schema")
    with psycopg.connect(
        dbname=selected.dbname,
        user=selected.user,
        password=selected.password,
        host=selected.host,
        port=selected.port,
        connect_timeout=10,
        options="-c statement_timeout=30000 -c lock_timeout=5000",
    ) as connection:
        connection.execute(RESULT_VIEW_SQL)
        consumer_role = os.getenv("BETTING_DB_USER") or selected.user
        connection.execute(
            sql.SQL("GRANT USAGE ON SCHEMA mlb TO {}").format(
                sql.Identifier(consumer_role)
            )
        )
        connection.execute(
            sql.SQL(
                "GRANT SELECT ON mlb.betting_game_results_v1, "
                "mlb.betting_player_results_v1 TO {}"
            ).format(sql.Identifier(consumer_role))
        )
