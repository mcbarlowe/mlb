"""Postgres storage for totals evaluation runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from psycopg import sql
from psycopg.types.json import Jsonb

from src.database import PostgresConfig, PostgresHandler

__all__ = [
    "ensure_totals_eval_tables",
    "insert_totals_eval_run",
]

TOTALS_EVAL_DDL = """
CREATE TABLE IF NOT EXISTS totals_eval_runs (
    run_id text PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    season integer NOT NULL,
    games_requested integer NOT NULL,
    games_evaluated integer NOT NULL,
    non_push_games integer NOT NULL,
    push_games integer NOT NULL,
    sims_per_game integer NOT NULL,
    seed integer NOT NULL,
    pa_calibration_path text,
    mlflow_tracking_uri text NOT NULL,
    outcome_run_dir text NOT NULL,
    contact_environment boolean NOT NULL,
    metrics jsonb NOT NULL,
    totals jsonb NOT NULL,
    calibration jsonb NOT NULL
);

CREATE TABLE IF NOT EXISTS totals_eval_games (
    run_id text NOT NULL REFERENCES totals_eval_runs(run_id) ON DELETE CASCADE,
    game_pk integer NOT NULL,
    season integer NOT NULL,
    point double precision NOT NULL,
    sim_prob_over double precision NOT NULL,
    sim_prob_under double precision NOT NULL,
    sim_prob_push double precision NOT NULL,
    market_prob_over double precision NOT NULL,
    sim_mean_total double precision NOT NULL,
    sim_total_stdev double precision NOT NULL,
    actual_total double precision NOT NULL,
    outcome text NOT NULL CHECK (outcome IN ('over', 'under', 'push')),
    actual_over integer CHECK (actual_over IN (0, 1)),
    sim_brier double precision,
    market_brier double precision,
    sim_log_loss double precision,
    market_log_loss double precision,
    PRIMARY KEY (run_id, game_pk)
);

CREATE INDEX IF NOT EXISTS totals_eval_games_season_game_idx
    ON totals_eval_games (season, game_pk);
"""

TOTALS_EVAL_EVOLUTION_DDL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'totals_eval_runs'
          AND column_name = 'edge_threshold'
    ) THEN
        ALTER TABLE totals_eval_runs ALTER COLUMN edge_threshold DROP NOT NULL;
        ALTER TABLE totals_eval_runs ALTER COLUMN edge_buckets DROP NOT NULL;
        ALTER TABLE totals_eval_runs ALTER COLUMN roi_by_edge DROP NOT NULL;
    END IF;
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'totals_eval_games'
          AND column_name = 'sim_edge_over'
    ) THEN
        ALTER TABLE totals_eval_games ALTER COLUMN sim_edge_over DROP NOT NULL;
        ALTER TABLE totals_eval_games ALTER COLUMN sim_edge_under DROP NOT NULL;
    END IF;
END
$$;
"""

RUN_COLUMNS = (
    "run_id",
    "season",
    "games_requested",
    "games_evaluated",
    "non_push_games",
    "push_games",
    "sims_per_game",
    "seed",
    "pa_calibration_path",
    "mlflow_tracking_uri",
    "outcome_run_dir",
    "contact_environment",
    "metrics",
    "totals",
    "calibration",
)

GAME_COLUMNS = (
    "run_id",
    "game_pk",
    "season",
    "point",
    "sim_prob_over",
    "sim_prob_under",
    "sim_prob_push",
    "market_prob_over",
    "sim_mean_total",
    "sim_total_stdev",
    "actual_total",
    "outcome",
    "actual_over",
    "sim_brier",
    "market_brier",
    "sim_log_loss",
    "market_log_loss",
)

JSONB_COLUMNS = frozenset({"metrics", "totals", "calibration"})


def ensure_totals_eval_tables(db: PostgresHandler) -> None:
    db.connection.execute(TOTALS_EVAL_DDL)
    needs_evolution = db.connection.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'totals_eval_runs'
              AND column_name = 'roi_by_edge'
              AND is_nullable = 'NO'
        )
        """
    ).fetchone()[0]
    if needs_evolution:
        db.connection.execute(TOTALS_EVAL_EVOLUTION_DDL)


def _run_value(run: Mapping[str, Any], column: str) -> object:
    if column in JSONB_COLUMNS:
        return Jsonb(run[column])
    return run[column]


def _game_value(row: Mapping[str, Any], column: str) -> object:
    return row[column]


def insert_totals_eval_run(
    *,
    run: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    db_config: PostgresConfig | None = None,
) -> tuple[int, int]:
    """Insert one totals-eval run and its game rows.

    Existing ``run_id``/``game_pk`` rows are preserved, so rerunning with an
    explicit ``--run-id`` is idempotent and never rewrites historical results.
    """
    run_query = sql.SQL(
        """
        INSERT INTO totals_eval_runs ({columns})
        VALUES ({placeholders})
        ON CONFLICT (run_id) DO NOTHING
        """
    ).format(
        columns=sql.SQL(", ").join(sql.Identifier(column) for column in RUN_COLUMNS),
        placeholders=sql.SQL(", ").join(sql.Placeholder() for _ in RUN_COLUMNS),
    )
    game_query = sql.SQL(
        """
        INSERT INTO totals_eval_games ({columns})
        VALUES ({placeholders})
        ON CONFLICT (run_id, game_pk) DO NOTHING
        """
    ).format(
        columns=sql.SQL(", ").join(sql.Identifier(column) for column in GAME_COLUMNS),
        placeholders=sql.SQL(", ").join(sql.Placeholder() for _ in GAME_COLUMNS),
    )
    with PostgresHandler(db_config) as db:
        ensure_totals_eval_tables(db)
        with db.connection.cursor() as cursor:
            cursor.execute(run_query, tuple(_run_value(run, column) for column in RUN_COLUMNS))
            run_inserted = max(0, cursor.rowcount)
            game_inserted = 0
            for row in rows:
                cursor.execute(
                    game_query,
                    tuple(_game_value(row, column) for column in GAME_COLUMNS),
                )
                game_inserted += max(0, cursor.rowcount)
    return run_inserted, game_inserted
