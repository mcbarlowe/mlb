from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import LiteralString, cast

import pandas as pd
import psycopg
from psycopg import sql
from psycopg.abc import Query

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class PostgresConfig:
    """Connection settings for the local PostgreSQL database."""

    dbname: str = "postgres"
    user: str | None = None
    password: str | None = None
    host: str | None = None
    port: int | None = None
    schema: str = "mlb"

    @classmethod
    def from_env(cls) -> PostgresConfig:
        """Build connection settings from MLB_DB_* environment variables."""

        port = os.getenv("MLB_DB_PORT")
        return cls(
            dbname=os.getenv("MLB_DB_NAME", "postgres"),
            user=os.getenv("MLB_DB_USER"),
            password=os.getenv("MLB_DB_PASSWORD"),
            host=os.getenv("MLB_DB_HOST"),
            port=int(port) if port else None,
            schema=os.getenv("MLB_DB_SCHEMA", "mlb"),
        )

    def describe(self) -> str:
        """Return a redaction-safe connection label."""

        parts = [f"dbname={self.dbname}", f"schema={self.schema.lower()}"]
        if self.host:
            parts.append(f"host={self.host}")
        else:
            parts.append("host=local-socket")
        if self.port is not None:
            parts.append(f"port={self.port}")
        return " ".join(parts)


class PostgresHandler:
    """Handler for PostgreSQL schema creation, loading, and querying."""

    managed_tables = (
        "teams",
        "players",
        "positions",
        "pitch_types",
        "event_types",
        "game_types",
        "venues",
        "games",
        "pitches",
        "linescore",
        "batting",
        "pitching",
        "fielding",
    )

    def __init__(self, config: PostgresConfig | None = None):
        self.config = config or PostgresConfig.from_env()
        self.schema = self._normalize_identifier(self.config.schema, "schema")
        self.connection = psycopg.connect(
            dbname=self.config.dbname,
            user=self.config.user,
            password=self.config.password,
            host=self.config.host,
            port=self.config.port,
            autocommit=True,
        )
        self._column_type_cache: dict[str, dict[str, str]] = {}
        self.ensure_schema()

    def _normalize_identifier(self, name: str, kind: str) -> str:
        normalized = name.lower()
        if not _IDENTIFIER_RE.fullmatch(normalized):
            raise ValueError(f"Invalid {kind} name: {name!r}")
        return normalized

    def _qualified_table(self, table_name: str) -> sql.Composed:
        normalized = self._normalize_identifier(table_name, "table")
        return sql.SQL("{}.{}").format(
            sql.Identifier(self.schema),
            sql.Identifier(normalized),
        )

    def ensure_schema(self):
        """Create the configured schema if it does not already exist."""

        self.connection.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self.schema))
        )
        self.connection.execute(
            sql.SQL("SET search_path TO {}, public").format(sql.Identifier(self.schema))
        )

    def reset_schema(self):
        """Drop and recreate the configured schema."""

        self.connection.execute(
            sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(self.schema))
        )
        self.ensure_schema()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.connection:
            self.connection.close()

    def _run_schema_file(self) -> None:
        """Execute the canonical schema.sql into the configured schema.

        Statements are schema-agnostic (unqualified); ensure_schema() has set
        search_path so objects land in self.schema. Split on ';' is safe here
        because schema.sql contains no dollar-quoted blocks or string literals
        with embedded semicolons.
        """

        from pathlib import Path

        ddl = Path(__file__).with_name("schema.sql").read_text()
        statements = [chunk.strip() for chunk in ddl.split(";") if chunk.strip()]
        with self.connection.cursor() as cursor:
            for statement in statements:
                cursor.execute(cast(Query, statement))

    def _schema_present(self) -> bool:
        return self.table_exists("games") and self.table_exists("pitches")

    def create_all_tables(self):
        """Create the full MLB schema (tables, partitions, keys, indexes).

        Idempotent: a no-op when the schema is already present, so it is safe to
        call at the start of every ETL/backfill run. The fresh-install schema
        reproduces the primary keys the idempotent loader relies on for its
        ON CONFLICT clauses, plus the season-partitioned pitches table.
        """

        if self._schema_present():
            print("✓ Schema already present; skipping DDL")
            return
        print("\nCreating MLB database schema from schema.sql ...")
        self._run_schema_file()
        print("✓ All tables created successfully!")

    def create_reference_tables(self):
        """Ensure the schema (including reference/lookup tables) exists."""

        if not self._schema_present():
            self._run_schema_file()
        print("✓ Reference tables ensured")




    def create_teams_table(self) -> None:
        self.create_all_tables()

    def create_players_table(self) -> None:
        self.create_all_tables()

    def create_positions_table(self) -> None:
        self.create_all_tables()

    def create_pitch_types_table(self) -> None:
        self.create_all_tables()

    def create_event_types_table(self) -> None:
        self.create_all_tables()

    def create_game_types_table(self) -> None:
        self.create_all_tables()

    def create_venues_table(self) -> None:
        self.create_all_tables()

    def create_games_table(self) -> None:
        self.create_all_tables()

    def create_pitches_table(self) -> None:
        self.create_all_tables()

    def create_linescore_table(self) -> None:
        self.create_all_tables()

    def create_batting_table(self) -> None:
        self.create_all_tables()

    def create_pitching_table(self) -> None:
        self.create_all_tables()

    def create_fielding_table(self) -> None:
        self.create_all_tables()

    def _column_types(self, table_name: str) -> dict[str, str]:
        if table_name not in self._column_type_cache:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    """,
                    (self.schema, table_name),
                )
                self._column_type_cache[table_name] = {
                    column_name: data_type
                    for column_name, data_type in cursor.fetchall()
                }
        return self._column_type_cache[table_name]
    def _pk_columns(self, table_name: str) -> list[str]:
        """Return primary-key column names (in key order) for a schema table."""

        qualified = f"{self.schema}.{self._normalize_identifier(table_name, 'table')}"
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT a.attname
                FROM pg_index i
                JOIN pg_attribute a
                  ON a.attrelid = i.indrelid AND a.attnum = ANY (i.indkey)
                WHERE i.indrelid = %s::regclass AND i.indisprimary
                ORDER BY array_position(i.indkey, a.attnum)
                """,
                (qualified,),
            )
            return [row[0] for row in cursor.fetchall()]

    def _cast_expr(self, column: str, data_type: str) -> sql.SQL:
        """Build a text->target cast that coerces '', 'None', and API sentinels to NULL."""

        c = '"' + column + '"'
        base = f"NULLIF(NULLIF({c}, 'None'), '')"
        if data_type in ("integer", "bigint", "smallint"):
            expr = f"({base})::numeric::{data_type}"
        elif data_type in ("numeric", "double precision", "real"):
            expr = (
                f"CASE WHEN {base} ~ '^-?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?$' "
                f"THEN ({base})::{data_type} END"
            )
        elif data_type == "date":
            expr = f"CASE WHEN {base} ~ '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}$' THEN ({base})::date END"
        elif data_type == "timestamp with time zone":
            expr = f"CASE WHEN {base} ~ '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}' THEN ({base})::timestamptz END"
        elif data_type == "timestamp without time zone":
            expr = f"CASE WHEN {base} ~ '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}' THEN ({base})::timestamp END"
        elif data_type == "boolean":
            expr = f"({base})::boolean"
        else:
            expr = base
        return sql.SQL(cast(LiteralString, expr))

    def insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = "append"):
        """Idempotently load a DataFrame.

        Rows are staged as text, then inserted with type-coercing casts and an
        ON CONFLICT clause keyed on the target primary key:
          * append  -> ON CONFLICT DO NOTHING (no duplicate rows on re-load)
          * replace -> ON CONFLICT DO UPDATE  (refresh values in place; never
                       TRUNCATE, which would CASCADE across foreign keys)
          * fail    -> raise if the table already holds rows
        """

        if df.empty:
            print(f"⚠ Warning: DataFrame is empty, skipping insert to {table_name}")
            return

        normalized_table = self._normalize_identifier(table_name, "table")
        if if_exists not in ("append", "replace", "fail"):
            raise ValueError(f"Unsupported if_exists mode: {if_exists}")
        if (
            if_exists == "fail"
            and self.table_exists(normalized_table)
            and self.get_row_count(normalized_table) > 0
        ):
            raise ValueError(f"Table {normalized_table} already contains data")

        normalized_columns = [self._normalize_identifier(column, "column") for column in df.columns]
        column_types = self._column_types(normalized_table)
        pk_columns = self._pk_columns(normalized_table)

        buffer = io.StringIO()
        df.to_csv(buffer, index=False, header=False, na_rep=r"\N")
        buffer.seek(0)

        qualified = self._qualified_table(normalized_table)
        staging = sql.Identifier(f"_stg_{normalized_table}")
        column_defs = sql.SQL(", ").join(
            sql.SQL("{} text").format(sql.Identifier(column)) for column in normalized_columns
        )
        column_idents = sql.SQL(", ").join(sql.Identifier(column) for column in normalized_columns)
        select_list = sql.SQL(", ").join(
            self._cast_expr(column, column_types.get(column, "text")) for column in normalized_columns
        )

        conflict = sql.SQL("")
        if pk_columns and all(column in normalized_columns for column in pk_columns):
            target = sql.SQL(", ").join(sql.Identifier(column) for column in pk_columns)
            update_columns = [c for c in normalized_columns if c not in pk_columns]
            if if_exists == "replace" and update_columns:
                setters = sql.SQL(", ").join(
                    sql.SQL("{col} = EXCLUDED.{col}").format(col=sql.Identifier(c))
                    for c in update_columns
                )
                conflict = sql.SQL(" ON CONFLICT ({}) DO UPDATE SET {}").format(target, setters)
            else:
                conflict = sql.SQL(" ON CONFLICT ({}) DO NOTHING").format(target)

        copy_sql = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT CSV, NULL '\\N')").format(
            staging, column_idents
        )
        insert_sql = sql.SQL("INSERT INTO {} ({}) SELECT {} FROM {}{}").format(
            qualified, column_idents, select_list, staging, conflict
        )

        with self.connection.cursor() as cursor:
            cursor.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(staging))
            cursor.execute(sql.SQL("CREATE TEMP TABLE {} ({})").format(staging, column_defs))
            try:
                with cursor.copy(copy_sql) as copy:
                    copy.write(buffer.getvalue())
                if if_exists == "replace" and pk_columns:
                    key_match = sql.SQL(" AND ").join(
                        sql.SQL("target.{}::text = stg.{}").format(
                            sql.Identifier(column),
                            sql.Identifier(column),
                        )
                        for column in pk_columns
                    )
                    delete_missing_sql = sql.SQL(
                        "DELETE FROM {} AS target "
                        "WHERE NOT EXISTS (SELECT 1 FROM {} AS stg WHERE {})"
                    ).format(qualified, staging, key_match)
                    cursor.execute(delete_missing_sql)
                cursor.execute(insert_sql)
            finally:
                cursor.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(staging))

        print(f"✓ Loaded {len(df)} rows into {normalized_table} (idempotent, if_exists={if_exists})")


    def query(self, query_text: str) -> pd.DataFrame:
        with self.connection.cursor() as cursor:
            cursor.execute(cast(Query, query_text))
            if cursor.description is None:
                return pd.DataFrame()
            rows = cursor.fetchall()
            columns = [description.name for description in cursor.description]
        return pd.DataFrame(rows, columns=columns)

    def table_exists(self, table_name: str) -> bool:
        normalized_table = self._normalize_identifier(table_name, "table")
        result = self.connection.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
            """,
            (self.schema, normalized_table),
        ).fetchone()
        return bool(result and result[0] > 0)

    def get_table_info(self, table_name: str) -> pd.DataFrame:
        normalized_table = self._normalize_identifier(table_name, "table")
        return self.query(
            f"""
            SELECT
                column_name,
                data_type,
                is_nullable,
                ordinal_position
            FROM information_schema.columns
            WHERE table_schema = '{self.schema}' AND table_name = '{normalized_table}'
            ORDER BY ordinal_position
            """
        )

    def get_row_count(self, table_name: str) -> int:
        normalized_table = self._normalize_identifier(table_name, "table")
        result = self.connection.execute(
            sql.SQL("SELECT COUNT(*) FROM {}").format(self._qualified_table(normalized_table))
        ).fetchone()
        if result is None:
            raise RuntimeError(f"Count query returned no rows for {normalized_table}")
        return int(result[0])

    def vacuum(self):
        for table_name in self.managed_tables:
            if self.table_exists(table_name):
                self.connection.execute(
                    sql.SQL("VACUUM ANALYZE {}").format(self._qualified_table(table_name))
                )
        print("✓ Vacuumed and analyzed managed tables")

    def export_table_to_parquet(self, table_name: str, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.query(f"SELECT * FROM {self._normalize_identifier(table_name, 'table')}").to_parquet(
            output_path,
            index=False,
        )
        print(f"✓ Exported {table_name} to {output_path}")
