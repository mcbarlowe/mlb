from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

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
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                sql.Identifier(self.schema)
            )
        )
        self.connection.execute(
            sql.SQL("SET search_path TO {}, public").format(sql.Identifier(self.schema))
        )

    def reset_schema(self):
        """Drop and recreate the configured schema."""

        self.connection.execute(
            sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                sql.Identifier(self.schema)
            )
        )
        self.ensure_schema()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.connection:
            self.connection.close()

    def create_pitches_table(self):
        schema = """
        CREATE TABLE IF NOT EXISTS pitches (
            game_pk INTEGER,
            season INTEGER,
            game_type VARCHAR,
            double_header VARCHAR,
            game_number INTEGER,
            game_date VARCHAR,
            day_night VARCHAR,
            away_team_id INTEGER,
            away_team_name VARCHAR,
            home_team_id INTEGER,
            home_team_name VARCHAR,
            venue_id INTEGER,
            venue_name VARCHAR,
            weather_condition VARCHAR,
            weather_temp DOUBLE PRECISION,
            weather_wind VARCHAR,
            event VARCHAR,
            event_type VARCHAR,
            description VARCHAR,
            rbi INTEGER,
            away_score INTEGER,
            home_score INTEGER,
            is_out BOOLEAN,
            at_bat_index INTEGER,
            half_inning VARCHAR,
            inning INTEGER,
            batter_id INTEGER,
            batter_name VARCHAR,
            bat_side VARCHAR,
            pitcher_id INTEGER,
            pitcher_name VARCHAR,
            throw_side VARCHAR,
            pitch_number INTEGER,
            pitch_call_description VARCHAR,
            is_in_play BOOLEAN,
            is_strike BOOLEAN,
            is_ball BOOLEAN,
            pitch_type VARCHAR,
            pitch_type_code VARCHAR,
            count_after_pitch VARCHAR,
            outs INTEGER,
            play_id VARCHAR,
            pitch_start_time VARCHAR,
            pitch_end_time VARCHAR,
            pitch_start_speed DOUBLE PRECISION,
            pitch_end_speed DOUBLE PRECISION,
            pitch_strike_zone_top DOUBLE PRECISION,
            pitch_strike_zone_bottom DOUBLE PRECISION,
            pitch_zone DOUBLE PRECISION,
            ay DOUBLE PRECISION,
            az DOUBLE PRECISION,
            pfxX DOUBLE PRECISION,
            pfxZ DOUBLE PRECISION,
            px DOUBLE PRECISION,
            pz DOUBLE PRECISION,
            vx0 DOUBLE PRECISION,
            vy0 DOUBLE PRECISION,
            vz0 DOUBLE PRECISION,
            x DOUBLE PRECISION,
            y DOUBLE PRECISION,
            x0 DOUBLE PRECISION,
            y0 DOUBLE PRECISION,
            z0 DOUBLE PRECISION,
            ax DOUBLE PRECISION,
            break_angle DOUBLE PRECISION,
            break_length DOUBLE PRECISION,
            break_y DOUBLE PRECISION,
            break_vertical DOUBLE PRECISION,
            break_vertical_induced DOUBLE PRECISION,
            break_horizontal DOUBLE PRECISION,
            spin_rate DOUBLE PRECISION,
            spin_direction VARCHAR,
            is_runner_on_first BOOLEAN,
            runner_on_first_id DOUBLE PRECISION,
            is_runner_on_second BOOLEAN,
            runner_on_second_id DOUBLE PRECISION,
            is_runner_on_third BOOLEAN,
            runner_on_third_id DOUBLE PRECISION,
            FOREIGN KEY (game_pk) REFERENCES games(game_pk)
        )
        """
        self.connection.execute(schema)
        print("✓ Created table: pitches")

    def create_linescore_table(self):
        schema = """
        CREATE TABLE IF NOT EXISTS linescore (
            game_pk INTEGER,
            inning INTEGER,
            inning_ordinal VARCHAR,
            team_type VARCHAR,
            runs INTEGER,
            hits INTEGER,
            errors INTEGER,
            left_on_base INTEGER,
            current_inning INTEGER,
            inning_state VARCHAR,
            inning_half VARCHAR,
            scheduled_innings INTEGER
        )
        """
        self.connection.execute(schema)
        print("✓ Created table: linescore")

    def create_batting_table(self):
        schema = """
        CREATE TABLE IF NOT EXISTS batting (
            game_pk INTEGER,
            team_type VARCHAR,
            player_id INTEGER,
            player_name VARCHAR,
            jersey_number VARCHAR,
            position_code VARCHAR,
            position_name VARCHAR,
            position_abbrev VARCHAR,
            batting_order VARCHAR,
            summary VARCHAR,
            gamesPlayed INTEGER,
            flyOuts INTEGER,
            groundOuts INTEGER,
            airOuts INTEGER,
            runs INTEGER,
            doubles INTEGER,
            triples INTEGER,
            homeRuns INTEGER,
            strikeOuts INTEGER,
            baseOnBalls INTEGER,
            intentionalWalks INTEGER,
            hits INTEGER,
            hitByPitch INTEGER,
            atBats INTEGER,
            caughtStealing INTEGER,
            stolenBases INTEGER,
            stolenBasePercentage NUMERIC,
            groundIntoDoublePlay INTEGER,
            groundIntoTriplePlay INTEGER,
            plateAppearances INTEGER,
            totalBases INTEGER,
            rbi INTEGER,
            leftOnBase INTEGER,
            sacBunts INTEGER,
            sacFlies INTEGER,
            catchersInterference INTEGER,
            pickoffs INTEGER,
            atBatsPerHomeRun NUMERIC,
            popOuts INTEGER,
            lineOuts INTEGER,
            note VARCHAR
        )
        """
        self.connection.execute(schema)
        print("✓ Created table: batting")

    def create_pitching_table(self):
        schema = """
        CREATE TABLE IF NOT EXISTS pitching (
            game_pk INTEGER,
            team_type VARCHAR,
            player_id INTEGER,
            player_name VARCHAR,
            jersey_number VARCHAR,
            position_code VARCHAR,
            position_name VARCHAR,
            position_abbrev VARCHAR,
            summary VARCHAR,
            gamesPlayed INTEGER,
            gamesStarted INTEGER,
            flyOuts INTEGER,
            groundOuts INTEGER,
            airOuts INTEGER,
            runs INTEGER,
            doubles INTEGER,
            triples INTEGER,
            homeRuns INTEGER,
            strikeOuts INTEGER,
            baseOnBalls INTEGER,
            intentionalWalks INTEGER,
            hits INTEGER,
            hitByPitch INTEGER,
            atBats INTEGER,
            caughtStealing INTEGER,
            stolenBases INTEGER,
            stolenBasePercentage NUMERIC,
            numberOfPitches INTEGER,
            inningsPitched VARCHAR,
            wins INTEGER,
            losses INTEGER,
            saves INTEGER,
            saveOpportunities INTEGER,
            holds INTEGER,
            blownSaves INTEGER,
            earnedRuns INTEGER,
            battersFaced INTEGER,
            outs INTEGER,
            gamesPitched INTEGER,
            completeGames INTEGER,
            shutouts INTEGER,
            pitchesThrown INTEGER,
            balls INTEGER,
            strikes INTEGER,
            strikePercentage NUMERIC,
            hitBatsmen INTEGER,
            balks INTEGER,
            wildPitches INTEGER,
            pickoffs INTEGER,
            rbi INTEGER,
            gamesFinished INTEGER,
            runsScoredPer9 NUMERIC,
            homeRunsPer9 NUMERIC,
            inheritedRunners INTEGER,
            inheritedRunnersScored INTEGER,
            catchersInterference INTEGER,
            sacBunts INTEGER,
            sacFlies INTEGER,
            passedBall INTEGER,
            popOuts INTEGER,
            lineOuts INTEGER,
            note VARCHAR
        )
        """
        self.connection.execute(schema)
        print("✓ Created table: pitching")

    def create_fielding_table(self):
        schema = """
        CREATE TABLE IF NOT EXISTS fielding (
            game_pk INTEGER,
            team_type VARCHAR,
            player_id INTEGER,
            player_name VARCHAR,
            jersey_number VARCHAR,
            position_code VARCHAR,
            position_name VARCHAR,
            position_abbrev VARCHAR,
            gamesStarted INTEGER,
            caughtStealing INTEGER,
            stolenBases INTEGER,
            stolenBasePercentage NUMERIC,
            caughtStealingPercentage NUMERIC,
            assists INTEGER,
            putOuts INTEGER,
            errors INTEGER,
            chances INTEGER,
            fielding VARCHAR,
            passedBall INTEGER,
            pickoffs INTEGER
        )
        """
        self.connection.execute(schema)
        print("✓ Created table: fielding")

    def create_reference_tables(self):
        positions_schema = """
        CREATE TABLE IF NOT EXISTS positions (
            code VARCHAR PRIMARY KEY,
            name VARCHAR,
            type VARCHAR,
            abbreviation VARCHAR
        )
        """
        self.connection.execute(positions_schema)
        print("✓ Created table: positions")

        pitch_types_schema = """
        CREATE TABLE IF NOT EXISTS pitch_types (
            code VARCHAR PRIMARY KEY,
            description VARCHAR
        )
        """
        self.connection.execute(pitch_types_schema)
        print("✓ Created table: pitch_types")

        event_types_schema = """
        CREATE TABLE IF NOT EXISTS event_types (
            code VARCHAR PRIMARY KEY,
            description VARCHAR
        )
        """
        self.connection.execute(event_types_schema)
        print("✓ Created table: event_types")

        game_types_schema = """
        CREATE TABLE IF NOT EXISTS game_types (
            id VARCHAR PRIMARY KEY,
            description VARCHAR
        )
        """
        self.connection.execute(game_types_schema)
        print("✓ Created table: game_types")

        venues_schema = """
        CREATE TABLE IF NOT EXISTS venues (
            venue_id INTEGER PRIMARY KEY,
            venue_name VARCHAR,
            venue_link VARCHAR,
            active BOOLEAN,
            season VARCHAR,
            address VARCHAR,
            city VARCHAR,
            state VARCHAR,
            state_abbrev VARCHAR,
            country VARCHAR,
            postal_code VARCHAR,
            latitude DOUBLE PRECISION,
            longitude DOUBLE PRECISION,
            elevation DOUBLE PRECISION,
            azimuth_angle DOUBLE PRECISION,
            timezone_id VARCHAR,
            timezone VARCHAR,
            timezone_offset DOUBLE PRECISION,
            capacity DOUBLE PRECISION,
            turf_type VARCHAR,
            roof_type VARCHAR,
            left_line DOUBLE PRECISION,
            left_center DOUBLE PRECISION,
            center DOUBLE PRECISION,
            right_center DOUBLE PRECISION,
            right_line DOUBLE PRECISION
        )
        """
        self.connection.execute(venues_schema)
        print("✓ Created table: venues")

    def create_teams_table(self):
        schema = """
        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY,
            team_name VARCHAR,
            team_code VARCHAR,
            file_code VARCHAR,
            abbreviation VARCHAR,
            team_name_short VARCHAR,
            location_name VARCHAR,
            first_year_of_play INTEGER,
            league_id INTEGER,
            league_name VARCHAR,
            division_id INTEGER,
            division_name VARCHAR,
            sport_id INTEGER,
            sport_name VARCHAR,
            venue_id INTEGER,
            venue_name VARCHAR,
            spring_league_id DOUBLE PRECISION,
            spring_league_name VARCHAR,
            spring_league_abbrev VARCHAR,
            parent_org_name VARCHAR,
            parent_org_id DOUBLE PRECISION,
            all_star_status BOOLEAN,
            active BOOLEAN
        )
        """
        self.connection.execute(schema)
        print("✓ Created table: teams")

    def create_players_table(self):
        schema = """
        CREATE TABLE IF NOT EXISTS players (
            player_id INTEGER PRIMARY KEY,
            full_name VARCHAR,
            first_name VARCHAR,
            last_name VARCHAR,
            middle_name VARCHAR,
            use_name VARCHAR,
            boxscore_name VARCHAR,
            nick_name VARCHAR,
            name_first_last VARCHAR,
            name_slug VARCHAR,
            first_last_name VARCHAR,
            last_first_name VARCHAR,
            last_init_name VARCHAR,
            init_last_name VARCHAR,
            full_fml_name VARCHAR,
            full_lfm_name VARCHAR,
            primary_number VARCHAR,
            birth_date VARCHAR,
            current_age INTEGER,
            birth_city VARCHAR,
            birth_state_province VARCHAR,
            birth_country VARCHAR,
            height VARCHAR,
            weight INTEGER,
            active BOOLEAN,
            primary_position_code VARCHAR,
            primary_position_name VARCHAR,
            primary_position_type VARCHAR,
            primary_position_abbrev VARCHAR,
            bat_side_code VARCHAR,
            bat_side_description VARCHAR,
            pitch_hand_code VARCHAR,
            pitch_hand_description VARCHAR,
            draft_year DOUBLE PRECISION,
            mlb_debut_date VARCHAR,
            strike_zone_top DOUBLE PRECISION,
            strike_zone_bottom DOUBLE PRECISION
        )
        """
        self.connection.execute(schema)
        print("✓ Created table: players")

    def create_games_table(self):
        schema = """
        CREATE TABLE IF NOT EXISTS games (
            game_pk INTEGER PRIMARY KEY,
            game_id VARCHAR,
            season VARCHAR,
            season_display VARCHAR,
            game_type VARCHAR,
            gameday_type VARCHAR,
            game_number INTEGER,
            double_header VARCHAR,
            tiebreaker VARCHAR,
            calendar_event_id VARCHAR,
            game_date VARCHAR,
            original_date VARCHAR,
            game_datetime VARCHAR,
            game_time VARCHAR,
            ampm VARCHAR,
            day_night VARCHAR,
            abstract_game_state VARCHAR,
            coded_game_state VARCHAR,
            detailed_state VARCHAR,
            status_code VARCHAR,
            start_time_tbd BOOLEAN,
            abstract_game_code VARCHAR,
            venue_id INTEGER,
            weather_condition VARCHAR,
            weather_temp VARCHAR,
            weather_wind VARCHAR,
            attendance DOUBLE PRECISION,
            first_pitch VARCHAR,
            game_duration_minutes DOUBLE PRECISION,
            away_team_id INTEGER,
            away_team_wins DOUBLE PRECISION,
            away_team_losses DOUBLE PRECISION,
            away_team_winning_percentage VARCHAR,
            away_team_division_leader BOOLEAN,
            away_team_games_played DOUBLE PRECISION,
            home_team_id INTEGER,
            home_team_wins DOUBLE PRECISION,
            home_team_losses DOUBLE PRECISION,
            home_team_winning_percentage VARCHAR,
            home_team_division_leader BOOLEAN,
            home_team_games_played DOUBLE PRECISION,
            away_probable_pitcher_id DOUBLE PRECISION,
            away_probable_pitcher_name VARCHAR,
            home_probable_pitcher_id DOUBLE PRECISION,
            home_probable_pitcher_name VARCHAR,
            has_challenges BOOLEAN,
            away_reviews_remaining DOUBLE PRECISION,
            away_reviews_used DOUBLE PRECISION,
            home_reviews_remaining DOUBLE PRECISION,
            home_reviews_used DOUBLE PRECISION,
            no_hitter BOOLEAN,
            perfect_game BOOLEAN,
            away_team_no_hitter BOOLEAN,
            away_team_perfect_game BOOLEAN,
            home_team_no_hitter BOOLEAN,
            home_team_perfect_game BOOLEAN,
            FOREIGN KEY (venue_id) REFERENCES venues(venue_id),
            FOREIGN KEY (away_team_id) REFERENCES teams(team_id),
            FOREIGN KEY (home_team_id) REFERENCES teams(team_id)
        )
        """
        self.connection.execute(schema)
        print("✓ Created table: games")

    def create_all_tables(self):
        print("\nCreating MLB database tables...")
        self.create_teams_table()
        self.create_players_table()
        self.create_reference_tables()
        self.create_games_table()
        self.create_pitches_table()
        self.create_linescore_table()
        self.create_batting_table()
        self.create_pitching_table()
        self.create_fielding_table()
        print("\n✓ All tables created successfully!")

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

    def insert_dataframe(
        self, df: pd.DataFrame, table_name: str, if_exists: str = "append"
    ):
        if df.empty:
            print(f"⚠ Warning: DataFrame is empty, skipping insert to {table_name}")
            return

        normalized_table = self._normalize_identifier(table_name, "table")
        if if_exists == "replace":
            self.connection.execute(
                sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE").format(
                    self._qualified_table(normalized_table)
                )
            )
        elif (
            if_exists == "fail"
            and self.table_exists(normalized_table)
            and self.get_row_count(normalized_table) > 0
        ):
            raise ValueError(f"Table {normalized_table} already contains data")
        elif if_exists != "append":
            raise ValueError(f"Unsupported if_exists mode: {if_exists}")

        normalized_columns = [
            self._normalize_identifier(column, "column") for column in df.columns
        ]
        df_to_load = df.copy()
        integer_column_types = {"smallint", "integer", "bigint"}
        column_types = self._column_types(normalized_table)
        for original_column, normalized_column in zip(df.columns, normalized_columns):
            if column_types.get(normalized_column) in integer_column_types:
                df_to_load[original_column] = df_to_load[original_column].astype(
                    "Int64"
                )

        buffer = io.StringIO()
        df_to_load.to_csv(buffer, index=False, header=False, na_rep=r"\N")
        buffer.seek(0)

        copy_sql = sql.SQL(
            "COPY {} ({}) FROM STDIN WITH (FORMAT CSV, NULL '\\N')"
        ).format(
            self._qualified_table(normalized_table),
            sql.SQL(", ").join(sql.Identifier(column) for column in normalized_columns),
        )

        with self.connection.cursor() as cursor, cursor.copy(copy_sql) as copy:
            copy.write(buffer.getvalue())

        print(f"✓ Inserted {len(df)} rows into {normalized_table}")

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
            sql.SQL("SELECT COUNT(*) FROM {}").format(
                self._qualified_table(normalized_table)
            )
        ).fetchone()
        if result is None:
            raise RuntimeError(f"Count query returned no rows for {normalized_table}")
        return int(result[0])

    def vacuum(self):
        for table_name in self.managed_tables:
            if self.table_exists(table_name):
                self.connection.execute(
                    sql.SQL("VACUUM ANALYZE {}").format(
                        self._qualified_table(table_name)
                    )
                )
        print("✓ Vacuumed and analyzed managed tables")

    def export_table_to_parquet(self, table_name: str, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.query(
            f"SELECT * FROM {self._normalize_identifier(table_name, 'table')}"
        ).to_parquet(
            output_path,
            index=False,
        )
        print(f"✓ Exported {table_name} to {output_path}")
