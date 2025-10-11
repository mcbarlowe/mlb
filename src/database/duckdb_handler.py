from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


class DuckDBHandler:
    """
    Handler for DuckDB database operations.

    Manages database connections, table creation, and data insertion
    for MLB transformed data.
    """

    def __init__(self, db_path: Path = None):
        """
        Initialize DuckDB handler.

        Args:
            db_path (Path, optional): Path to database file. If None, uses in-memory database.
        """
        if db_path:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = duckdb.connect(str(self.db_path))
        else:
            self.db_path = None
            self.connection = duckdb.connect(":memory:")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()

    def create_pitches_table(self):
        """Create table for pitch-level data from GameFeedData."""
        schema = """
        CREATE TABLE IF NOT EXISTS pitches (
            game_pk INTEGER,
            season INTEGER,
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
            pitch_start_speed DOUBLE,
            pitch_end_speed DOUBLE,
            pitch_strike_zone_top DOUBLE,
            pitch_strike_zone_bottom DOUBLE,
            pitch_zone DOUBLE,
            ay DOUBLE,
            az DOUBLE,
            pfxX DOUBLE,
            pfxZ DOUBLE,
            px DOUBLE,
            pz DOUBLE,
            vx0 DOUBLE,
            vy0 DOUBLE,
            vz0 DOUBLE,
            x DOUBLE,
            y DOUBLE,
            x0 DOUBLE,
            y0 DOUBLE,
            z0 DOUBLE,
            ax DOUBLE,
            break_angle DOUBLE,
            break_length DOUBLE,
            break_y DOUBLE,
            break_vertical DOUBLE,
            break_vertical_induced DOUBLE,
            break_horizontal DOUBLE,
            spin_rate DOUBLE,
            spin_direction VARCHAR,
            is_runner_on_first BOOLEAN,
            runner_on_first_id DOUBLE,
            is_runner_on_second BOOLEAN,
            runner_on_second_id DOUBLE,
            is_runner_on_third BOOLEAN,
            runner_on_third_id DOUBLE
        )
        """
        self.connection.execute(schema)
        print("✓ Created table: pitches")

    def create_linescore_table(self):
        """Create table for linescore data."""
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
        """Create table for batting statistics."""
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
            gamesPlayed INTEGER,
            flyOuts INTEGER,
            groundOuts INTEGER,
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
            groundIntoDoublePlay INTEGER,
            groundIntoTriplePlay INTEGER,
            totalBases INTEGER,
            rbi INTEGER,
            leftOnBase INTEGER,
            sacBunts INTEGER,
            sacFlies INTEGER,
            catchersInterference INTEGER,
            pickoffs INTEGER,
            note VARCHAR
        )
        """
        self.connection.execute(schema)
        print("✓ Created table: batting")

    def create_pitching_table(self):
        """Create table for pitching statistics."""
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
            gamesPlayed INTEGER,
            gamesStarted INTEGER,
            flyOuts INTEGER,
            groundOuts INTEGER,
            runs INTEGER,
            doubles INTEGER,
            triples INTEGER,
            homeRuns INTEGER,
            strikeOuts INTEGER,
            baseOnBalls INTEGER,
            intentionalWalks INTEGER,
            hits INTEGER,
            atBats INTEGER,
            caughtStealing INTEGER,
            stolenBases INTEGER,
            numberOfPitches INTEGER,
            inningsPitched VARCHAR,
            wins INTEGER,
            losses INTEGER,
            saves INTEGER,
            saveOpportunites INTEGER,
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
            hitBatsmen INTEGER,
            wildPitches INTEGER,
            pickoffs INTEGER,
            airOuts INTEGER,
            rbi INTEGER,
            gamesFinished INTEGER,
            inheritedRunners INTEGER,
            inheritedRunnersScored INTEGER,
            catchersInterference INTEGER,
            sacBunts INTEGER,
            sacFlies INTEGER,
            note VARCHAR
        )
        """
        self.connection.execute(schema)
        print("✓ Created table: pitching")

    def create_fielding_table(self):
        """Create table for fielding statistics."""
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
            assists INTEGER,
            putOuts INTEGER,
            errors INTEGER,
            chances INTEGER,
            caughtStealing INTEGER,
            passedBall INTEGER,
            stolenBases INTEGER,
            stolenBasePercentage DOUBLE,
            pickoffs INTEGER
        )
        """
        self.connection.execute(schema)
        print("✓ Created table: fielding")

    def create_reference_tables(self):
        """Create tables for reference data."""
        # Positions
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

        # Pitch Types
        pitch_types_schema = """
        CREATE TABLE IF NOT EXISTS pitch_types (
            code VARCHAR PRIMARY KEY,
            description VARCHAR
        )
        """
        self.connection.execute(pitch_types_schema)
        print("✓ Created table: pitch_types")

        # Event Types
        event_types_schema = """
        CREATE TABLE IF NOT EXISTS event_types (
            code VARCHAR PRIMARY KEY,
            description VARCHAR
        )
        """
        self.connection.execute(event_types_schema)
        print("✓ Created table: event_types")

        # Game Types
        game_types_schema = """
        CREATE TABLE IF NOT EXISTS game_types (
            id VARCHAR PRIMARY KEY,
            description VARCHAR
        )
        """
        self.connection.execute(game_types_schema)
        print("✓ Created table: game_types")

        # Venues
        venues_schema = """
        CREATE TABLE IF NOT EXISTS venues (
            id INTEGER PRIMARY KEY,
            name VARCHAR,
            city VARCHAR,
            state VARCHAR,
            stateAbbrev VARCHAR,
            latitude DOUBLE,
            longitude DOUBLE,
            timeZone_id VARCHAR,
            timeZone_offset VARCHAR,
            timeZone_tz VARCHAR
        )
        """
        self.connection.execute(venues_schema)
        print("✓ Created table: venues")

    def create_all_tables(self):
        """Create all tables in the database."""
        print("\nCreating MLB database tables...")
        self.create_pitches_table()
        self.create_linescore_table()
        self.create_batting_table()
        self.create_pitching_table()
        self.create_fielding_table()
        self.create_reference_tables()
        print("\n✓ All tables created successfully!")

    def insert_dataframe(self, df: pd.DataFrame, table_name: str,
                        if_exists: str = "append"):
        """
        Insert DataFrame into specified table.

        Args:
            df (pd.DataFrame): DataFrame to insert
            table_name (str): Target table name
            if_exists (str): How to behave if table exists ('append', 'replace', 'fail')
        """
        if df.empty:
            print(f"⚠ Warning: DataFrame is empty, skipping insert to {table_name}")
            return

        # Use DuckDB's native DataFrame insertion
        self.connection.execute(f"INSERT INTO {table_name} SELECT * FROM df")
        print(f"✓ Inserted {len(df)} rows into {table_name}")

    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.

        Args:
            sql (str): SQL query to execute

        Returns:
            pd.DataFrame: Query results
        """
        return self.connection.execute(sql).fetchdf()

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name (str): Name of the table

        Returns:
            bool: True if table exists
        """
        result = self.connection.execute(
            f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
        ).fetchone()
        return result[0] > 0

    def get_table_info(self, table_name: str) -> pd.DataFrame:
        """
        Get schema information for a table.

        Args:
            table_name (str): Name of the table

        Returns:
            pd.DataFrame: Table schema information
        """
        return self.connection.execute(f"DESCRIBE {table_name}").fetchdf()

    def get_row_count(self, table_name: str) -> int:
        """
        Get row count for a table.

        Args:
            table_name (str): Name of the table

        Returns:
            int: Number of rows
        """
        result = self.connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        return result[0]

    def vacuum(self):
        """Optimize database by reclaiming space."""
        self.connection.execute("VACUUM")
        print("✓ Database optimized")

    def export_table_to_parquet(self, table_name: str, output_path: Path):
        """
        Export table to Parquet file.

        Args:
            table_name (str): Table to export
            output_path (Path): Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection.execute(
            f"COPY {table_name} TO '{output_path}' (FORMAT PARQUET)"
        )
        print(f"✓ Exported {table_name} to {output_path}")
