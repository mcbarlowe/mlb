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
        """Create table for pitch-level data from GameFeedData with FK to games."""
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
            weather_temp DOUBLE,
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
            runner_on_third_id DOUBLE,
            FOREIGN KEY (game_pk) REFERENCES games(game_pk)
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
            stolenBasePercentage VARCHAR,
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
            atBatsPerHomeRun VARCHAR,
            popOuts INTEGER,
            lineOuts INTEGER,
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
            stolenBasePercentage VARCHAR,
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
            strikePercentage VARCHAR,
            hitBatsmen INTEGER,
            balks INTEGER,
            wildPitches INTEGER,
            pickoffs INTEGER,
            rbi INTEGER,
            gamesFinished INTEGER,
            runsScoredPer9 VARCHAR,
            homeRunsPer9 VARCHAR,
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
            gamesStarted INTEGER,
            caughtStealing INTEGER,
            stolenBases INTEGER,
            stolenBasePercentage VARCHAR,
            caughtStealingPercentage VARCHAR,
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
            latitude DOUBLE,
            longitude DOUBLE,
            elevation DOUBLE,
            azimuth_angle DOUBLE,
            timezone_id VARCHAR,
            timezone VARCHAR,
            timezone_offset DOUBLE,
            capacity DOUBLE,
            turf_type VARCHAR,
            roof_type VARCHAR,
            left_line DOUBLE,
            left_center DOUBLE,
            center DOUBLE,
            right_center DOUBLE,
            right_line DOUBLE
        )
        """
        self.connection.execute(venues_schema)
        print("✓ Created table: venues")

    def create_teams_table(self):
        """Create table for team dimension data."""
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
            spring_league_id DOUBLE,
            spring_league_name VARCHAR,
            spring_league_abbrev VARCHAR,
            parent_org_name VARCHAR,
            parent_org_id DOUBLE,
            all_star_status BOOLEAN,
            active BOOLEAN
        )
        """
        self.connection.execute(schema)
        print("✓ Created table: teams")

    def create_players_table(self):
        """Create table for player dimension data."""
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
            draft_year DOUBLE,
            mlb_debut_date VARCHAR,
            strike_zone_top DOUBLE,
            strike_zone_bottom DOUBLE
        )
        """
        self.connection.execute(schema)
        print("✓ Created table: players")

    def create_games_table(self):
        """Create table for game fact data with foreign keys to teams and venues."""
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
            attendance DOUBLE,
            first_pitch VARCHAR,
            game_duration_minutes DOUBLE,
            away_team_id INTEGER,
            away_team_wins DOUBLE,
            away_team_losses DOUBLE,
            away_team_winning_percentage VARCHAR,
            away_team_division_leader BOOLEAN,
            away_team_games_played DOUBLE,
            home_team_id INTEGER,
            home_team_wins DOUBLE,
            home_team_losses DOUBLE,
            home_team_winning_percentage VARCHAR,
            home_team_division_leader BOOLEAN,
            home_team_games_played DOUBLE,
            away_probable_pitcher_id DOUBLE,
            away_probable_pitcher_name VARCHAR,
            home_probable_pitcher_id DOUBLE,
            home_probable_pitcher_name VARCHAR,
            has_challenges BOOLEAN,
            away_reviews_remaining DOUBLE,
            away_reviews_used DOUBLE,
            home_reviews_remaining DOUBLE,
            home_reviews_used DOUBLE,
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
        """Create all tables in the database."""
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
