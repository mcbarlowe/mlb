"""
Example usage of the PostgreSQL-backed MLB database.

This module demonstrates how to:
1. Create PostgreSQL tables for the MLB schema
2. Load data from the MLB API into the database
3. Query the database for analysis
"""

from pathlib import Path

from src.data import (
    BoxscoreData,
    GameFeedData,
    LinescoreData,
    PlayerData,
    ReferenceData,
    TeamData,
)
from src.database import PostgresConfig, PostgresHandler
from src.endpoints.game_feed import GameFeed
from src.endpoints.pitch_types import PitchTypes
from src.endpoints.positions import Positions


def default_db_config() -> PostgresConfig:
    """Return the repository's default local PostgreSQL config."""

    return PostgresConfig.from_env()


def example_create_database():
    """Example: create all PostgreSQL tables in the configured schema."""

    db_config = default_db_config()

    with PostgresHandler(db_config) as db:
        db.create_all_tables()

    print(f"\n✓ Database schema ready at {db_config.describe()}")


def example_load_game_data():
    """Example: fetch game data and load it with supporting dimensions."""

    db_config = default_db_config()
    game_pk = 716828

    game_feed = GameFeed()
    game_data = game_feed.get(game_pk=game_pk)

    with PostgresHandler(db_config) as db:
        if not db.table_exists("pitches"):
            db.create_all_tables()

        team_transformer = TeamData()
        teams_df = team_transformer.transform(game_data)
        team_transformer.save_to_db(teams_df, db)

        player_transformer = PlayerData()
        players_df = player_transformer.transform(game_data)
        player_transformer.save_to_db(players_df, db)

        pitch_transformer = GameFeedData()
        pitches_df = pitch_transformer.transform(game_data, game_id=game_pk, season=2024)
        pitch_transformer.save_to_db(pitches_df, db)

        linescore_transformer = LinescoreData()
        linescore_df = linescore_transformer.transform(game_data, game_pk=game_pk)
        linescore_transformer.save_to_db(linescore_df, db)

        boxscore_transformer = BoxscoreData()
        boxscore_data = boxscore_transformer.transform_all(game_data, game_pk=game_pk)
        for table_name, df in boxscore_data.items():
            if not df.empty:
                stat_type = table_name.split("_")[1]
                boxscore_transformer.save_to_db(df, stat_type, db)

    print(f"\n✓ Loaded game {game_pk} into {db_config.describe()}")


def example_load_reference_data():
    """Example: load reference data into the configured schema."""

    db_config = default_db_config()

    with PostgresHandler(db_config) as db:
        if not db.table_exists("positions"):
            db.create_reference_tables()

        ref_transformer = ReferenceData()

        positions_api = Positions()
        positions_data = positions_api.get()
        positions_df = ref_transformer.transform(positions_data)
        ref_transformer.save_to_db(positions_df, "positions", db)

        pitch_types_api = PitchTypes()
        pitch_types_data = pitch_types_api.get()
        pitch_types_df = ref_transformer.transform(pitch_types_data)
        ref_transformer.save_to_db(pitch_types_df, "pitch_types", db)

    print(f"\n✓ Loaded reference data into {db_config.describe()}")


def example_query_database():
    """Example: query the configured PostgreSQL schema for analysis."""

    db_config = default_db_config()

    with PostgresHandler(db_config) as db:
        print("\n=== Pitches by Type ===")
        query = """
        SELECT
            pitch_type,
            COUNT(*) as pitch_count,
            ROUND(AVG(pitch_start_speed), 2) as avg_velocity
        FROM pitches
        WHERE pitch_type IS NOT NULL
        GROUP BY pitch_type
        ORDER BY pitch_count DESC
        """
        result = db.query(query)
        print(result.to_string())

        print("\n=== Top 10 Batters by Hits ===")
        query = """
        SELECT
            player_name,
            SUM(hits) as total_hits,
            SUM(atBats) as total_at_bats,
            ROUND(CAST(SUM(hits) AS FLOAT) / NULLIF(SUM(atBats), 0), 3) as batting_avg
        FROM batting
        GROUP BY player_name
        ORDER BY total_hits DESC
        LIMIT 10
        """
        result = db.query(query)
        print(result.to_string())

        print("\n=== Top 10 Pitchers by Strikeouts ===")
        query = """
        SELECT
            player_name,
            SUM(strikeOuts) as total_strikeouts,
            SUM(CAST(inningsPitched AS FLOAT)) as total_innings,
            ROUND(SUM(strikeOuts) * 9.0 / NULLIF(SUM(CAST(inningsPitched AS FLOAT)), 0), 2) as k_per_9
        FROM pitching
        GROUP BY player_name
        ORDER BY total_strikeouts DESC
        LIMIT 10
        """
        result = db.query(query)
        print(result.to_string())

        print("\n=== Pitch Location Summary ===")
        query = """
        SELECT
            pitch_zone,
            COUNT(*) as pitch_count,
            SUM(CASE WHEN is_strike THEN 1 ELSE 0 END) as strikes,
            ROUND(100.0 * SUM(CASE WHEN is_strike THEN 1 ELSE 0 END) / COUNT(*), 1) as strike_pct
        FROM pitches
        WHERE pitch_zone IS NOT NULL
        GROUP BY pitch_zone
        ORDER BY pitch_zone
        """
        result = db.query(query)
        print(result.to_string())


def example_query_with_dimensions():
    """Example: query the database using dimension-table joins."""

    db_config = default_db_config()

    with PostgresHandler(db_config) as db:
        print("\n=== Pitches with Player Details ===")
        query = """
        SELECT
            p.game_pk,
            p.inning,
            p.half_inning,
            batter.full_name as batter_name,
            batter.bat_side_code,
            pitcher.full_name as pitcher_name,
            pitcher.pitch_hand_code,
            p.pitch_type,
            p.pitch_start_speed,
            p.pitch_call_description
        FROM pitches p
        JOIN players batter ON p.batter_id = batter.player_id
        JOIN players pitcher ON p.pitcher_id = pitcher.player_id
        LIMIT 10
        """
        result = db.query(query)
        print(result.to_string())

        print("\n=== Games by Team Matchup ===")
        query = """
        SELECT DISTINCT
            p.game_pk,
            p.game_date,
            away.team_name as away_team,
            home.team_name as home_team,
            p.venue_name,
            p.weather_condition
        FROM pitches p
        JOIN teams away ON p.away_team_id = away.team_id
        JOIN teams home ON p.home_team_id = home.team_id
        ORDER BY p.game_date DESC
        LIMIT 10
        """
        result = db.query(query)
        print(result.to_string())

        print("\n=== Top Pitchers by Average Velocity ===")
        query = """
        SELECT
            pl.full_name,
            pl.primary_position_name,
            COUNT(*) as total_pitches,
            ROUND(AVG(p.pitch_start_speed), 2) as avg_velocity,
            MAX(p.pitch_start_speed) as max_velocity
        FROM pitches p
        JOIN players pl ON p.pitcher_id = pl.player_id
        WHERE p.pitch_start_speed IS NOT NULL
        GROUP BY pl.player_id, pl.full_name, pl.primary_position_name
        HAVING COUNT(*) >= 10
        ORDER BY avg_velocity DESC
        LIMIT 10
        """
        result = db.query(query)
        print(result.to_string())


def example_database_info():
    """Example: inspect row counts and table metadata."""

    db_config = default_db_config()

    with PostgresHandler(db_config) as db:
        tables = ["teams", "players", "pitches", "linescore", "batting", "pitching", "fielding"]

        print("\n=== Database Statistics ===")
        for table in tables:
            if db.table_exists(table):
                count = db.get_row_count(table)
                print(f"{table}: {count:,} rows")
            else:
                print(f"{table}: Table does not exist")

        print("\n=== Pitches Table Schema ===")
        if db.table_exists("pitches"):
            schema = db.get_table_info("pitches")
            print(schema.to_string())


def example_batch_load_games():
    """Example: load multiple games into the configured schema."""

    db_config = default_db_config()
    game_pks = [716828, 716829, 716830]

    game_feed = GameFeed()
    pitch_transformer = GameFeedData()

    with PostgresHandler(db_config) as db:
        if not db.table_exists("pitches"):
            db.create_all_tables()

        for game_pk in game_pks:
            try:
                print(f"\nProcessing game {game_pk}...")

                game_data = game_feed.get(game_pk=game_pk)
                season = int(game_data["gameData"]["game"]["season"])
                pitches_df = pitch_transformer.transform(game_data, game_id=game_pk, season=season)
                pitch_transformer.save_to_db(pitches_df, db)

                print(f"✓ Loaded {len(pitches_df)} pitches from game {game_pk}")

            except Exception as e:
                print(f"✗ Error processing game {game_pk}: {e}")

        total_pitches = db.get_row_count("pitches")
        print(f"\n✓ Total pitches in database: {total_pitches:,}")


def example_export_to_parquet():
    """Example: export PostgreSQL tables back to Parquet files."""

    db_config = default_db_config()
    export_dir = Path("data/exports")

    with PostgresHandler(db_config) as db:
        tables = ["pitches", "linescore", "batting", "pitching", "fielding"]

        for table in tables:
            if db.table_exists(table):
                output_path = export_dir / f"{table}.parquet"
                db.export_table_to_parquet(table, output_path)

    print(f"\n✓ Exported tables to {export_dir}")


if __name__ == "__main__":
    print("=== PostgreSQL Database Examples ===\n")
    print("Uncomment the function calls below to run examples:\n")

    # Step 1: Create database with tables
    # example_create_database()

    # Step 2: Load reference data
    # example_load_reference_data()

    # Step 3: Load game data (includes dimension tables)
    # example_load_game_data()

    # Step 4: Load multiple games
    # example_batch_load_games()

    # Step 5: Query the database
    # example_query_database()

    # Step 6: Query with dimension joins
    # example_query_with_dimensions()

    # Step 7: Show database info
    # example_database_info()

    # Step 8: Export to Parquet
    # example_export_to_parquet()

    print("\nUncomment the function calls in __main__ to run examples.")
