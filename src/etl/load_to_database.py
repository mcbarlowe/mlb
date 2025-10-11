"""
ETL script to load MLB data from JSON files into DuckDB database.

This script reads raw JSON files and processed parquet files,
transforms the data, and loads it into a DuckDB database.
"""

import json
from pathlib import Path

from tqdm import tqdm

from src.data import BoxscoreData, GameFeedData, LinescoreData
from src.database import DuckDBHandler


def load_raw_livefeeds_to_db(db_path: Path, raw_data_path: Path = None):
    """
    Load raw live feed JSON files into database.

    Args:
        db_path (Path): Path to DuckDB database
        raw_data_path (Path): Path to raw live feed JSON files
    """
    if raw_data_path is None:
        raw_data_path = Path("data/raw/livefeeds/")

    if not raw_data_path.exists():
        print(f"⚠ Warning: {raw_data_path} does not exist")
        return

    # Get all JSON files
    json_files = list(raw_data_path.glob("**/*.json"))

    if not json_files:
        print(f"⚠ Warning: No JSON files found in {raw_data_path}")
        return

    print(f"\nFound {len(json_files)} game files to process")

    # Initialize transformers
    pitch_transformer = GameFeedData()
    linescore_transformer = LinescoreData()
    boxscore_transformer = BoxscoreData()

    with DuckDBHandler(db_path) as db:
        # Ensure tables exist
        if not db.table_exists("pitches"):
            db.create_all_tables()

        # Process each game file
        for file_path in tqdm(json_files, desc="Loading games to database"):
            try:
                season = file_path.parent.stem
                game_pk = int(file_path.stem)

                # Load JSON
                with open(file_path, "r") as f:
                    game_data = json.load(f)

                # Transform and load pitch data
                pitches_df = pitch_transformer.transform(
                    game_data, game_id=game_pk, season=season
                )
                pitch_transformer.save_to_db(pitches_df, db)

                # Transform and load linescore data
                linescore_df = linescore_transformer.transform(game_data, game_pk=game_pk)
                linescore_transformer.save_to_db(linescore_df, db)

                # Transform and load boxscore data
                boxscore_data = boxscore_transformer.transform_all(game_data, game_pk=game_pk)
                for table_name, df in boxscore_data.items():
                    if not df.empty:
                        stat_type = table_name.split("_")[1]
                        boxscore_transformer.save_to_db(df, stat_type, db)

            except Exception as e:
                tqdm.write(f"✗ Error processing {file_path}: {e}")

        # Print final statistics
        print("\n=== Database Statistics ===")
        tables = ["pitches", "linescore", "batting", "pitching", "fielding"]
        for table in tables:
            if db.table_exists(table):
                count = db.get_row_count(table)
                print(f"{table}: {count:,} rows")


def load_reference_data_to_db(db_path: Path):
    """
    Load reference data into database.

    Args:
        db_path (Path): Path to DuckDB database
    """
    from src.data import ReferenceData
    from src.endpoints.event_types import EventTypes
    from src.endpoints.game_types import GameTypes
    from src.endpoints.pitch_types import PitchTypes
    from src.endpoints.positions import Positions

    print("\nLoading reference data...")

    ref_transformer = ReferenceData()

    with DuckDBHandler(db_path) as db:
        # Ensure reference tables exist
        if not db.table_exists("positions"):
            db.create_reference_tables()

        # Load positions
        try:
            positions_api = Positions()
            positions_data = positions_api.get()
            positions_df = ref_transformer.transform(positions_data)
            ref_transformer.save_to_db(positions_df, "positions", db)
            print("✓ Loaded positions")
        except Exception as e:
            print(f"✗ Error loading positions: {e}")

        # Load pitch types
        try:
            pitch_types_api = PitchTypes()
            pitch_types_data = pitch_types_api.get()
            pitch_types_df = ref_transformer.transform(pitch_types_data)
            ref_transformer.save_to_db(pitch_types_df, "pitch_types", db)
            print("✓ Loaded pitch types")
        except Exception as e:
            print(f"✗ Error loading pitch types: {e}")

        # Load event types
        try:
            event_types_api = EventTypes()
            event_types_data = event_types_api.get()
            event_types_df = ref_transformer.transform(event_types_data)
            ref_transformer.save_to_db(event_types_df, "event_types", db)
            print("✓ Loaded event types")
        except Exception as e:
            print(f"✗ Error loading event types: {e}")

        # Load game types
        try:
            game_types_api = GameTypes()
            game_types_data = game_types_api.get()
            game_types_df = ref_transformer.transform(game_types_data)
            ref_transformer.save_to_db(game_types_df, "game_types", db)
            print("✓ Loaded game types")
        except Exception as e:
            print(f"✗ Error loading game types: {e}")


def create_indexes(db_path: Path):
    """
    Create indexes on commonly queried columns for better performance.

    Args:
        db_path (Path): Path to DuckDB database
    """
    print("\nCreating indexes...")

    with DuckDBHandler(db_path) as db:
        # Indexes for pitches table
        if db.table_exists("pitches"):
            db.connection.execute("CREATE INDEX IF NOT EXISTS idx_pitches_game_pk ON pitches(game_pk)")
            db.connection.execute("CREATE INDEX IF NOT EXISTS idx_pitches_pitcher ON pitches(pitcher_id)")
            db.connection.execute("CREATE INDEX IF NOT EXISTS idx_pitches_batter ON pitches(batter_id)")
            print("✓ Created indexes on pitches table")

        # Indexes for batting table
        if db.table_exists("batting"):
            db.connection.execute("CREATE INDEX IF NOT EXISTS idx_batting_game_pk ON batting(game_pk)")
            db.connection.execute("CREATE INDEX IF NOT EXISTS idx_batting_player ON batting(player_id)")
            print("✓ Created indexes on batting table")

        # Indexes for pitching table
        if db.table_exists("pitching"):
            db.connection.execute("CREATE INDEX IF NOT EXISTS idx_pitching_game_pk ON pitching(game_pk)")
            db.connection.execute("CREATE INDEX IF NOT EXISTS idx_pitching_player ON pitching(player_id)")
            print("✓ Created indexes on pitching table")


def main():
    """
    Main ETL function to load all data into database.
    """
    db_path = Path("data/mlb.duckdb")

    print("=== MLB Data ETL to DuckDB ===")
    print(f"Database: {db_path}\n")

    # Step 1: Create database and tables
    print("Step 1: Creating database and tables...")
    with DuckDBHandler(db_path) as db:
        db.create_all_tables()

    # Step 2: Load reference data
    print("\nStep 2: Loading reference data...")
    load_reference_data_to_db(db_path)

    # Step 3: Load game data
    print("\nStep 3: Loading game data...")
    load_raw_livefeeds_to_db(db_path)

    # Step 4: Create indexes
    print("\nStep 4: Creating indexes...")
    create_indexes(db_path)

    # Step 5: Optimize database
    print("\nStep 5: Optimizing database...")
    with DuckDBHandler(db_path) as db:
        db.vacuum()

    print(f"\n✓ ETL Complete! Database ready at {db_path}")


if __name__ == "__main__":
    main()
