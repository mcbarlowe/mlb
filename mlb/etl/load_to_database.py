"""
ETL script to load MLB data from JSON files into a local PostgreSQL schema.

This script reads raw JSON files, transforms the data, and loads it into the
configured PostgreSQL database.

Uses async methods where applicable for faster API fetches.
"""

import asyncio
import json
from pathlib import Path

from tqdm import tqdm

from mlb.data import BoxscoreData, GameFeedData, LinescoreData, PlayerData, TeamData
from mlb.data.game_data import GameData
from mlb.data.venue_data import VenueData
from mlb.database import PostgresConfig, PostgresHandler


def load_raw_livefeeds_to_db(db_config: PostgresConfig, raw_data_path: Path | None = None):
    """Load raw live feed JSON files into the configured PostgreSQL schema."""

    if raw_data_path is None:
        raw_data_path = Path("data/raw/livefeeds/")

    if not raw_data_path.exists():
        print(f"Warning: {raw_data_path} does not exist")
        return

    json_files = list(raw_data_path.glob("**/*.json"))
    if not json_files:
        print(f"Warning: No JSON files found in {raw_data_path}")
        return

    print(f"\nFound {len(json_files)} game files to process")

    pitch_transformer = GameFeedData()
    linescore_transformer = LinescoreData()
    boxscore_transformer = BoxscoreData()
    team_transformer = TeamData()
    venue_transformer = VenueData()
    game_transformer = GameData()
    player_transformer = PlayerData()

    seen_teams = set()
    seen_venues = set()
    seen_players = set()

    with PostgresHandler(db_config) as db:
        if not db.table_exists("pitches"):
            db.create_all_tables()

        for file_path in tqdm(json_files, desc="Loading games to database"):
            try:
                season = int(file_path.parent.stem)
                game_pk = int(file_path.stem)

                with open(file_path, "r") as f:
                    game_data = json.load(f)

                teams_df = team_transformer.transform(game_data)
                if not teams_df.empty:
                    new_teams_df = teams_df[~teams_df["team_id"].isin(seen_teams)]
                    if not new_teams_df.empty:
                        team_transformer.save_to_db(new_teams_df, db)
                        seen_teams.update(new_teams_df["team_id"].tolist())

                venue_df = venue_transformer.transform(game_data)
                if not venue_df.empty:
                    venue_id = venue_df.iloc[0]["venue_id"]
                    if venue_id not in seen_venues:
                        venue_transformer.save_to_db(venue_df, db)
                        seen_venues.add(venue_id)

                players_df = player_transformer.transform(game_data)
                if not players_df.empty:
                    new_players_df = players_df[~players_df["player_id"].isin(seen_players)]
                    if not new_players_df.empty:
                        player_transformer.save_to_db(new_players_df, db)
                        seen_players.update(new_players_df["player_id"].tolist())

                game_df = game_transformer.transform(game_data)
                if not game_df.empty:
                    game_transformer.save_to_db(game_df, db)

                pitches_df = pitch_transformer.transform(
                    game_data,
                    game_id=game_pk,
                    season=season,
                )
                pitch_transformer.save_to_db(pitches_df, db)

                linescore_df = linescore_transformer.transform(game_data, game_pk=game_pk)
                linescore_transformer.save_to_db(linescore_df, db)

                boxscore_data = boxscore_transformer.transform_all(game_data, game_pk=game_pk)
                for table_name, df in boxscore_data.items():
                    if not df.empty:
                        stat_type = table_name.split("_")[1]
                        boxscore_transformer.save_to_db(df, stat_type, db)

            except (KeyError, TypeError, ValueError) as e:
                tqdm.write(f"Error processing {file_path}: {e}")

        print("\n=== Database Statistics ===")
        tables = [
            "teams",
            "venues",
            "players",
            "games",
            "pitches",
            "linescore",
            "batting",
            "pitching",
            "fielding",
        ]
        for table in tables:
            if db.table_exists(table):
                count = db.get_row_count(table)
                print(f"{table}: {count:,} rows")


async def load_reference_data_to_db_async(db_config: PostgresConfig):
    """Load reference data into the configured PostgreSQL schema."""

    from mlb.data import ReferenceData
    from mlb.endpoints.event_types import EventTypes
    from mlb.endpoints.game_types import GameTypes
    from mlb.endpoints.pitch_types import PitchTypes
    from mlb.endpoints.positions import Positions

    print("\nLoading reference data...")

    ref_transformer = ReferenceData()

    async with Positions() as positions_api, \
               PitchTypes() as pitch_types_api, \
               EventTypes() as event_types_api, \
               GameTypes() as game_types_api:

        results = await asyncio.gather(
            positions_api.get_async(),
            pitch_types_api.get_async(),
            event_types_api.get_async(),
            game_types_api.get_async(),
            return_exceptions=True,
        )

    positions_data, pitch_types_data, event_types_data, game_types_data = results

    with PostgresHandler(db_config) as db:
        if not db.table_exists("positions"):
            db.create_reference_tables()

        if isinstance(positions_data, BaseException):
            print(f"Error loading positions: {positions_data}")
        else:
            positions_df = ref_transformer.transform(positions_data)
            ref_transformer.save_to_db(positions_df, "positions", db, if_exists="replace")
            print("Loaded positions")

        if isinstance(pitch_types_data, BaseException):
            print(f"Error loading pitch types: {pitch_types_data}")
        else:
            pitch_types_df = ref_transformer.transform(pitch_types_data)
            ref_transformer.save_to_db(pitch_types_df, "pitch_types", db, if_exists="replace")
            print("Loaded pitch types")

        if isinstance(event_types_data, BaseException):
            print(f"Error loading event types: {event_types_data}")
        else:
            event_types_df = ref_transformer.transform(event_types_data)
            ref_transformer.save_to_db(event_types_df, "event_types", db, if_exists="replace")
            print("Loaded event types")

        if isinstance(game_types_data, BaseException):
            print(f"Error loading game types: {game_types_data}")
        else:
            game_types_df = ref_transformer.transform(game_types_data)
            ref_transformer.save_to_db(game_types_df, "game_types", db, if_exists="replace")
            print("Loaded game types")


def load_reference_data_to_db(db_config: PostgresConfig):
    """Load reference data using async API calls internally."""

    asyncio.run(load_reference_data_to_db_async(db_config))



def create_indexes(db_config: PostgresConfig):
    """Create indexes on commonly queried columns."""

    print("\nCreating indexes...")

    with PostgresHandler(db_config) as db:
        if db.table_exists("pitches"):
            db.connection.execute("CREATE INDEX IF NOT EXISTS idx_pitches_game_pk ON pitches(game_pk)")
            db.connection.execute("CREATE INDEX IF NOT EXISTS idx_pitches_pitcher ON pitches(pitcher_id)")
            db.connection.execute("CREATE INDEX IF NOT EXISTS idx_pitches_batter ON pitches(batter_id)")
            print("Created indexes on pitches table")

        if db.table_exists("batting"):
            db.connection.execute("CREATE INDEX IF NOT EXISTS idx_batting_game_pk ON batting(game_pk)")
            db.connection.execute("CREATE INDEX IF NOT EXISTS idx_batting_player ON batting(player_id)")
            print("Created indexes on batting table")

        if db.table_exists("pitching"):
            db.connection.execute("CREATE INDEX IF NOT EXISTS idx_pitching_game_pk ON pitching(game_pk)")
            db.connection.execute("CREATE INDEX IF NOT EXISTS idx_pitching_player ON pitching(player_id)")
            print("Created indexes on pitching table")



def main():
    """Load the repository's MLB data into the configured PostgreSQL schema."""

    db_config = PostgresConfig.from_env()

    print("=== MLB Data ETL to PostgreSQL ===")
    print(f"Target: {db_config.describe()}\n")

    print("Step 1: Creating database tables...")
    with PostgresHandler(db_config) as db:
        db.create_all_tables()

    print("\nStep 2: Loading reference data...")
    load_reference_data_to_db(db_config)

    print("\nStep 3: Loading game data...")
    load_raw_livefeeds_to_db(db_config)

    print("\nStep 4: Creating indexes...")
    create_indexes(db_config)

    print("\nStep 5: Vacuuming and analyzing managed tables...")
    with PostgresHandler(db_config) as db:
        db.vacuum()

    print(f"\nETL Complete! Database ready in {db_config.describe()}")


if __name__ == "__main__":
    main()
