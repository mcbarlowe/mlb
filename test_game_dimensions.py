"""
Test script for game, team, and venue dimension tables.

This script tests the extraction and loading of dimension and fact tables
from MLB live feed data into a local PostgreSQL schema.
"""

import json
from dataclasses import replace
from pathlib import Path

from src.data.game_data import GameData
from src.data.team_data import TeamData
from src.data.venue_data import VenueData
from src.database import PostgresConfig, PostgresHandler

SAMPLE_FILE = Path("example_json_files/example_live_feed.json")
TEST_SCHEMA = "mlb_test_dimensions"


def test_dimension_extraction():
    """Test extraction of game, team, and venue data from a sample live feed."""

    print(f"Loading sample data from: {SAMPLE_FILE}")
    with open(SAMPLE_FILE, "r") as f:
        live_feed_data = json.load(f)

    print("\n" + "=" * 60)
    print("EXTRACTING DIMENSION AND FACT DATA")
    print("=" * 60)

    print("\n1. Extracting Venue Dimension...")
    venue_extractor = VenueData()
    venue_df = venue_extractor.transform(live_feed_data)
    print(f"   ✓ Extracted {len(venue_df)} venue record(s)")
    print(f"   Venue: {venue_df['venue_name'].iloc[0]}")

    print("\n2. Extracting Team Dimension...")
    team_extractor = TeamData()
    team_df = team_extractor.transform(live_feed_data)
    print(f"   ✓ Extracted {len(team_df)} team record(s)")
    for team_number, row in enumerate(team_df.itertuples(index=False), start=1):
        print(f"   Team {team_number}: {row.team_name} (ID: {row.team_id})")

    print("\n3. Extracting Game Fact...")
    game_extractor = GameData()
    game_df = game_extractor.transform(live_feed_data)
    print(f"   ✓ Extracted {len(game_df)} game record(s)")
    print(f"   Game: {game_df['game_id'].iloc[0]} (PK: {game_df['game_pk'].iloc[0]})")
    print(f"   Date: {game_df['game_date'].iloc[0]}")
    print(f"   Away Team ID: {game_df['away_team_id'].iloc[0]}")
    print(f"   Home Team ID: {game_df['home_team_id'].iloc[0]}")
    print(f"   Venue ID: {game_df['venue_id'].iloc[0]}")

    print("\n" + "=" * 60)
    print("LOADING DATA INTO POSTGRES")
    print("=" * 60)

    db_config = replace(PostgresConfig.from_env(), schema=TEST_SCHEMA)
    print(f"\nUsing test schema: {db_config.describe()}")

    with PostgresHandler(db_config) as db:
        db.reset_schema()

        print("\nCreating tables...")
        db.create_teams_table()
        db.create_reference_tables()
        db.create_games_table()

        print("\n4. Inserting Team Dimension...")
        team_extractor.save_to_db(team_df, db, if_exists="append")

        print("\n5. Inserting Venue Dimension...")
        venue_extractor.save_to_db(venue_df, db, if_exists="append")

        print("\n6. Inserting Game Fact...")
        game_extractor.save_to_db(game_df, db, if_exists="append")

        print("\n" + "=" * 60)
        print("VERIFYING DATA")
        print("=" * 60)

        print("\n7. Querying Teams Table...")
        teams_query = "SELECT team_id, team_name, abbreviation, league_name, division_name FROM teams"
        teams_result = db.query(teams_query)
        print(teams_result.to_string(index=False))

        print("\n8. Querying Venues Table...")
        venues_query = "SELECT venue_id, venue_name, city, state_abbrev, capacity FROM venues"
        venues_result = db.query(venues_query)
        print(venues_result.to_string(index=False))

        print("\n9. Querying Games Table...")
        games_query = """
        SELECT
            game_pk,
            game_date,
            away_team_id,
            home_team_id,
            venue_id,
            attendance,
            game_duration_minutes
        FROM games
        """
        games_result = db.query(games_query)
        print(games_result.to_string(index=False))

        print("\n10. Querying with JOIN (Star Schema)...")
        join_query = """
        SELECT
            g.game_pk,
            g.game_date,
            away.team_name as away_team,
            away.abbreviation as away_abbr,
            g.away_team_wins || '-' || g.away_team_losses as away_record,
            home.team_name as home_team,
            home.abbreviation as home_abbr,
            g.home_team_wins || '-' || g.home_team_losses as home_record,
            v.venue_name,
            v.city,
            g.attendance
        FROM games g
        JOIN teams away ON g.away_team_id = away.team_id
        JOIN teams home ON g.home_team_id = home.team_id
        JOIN venues v ON g.venue_id = v.venue_id
        """
        join_result = db.query(join_query)
        print(join_result.to_string(index=False))

        print("\n" + "=" * 60)
        print("TABLE ROW COUNTS")
        print("=" * 60)
        print(f"Teams: {db.get_row_count('teams')}")
        print(f"Venues: {db.get_row_count('venues')}")
        print(f"Games: {db.get_row_count('games')}")

    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nTest schema ready at: {db_config.describe()}")


if __name__ == "__main__":
    test_dimension_extraction()
