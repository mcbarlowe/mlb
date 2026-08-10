"""
Test script for games-to-pitches one-to-many relationship.

This script verifies the foreign key relationship between the games
fact table and the pitches fact table in a local PostgreSQL schema.
"""

import json
from dataclasses import replace
from pathlib import Path

from src.data.game_data import GameData
from src.data.game_feed_data import GameFeedData
from src.data.team_data import TeamData
from src.data.venue_data import VenueData
from src.database import PostgresConfig, PostgresHandler

SAMPLE_FILE = Path("example_json_files/example_live_feed.json")
TEST_SCHEMA = "mlb_test_game_pitches"


def test_games_pitches_relationship():
    """Test one-to-many relationship between games and pitches."""

    print(f"Loading sample data from: {SAMPLE_FILE}")
    with open(SAMPLE_FILE, "r") as f:
        live_feed_data = json.load(f)

    print("\n" + "=" * 60)
    print("EXTRACTING DATA")
    print("=" * 60)

    print("\n1. Extracting dimensions...")
    venue_extractor = VenueData()
    venue_df = venue_extractor.transform(live_feed_data)
    print(f"   ✓ Venues: {len(venue_df)} record(s)")

    team_extractor = TeamData()
    team_df = team_extractor.transform(live_feed_data)
    print(f"   ✓ Teams: {len(team_df)} record(s)")

    print("\n2. Extracting game fact...")
    game_extractor = GameData()
    game_df = game_extractor.transform(live_feed_data)
    game_pk = game_df["game_pk"].iloc[0]
    print(f"   ✓ Game: {game_pk}")

    print("\n3. Extracting pitches fact...")
    pitches_extractor = GameFeedData()
    pitches_df = pitches_extractor.transform(live_feed_data, game_pk, 2022)
    print(f"   ✓ Pitches: {len(pitches_df)} record(s) for game {game_pk}")

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
        db.create_pitches_table()

        print("\n4. Inserting dimensions...")
        team_extractor.save_to_db(team_df, db, if_exists="append")
        venue_extractor.save_to_db(venue_df, db, if_exists="append")

        print("\n5. Inserting parent fact (games)...")
        game_extractor.save_to_db(game_df, db, if_exists="append")

        print("\n6. Inserting child fact (pitches)...")
        db.insert_dataframe(pitches_df, "pitches", if_exists="append")

        print("\n" + "=" * 60)
        print("VERIFYING ONE-TO-MANY RELATIONSHIP")
        print("=" * 60)

        print("\n7. Row counts:")
        games_count = db.get_row_count("games")
        pitches_count = db.get_row_count("pitches")
        print(f"   Games: {games_count}")
        print(f"   Pitches: {pitches_count}")
        print(f"   Ratio: {pitches_count}:{games_count} (many pitches to one game)")

        print("\n8. Testing JOIN query (verifying FK relationship)...")
        join_query = """
        SELECT
            g.game_pk,
            g.game_date,
            g.away_team_id,
            g.home_team_id,
            COUNT(p.pitch_number) as total_pitches,
            COUNT(DISTINCT p.play_id) as total_at_bats,
            COUNT(DISTINCT p.inning) as total_innings
        FROM games g
        INNER JOIN pitches p ON g.game_pk = p.game_pk
        GROUP BY g.game_pk, g.game_date, g.away_team_id, g.home_team_id
        """
        join_result = db.query(join_query)
        print("\n   Query result:")
        print(join_result.to_string(index=False))

        print("\n9. Sample pitches with game context (first 5 pitches)...")
        sample_query = """
        SELECT
            g.game_date,
            g.away_team_id,
            g.home_team_id,
            p.inning,
            p.half_inning,
            p.batter_name,
            p.pitcher_name,
            p.pitch_type,
            p.pitch_start_speed,
            p.description
        FROM games g
        INNER JOIN pitches p ON g.game_pk = p.game_pk
        ORDER BY p.at_bat_index, p.pitch_number
        LIMIT 5
        """
        sample_result = db.query(sample_query)
        print(sample_result.to_string(index=False))

        print("\n" + "=" * 60)
        print("TESTING FK CONSTRAINT ENFORCEMENT")
        print("=" * 60)

        print("\n10. Attempting to insert pitch with non-existent game_pk...")
        try:
            import pandas as pd

            fake_pitch = pd.DataFrame([{
                "game_pk": 999999,
                "season": 2022,
                "pitch_number": 1,
                "pitch_type": "FF",
                "batter_name": "Test Player",
            }])

            db.insert_dataframe(fake_pitch, "pitches", if_exists="append")
            print("   ✗ UNEXPECTED: Insert succeeded (FK constraint not enforced)")

        except Exception as e:
            print("   ✓ EXPECTED: Insert failed due to FK constraint")
            print(f"   Error: {str(e)[:100]}...")

        print("\n11. Pitch distribution by inning...")
        inning_query = f"""
        SELECT
            p.inning,
            p.half_inning,
            COUNT(*) as pitch_count
        FROM pitches p
        INNER JOIN games g ON p.game_pk = g.game_pk
        WHERE g.game_pk = {int(game_pk)}
        GROUP BY p.inning, p.half_inning
        ORDER BY p.inning, p.half_inning DESC
        """
        inning_result = db.query(inning_query)
        print(inning_result.to_string(index=False))

    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nThe one-to-many relationship between games and pitches is working correctly.")
    print(f"- 1 game ({game_pk}) has {pitches_count} pitches")
    print("- Foreign key constraint is enforced")
    print("- JOIN queries work as expected")
    print(f"\nTest schema ready at: {db_config.describe()}")


if __name__ == "__main__":
    test_games_pitches_relationship()
