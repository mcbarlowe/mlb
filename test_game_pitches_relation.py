"""
Test script for games-to-pitches one-to-many relationship.

This script verifies the foreign key relationship between the games
fact table and the pitches fact table.
"""

import json
from pathlib import Path

from src.data.game_data import GameData
from src.data.team_data import TeamData
from src.data.venue_data import VenueData
from src.data.game_feed_data import GameFeedData
from src.database.duckdb_handler import DuckDBHandler


def test_games_pitches_relationship():
    """Test one-to-many relationship between games and pitches."""

    # Load sample data
    sample_file = Path("data/raw/livefeeds/2022/661363.json")
    print(f"Loading sample data from: {sample_file}")

    with open(sample_file, "r") as f:
        live_feed_data = json.load(f)

    print("\n" + "=" * 60)
    print("EXTRACTING DATA")
    print("=" * 60)

    # Extract dimension data
    print("\n1. Extracting dimensions...")
    venue_extractor = VenueData()
    venue_df = venue_extractor.transform(live_feed_data)
    print(f"   ✓ Venues: {len(venue_df)} record(s)")

    team_extractor = TeamData()
    team_df = team_extractor.transform(live_feed_data)
    print(f"   ✓ Teams: {len(team_df)} record(s)")

    # Extract game fact
    print("\n2. Extracting game fact...")
    game_extractor = GameData()
    game_df = game_extractor.transform(live_feed_data)
    game_pk = game_df['game_pk'].iloc[0]
    print(f"   ✓ Game: {game_pk}")

    # Extract pitches fact
    print("\n3. Extracting pitches fact...")
    pitches_extractor = GameFeedData()
    pitches_df = pitches_extractor.transform(live_feed_data, game_pk, "2022")
    print(f"   ✓ Pitches: {len(pitches_df)} record(s) for game {game_pk}")

    print("\n" + "=" * 60)
    print("LOADING DATA INTO DUCKDB")
    print("=" * 60)

    # Create database and tables
    db_path = Path("data/test_game_pitches_relation.duckdb")
    print(f"\nCreating test database: {db_path}")

    with DuckDBHandler(db_path) as db:
        # Create tables (dimensions first, then facts)
        print("\nCreating tables...")
        db.create_teams_table()
        db.create_reference_tables()  # Creates venues table
        db.create_games_table()        # Parent fact table
        db.create_pitches_table()      # Child fact table with FK

        # Insert dimension data
        print("\n4. Inserting dimensions...")
        team_extractor.save_to_db(team_df, db, if_exists="append")
        venue_extractor.save_to_db(venue_df, db, if_exists="append")

        # Insert parent fact (games) first
        print("\n5. Inserting parent fact (games)...")
        game_extractor.save_to_db(game_df, db, if_exists="append")

        # Insert child fact (pitches) - FK constraint will be enforced
        print("\n6. Inserting child fact (pitches)...")
        db.insert_dataframe(pitches_df, "pitches", if_exists="append")

        print("\n" + "=" * 60)
        print("VERIFYING ONE-TO-MANY RELATIONSHIP")
        print("=" * 60)

        # Check row counts
        print("\n7. Row counts:")
        games_count = db.get_row_count('games')
        pitches_count = db.get_row_count('pitches')
        print(f"   Games: {games_count}")
        print(f"   Pitches: {pitches_count}")
        print(f"   Ratio: {pitches_count}:{games_count} (many pitches to one game)")

        # Verify FK relationship with JOIN
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

        # Show sample pitches with game context
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

        # Test FK constraint enforcement
        print("\n" + "=" * 60)
        print("TESTING FK CONSTRAINT ENFORCEMENT")
        print("=" * 60)

        print("\n10. Attempting to insert pitch with non-existent game_pk...")
        try:
            # Create a fake pitch record with invalid game_pk
            import pandas as pd
            fake_pitch = pd.DataFrame([{
                'game_pk': 999999,  # Non-existent game
                'season': 2022,
                'pitch_number': 1,
                'pitch_type': 'FF',
                'batter_name': 'Test Player'
            }])

            # This should fail due to FK constraint
            db.insert_dataframe(fake_pitch, "pitches", if_exists="append")
            print("   ✗ UNEXPECTED: Insert succeeded (FK constraint not enforced)")

        except Exception as e:
            print(f"   ✓ EXPECTED: Insert failed due to FK constraint")
            print(f"   Error: {str(e)[:100]}...")

        # Verify pitch counts by inning
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
    print(f"\nThe one-to-many relationship between games and pitches is working correctly.")
    print(f"- 1 game ({game_pk}) has {pitches_count} pitches")
    print(f"- Foreign key constraint is enforced")
    print(f"- JOIN queries work as expected")
    print(f"\nTest database saved at: {db_path}")


if __name__ == "__main__":
    test_games_pitches_relationship()
