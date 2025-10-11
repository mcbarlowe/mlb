"""Quick verification of mlb.duckdb contents."""

from pathlib import Path
from src.database.duckdb_handler import DuckDBHandler

db_path = Path("data/mlb.duckdb")

print("=" * 60)
print("MLB DATABASE VERIFICATION")
print("=" * 60)

with DuckDBHandler(db_path) as db:
    print(f"\nDatabase: {db_path}")
    print(f"Size: {db_path.stat().st_size / (1024**2):.2f} MB")

    print("\n" + "=" * 60)
    print("TABLE ROW COUNTS")
    print("=" * 60)

    tables = ["teams", "venues", "players", "games", "pitches", "linescore", "batting", "pitching", "fielding"]
    for table in tables:
        if db.table_exists(table):
            count = db.get_row_count(table)
            print(f"{table:15s}: {count:>10,} rows")
        else:
            print(f"{table:15s}: TABLE NOT FOUND")

    # Sample query to verify relationships
    if db.table_exists("games") and db.table_exists("teams") and db.table_exists("venues"):
        print("\n" + "=" * 60)
        print("SAMPLE GAMES WITH RELATIONSHIPS")
        print("=" * 60)

        query = """
        SELECT
            g.game_pk,
            g.game_date,
            away.abbreviation as away,
            home.abbreviation as home,
            v.venue_name,
            g.attendance
        FROM games g
        JOIN teams away ON g.away_team_id = away.team_id
        JOIN teams home ON g.home_team_id = home.team_id
        JOIN venues v ON g.venue_id = v.venue_id
        ORDER BY g.game_date DESC
        LIMIT 10
        """
        result = db.query(query)
        print("\n" + result.to_string(index=False))

    # Check games-to-pitches relationship
    if db.table_exists("games") and db.table_exists("pitches"):
        print("\n" + "=" * 60)
        print("GAMES-TO-PITCHES RELATIONSHIP")
        print("=" * 60)

        query = """
        SELECT
            COUNT(DISTINCT g.game_pk) as total_games,
            COUNT(p.pitch_number) as total_pitches,
            ROUND(COUNT(p.pitch_number) * 1.0 / COUNT(DISTINCT g.game_pk), 1) as avg_pitches_per_game
        FROM games g
        LEFT JOIN pitches p ON g.game_pk = p.game_pk
        """
        result = db.query(query)
        print("\n" + result.to_string(index=False))

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
