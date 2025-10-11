"""
Example usage of MLB data transformation classes.

This module demonstrates how to use the data transformation classes
to convert API responses into tabular formats.
"""

from pathlib import Path

from src.data import BoxscoreData, GameFeedData, LinescoreData, PlayerData, ReferenceData, TeamData
from src.endpoints.game_feed import GameFeed
from src.endpoints.pitch_types import PitchTypes
from src.endpoints.positions import Positions


def example_game_feed_transform():
    """
    Example: Extract and transform game feed data to pitch-level parquet.
    """
    # Fetch game data
    game_feed = GameFeed()
    game_data = game_feed.get(game_pk=716828)  # Example game

    # Transform to pitch-level DataFrame
    transformer = GameFeedData()
    pitches_df = transformer.transform(game_data, game_pk=716828, season=2024)

    # Save to parquet
    output_path = Path("data/processed/pitches/716828.parquet")
    transformer.save(pitches_df, output_path, format="parquet")

    print(f"Saved {len(pitches_df)} pitch records to {output_path}")
    print(f"Columns: {list(pitches_df.columns)}")


def example_linescore_transform():
    """
    Example: Extract and transform linescore data.
    """
    # Fetch game data
    game_feed = GameFeed()
    game_data = game_feed.get(game_pk=716828)

    # Transform linescore
    transformer = LinescoreData()
    linescore_df = transformer.transform(game_data, game_pk=716828)

    # Save to CSV
    output_path = Path("data/processed/linescores/716828.csv")
    transformer.save(linescore_df, output_path, format="csv")

    print(f"Saved {len(linescore_df)} inning records to {output_path}")
    print(linescore_df.head())


def example_boxscore_transform():
    """
    Example: Extract and transform boxscore data for all players.
    """
    # Fetch game data
    game_feed = GameFeed()
    game_data = game_feed.get(game_pk=716828)

    # Transform boxscore
    transformer = BoxscoreData()
    boxscore_data = transformer.transform_all(game_data, game_pk=716828)

    # Save each DataFrame
    base_path = Path("data/processed/boxscores/716828")

    for data_type, df in boxscore_data.items():
        if not df.empty:
            output_path = base_path / f"{data_type}.parquet"
            transformer.save(df, output_path, format="parquet")
            print(f"Saved {data_type}: {len(df)} records")


def example_reference_data_transform():
    """
    Example: Extract and transform reference data (positions, pitch types, etc.).
    """
    # Fetch positions reference data
    positions = Positions()
    positions_data = positions.get()

    transformer = ReferenceData()
    positions_df = transformer.transform(positions_data)

    # Save to parquet
    output_path = Path("data/reference/positions.parquet")
    transformer.save(positions_df, output_path, format="parquet")

    print(f"Saved {len(positions_df)} position records")
    print(positions_df.head())

    # Fetch pitch types reference data
    pitch_types = PitchTypes()
    pitch_types_data = pitch_types.get()

    pitch_types_df = transformer.transform(pitch_types_data)
    output_path = Path("data/reference/pitch_types.parquet")
    transformer.save(pitch_types_df, output_path, format="parquet")

    print(f"Saved {len(pitch_types_df)} pitch type records")
    print(pitch_types_df.head())


def example_team_dimension_transform():
    """
    Example: Extract team dimension data from game feed.
    """
    # Fetch game data
    game_feed = GameFeed()
    game_data = game_feed.get(game_pk=716828)

    # Transform to team dimension
    transformer = TeamData()
    teams_df = transformer.transform(game_data)

    # Save to parquet
    output_path = Path("data/processed/dimensions/teams.parquet")
    transformer.save(teams_df, output_path, format="parquet")

    print(f"Saved {len(teams_df)} team records to {output_path}")
    print(teams_df[["team_id", "team_name", "abbreviation", "league_name", "division_name"]])


def example_player_dimension_transform():
    """
    Example: Extract player dimension data from game feed.
    """
    # Fetch game data
    game_feed = GameFeed()
    game_data = game_feed.get(game_pk=716828)

    # Transform to player dimension
    transformer = PlayerData()
    players_df = transformer.transform(game_data)

    # Save to parquet
    output_path = Path("data/processed/dimensions/players.parquet")
    transformer.save(players_df, output_path, format="parquet")

    print(f"Saved {len(players_df)} player records to {output_path}")
    print(players_df[["player_id", "full_name", "primary_position_name", "bat_side_code", "pitch_hand_code"]].head(10))


def example_batch_processing():
    """
    Example: Batch process multiple games.
    """
    import json

    game_pks = [716828, 716829, 716830]  # Example game IDs

    transformer = GameFeedData()
    game_feed = GameFeed()

    for game_pk in game_pks:
        try:
            # Fetch and transform
            game_data = game_feed.get(game_pk=game_pk)
            pitches_df = transformer.transform(game_data, game_pk=game_pk)

            # Save
            output_path = Path(f"data/processed/batch/{game_pk}.parquet")
            transformer.save(pitches_df, output_path)

            print(f"Processed game {game_pk}: {len(pitches_df)} pitches")

        except Exception as e:
            print(f"Error processing game {game_pk}: {e}")


if __name__ == "__main__":
    print("=== Game Feed Transform ===")
    # example_game_feed_transform()

    print("\n=== Linescore Transform ===")
    # example_linescore_transform()

    print("\n=== Boxscore Transform ===")
    # example_boxscore_transform()

    print("\n=== Reference Data Transform ===")
    # example_reference_data_transform()

    print("\n=== Team Dimension Transform ===")
    # example_team_dimension_transform()

    print("\n=== Player Dimension Transform ===")
    # example_player_dimension_transform()

    print("\n=== Batch Processing ===")
    # example_batch_processing()

    print("\nUncomment the function calls above to run examples.")
