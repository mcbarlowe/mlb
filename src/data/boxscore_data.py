from pathlib import Path
from typing import Literal

import pandas as pd


class BoxscoreData:
    """
    Data transformation class for MLB Boxscore data.

    Transforms boxscore JSON into player-level statistics tables.
    """

    def transform_batting(self, data: dict, game_pk: int = None,
                          team_type: Literal["away", "home"] = "away") -> pd.DataFrame:
        """
        Transform batting boxscore data into DataFrame.

        Args:
            data (dict): Raw GUMBO JSON response
            game_pk (int, optional): Game identifier
            team_type (str): 'away' or 'home'

        Returns:
            pd.DataFrame: Player batting statistics
        """
        boxscore = data.get("liveData", {}).get("boxscore", {})
        teams = boxscore.get("teams", {})
        team_data = teams.get(team_type, {})
        players = team_data.get("players", {})

        rows = []
        for player_key, player_data in players.items():
            stats = player_data.get("stats", {}).get("batting", {})
            if not stats:  # Skip if no batting stats
                continue

            person = player_data.get("person", {})
            position = player_data.get("position", {})

            row = {
                "game_pk": game_pk,
                "team_type": team_type,
                "player_id": person.get("id"),
                "player_name": person.get("fullName"),
                "jersey_number": player_data.get("jerseyNumber"),
                "position_code": position.get("code"),
                "position_name": position.get("name"),
                "position_abbrev": position.get("abbreviation"),
                "batting_order": player_data.get("battingOrder"),
                **stats  # Unpack all batting stats
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def transform_pitching(self, data: dict, game_pk: int = None,
                           team_type: Literal["away", "home"] = "away") -> pd.DataFrame:
        """
        Transform pitching boxscore data into DataFrame.

        Args:
            data (dict): Raw GUMBO JSON response
            game_pk (int, optional): Game identifier
            team_type (str): 'away' or 'home'

        Returns:
            pd.DataFrame: Player pitching statistics
        """
        boxscore = data.get("liveData", {}).get("boxscore", {})
        teams = boxscore.get("teams", {})
        team_data = teams.get(team_type, {})
        players = team_data.get("players", {})

        rows = []
        for player_key, player_data in players.items():
            stats = player_data.get("stats", {}).get("pitching", {})
            if not stats:  # Skip if no pitching stats
                continue

            person = player_data.get("person", {})
            position = player_data.get("position", {})

            row = {
                "game_pk": game_pk,
                "team_type": team_type,
                "player_id": person.get("id"),
                "player_name": person.get("fullName"),
                "jersey_number": player_data.get("jerseyNumber"),
                "position_code": position.get("code"),
                "position_name": position.get("name"),
                "position_abbrev": position.get("abbreviation"),
                **stats  # Unpack all pitching stats
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def transform_fielding(self, data: dict, game_pk: int = None,
                           team_type: Literal["away", "home"] = "away") -> pd.DataFrame:
        """
        Transform fielding boxscore data into DataFrame.

        Args:
            data (dict): Raw GUMBO JSON response
            game_pk (int, optional): Game identifier
            team_type (str): 'away' or 'home'

        Returns:
            pd.DataFrame: Player fielding statistics
        """
        boxscore = data.get("liveData", {}).get("boxscore", {})
        teams = boxscore.get("teams", {})
        team_data = teams.get(team_type, {})
        players = team_data.get("players", {})

        rows = []
        for player_key, player_data in players.items():
            stats = player_data.get("stats", {}).get("fielding", {})
            if not stats:  # Skip if no fielding stats
                continue

            person = player_data.get("person", {})
            position = player_data.get("position", {})

            row = {
                "game_pk": game_pk,
                "team_type": team_type,
                "player_id": person.get("id"),
                "player_name": person.get("fullName"),
                "jersey_number": player_data.get("jerseyNumber"),
                "position_code": position.get("code"),
                "position_name": position.get("name"),
                "position_abbrev": position.get("abbreviation"),
                **stats  # Unpack all fielding stats
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def transform_all(self, data: dict, game_pk: int = None) -> dict:
        """
        Transform all boxscore data (batting, pitching, fielding) for both teams.

        Args:
            data (dict): Raw GUMBO JSON response
            game_pk (int, optional): Game identifier

        Returns:
            dict: Dictionary containing DataFrames for each stat type and team
        """
        result = {}

        for team_type in ["away", "home"]:
            result[f"{team_type}_batting"] = self.transform_batting(data, game_pk, team_type)
            result[f"{team_type}_pitching"] = self.transform_pitching(data, game_pk, team_type)
            result[f"{team_type}_fielding"] = self.transform_fielding(data, game_pk, team_type)

        return result

    def save(self, df: pd.DataFrame, output_path: Path, format: str = "parquet") -> None:
        """
        Save DataFrame to file.

        Args:
            df (pd.DataFrame): DataFrame to save
            output_path (Path): Output file path
            format (str): Output format ('parquet', 'csv', or 'json')

        Raises:
            ValueError: If format is not supported
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "parquet":
            df.to_parquet(output_path, index=False)
        elif format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "json":
            df.to_json(output_path, orient="records", lines=True)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'parquet', 'csv', or 'json'.")

    def save_to_db(self, df: pd.DataFrame, table_name: str, db_handler,
                   if_exists: str = "append") -> None:
        """
        Save DataFrame to DuckDB database.

        Args:
            df (pd.DataFrame): DataFrame to save
            table_name (str): Table name ('batting', 'pitching', or 'fielding')
            db_handler: DuckDBHandler instance
            if_exists (str): How to behave if table exists ('append', 'replace', 'fail')
        """
        db_handler.insert_dataframe(df, table_name, if_exists=if_exists)
