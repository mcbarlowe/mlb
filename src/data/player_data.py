from pathlib import Path
from typing import Optional

import pandas as pd


class PlayerData:
    """
    Data transformation class for MLB Player dimension data.

    Extracts player information from GUMBO gameData.players node
    to create a normalized player dimension table.
    """

    def __init__(self):
        """Initialize the PlayerData transformer."""
        self.data_types = {
            "player_id": int,
            "full_name": str,
            "first_name": str,
            "last_name": str,
            "middle_name": str,
            "use_name": str,
            "boxscore_name": str,
            "nick_name": str,
            "name_first_last": str,
            "name_slug": str,
            "first_last_name": str,
            "last_first_name": str,
            "last_init_name": str,
            "init_last_name": str,
            "full_fml_name": str,
            "full_lfm_name": str,
            "primary_number": str,
            "birth_date": str,
            "current_age": int,
            "birth_city": str,
            "birth_state_province": str,
            "birth_country": str,
            "height": str,
            "weight": int,
            "active": bool,
            "primary_position_code": str,
            "primary_position_name": str,
            "primary_position_type": str,
            "primary_position_abbrev": str,
            "bat_side_code": str,
            "bat_side_description": str,
            "pitch_hand_code": str,
            "pitch_hand_description": str,
            "draft_year": float,
            "mlb_debut_date": str,
            "strike_zone_top": float,
            "strike_zone_bottom": float,
        }

    def _extract_player_info(self, player: dict) -> dict:
        """
        Extract player information from a player object.

        Args:
            player (dict): Player object from gameData.players

        Returns:
            dict: Flattened player information
        """
        player_info = {}

        # Basic info
        player_info["player_id"] = player.get("id")
        player_info["full_name"] = player.get("fullName")
        player_info["first_name"] = player.get("firstName")
        player_info["last_name"] = player.get("lastName")
        player_info["middle_name"] = player.get("middleName")
        player_info["use_name"] = player.get("useName")
        player_info["boxscore_name"] = player.get("boxscoreName")
        player_info["nick_name"] = player.get("nickName")

        # Name variations
        player_info["name_first_last"] = player.get("nameFirstLast")
        player_info["name_slug"] = player.get("nameSlug")
        player_info["first_last_name"] = player.get("firstLastName")
        player_info["last_first_name"] = player.get("lastFirstName")
        player_info["last_init_name"] = player.get("lastInitName")
        player_info["init_last_name"] = player.get("initLastName")
        player_info["full_fml_name"] = player.get("fullFMLName")
        player_info["full_lfm_name"] = player.get("fullLFMName")

        # Jersey and physical info
        player_info["primary_number"] = player.get("primaryNumber")
        player_info["birth_date"] = player.get("birthDate")
        player_info["current_age"] = player.get("currentAge")
        player_info["birth_city"] = player.get("birthCity")
        player_info["birth_state_province"] = player.get("birthStateProvince")
        player_info["birth_country"] = player.get("birthCountry")
        player_info["height"] = player.get("height")
        player_info["weight"] = player.get("weight")
        player_info["active"] = player.get("active", True)

        # Position info
        primary_position = player.get("primaryPosition", {})
        player_info["primary_position_code"] = primary_position.get("code")
        player_info["primary_position_name"] = primary_position.get("name")
        player_info["primary_position_type"] = primary_position.get("type")
        player_info["primary_position_abbrev"] = primary_position.get("abbreviation")

        # Bat/Pitch hand
        bat_side = player.get("batSide", {})
        player_info["bat_side_code"] = bat_side.get("code")
        player_info["bat_side_description"] = bat_side.get("description")

        pitch_hand = player.get("pitchHand", {})
        player_info["pitch_hand_code"] = pitch_hand.get("code")
        player_info["pitch_hand_description"] = pitch_hand.get("description")

        # Career info
        player_info["draft_year"] = player.get("draftYear")
        player_info["mlb_debut_date"] = player.get("mlbDebutDate")

        # Strike zone
        player_info["strike_zone_top"] = player.get("strikeZoneTop")
        player_info["strike_zone_bottom"] = player.get("strikeZoneBottom")

        return player_info

    def transform(self, data: dict) -> pd.DataFrame:
        """
        Transform gameData.players into player dimension DataFrame.

        Args:
            data (dict): Raw GUMBO JSON response

        Returns:
            pd.DataFrame: Player dimension data for all players in the game
        """
        game_data = data.get("gameData", {})
        players = game_data.get("players", {})

        players_list = []

        # Iterate through all players (keys are "ID{player_id}")
        for player_key, player_data in players.items():
            if player_key.startswith("ID"):
                player_info = self._extract_player_info(player_data)
                players_list.append(player_info)

        players_df = pd.DataFrame(players_list)

        if not players_df.empty:
            # Apply data types
            players_df = players_df.astype(self.data_types)

        return players_df

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

    def save_to_db(self, df: pd.DataFrame, db_handler, if_exists: str = "append") -> None:
        """
        Save DataFrame to DuckDB database.

        Args:
            df (pd.DataFrame): DataFrame to save
            db_handler: DuckDBHandler instance
            if_exists (str): How to behave if table exists ('append', 'replace', 'fail')
        """
        db_handler.insert_dataframe(df, "players", if_exists=if_exists)
