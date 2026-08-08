from pathlib import Path

import pandas as pd

from src.data._type_utils import coerce_dataframe_types


class GameFeedData:
    """
    Data transformation class for MLB Game Feed (GUMBO) responses.

    Transforms nested JSON game feed data into flat pitch-level DataFrames
    suitable for analysis and storage.
    """

    def __init__(self):
        """Initialize the GameFeedData transformer."""
        self.data_types = {
            "game_pk": int,
            "season": int,
            "game_type": str,
            "game_date": str,
            "day_night": str,
            "double_header": str,
            "game_number": int,
            "away_team_id": int,
            "away_team_name": str,
            "home_team_id": int,
            "home_team_name": str,
            "venue_id": int,
            "venue_name": str,
            "weather_condition": str,
            "weather_temp": float,
            "weather_wind": str,
            "event": str,
            "event_type": str,
            "description": str,
            "rbi": int,
            "away_score": int,
            "home_score": int,
            "is_out": bool,
            "at_bat_index": int,
            "half_inning": str,
            "inning": int,
            "batter_id": int,
            "batter_name": str,
            "bat_side": str,
            "pitcher_id": int,
            "pitcher_name": str,
            "throw_side": str,
            "pitch_number": int,
            "pitch_call_description": str,
            "is_in_play": bool,
            "is_strike": bool,
            "is_ball": bool,
            "pitch_type": str,
            "pitch_type_code": str,
            "count_after_pitch": str,
            "outs": int,
            "play_id": str,
            "pitch_start_time": str,
            "pitch_end_time": str,
            "pitch_start_speed": float,
            "pitch_end_speed": float,
            "pitch_strike_zone_top": float,
            "pitch_strike_zone_bottom": float,
            "pitch_zone": float,
            "ay": float,
            "az": float,
            "pfxX": float,
            "pfxZ": float,
            "px": float,
            "pz": float,
            "vx0": float,
            "vy0": float,
            "vz0": float,
            "x": float,
            "y": float,
            "x0": float,
            "y0": float,
            "z0": float,
            "ax": float,
            "break_angle": float,
            "break_length": float,
            "break_y": float,
            "break_vertical": float,
            "break_vertical_induced": float,
            "break_horizontal": float,
            "spin_rate": float,
            "spin_direction": str,
            "is_runner_on_first": bool,
            "runner_on_first_id": float,
            "is_runner_on_second": bool,
            "runner_on_second_id": float,
            "is_runner_on_third": bool,
            "runner_on_third_id": float,
        }

    def _process_game_data(self, data: dict) -> dict:
        """
        Process gameData node and extract game-level metadata.

        Args:
            data (dict): Full GUMBO JSON response containing gameData node

        Returns:
            dict: Dictionary of game-level metadata fields
        """
        game_data = data.get("gameData", {})
        game_info = {}

        # Extract from game node
        game = game_data.get("game", {})
        game_info["game_pk"] = game.get("pk")
        game_info["season"] = game.get("season")
        game_info["game_type"] = game.get("type")
        game_info["double_header"] = game.get("doubleHeader")
        game_info["game_number"] = game.get("gameNumber", 1)

        # Extract from datetime node
        datetime_info = game_data.get("datetime", {})
        game_info["game_date"] = datetime_info.get("dateTime")
        game_info["day_night"] = datetime_info.get("dayNight")

        # Extract from teams node
        teams = game_data.get("teams", {})
        away_team = teams.get("away", {})
        home_team = teams.get("home", {})

        game_info["away_team_id"] = away_team.get("id")
        game_info["away_team_name"] = away_team.get("name")
        game_info["home_team_id"] = home_team.get("id")
        game_info["home_team_name"] = home_team.get("name")

        # Extract from venue node
        venue = game_data.get("venue", {})
        game_info["venue_id"] = venue.get("id")
        game_info["venue_name"] = venue.get("name")

        # Extract from weather node
        weather = game_data.get("weather", {})
        game_info["weather_condition"] = weather.get("condition")
        game_info["weather_temp"] = weather.get("temp")
        game_info["weather_wind"] = weather.get("wind")

        return game_info

    def _process_pitch_data(self, pitch: dict) -> dict:
        """
        Process individual pitch data and return a dict representing one row.

        Args:
            pitch (dict): Data of a single pitch in game to process.

        Returns:
            dict: A dictionary of processed pitch information.
        """
        row = {}

        pitch_data = pitch.get("pitchData", {})
        pitch_coordinates = pitch_data.get("coordinates", {})
        pitch_breaks = pitch_data.get("breaks", {})
        pitch_details = pitch.get("details", {})
        count = pitch.get("count", {})

        row["pitch_number"] = pitch.get("pitchNumber", 0)
        row["pitch_call_description"] = pitch_details.get("description")
        row["is_in_play"] = pitch_details.get("isInPlay", False)
        row["is_strike"] = pitch_details.get("isStrike", False)
        row["is_ball"] = pitch_details.get("isBall", False)
        row["pitch_type"] = pitch_details.get("type", {}).get("description")
        row["pitch_type_code"] = pitch_details.get("type", {}).get("code")
        row["is_out"] = pitch_details.get("isOut", False)

        row["count_after_pitch"] = f"{count.get('balls', 0)}-{count.get('strikes')}"
        row["outs"] = pitch.get("about", {}).get("outs", 0)
        row["play_id"] = pitch.get("playId")
        row["pitch_start_time"] = pitch.get("startTime")
        row["pitch_end_time"] = pitch.get("endTime")

        row["pitch_start_speed"] = pitch_data.get("startSpeed")
        row["pitch_end_speed"] = pitch_data.get("endSpeed")
        row["pitch_strike_zone_top"] = pitch_data.get("strikeZoneTop")
        row["pitch_strike_zone_bottom"] = pitch_data.get("strikeZoneBottom")
        row["pitch_zone"] = pitch_data.get("zone", 0)
        row["ay"] = pitch_coordinates.get("aY")
        row["az"] = pitch_coordinates.get("aZ")
        row["pfxX"] = pitch_coordinates.get("pfxX")
        row["pfxZ"] = pitch_coordinates.get("pfxZ")
        row["px"] = pitch_coordinates.get("pX")
        row["pz"] = pitch_coordinates.get("pZ")
        row["vx0"] = pitch_coordinates.get("vX0")
        row["vy0"] = pitch_coordinates.get("vY0")
        row["vz0"] = pitch_coordinates.get("vZ0")
        row["x"] = pitch_coordinates.get("x")
        row["y"] = pitch_coordinates.get("y")
        row["x0"] = pitch_coordinates.get("x0")
        row["y0"] = pitch_coordinates.get("y0")
        row["z0"] = pitch_coordinates.get("z0")
        row["ax"] = pitch_coordinates.get("aX")
        row["break_angle"] = pitch_breaks.get("breakAngle")
        row["break_length"] = pitch_breaks.get("breakLength")
        row["break_y"] = pitch_breaks.get("breakY")
        row["break_vertical"] = pitch_breaks.get("breakVertical")
        row["break_vertical_induced"] = pitch_breaks.get("breakVerticalInduced")
        row["break_horizontal"] = pitch_breaks.get("breakHorizontal")
        row["spin_rate"] = pitch_data.get("spinRate")
        row["spin_direction"] = pitch_data.get("spinDirection")

        return row

    def _process_play_data(self, play: dict, game_info: dict | None = None) -> list:
        """
        Process individual play data and return pitch-level records.

        Args:
            play (dict): Data of a single play in game to filter pitches out.
            game_info (dict, optional): Game-level metadata to include with each pitch.

        Returns:
            list: A list of dictionaries with processed play+pitch information.
        """
        pitches = []

        ab_result = play.get("result", {})
        ab_about = play.get("about", {})
        ab_matchup = play.get("matchup", {})
        ab_runners = play.get("runners", [])

        play_info = {}
        play_info["event"] = ab_result.get("event")
        play_info["event_type"] = ab_result.get("eventType")
        play_info["description"] = ab_result.get("description")
        play_info["rbi"] = ab_result.get("rbi", 0)
        play_info["away_score"] = ab_result.get("awayScore", 0)
        play_info["home_score"] = ab_result.get("homeScore", 0)
        play_info["is_out"] = ab_result.get("isOut", False)
        play_info["at_bat_index"] = ab_about.get("atBatIndex")
        play_info["half_inning"] = ab_about.get("halfInning")
        play_info["inning"] = ab_about.get("inning")
        play_info["batter_id"] = ab_matchup.get("batter", {}).get("id")
        play_info["batter_name"] = ab_matchup.get("batter", {}).get("fullName")
        play_info["bat_side"] = ab_matchup.get("batSide", {}).get("code")
        play_info["pitcher_id"] = ab_matchup.get("pitcher", {}).get("id")
        play_info["pitcher_name"] = ab_matchup.get("pitcher", {}).get("fullName")
        play_info["throw_side"] = ab_matchup.get("pitchHand", {}).get("code")

        # Initialize runner columns to False (will be set to True if runner exists)
        play_info["is_runner_on_first"] = False
        play_info["runner_on_first_id"] = None
        play_info["is_runner_on_second"] = False
        play_info["runner_on_second_id"] = None
        play_info["is_runner_on_third"] = False
        play_info["runner_on_third_id"] = None

        # Check runners' 'start' field to determine who was on base at the START of the at-bat
        # The 'start' field shows where each runner was positioned when the play began
        for runner in ab_runners:
            movement = runner.get("movement", {})
            runner_details = runner.get("details", {})
            runner_info = runner_details.get("runner", {})
            start_base = movement.get("start")
            runner_id = runner_info.get("id")

            if start_base == "1B":
                play_info["is_runner_on_first"] = True
                play_info["runner_on_first_id"] = runner_id
            elif start_base == "2B":
                play_info["is_runner_on_second"] = True
                play_info["runner_on_second_id"] = runner_id
            elif start_base == "3B":
                play_info["is_runner_on_third"] = True
                play_info["runner_on_third_id"] = runner_id

        pitch_indices = play.get("pitchIndex", [])
        play_events = play.get("playEvents", [])
        for index in pitch_indices:
            if index >= len(play_events):
                continue
            pitch = play_events[index]
            pitch_info = self._process_pitch_data(pitch)

            # Merge game_info, play_info, and pitch_info
            combined_info = {}
            if game_info:
                combined_info.update(game_info)
            combined_info.update(play_info)
            combined_info.update(pitch_info)

            pitches.append(combined_info)

        return pitches

    def _process_plays_data(self, plays: list, game_info: dict | None = None) -> pd.DataFrame:
        """
        Process all plays and return structured DataFrame.

        Args:
            plays (list): List of all plays in the game.
            game_info (dict, optional): Game-level metadata to include with each pitch.

        Returns:
            pd.DataFrame: DataFrame with pitch-level records.
        """
        rows = []
        for play in plays:
            rows.extend(self._process_play_data(play, game_info))

        return pd.DataFrame(rows)

    def transform(self, data: dict, game_id: int | None = None,
                  season: int | None = None) -> pd.DataFrame:
        """
        Transform game feed JSON into pitch-level DataFrame.

        Args:
            data (dict): Raw GUMBO JSON response
            game_id (int, optional): Game identifier (deprecated - extracted from gameData)
            season (int, optional): Season year (deprecated - extracted from gameData)

        Returns:
            pd.DataFrame: Transformed pitch-level data
        """
        # Extract game-level metadata from gameData node
        game_info = self._process_game_data(data)

        # Override with parameters if provided (for backwards compatibility)
        if game_id is not None:
            game_info["game_pk"] = game_id
        if season is not None:
            game_info["season"] = season

        live_data = data.get("liveData", {})
        plays = live_data.get("plays", {}).get("allPlays", [])

        pitches_df = self._process_plays_data(plays, game_info)

        if pitches_df.empty:
            return pd.DataFrame(columns=self.data_types.keys())

        # Handle missing runner columns
        if "is_runner_on_third" not in pitches_df.columns:
            pitches_df["is_runner_on_third"] = False
            pitches_df["runner_on_third_id"] = None

        if "is_runner_on_second" not in pitches_df.columns:
            pitches_df["is_runner_on_second"] = False
            pitches_df["runner_on_second_id"] = None

        if "is_runner_on_first" not in pitches_df.columns:
            pitches_df["is_runner_on_first"] = False
            pitches_df["runner_on_first_id"] = None

        pitches_df = coerce_dataframe_types(pitches_df, self.data_types)

        return pitches_df

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
        Save DataFrame to the configured PostgreSQL database.

        Args:
            df (pd.DataFrame): DataFrame to save
            db_handler: PostgresHandler instance
            if_exists (str): How to behave if table exists ('append', 'replace', 'fail')
        """
        db_handler.insert_dataframe(df, "pitches", if_exists=if_exists)
