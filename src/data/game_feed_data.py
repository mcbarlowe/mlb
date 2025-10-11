from pathlib import Path
from typing import Optional

import pandas as pd


class GameFeedData:
    """
    Data transformation class for MLB Game Feed (GUMBO) responses.

    Transforms nested JSON game feed data into flat pitch-level DataFrames
    suitable for analysis and storage.
    """

    def __init__(self):
        """Initialize the GameFeedData transformer."""
        self.data_types = {
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

    def _process_play_data(self, play: dict) -> list:
        """
        Process individual play data and return pitch-level records.

        Args:
            play (dict): Data of a single play in game to filter pitches out.

        Returns:
            list: A list of dictionaries with processed play+pitch information.
        """
        pitches = []

        ab_result = play["result"]
        ab_about = play["about"]
        ab_matchup = play["matchup"]
        ab_runners = play.get("runners", [])

        play_info = {}
        play_info["event"] = ab_result.get("event")
        play_info["event_type"] = ab_result["eventType"]
        play_info["description"] = ab_result["description"]
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

        for runner in ab_runners:
            if runner["movement"]["originBase"] == "1B":
                play_info["is_runner_on_first"] = True
                play_info["runner_on_first_id"] = runner["details"]["runner"]["id"]
            elif runner["movement"]["originBase"] == "2B":
                play_info["is_runner_on_second"] = True
                play_info["runner_on_second_id"] = runner["details"]["runner"]["id"]
            elif runner["movement"]["originBase"] == "3B":
                play_info["is_runner_on_third"] = True
                play_info["runner_on_third_id"] = runner["details"]["runner"]["id"]

        for index in play["pitchIndex"]:
            pitch = play["playEvents"][index]
            pitch_info = self._process_pitch_data(pitch)
            pitch_info = {**play_info, **pitch_info}  # Merge play and pitch info
            pitches.append(pitch_info)

        return pitches

    def _process_plays_data(self, plays: list) -> pd.DataFrame:
        """
        Process all plays and return structured DataFrame.

        Args:
            plays (list): List of all plays in the game.

        Returns:
            pd.DataFrame: DataFrame with pitch-level records.
        """
        rows = []
        for play in plays:
            rows.extend(self._process_play_data(play))

        return pd.DataFrame(rows)

    def transform(self, data: dict, game_id: Optional[int] = None,
                  season: Optional[int] = None) -> pd.DataFrame:
        """
        Transform game feed JSON into pitch-level DataFrame.

        Args:
            data (dict): Raw GUMBO JSON response
            game_id (int, optional): Game identifier
            season (int, optional): Season year

        Returns:
            pd.DataFrame: Transformed pitch-level data
        """
        live_data = data["liveData"]
        plays = live_data["plays"]["allPlays"]

        pitches_df = self._process_plays_data(plays)

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

        # Apply data types
        pitches_df = pitches_df.astype(self.data_types)

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
        Save DataFrame to DuckDB database.

        Args:
            df (pd.DataFrame): DataFrame to save
            db_handler: DuckDBHandler instance
            if_exists (str): How to behave if table exists ('append', 'replace', 'fail')
        """
        db_handler.insert_dataframe(df, "pitches", if_exists=if_exists)
