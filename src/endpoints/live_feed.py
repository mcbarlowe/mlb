import asyncio
from collections.abc import Callable

import pandas as pd
import requests

from src.endpoints.base_api import AsyncBaseAPI, BaseAPI


class LiveFeed(BaseAPI, AsyncBaseAPI):
    """
    Endpoint wrapper for fetching live feeds from the MLB API.

    Supports both sync and async operations:
    - Sync: live_feed.extract(game_id)
    - Async: async with LiveFeed() as lf: await lf.extract_async(game_id)
    """

    def __init__(self, concurrency_limit: int = 15, *args, **kwargs) -> None:
        """
        Initialize the LiveFeed endpoint.

        Args:
            concurrency_limit: Maximum concurrent requests for async operations
        """
        BaseAPI.__init__(self, *args, **kwargs)
        AsyncBaseAPI.__init__(self, concurrency_limit, *args, **kwargs)
        self.base_url = "https://statsapi.mlb.com/api/v1.1/game"

    def extract(self, game_id, *args, **kwargs) -> dict:
        """
        Handle GET requests to fetch live feeds (sync).
        """
        url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
        headers = {"Accept-Encoding": "gzip"}

        response = requests.get(url, params=kwargs, headers=headers)

        try:
            response.raise_for_status()  # Raise an error for bad responses
            return response.json()
        except requests.exceptions.HTTPError as e:
            raise Exception(f"HTTP error occurred: {e}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request error occurred: {e}")

    async def extract_async(self, game_id: int, **kwargs) -> dict:
        """
        Handle GET requests to fetch live feeds (async).

        Args:
            game_id: MLBAM unique game identifier
            **kwargs: Additional query parameters

        Returns:
            dict: Complete game state JSON response
        """
        url = f"{self.base_url}/{game_id}/feed/live"
        headers = {"Accept-Encoding": "gzip"}

        return await self._request_with_retry(url, params=kwargs, headers=headers)

    async def extract_many_async(
        self,
        game_ids: list[int],
        on_success: Callable[[int, dict], None] | None = None,
        on_error: Callable[[int, Exception], None] | None = None,
    ) -> list[dict | None]:
        """
        Extract multiple game feeds concurrently.

        Args:
            game_ids: List of MLBAM game identifiers to fetch
            on_success: Callback for successful fetches (game_id, result)
            on_error: Callback for failed fetches (game_id, exception)

        Returns:
            list: List of results in same order as game_ids (None for failed items)
        """
        async def fetch_one(game_id: int) -> dict | None:
            try:
                result = await self.extract_async(game_id)
                if on_success:
                    on_success(game_id, result)
                return result
            except Exception as e:
                if on_error:
                    on_error(game_id, e)
                return None

        tasks = [fetch_one(game_id) for game_id in game_ids]
        return await asyncio.gather(*tasks)

    def _process_pitch_data(self, pitch: dict) -> dict:
        """
        Process individual pitch data and return a dict representing one row
        which is the equivalent of one pitch in an at bat.

        Args:
            pitch (dict): data of a single pitch in game to process.
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
        Process individual play data and return a dict representing one row
        which is the equivalent of one at bat in a game

        Args:
            play (dict): data of a single play in game to filter pitches out.
        Returns:
            dict: A dictionary of processed play information.
        """
        pitches = []

        # todo add pitch info here
        ab_result = play["result"]
        ab_about = play["about"]
        ab_matchup = play["matchup"]
        ab_runners = play.get("runners")
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
            start_base = runner["movement"].get("start")
            if start_base == "1B":
                play_info["is_runner_on_first"] = True
                play_info["runner_on_first_id"] = runner["details"]["runner"]["id"]
            elif start_base == "2B":
                play_info["is_runner_on_second"] = True
                play_info["runner_on_second_id"] = runner["details"]["runner"]["id"]
            elif start_base == "3B":
                play_info["is_runner_on_third"] = True
                play_info["runner_on_third_id"] = runner["details"]["runner"]["id"]

        for index in play["pitchIndex"]:
            pitch = play["playEvents"][index]
            pitch_info = self._process_pitch_data(pitch)
            pitch_info = {**play_info, **pitch_info}  # Merge play and pitch info
            pitches.append(pitch_info)

        return pitches

    def _process_plays_data(self, plays: dict) -> "pd.DataFrame":
        """
        Process individual pitch data and return structured information.
        Args:
            plays (dict): data of all plays in game to filter pitches out.
        Returns:
            list: A list of processed pitch information.
        """
        rows = []
        for play in plays:
            rows.extend(self._process_play_data(play))

        pitch_df = pd.DataFrame(rows)

        return pitch_df

    def transform(self, data: dict, game_id, season) -> "pd.DataFrame":
        data_types = {
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
        live_data = data["liveData"]
        plays = live_data["plays"]["allPlays"]

        pitches_df = self._process_plays_data(plays)
        if "is_runner_on_third" not in pitches_df.columns:
            pitches_df["is_runner_on_third"] = False
            pitches_df["runner_on_third_id"] = None

        if "is_runner_on_second" not in pitches_df.columns:
            pitches_df["is_runner_on_second"] = False
            pitches_df["runner_on_second_id"] = None
        # pitches_df = pitches_df.fillna(
        #    {
        #        "is_runner_on_first": False,
        #        "is_runner_on_second": False,
        #        "is_runner_on_third": False,
        #    },
        # )
        pitches_df = pitches_df.astype(data_types)

        return pitches_df

    def load(self, data: dict, *args, **kwargs) -> None:
        pass

    def etl(self, game_id) -> None:
        data = self.extract(game_id)
        transformed_data = self.transform(data)
        self.load(transformed_data)
