from pathlib import Path
from typing import Optional

import pandas as pd


class GameData:
    """
    Data transformation class for MLB Game fact table.

    Extracts game information from GUMBO gameData node to create a
    normalized game fact table with foreign keys to team and venue dimensions.
    """

    def __init__(self):
        """Initialize the GameData transformer."""
        self.data_types = {
            "game_pk": int,
            "game_id": str,
            "season": str,
            "season_display": str,
            "game_type": str,
            "gameday_type": str,
            "game_number": int,
            "double_header": str,
            "tiebreaker": str,
            "calendar_event_id": str,
            "game_date": str,
            "original_date": str,
            "game_datetime": str,
            "game_time": str,
            "ampm": str,
            "day_night": str,
            "abstract_game_state": str,
            "coded_game_state": str,
            "detailed_state": str,
            "status_code": str,
            "start_time_tbd": bool,
            "abstract_game_code": str,
            "venue_id": int,
            "weather_condition": str,
            "weather_temp": str,
            "weather_wind": str,
            "attendance": float,
            "first_pitch": str,
            "game_duration_minutes": float,
            "away_team_id": int,
            "away_team_wins": float,
            "away_team_losses": float,
            "away_team_winning_percentage": str,
            "away_team_division_leader": bool,
            "away_team_games_played": float,
            "home_team_id": int,
            "home_team_wins": float,
            "home_team_losses": float,
            "home_team_winning_percentage": str,
            "home_team_division_leader": bool,
            "home_team_games_played": float,
            "away_probable_pitcher_id": float,
            "away_probable_pitcher_name": str,
            "home_probable_pitcher_id": float,
            "home_probable_pitcher_name": str,
            "has_challenges": bool,
            "away_reviews_remaining": float,
            "away_reviews_used": float,
            "home_reviews_remaining": float,
            "home_reviews_used": float,
            "no_hitter": bool,
            "perfect_game": bool,
            "away_team_no_hitter": bool,
            "away_team_perfect_game": bool,
            "home_team_no_hitter": bool,
            "home_team_perfect_game": bool,
        }

    def _extract_game_info(self, game_data: dict) -> dict:
        """
        Extract game fact information from gameData object.

        Args:
            game_data (dict): gameData object from live feed

        Returns:
            dict: Flattened game information with foreign keys
        """
        game_info = {}

        # Extract game identification fields
        game = game_data.get("game", {})
        game_info["game_pk"] = game.get("pk")
        game_info["game_id"] = game.get("id")
        game_info["season"] = game.get("season")
        game_info["season_display"] = game.get("seasonDisplay")
        game_info["game_type"] = game.get("type")
        game_info["gameday_type"] = game.get("gamedayType")
        game_info["game_number"] = game.get("gameNumber")
        game_info["double_header"] = game.get("doubleHeader")
        game_info["tiebreaker"] = game.get("tiebreaker")
        game_info["calendar_event_id"] = game.get("calendarEventID")

        # Extract datetime fields
        datetime_info = game_data.get("datetime", {})
        game_info["game_date"] = datetime_info.get("officialDate")
        game_info["original_date"] = datetime_info.get("originalDate")
        game_info["game_datetime"] = datetime_info.get("dateTime")
        game_info["game_time"] = datetime_info.get("time")
        game_info["ampm"] = datetime_info.get("ampm")
        game_info["day_night"] = datetime_info.get("dayNight")

        # Extract status fields
        status = game_data.get("status", {})
        game_info["abstract_game_state"] = status.get("abstractGameState")
        game_info["coded_game_state"] = status.get("codedGameState")
        game_info["detailed_state"] = status.get("detailedState")
        game_info["status_code"] = status.get("statusCode")
        game_info["start_time_tbd"] = status.get("startTimeTBD")
        game_info["abstract_game_code"] = status.get("abstractGameCode")

        # Extract venue FK
        venue = game_data.get("venue", {})
        game_info["venue_id"] = venue.get("id")

        # Extract weather fields
        weather = game_data.get("weather", {})
        game_info["weather_condition"] = weather.get("condition")
        game_info["weather_temp"] = weather.get("temp")
        game_info["weather_wind"] = weather.get("wind")

        # Extract game info fields
        game_info_node = game_data.get("gameInfo", {})
        game_info["attendance"] = game_info_node.get("attendance")
        game_info["first_pitch"] = game_info_node.get("firstPitch")
        game_info["game_duration_minutes"] = game_info_node.get("gameDurationMinutes")

        # Extract team FKs and records
        teams = game_data.get("teams", {})

        # Away team
        away_team = teams.get("away", {})
        game_info["away_team_id"] = away_team.get("id")

        away_record = away_team.get("record", {})
        game_info["away_team_wins"] = away_record.get("wins")
        game_info["away_team_losses"] = away_record.get("losses")
        game_info["away_team_winning_percentage"] = away_record.get("winningPercentage")
        game_info["away_team_division_leader"] = away_record.get("divisionLeader")
        game_info["away_team_games_played"] = away_record.get("gamesPlayed")

        # Home team
        home_team = teams.get("home", {})
        game_info["home_team_id"] = home_team.get("id")

        home_record = home_team.get("record", {})
        game_info["home_team_wins"] = home_record.get("wins")
        game_info["home_team_losses"] = home_record.get("losses")
        game_info["home_team_winning_percentage"] = home_record.get("winningPercentage")
        game_info["home_team_division_leader"] = home_record.get("divisionLeader")
        game_info["home_team_games_played"] = home_record.get("gamesPlayed")

        # Extract probable pitchers
        probable_pitchers = game_data.get("probablePitchers", {})

        away_pitcher = probable_pitchers.get("away", {})
        game_info["away_probable_pitcher_id"] = away_pitcher.get("id")
        game_info["away_probable_pitcher_name"] = away_pitcher.get("fullName")

        home_pitcher = probable_pitchers.get("home", {})
        game_info["home_probable_pitcher_id"] = home_pitcher.get("id")
        game_info["home_probable_pitcher_name"] = home_pitcher.get("fullName")

        # Extract review/challenge info
        review = game_data.get("review", {})
        game_info["has_challenges"] = review.get("hasChallenges")

        away_review = review.get("away", {})
        game_info["away_reviews_remaining"] = away_review.get("remaining")
        game_info["away_reviews_used"] = away_review.get("used")

        home_review = review.get("home", {})
        game_info["home_reviews_remaining"] = home_review.get("remaining")
        game_info["home_reviews_used"] = home_review.get("used")

        # Extract flags
        flags = game_data.get("flags", {})
        game_info["no_hitter"] = flags.get("noHitter")
        game_info["perfect_game"] = flags.get("perfectGame")
        game_info["away_team_no_hitter"] = flags.get("awayTeamNoHitter")
        game_info["away_team_perfect_game"] = flags.get("awayTeamPerfectGame")
        game_info["home_team_no_hitter"] = flags.get("homeTeamNoHitter")
        game_info["home_team_perfect_game"] = flags.get("homeTeamPerfectGame")

        return game_info

    def transform(self, data: dict) -> pd.DataFrame:
        """
        Transform gameData into game fact table DataFrame.

        Args:
            data (dict): Raw GUMBO JSON response

        Returns:
            pd.DataFrame: Game fact table with one row and FKs to dimensions
        """
        game_data = data.get("gameData", {})

        if not game_data:
            return pd.DataFrame()

        game_info = self._extract_game_info(game_data)
        game_df = pd.DataFrame([game_info])

        if not game_df.empty:
            # Apply data types
            game_df = game_df.astype(self.data_types)

        return game_df

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
        db_handler.insert_dataframe(df, "games", if_exists=if_exists)
