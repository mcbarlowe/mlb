from pathlib import Path
from typing import Optional

import pandas as pd


class TeamData:
    """
    Data transformation class for MLB Team dimension data.

    Extracts team information from GUMBO gameData.teams node
    to create a normalized team dimension table.
    """

    def __init__(self):
        """Initialize the TeamData transformer."""
        self.data_types = {
            "team_id": int,
            "team_name": str,
            "team_code": str,
            "file_code": str,
            "abbreviation": str,
            "team_name_short": str,
            "location_name": str,
            "first_year_of_play": int,
            "league_id": int,
            "league_name": str,
            "division_id": int,
            "division_name": str,
            "sport_id": int,
            "sport_name": str,
            "venue_id": int,
            "venue_name": str,
            "spring_league_id": float,
            "spring_league_name": str,
            "spring_league_abbrev": str,
            "parent_org_name": str,
            "parent_org_id": float,
            "all_star_status": bool,
            "active": bool,
        }

    def _extract_team_info(self, team: dict, team_type: str) -> dict:
        """
        Extract team information from a team object.

        Args:
            team (dict): Team object from gameData.teams
            team_type (str): 'away' or 'home'

        Returns:
            dict: Flattened team information
        """
        team_info = {}

        team_info["team_id"] = team.get("id")
        team_info["team_name"] = team.get("name")
        team_info["team_code"] = team.get("teamCode")
        team_info["file_code"] = team.get("fileCode")
        team_info["abbreviation"] = team.get("abbreviation")
        team_info["team_name_short"] = team.get("teamName")
        team_info["location_name"] = team.get("locationName")
        team_info["first_year_of_play"] = team.get("firstYearOfPlay")

        # League info
        league = team.get("league", {})
        team_info["league_id"] = league.get("id")
        team_info["league_name"] = league.get("name")

        # Division info
        division = team.get("division", {})
        team_info["division_id"] = division.get("id")
        team_info["division_name"] = division.get("name")

        # Sport info
        sport = team.get("sport", {})
        team_info["sport_id"] = sport.get("id")
        team_info["sport_name"] = sport.get("name")

        # Venue info
        venue = team.get("venue", {})
        team_info["venue_id"] = venue.get("id")
        team_info["venue_name"] = venue.get("name")

        # Spring League (MLB games only)
        spring_league = team.get("springLeague", {})
        team_info["spring_league_id"] = spring_league.get("id")
        team_info["spring_league_name"] = spring_league.get("name")
        team_info["spring_league_abbrev"] = spring_league.get("abbreviation")

        # MiLB parent org info
        team_info["parent_org_name"] = team.get("parentOrgName")
        team_info["parent_org_id"] = team.get("parentOrgId")

        # Status flags
        team_info["all_star_status"] = team.get("allStarStatus", False)
        team_info["active"] = team.get("active", True)

        return team_info

    def transform(self, data: dict) -> pd.DataFrame:
        """
        Transform gameData.teams into team dimension DataFrame.

        Args:
            data (dict): Raw GUMBO JSON response

        Returns:
            pd.DataFrame: Team dimension data with both away and home teams
        """
        game_data = data.get("gameData", {})
        teams = game_data.get("teams", {})

        teams_list = []

        # Extract away team
        if "away" in teams:
            away_team = self._extract_team_info(teams["away"], "away")
            teams_list.append(away_team)

        # Extract home team
        if "home" in teams:
            home_team = self._extract_team_info(teams["home"], "home")
            teams_list.append(home_team)

        teams_df = pd.DataFrame(teams_list)

        if not teams_df.empty:
            # Apply data types
            teams_df = teams_df.astype(self.data_types)

        return teams_df

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
        db_handler.insert_dataframe(df, "teams", if_exists=if_exists)
