from pathlib import Path

import pandas as pd

from mlb.data._type_utils import coerce_dataframe_types


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

    def _extract_team_info(self, team: dict) -> dict:
        """Extract team information from a team object."""
        team_info = {}

        team_info["team_id"] = team.get("id")
        team_info["team_name"] = team.get("name")
        team_info["team_code"] = team.get("teamCode")
        team_info["file_code"] = team.get("fileCode")
        team_info["abbreviation"] = team.get("abbreviation")
        team_info["team_name_short"] = team.get("teamName")
        team_info["location_name"] = team.get("locationName")
        team_info["first_year_of_play"] = team.get("firstYearOfPlay")

        league = team.get("league", {})
        team_info["league_id"] = league.get("id")
        team_info["league_name"] = league.get("name")

        division = team.get("division", {})
        team_info["division_id"] = division.get("id")
        team_info["division_name"] = division.get("name")

        sport = team.get("sport", {})
        team_info["sport_id"] = sport.get("id")
        team_info["sport_name"] = sport.get("name")

        venue = team.get("venue", {})
        team_info["venue_id"] = venue.get("id")
        team_info["venue_name"] = venue.get("name")

        spring_league = team.get("springLeague", {})
        team_info["spring_league_id"] = spring_league.get("id")
        team_info["spring_league_name"] = spring_league.get("name")
        team_info["spring_league_abbrev"] = spring_league.get("abbreviation")

        team_info["parent_org_name"] = team.get("parentOrgName")
        team_info["parent_org_id"] = team.get("parentOrgId")

        team_info["all_star_status"] = team.get("allStarStatus", False)
        team_info["active"] = team.get("active", True)

        return team_info

    def transform(self, data: dict) -> pd.DataFrame:
        """Transform gameData.teams into a team dimension DataFrame."""
        game_data = data.get("gameData", {})
        teams = game_data.get("teams", {})

        teams_list = []

        if "away" in teams:
            teams_list.append(self._extract_team_info(teams["away"]))

        if "home" in teams:
            teams_list.append(self._extract_team_info(teams["home"]))

        teams_df = pd.DataFrame(teams_list)

        if not teams_df.empty:
            teams_df = coerce_dataframe_types(teams_df, self.data_types)

        return teams_df

    def save(self, df: pd.DataFrame, output_path: Path, format: str = "parquet") -> None:
        """Save DataFrame to file."""
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
        """Save DataFrame to the configured PostgreSQL database."""
        db_handler.insert_dataframe(df, "teams", if_exists=if_exists)
