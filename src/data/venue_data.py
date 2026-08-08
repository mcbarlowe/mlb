from pathlib import Path

import pandas as pd

from src.data._type_utils import coerce_dataframe_types


class VenueData:
    """
    Data transformation class for MLB Venue dimension data.

    Extracts venue information from GUMBO gameData.venue node
    to create a normalized venue dimension table.
    """

    def __init__(self):
        """Initialize the VenueData transformer."""
        self.data_types = {
            "venue_id": int,
            "venue_name": str,
            "venue_link": str,
            "active": bool,
            "season": str,
            "address": str,
            "city": str,
            "state": str,
            "state_abbrev": str,
            "country": str,
            "postal_code": str,
            "latitude": float,
            "longitude": float,
            "elevation": float,
            "azimuth_angle": float,
            "timezone_id": str,
            "timezone": str,
            "timezone_offset": float,
            "capacity": float,
            "turf_type": str,
            "roof_type": str,
            "left_line": float,
            "left_center": float,
            "center": float,
            "right_center": float,
            "right_line": float,
        }

    def _extract_venue_info(self, venue: dict) -> dict:
        """
        Extract venue information from the venue object.

        Args:
            venue (dict): Venue object from gameData.venue

        Returns:
            dict: Flattened venue information
        """
        venue_info = {}

        # Basic info
        venue_info["venue_id"] = venue.get("id")
        venue_info["venue_name"] = venue.get("name")
        venue_info["venue_link"] = venue.get("link")
        venue_info["active"] = venue.get("active")
        venue_info["season"] = venue.get("season")

        # Location info
        location = venue.get("location", {})
        venue_info["address"] = location.get("address1")
        venue_info["city"] = location.get("city")
        venue_info["state"] = location.get("state")
        venue_info["state_abbrev"] = location.get("stateAbbrev")
        venue_info["country"] = location.get("country")
        venue_info["postal_code"] = location.get("postalCode")

        # Coordinates
        coordinates = location.get("defaultCoordinates", {})
        venue_info["latitude"] = coordinates.get("latitude")
        venue_info["longitude"] = coordinates.get("longitude")
        venue_info["elevation"] = location.get("elevation")
        venue_info["azimuth_angle"] = location.get("azimuthAngle")

        # Timezone
        timezone = venue.get("timeZone", {})
        venue_info["timezone_id"] = timezone.get("id")
        venue_info["timezone"] = timezone.get("tz")
        venue_info["timezone_offset"] = timezone.get("offset")

        # Field dimensions and info
        field_info = venue.get("fieldInfo", {})
        venue_info["capacity"] = field_info.get("capacity")
        venue_info["turf_type"] = field_info.get("turfType")
        venue_info["roof_type"] = field_info.get("roofType")
        venue_info["left_line"] = field_info.get("leftLine")
        venue_info["left_center"] = field_info.get("leftCenter")
        venue_info["center"] = field_info.get("center")
        venue_info["right_center"] = field_info.get("rightCenter")
        venue_info["right_line"] = field_info.get("rightLine")

        return venue_info

    def transform(self, data: dict) -> pd.DataFrame:
        """
        Transform gameData.venue into venue dimension DataFrame.

        Args:
            data (dict): Raw GUMBO JSON response

        Returns:
            pd.DataFrame: Venue dimension data with one row
        """
        game_data = data.get("gameData", {})
        venue = game_data.get("venue", {})

        if not venue:
            return pd.DataFrame()

        venue_info = self._extract_venue_info(venue)
        venue_df = pd.DataFrame([venue_info])

        if not venue_df.empty:
            venue_df = coerce_dataframe_types(venue_df, self.data_types)

        return venue_df

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
        db_handler.insert_dataframe(df, "venues", if_exists=if_exists)
