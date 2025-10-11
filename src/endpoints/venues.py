import requests

from src.endpoints.base_api import BaseAPI


class Venues(BaseAPI):
    """
    Endpoint wrapper for the MLB Venues API.

    Returns information about MLB stadiums and venues.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the Venues endpoint.
        """
        super().__init__(*args, **kwargs)
        self.base_url = "https://statsapi.mlb.com/api/v1/venues"

    def get(self, season: int = None, *args, **kwargs) -> dict:
        """
        Fetch venue information.

        Args:
            season (int, optional): Season year to filter venues
            **kwargs: Additional query parameters

        Returns:
            dict: List of venues with location, coordinates, and timezone info

        Raises:
            Exception: If HTTP error or request error occurs
        """
        params = kwargs.copy()
        if season:
            params['season'] = season

        response = requests.get(self.base_url, params=params)

        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            raise Exception(f"HTTP error occurred: {e}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request error occurred: {e}")
