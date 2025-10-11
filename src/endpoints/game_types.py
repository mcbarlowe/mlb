import requests

from src.endpoints.base_api import BaseAPI


class GameTypes(BaseAPI):
    """
    Endpoint wrapper for the MLB Game Types API.

    Returns available game types (e.g., Regular Season, Playoffs, Spring Training).
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the GameTypes endpoint.
        """
        super().__init__(*args, **kwargs)
        self.base_url = "https://statsapi.mlb.com/api/v1/gameTypes"

    def get(self, *args, **kwargs) -> dict:
        """
        Fetch available game types.

        Args:
            **kwargs: Additional query parameters

        Returns:
            dict: List of game types

        Raises:
            Exception: If HTTP error or request error occurs
        """
        response = requests.get(self.base_url, params=kwargs)

        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            raise Exception(f"HTTP error occurred: {e}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request error occurred: {e}")
