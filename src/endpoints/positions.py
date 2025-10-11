import requests

from src.endpoints.base_api import BaseAPI


class Positions(BaseAPI):
    """
    Endpoint wrapper for the MLB Positions API.

    Returns available player positions with codes, names, and abbreviations.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the Positions endpoint.
        """
        super().__init__(*args, **kwargs)
        self.base_url = "https://statsapi.mlb.com/api/v1/positions"

    def get(self, *args, **kwargs) -> dict:
        """
        Fetch available player positions.

        Args:
            **kwargs: Additional query parameters

        Returns:
            dict: List of positions with codes, names, types, and abbreviations

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
