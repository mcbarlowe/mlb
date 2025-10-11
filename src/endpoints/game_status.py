import requests

from src.endpoints.base_api import BaseAPI


class GameStatus(BaseAPI):
    """
    Endpoint wrapper for the MLB Game Status API.

    Returns available game status codes and descriptions.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the GameStatus endpoint.
        """
        super().__init__(*args, **kwargs)
        self.base_url = "https://statsapi.mlb.com/api/v1/gameStatus"

    def get(self, *args, **kwargs) -> dict:
        """
        Fetch available game status codes.

        Args:
            **kwargs: Additional query parameters

        Returns:
            dict: List of game status codes and descriptions

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
