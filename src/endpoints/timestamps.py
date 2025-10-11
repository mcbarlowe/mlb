import requests

from src.endpoints.base_api import BaseAPI


class Timestamps(BaseAPI):
    """
    Endpoint wrapper for fetching game update timestamps.

    Returns a list of timestamps for all updates during a selected game.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the Timestamps endpoint.
        """
        super().__init__(*args, **kwargs)
        self.base_url = "https://statsapi.mlb.com/api/v1.1/game"

    def get(self, game_pk: int, *args, **kwargs) -> dict:
        """
        Fetch list of update timestamps for a specific game.

        Args:
            game_pk (int): MLBAM unique game identifier
            **kwargs: Additional query parameters

        Returns:
            dict: List of timestamps for game updates

        Raises:
            Exception: If HTTP error or request error occurs
        """
        url = f"{self.base_url}/{game_pk}/feed/live/timestamps"

        response = requests.get(url, params=kwargs)

        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            raise Exception(f"HTTP error occurred: {e}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request error occurred: {e}")
