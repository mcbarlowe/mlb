import requests

from src.endpoints.base_api import BaseAPI


class GameFeed(BaseAPI):
    """
    Endpoint wrapper for the MLB Game Feed API (GUMBO).

    The GUMBO (Grand Unified Master Baseball Object) provides complete
    game state information including plays, linescore, boxscore, and more.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the GameFeed endpoint.
        """
        super().__init__(*args, **kwargs)
        self.base_url = "https://statsapi.mlb.com/api/v1.1/game"

    def get(self, game_pk: int, timecode: str = None, *args, **kwargs) -> dict:
        """
        Fetch live game feed data for a specific game.

        Args:
            game_pk (int): MLBAM unique game identifier
            timecode (str, optional): Specific point in time (format: yyyymmdd_######)
            **kwargs: Additional query parameters

        Returns:
            dict: Complete game state JSON response

        Raises:
            Exception: If HTTP error or request error occurs
        """
        url = f"{self.base_url}/{game_pk}/feed/live"

        # Add timecode if provided
        params = kwargs.copy()
        if timecode:
            params['timecode'] = timecode

        # Use gzip encoding for efficiency
        headers = {"Accept-Encoding": "gzip"}

        response = requests.get(url, params=params, headers=headers)

        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            raise Exception(f"HTTP error occurred: {e}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request error occurred: {e}")
