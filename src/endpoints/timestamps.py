import requests

from src.endpoints.base_api import AsyncBaseAPI, BaseAPI


class Timestamps(BaseAPI, AsyncBaseAPI):
    """
    Endpoint wrapper for fetching game update timestamps.

    Returns a list of timestamps for all updates during a selected game.
    Supports both sync and async operations.
    """

    def __init__(self, concurrency_limit: int = 15, *args, **kwargs) -> None:
        """
        Initialize the Timestamps endpoint.
        """
        BaseAPI.__init__(self, *args, **kwargs)
        AsyncBaseAPI.__init__(self, concurrency_limit, *args, **kwargs)
        self.base_url = "https://statsapi.mlb.com/api/v1.1/game"

    def get(self, game_pk: int, *args, **kwargs) -> dict:
        """
        Fetch list of update timestamps for a specific game (sync).

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

    async def get_async(self, game_pk: int, **kwargs) -> dict:
        """
        Fetch list of update timestamps for a specific game (async).

        Args:
            game_pk (int): MLBAM unique game identifier
            **kwargs: Additional query parameters

        Returns:
            dict: List of timestamps for game updates
        """
        url = f"{self.base_url}/{game_pk}/feed/live/timestamps"
        return await self._request_with_retry(url, params=kwargs)
