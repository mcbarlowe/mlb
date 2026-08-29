import requests

from mlb.endpoints.base_api import AsyncBaseAPI, BaseAPI


class Schedule(BaseAPI, AsyncBaseAPI):
    """
    Endpoint wrapper for the MLB Schedule API.

    Supports both sync and async operations:
    - Sync: schedule.get(sportId=1, season=2024)
    - Async: async with Schedule() as s: await s.get_async(sportId=1, season=2024)
    """

    def __init__(self, concurrency_limit: int = 15, *args, **kwargs) -> None:
        """
        Initialize the Schedule endpoint.
        """
        BaseAPI.__init__(self, *args, **kwargs)
        AsyncBaseAPI.__init__(self, concurrency_limit, *args, **kwargs)
        self.base_url = "https://statsapi.mlb.com/api/v1/schedule/games"

    def get(self, *args, **kwargs) -> dict:
        """
        Handle GET requests to fetch the schedule (sync).
        """
        response = requests.get(self.base_url, params=kwargs)
        try:
            response.raise_for_status()  # Raise an error for bad responses
            return response.json()
        except requests.exceptions.HTTPError as e:
            raise Exception(f"HTTP error occurred: {e}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request error occurred: {e}")

    async def get_async(self, **kwargs) -> dict:
        """
        Handle GET requests to fetch the schedule (async).

        Args:
            **kwargs: Query parameters (e.g., sportId, season)

        Returns:
            dict: Schedule JSON response
        """
        return await self._request_with_retry(self.base_url, params=kwargs)
