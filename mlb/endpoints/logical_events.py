import requests

from mlb.endpoints.base_api import AsyncBaseAPI, BaseAPI


class LogicalEvents(BaseAPI, AsyncBaseAPI):
    """
    Endpoint wrapper for the MLB Logical Events API.

    Returns available logical event types.
    Supports both sync and async operations.
    """

    def __init__(self, concurrency_limit: int = 15, *args, **kwargs) -> None:
        """
        Initialize the LogicalEvents endpoint.
        """
        BaseAPI.__init__(self, *args, **kwargs)
        AsyncBaseAPI.__init__(self, concurrency_limit, *args, **kwargs)
        self.base_url = "https://statsapi.mlb.com/api/v1/logicalEvents"

    def get(self, *args, **kwargs) -> dict:
        """
        Fetch available logical events (sync).

        Args:
            **kwargs: Additional query parameters

        Returns:
            dict: List of logical events

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

    async def get_async(self, **kwargs) -> dict:
        """
        Fetch available logical events (async).

        Returns:
            dict: List of logical events
        """
        return await self._request_with_retry(self.base_url, params=kwargs)
