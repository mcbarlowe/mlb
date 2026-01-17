import requests

from src.endpoints.base_api import AsyncBaseAPI, BaseAPI


class Venues(BaseAPI, AsyncBaseAPI):
    """
    Endpoint wrapper for the MLB Venues API.

    Returns information about MLB stadiums and venues.
    Supports both sync and async operations.
    """

    def __init__(self, concurrency_limit: int = 15, *args, **kwargs) -> None:
        """
        Initialize the Venues endpoint.
        """
        BaseAPI.__init__(self, *args, **kwargs)
        AsyncBaseAPI.__init__(self, concurrency_limit, *args, **kwargs)
        self.base_url = "https://statsapi.mlb.com/api/v1/venues"

    def get(self, season: int = None, *args, **kwargs) -> dict:
        """
        Fetch venue information (sync).

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

    async def get_async(self, season: int = None, **kwargs) -> dict:
        """
        Fetch venue information (async).

        Args:
            season (int, optional): Season year to filter venues
            **kwargs: Additional query parameters

        Returns:
            dict: List of venues with location, coordinates, and timezone info
        """
        params = kwargs.copy()
        if season:
            params['season'] = season

        return await self._request_with_retry(self.base_url, params=params)
