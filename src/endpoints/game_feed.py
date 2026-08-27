import asyncio
from collections.abc import Callable

import requests

from src.endpoints.base_api import AsyncBaseAPI, BaseAPI


class GameFeed(BaseAPI, AsyncBaseAPI):
    """
    Endpoint wrapper for the MLB Game Feed API (GUMBO).

    The GUMBO (Grand Unified Master Baseball Object) provides complete
    game state information including plays, linescore, boxscore, and more.

    Supports both sync and async operations:
    - Sync: game_feed.get(game_pk)
    - Async: async with GameFeed() as gf: await gf.get_async(game_pk)
    """

    def __init__(self, concurrency_limit: int = 15, *args, **kwargs) -> None:
        """
        Initialize the GameFeed endpoint.

        Args:
            concurrency_limit: Maximum concurrent requests for async operations
        """
        BaseAPI.__init__(self, *args, **kwargs)
        AsyncBaseAPI.__init__(self, concurrency_limit, *args, **kwargs)
        self.base_url = "https://statsapi.mlb.com/api/v1.1/game"

    def get(self, game_pk: int, timecode: str | None = None, *args, **kwargs) -> dict:
        """
        Fetch live game feed data for a specific game (sync).

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

    async def get_async(
        self, game_pk: int, timecode: str | None = None, **kwargs
    ) -> dict:
        """
        Fetch live game feed data for a specific game (async).

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

        params = kwargs.copy()
        if timecode:
            params['timecode'] = timecode

        headers = {"Accept-Encoding": "gzip"}

        return await self._request_with_retry(url, params=params, headers=headers)

    async def get_many_async(
        self,
        game_pks: list[int],
        on_success: Callable[[int, dict], None] | None = None,
        on_error: Callable[[int, Exception], None] | None = None,
    ) -> list[dict | None]:
        """
        Fetch multiple game feeds concurrently.

        Args:
            game_pks: List of MLBAM game identifiers to fetch
            on_success: Callback for successful fetches (game_pk, result)
            on_error: Callback for failed fetches (game_pk, exception)

        Returns:
            list: List of results in same order as game_pks (None for failed items)
        """
        async def fetch_one(game_pk: int) -> dict | None:
            try:
                result = await self.get_async(game_pk)
                if on_success:
                    on_success(game_pk, result)
                return result
            except Exception as e:
                if on_error:
                    on_error(game_pk, e)
                return None

        tasks = [fetch_one(game_pk) for game_pk in game_pks]
        return await asyncio.gather(*tasks)
