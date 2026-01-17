import asyncio
import random
from typing import Callable, Optional

import httpx


class BaseAPI:
    """
    Base class for API endpoints.
    This class can be extended to create specific API endpoints.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the base API endpoint.
        """
        pass

    def get(self, *args, **kwargs):
        """
        Handle GET requests.
        Override this method in subclasses to implement specific logic.
        """
        raise NotImplementedError("GET method not implemented.")

    def post(self, *args, **kwargs):
        """
        Handle POST requests.
        Override this method in subclasses to implement specific logic.
        """
        raise NotImplementedError("POST method not implemented.")


class AsyncBaseAPI:
    """
    Async base class for API endpoints with rate limiting and connection pooling.

    Provides:
    - asyncio.Semaphore for concurrency control
    - httpx.AsyncClient for connection pooling
    - Async context manager protocol for resource management
    - Retry logic with exponential backoff
    """

    def __init__(self, concurrency_limit: int = 15, *args, **kwargs):
        """
        Initialize the async base API endpoint.

        Args:
            concurrency_limit: Maximum concurrent requests (default 15)
        """
        self.concurrency_limit = concurrency_limit
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._client: Optional[httpx.AsyncClient] = None
        self.base_url: str = ""

    async def __aenter__(self):
        """Async context manager entry - initialize resources."""
        self._semaphore = asyncio.Semaphore(self.concurrency_limit)
        self._client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(
                max_connections=self.concurrency_limit + 5,
                max_keepalive_connections=self.concurrency_limit,
            ),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._semaphore = None

    async def _request_with_retry(
        self,
        url: str,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> dict:
        """
        Make an HTTP GET request with retry logic and exponential backoff.

        Args:
            url: The URL to request
            params: Optional query parameters
            headers: Optional request headers
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff (seconds)

        Returns:
            dict: JSON response

        Raises:
            Exception: If all retries fail
        """
        if self._client is None:
            raise RuntimeError(
                "AsyncBaseAPI must be used as an async context manager"
            )

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                async with self._semaphore:
                    response = await self._client.get(
                        url, params=params, headers=headers
                    )
                    response.raise_for_status()
                    return response.json()

            except httpx.HTTPStatusError as e:
                last_exception = e
                # Don't retry 4xx errors (except 429)
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    raise Exception(f"HTTP error occurred: {e}")

                # Retry on 429 (rate limit) or 5xx errors
                if attempt < max_retries:
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                    continue
                raise Exception(f"HTTP error occurred after {max_retries} retries: {e}")

            except httpx.RequestError as e:
                last_exception = e
                if attempt < max_retries:
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                    continue
                raise Exception(f"Request error occurred after {max_retries} retries: {e}")

        raise Exception(f"Request failed after {max_retries} retries: {last_exception}")

    async def get_async(self, *args, **kwargs) -> dict:
        """
        Handle async GET requests.
        Override this method in subclasses to implement specific logic.
        """
        raise NotImplementedError("get_async method not implemented.")

    async def get_many_async(
        self,
        items: list,
        on_success: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ) -> list:
        """
        Fetch multiple items concurrently.
        Override this method in subclasses for batch operations.

        Args:
            items: List of items to fetch
            on_success: Callback for successful fetches (item, result)
            on_error: Callback for failed fetches (item, exception)

        Returns:
            list: List of results (None for failed items)
        """
        raise NotImplementedError("get_many_async method not implemented.")
