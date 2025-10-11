import requests

from src.endpoints.base_api import BaseAPI


class LogicalEvents(BaseAPI):
    """
    Endpoint wrapper for the MLB Logical Events API.

    Returns available logical event types.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the LogicalEvents endpoint.
        """
        super().__init__(*args, **kwargs)
        self.base_url = "https://statsapi.mlb.com/api/v1/logicalEvents"

    def get(self, *args, **kwargs) -> dict:
        """
        Fetch available logical events.

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
