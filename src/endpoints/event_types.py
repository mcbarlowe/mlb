import requests

from src.endpoints.base_api import BaseAPI


class EventTypes(BaseAPI):
    """
    Endpoint wrapper for the MLB Event Types API.

    Returns available event types (e.g., atBat, substitutions, stolen base, etc.).
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the EventTypes endpoint.
        """
        super().__init__(*args, **kwargs)
        self.base_url = "https://statsapi.mlb.com/api/v1/eventTypes"

    def get(self, *args, **kwargs) -> dict:
        """
        Fetch available event types.

        Args:
            **kwargs: Additional query parameters

        Returns:
            dict: List of event types

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
