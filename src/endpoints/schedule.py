import requests

from src.endpoints.base_api import BaseAPI


class Schedule(BaseAPI):
    """
    Endpoint wrapper for the MLB Schedule API.
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the Schedule endpoint.
        """
        super().__init__(*args, **kwargs)
        self.base_url = "https://statsapi.mlb.com/api/v1/schedule/games"

    def get(self, *args, **kwargs) -> dict:
        """
        Handle GET requests to fetch the schedule.
        """
        response = requests.get(self.base_url, params=kwargs)
        try:
            response.raise_for_status()  # Raise an error for bad responses
            return response.json()
        except requests.exceptions.HTTPError as e:
            raise Exception(f"HTTP error occurred: {e}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request error occurred: {e}")
