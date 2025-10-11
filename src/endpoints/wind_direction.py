import requests

from src.endpoints.base_api import BaseAPI


class WindDirection(BaseAPI):
    """
    Endpoint wrapper for the MLB Wind Direction API.

    Returns available wind direction codes and descriptions.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the WindDirection endpoint.
        """
        super().__init__(*args, **kwargs)
        self.base_url = "https://statsapi.mlb.com/api/v1/windDirection"

    def get(self, *args, **kwargs) -> dict:
        """
        Fetch available wind direction values.

        Args:
            **kwargs: Additional query parameters

        Returns:
            dict: List of wind direction codes and descriptions

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
