import requests

from src.endpoints.base_api import BaseAPI


class PitchTypes(BaseAPI):
    """
    Endpoint wrapper for the MLB Pitch Types API.

    Returns available pitch type classifications (e.g., Fastball, Curveball, etc.).
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the PitchTypes endpoint.
        """
        super().__init__(*args, **kwargs)
        self.base_url = "https://statsapi.mlb.com/api/v1/pitchTypes"

    def get(self, *args, **kwargs) -> dict:
        """
        Fetch available pitch types.

        Args:
            **kwargs: Additional query parameters

        Returns:
            dict: List of pitch types with codes and descriptions

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
