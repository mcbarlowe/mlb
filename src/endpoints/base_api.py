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
