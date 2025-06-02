"""Exceptions for the language model server and client."""

from abc import ABC
from typing import Optional

from requests import Response

from nip.language_model_server.types import LanguageModelErrorResponse


class LanguageModelServerError(Exception, ABC):
    """Base exception for errors encountered by the language model server."""

    status_code = 500

    def to_dict(self) -> LanguageModelErrorResponse:
        """Convert the exception to a dictionary for JSON serialization.

        Returns
        -------
        dict_representation : LanguageModelErrorResponse
            A dictionary representation of the exception, suitable for JSON
            serialization.
        """
        return {
            "error": self.__class__.__name__,
            "message": str(self),
        }


class VllmNotInstalledError(LanguageModelServerError):
    """Exception raised when vLLM is not installed."""

    def __init__(self):
        super().__init__("vLLM is not installed on the server.")


class VllmServerNotRunningError(LanguageModelServerError):
    """Exception raised when trying to stop a vLLM server that is not running."""

    status_code = 404

    def __init__(self):
        super().__init__("vLLM server is not running. Cannot stop it.")


class LanguageModelClientError(Exception, ABC):
    """Base exception for errors encountered by the language model client."""


class BadResponseError(LanguageModelClientError):
    """Exception raised when the server returns an invalid response.

    Parameters
    ----------
    message : str
        A message describing the error.
    response : Optional[Response], optional
        The response object from the server, if available. This can be useful for
        debugging or logging purposes.
    """

    def __init__(self, message: str, response: Optional[Response] = None):
        super().__init__(message)
        self.response = response


class TimeoutError(LanguageModelClientError):
    """Exception raised when a request to the language model server times out.

    Parameters
    ----------
    message : str
        A message describing the timeout error.
    """
