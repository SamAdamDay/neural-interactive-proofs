"""Exceptions for the language model server and client."""

from abc import ABC
from typing import Optional

from requests import Response

from nip.language_model_server.types import VllmServerStatus


class LanguageModelServerError(Exception, ABC):
    """Base exception for errors encountered by the language model server."""

    status_code = 500


class VllmNotInstalledError(LanguageModelServerError):
    """Exception raised when vLLM is not installed."""

    status_code = 503

    def __init__(self):
        super().__init__("vLLM is not installed on the server.")


class VllmNoGpusError(LanguageModelServerError):
    """Exception raised when no GPUs are available for the vLLM server."""

    status_code = 503

    def __init__(self):
        super().__init__(
            "No GPUs available for the vLLM server. Please check your setup."
        )


class VllmModelNotFoundError(LanguageModelServerError):
    """Exception raised when the specified vLLM model is not found."""

    status_code = 404

    def __init__(self, model_name: str, error: Optional[Exception] = None):
        message = f"vLLM model '{model_name}' not found on Hugging Face."
        if error:
            message += f" Error: {error!s}"
        super().__init__(message)
        self.model_name = model_name


class VllmServerNotRunningError(LanguageModelServerError):
    """Exception raised when trying to stop a vLLM server that is not running."""

    status_code = 404

    def __init__(self):
        super().__init__("vLLM server is not running. Cannot stop it.")


class VllmServerError(LanguageModelServerError):
    """Exception raised when there is an error with the vLLM server.

    Parameters
    ----------
    status: VllmServerStatus
        The status of the vLLM server
    """

    def __init__(self, status: VllmServerStatus):
        super().__init__(f"vLLM server error. Status: {status!r}")
        self.status = status


class AccelerateConfigNotFoundError(LanguageModelServerError):
    """Exception raised when the accelerate configuration file is not found."""

    status_code = 500

    def __init__(self, config_path: str):
        super().__init__(f"Accelerate configuration file '{config_path}' not found.")
        self.config_path = config_path


class MaxTrainingJobsReachedError(LanguageModelServerError):
    """Exception raised when the maximum number of training jobs is reached."""

    status_code = 429

    def __init__(self, max_jobs: int):
        super().__init__(f"Maximum number of training jobs ({max_jobs}) reached.")
        self.max_jobs = max_jobs


class TrainingJobNotFoundServerError(LanguageModelServerError):
    """Exception raised when a training job is not found."""

    status_code = 404

    def __init__(self, job_id: str):
        super().__init__(f"Training job with ID '{job_id}' not found.")
        self.job_id = job_id


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


class TrainingJobNotFoundClientError(LanguageModelClientError):
    """Exception raised when a training job is not found."""

    def __init__(self, job_id: str):
        super().__init__(f"Training job with ID '{job_id}' not found.")
        self.job_id = job_id


class ClientTimeoutError(LanguageModelClientError):
    """Exception raised when a request to the language model server times out.

    Parameters
    ----------
    message : str
        A message describing the timeout error.
    """
