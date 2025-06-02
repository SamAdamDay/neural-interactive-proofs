"""Types for the language model server, specifying request and response structures."""

from typing import TypedDict, Literal, TypeAlias

ServerStatus: TypeAlias = Literal[
    "online",
    "not_started",
    "exited",
    "not_accepting_connections",
    "server_error",
    "other_error",
]
"""The status of the vLLM server. One of:

- "online": The server is running and accepting connections.
- "not_started": The server has not been started.
- "exited": The server has exited unexpectedly.
- "not_accepting_connections": The server is running but not accepting connections. This
  can happen if the server is still starting up or if it has crashed.
- "server_error": A 5xx error occurred when trying to connect to the server.
- "other_error": Any other error occurred when trying to connect to the server.
"""


class LanguageModelErrorResponse(TypedDict):
    """A typed dictionary for the error response from the language model server."""

    error: str
    """The type of error that occurred."""

    message: str
    """A human-readable message describing the error."""


class VllmStartResponse(TypedDict):
    """A typed dictionary for the response when starting the vLLM server."""

    message: str
    """A message indicating the result of the start operation."""

    model_name: str
    """The name of the model that the vLLM server is serving."""

    port: int
    """The port on which the vLLM server is running."""


class VllmStopResponse(TypedDict):
    """A typed dictionary for the response when stopping the vLLM server."""

    message: str
    """A message indicating the result of the stop operation."""


class VllmStatusResponse(TypedDict):
    """A typed dictionary for the response when checking the vLLM server status."""

    status: ServerStatus
    """The status of the vLLM server, as defined in ServerStatus."""

    error: str | None
    """An error message if the server is not online, otherwise None."""
