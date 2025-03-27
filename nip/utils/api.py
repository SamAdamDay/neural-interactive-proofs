"""Utilities for dealing with APIs."""

from abc import ABC
from typing import Optional, Any
from inspect import signature


class GenerationError(Exception, ABC):
    """Base class for exceptions raised during generation of the next message.

    Parameters
    ----------
    num_retries : int, optional
        The number of retries that have been attempted, or ``None`` if this data is not
        recorded.
    """

    @property
    def display_message(self) -> str:
        """The message to display to the user."""
        if self.num_retries is None:
            return "Generation failed"
        else:
            return f"Generation failed after {self.num_retries} retries"

    def __init__(self, num_retries: Optional[int] = None):
        self.num_retries = num_retries
        super().__init__(self.display_message)

    def copy_with_retries(self, num_retries: int) -> "GenerationError":
        """Return a copy of the error with the number of retries updated.

        Parameters
        ----------
        num_retries : int
            The number of retries to set.

        Returns
        -------
        GenerationError
            A copy of the error with the number of retries set to ``num_retries``.
        """

        kwargs = {
            key: getattr(self, key)
            for key in signature(self.__class__).parameters.keys()
        }
        kwargs["num_retries"] = num_retries
        return self.__class__(**kwargs)


class ConnectionError(GenerationError, ABC):
    """Base class for exceptions regarding connection errors.

    Parameters
    ----------
    message : str
        The error message.
    metadata : Any, optional
        Any additional metadata provided by the API.
    num_retries : int, optional
        The number of retries that have been attempted, or ``None`` if this data is not
        recorded.
    """

    name: str = "Connection Error"
    """The human-readable name of the error."""

    code: Optional[str] = None
    """The error code, if available."""

    @property
    def display_message(self) -> str:
        """The message to display to the user."""
        message = self.name
        if self.code is not None:
            message += f" (code: {self.code})"
        message += f": {self.message}"
        if self.num_retries is not None:
            message += f"\n Number of retries: {self.num_retries}"
        if self.metadata is not None:
            message += f"\n Metadata: {self.metadata}"
        return message

    def __init__(
        self,
        message: str,
        metadata: Optional[Any] = None,
        num_retries: Optional[int] = None,
    ):
        self.message = message
        self.metadata = metadata
        super().__init__(num_retries=num_retries)


class GenericConnectionError(ConnectionError):
    """A generic connection error, which doesn't have its own specific class."""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        metadata: Optional[Any] = None,
        num_retries: Optional[int] = None,
    ):
        self.code = code
        super().__init__(
            message=message,
            metadata=metadata,
            num_retries=num_retries,
        )


class RateLimitError(ConnectionError):
    """Raised when the rate limit is exceeded."""

    name = "Rate Limit Error"
    code = 429


class TimeoutError(ConnectionError):
    """Raised when the request times out."""

    name = "Timeout Error"
    code = 408


class InsufficientCreditsError(ConnectionError):
    """Raised when the user does not have enough credits to make a request."""

    name = "Insufficient Credits Error"
    code = 402


class NotGuessedError(GenerationError):
    """Raised when the agent has not made a decision within the max number of turns."""


class ContentFilterError(GenerationError):
    """Raised when the agent's response is blocked by a content filter."""


class ContentIsNoneError(GenerationError):
    """Raised when the content of the generated message is ``None``."""


class UnknownFinishReasonError(GenerationError):
    """Raised when the agent's finishes generating for an unknown reason."""

    def __init__(self, reason: str, num_retries: Optional[int] = None):
        self.reason = reason
        self.num_retries = num_retries
        if num_retries is None:
            super(GenerationError, self).__init__(
                f"Generation failed with reason {reason!r}"
            )
        else:
            super(GenerationError, self).__init__(
                f"Generation failed after {num_retries} retries with reason {reason!r}"
            )


class InvalidResponseError(GenerationError):
    """Raised when the agent's response is invalid."""

    def __init__(self, response_text: str, num_retries: Optional[int] = None):
        self.response_text = response_text
        self.num_retries = num_retries
        if num_retries is None:
            super(GenerationError, self).__init__(
                f"Invalid generation. Response: {response_text!r}"
            )
        else:
            super(GenerationError, self).__init__(
                f"Invalid generation after {num_retries} retries. Response: "
                f"{response_text!r}"
            )


class InvalidDecisionError(InvalidResponseError):
    """Raised when the agent's decision is invalid (i.e. not accept or reject)."""


class NotAllActiveChannelsInResponseError(InvalidResponseError):
    """Raised when the response does not contain messages for all active channels."""
