"""Utilities for dealing with APIs."""

from abc import ABC
from typing import Optional
from inspect import signature


class GenerationError(Exception, ABC):
    """Base class for exceptions raised during generation of the next message."""

    def __init__(self, num_retries: Optional[int] = None):
        self.num_retries = num_retries
        if num_retries is None:
            super().__init__("Generation failed")
        else:
            super().__init__(f"Generation failed after {num_retries} retries")

    def copy_with_retries(self, num_retries: int) -> "GenerationError":
        """Return a copy of the error with the number of retries updated.

        Parameters
        ----------
        num_retries : int
            The number of retries to set.

        Returns
        -------
        GenerationError
            A copy of the error with the number of retries set to `num_retries`.
        """

        kwargs = {
            key: getattr(self, key)
            for key in signature(self.__class__).parameters.keys()
        }
        kwargs["num_retries"] = num_retries
        return self.__class__(**kwargs)


class NotGuessedError(GenerationError):
    """Raised when the agent has not made a decision within the max number of turns."""


class ContentFilterError(GenerationError):
    """Raised when the agent's response is blocked by a content filter."""


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
