"""Utilities for dealing with APIs."""

from abc import ABC


class GenerationError(Exception, ABC):
    """Base class for exceptions raised during generation of the next message"""

    def __init__(self, num_retries: int):
        self.num_retries = num_retries
        super().__init__(f"Generation failed after {num_retries} retries")


class NotGuessedError(GenerationError):
    """Raised when the agent has not made a decision within the max number of turns"""


class ContentFilterError(GenerationError):
    """Raised when the agent's response is blocked by a content filter"""


class UnknownFinishReasonError(GenerationError):
    """Raised when the agent's finishes generating for an unknown reason"""

    def __init__(self, num_retries: int, reason: str):
        self.reason = reason
        self.num_retries = num_retries
        super(GenerationError, self).__init__(
            f"Generation failed after {num_retries} retries with reason {reason!r}"
        )


class InvalidResponseError(GenerationError):
    """Raised when the agent's response is invalid"""

    def __init__(self, num_retries: int, response_text: str):
        self.response_text = response_text
        self.num_retries = num_retries
        super(GenerationError, self).__init__(
            f"Invalid generation after {num_retries} retries. Response: "
            "{response_text!r}"
        )


class InvalidDecisionError(InvalidResponseError):
    """Raised when the agent's decision is invalid (i.e. not accept or reject)"""
