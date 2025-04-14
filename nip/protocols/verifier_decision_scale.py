"""Classes for handling the decision scale of a verifier.

A verifier scale allows for more nuanced decisions than just "accept" or "reject". This
module contains the classes and functions necessary to handle the verifier decision
scale in a protocol.

Verifier scales are only relevant for text-based protocols.
"""

from abc import ABC, abstractmethod
from typing import Literal
from functools import cached_property

from nip.parameters import HyperParameters
from nip.parameters.types import VerifierDecisionScaleType


class VerifierDecisionParseError(Exception):
    """Exception raised when the verifier decision cannot be parsed.

    Parameters
    ----------
    decision_text : str
        The decision text from the verifier model.
    """

    def __init__(self, decision_text: str) -> None:
        super().__init__(f"Verifier decision could not be parsed: {decision_text!r}")
        self.decision_text = decision_text


class VerifierDecisionScaleHandler(ABC):
    """Base class for handling the verifier decision scale.

    Parameters
    ----------
    hyper_params : HyperParameters
        The hyperparameters for the experiment.
    """

    def __init__(self, hyper_params: HyperParameters) -> None:
        self.hyper_params = hyper_params

    @property
    @abstractmethod
    def possible_decision_texts(self) -> tuple[str]:
        """The possible decision texts from the verifier model in order.

        The decision texts should be ordered from reject to accept.
        """

    @abstractmethod
    def extract_decision(self, decision_text: str) -> tuple[Literal[0, 1, 3], float]:
        """Extract the discrete decision and float decision from the decision text.

        Parameters
        ----------
        decision_text : str
            The decision text from the verifier model.

        Returns
        -------
        discrete_decision : Literal[0, 1, 3]
            The discrete decision from the verifier model, with the following meanings:
            - 0: reject
            - 1: accept
            - 3: neither accept nor reject
        continuous_decision : float
            The continuous decision from the verifier model. This is a number between -1 and
            1, where -1 is "reject" and 1 is "accept". This is a more fine-grained
            version of ``discrete_decision``.

        Raises
        ------
        VerifierDecisionParseError
            If the decision text cannot be parsed.
        """


VERIFIER_DECISION_SCALE_HANDLERS: dict[
    VerifierDecisionScaleType, type[VerifierDecisionScaleHandler]
] = {}


def register_verifier_decision_scale_handler(
    decision_scale_type: VerifierDecisionScaleType,
) -> type[VerifierDecisionScaleHandler]:
    """Register a verifier decision scale handler.

    Parameters
    ----------
    decision_scale_type : VerifierDecisionScaleType
        The decision scale type to register the handler for.

    Returns
    -------
    handler_class : type[VerifierDecisionScaleHandler]
        The class of the handler.
    """

    def decorator(
        handler_class: type[VerifierDecisionScaleHandler],
    ) -> type[VerifierDecisionScaleHandler]:
        VERIFIER_DECISION_SCALE_HANDLERS[decision_scale_type] = handler_class
        return handler_class

    return decorator


def build_verifier_decision_scale_handler(
    hyper_params: HyperParameters,
) -> VerifierDecisionScaleHandler:
    """Build the verifier decision scale handler.

    Parameters
    ----------
    hyper_params : HyperParameters
        The hyperparameters for the experiment.

    Returns
    -------
    handler : VerifierDecisionScaleHandler
        The verifier decision scale handler.
    """

    decision_scale_type = hyper_params.protocol_common.verifier_decision_scale
    handler_class = VERIFIER_DECISION_SCALE_HANDLERS.get(decision_scale_type)
    return handler_class(hyper_params)


@register_verifier_decision_scale_handler("accept_reject")
class AcceptRejectVerifierDecisionScaleHandler(VerifierDecisionScaleHandler):
    """Handler for the accept/reject verifier decision scale.

    The decision text is expected to be either "accept" or "reject".
    """

    possible_decision_texts = ("accept", "reject")

    def extract_decision(self, decision_text: str) -> tuple[Literal[0, 1], float]:
        """Extract the discrete decision and float decision from the decision text.

        Parameters
        ----------
        decision_text : str
            The decision text from the verifier model.

        Returns
        -------
        discrete_decision : Literal[0, 1]
            The discrete decision from the verifier model, with the following meanings:
            - 0: reject
            - 1: accept
        continuous_decision : float
            The continuous decision from the verifier model. This is a number between -1 and
            1, where -1 is "reject" and 1 is "accept". This is a more fine-grained
            version of ``discrete_decision``.

        Raises
        ------
        VerifierDecisionParseError
            If the decision text cannot be parsed.
        """
        decision_text = decision_text.strip().lower().split(" ")[0]

        if decision_text == "accept":
            return 1, 1.0
        elif decision_text == "reject":
            return 0, -1.0
        else:
            raise VerifierDecisionParseError(decision_text)


@register_verifier_decision_scale_handler("likert_scale")
@register_verifier_decision_scale_handler("likert_scale_no_undecided")
class LikertScaleVerifierDecisionScaleHandler(VerifierDecisionScaleHandler):
    """Handler for the Likert scale verifier decision scale.

    The decision text is expected to be one of the following:
    - "strongly agree"
    - "agree"
    - "undecided" (not for "likert_scale_no_undecided")
    - "disagree"
    - "strongly disagree"
    """

    @cached_property
    def possible_decision_texts(self) -> tuple[str]:
        """The possible decision texts from the verifier model."""
        if self.hyper_params.protocol_common.verifier_decision_scale == "likert_scale":
            return (
                "strongly disagree",
                "disagree",
                "neither agree nor disagree",
                "agree",
                "strongly agree",
            )
        else:
            return (
                "strongly disagree",
                "disagree",
                "agree",
                "strongly agree",
            )

    def extract_decision(self, decision_text: str) -> tuple[Literal[0, 1, 3], float]:
        """Extract the discrete decision and float decision from the decision text.

        Parameters
        ----------
        decision_text : str
            The decision text from the verifier model.

        Returns
        -------
        discrete_decision : Literal[0, 1, 3]
            The discrete decision from the verifier model, with the following meanings:
            - 0: reject
            - 1: accept
            - 3: neither accept nor reject
        continuous_decision : float
            The continuous decision from the verifier model. This is a number between -1 and
            1, where -1 is "reject" and 1 is "accept". This is a more fine-grained
            version of ``discrete_decision``.

        Raises
        ------
        VerifierDecisionParseError
            If the decision text cannot be parsed.
        """
        decision_text_split = decision_text.strip().lower().split(" ")

        if decision_text_split[:2] == ["strongly", "agree"]:
            return 1, 1.0
        elif decision_text_split[0] == "agree":
            return 1, 0.5
        elif (
            self.hyper_params.protocol_common.verifier_decision_scale == "likert_scale"
            and decision_text_split[:4] == ["neither", "agree", "nor", "disagree"]
        ):
            return 0, 0.0
        elif decision_text_split[0] == "disagree":
            return 0, -0.5
        elif decision_text_split[:2] == ["strongly", "disagree"]:
            return 0, -1.0
        else:
            raise VerifierDecisionParseError(decision_text)


@register_verifier_decision_scale_handler("out_of_10")
class OutOf10VerifierDecisionScaleHandler(VerifierDecisionScaleHandler):
    """Handler for the out of 10 verifier decision scale.

    The decision text is expected to be a number between 0 and 10.
    """

    possible_decision_texts = tuple(str(i) for i in range(11))

    def extract_decision(self, decision_text: str) -> tuple[Literal[0, 1, 3], float]:
        """Extract the discrete decision and float decision from the decision text.

        Parameters
        ----------
        decision_text : str
            The decision text from the verifier model.

        Returns
        -------
        discrete_decision : Literal[0, 1, 3]
            The discrete decision from the verifier model, with the following meanings:
            - 0: reject
            - 1: accept
            - 3: neither accept nor reject
        continuous_decision : float
            The continuous decision from the verifier model. This is a number between -1 and
            1, where -1 is "reject" and 1 is "accept". This is a more fine-grained
            version of ``discrete_decision``.

        Raises
        ------
        VerifierDecisionParseError
            If the decision text cannot be parsed.
        """

        first_word = decision_text.strip().split(" ")[0]
        try:
            decision_value = float(first_word)
        except ValueError:
            raise VerifierDecisionParseError(decision_text) from None
        if not (0 <= decision_value <= 10):
            raise VerifierDecisionParseError(decision_text)

        if decision_value < 5:
            discrete_decision = 0
        elif decision_value == 5:
            discrete_decision = 3
        else:
            discrete_decision = 1
        continuous_decision = (decision_value / 10) * 2 - 1

        return discrete_decision, continuous_decision


@register_verifier_decision_scale_handler("out_of_100")
class OutOf100VerifierDecisionScaleHandler(VerifierDecisionScaleHandler):
    """Handler for the out of 100 verifier decision scale.

    The decision text is expected to be a number between 0 and 100.
    """

    possible_decision_texts = tuple(str(i) for i in range(101))

    def extract_decision(self, decision_text: str) -> tuple[Literal[0, 1, 3], float]:
        """Extract the discrete decision and float decision from the decision text.

        Parameters
        ----------
        decision_text : str
            The decision text from the verifier model.

        Returns
        -------
        discrete_decision : Literal[0, 1, 3]
            The discrete decision from the verifier model, with the following meanings:
            - 0: reject
            - 1: accept
            - 3: neither accept nor reject
        continuous_decision : float
            The continuous decision from the verifier model. This is a number between -1 and
            1, where -1 is "reject" and 1 is "accept". This is a more fine-grained
            version of ``discrete_decision``.

        Raises
        ------
        VerifierDecisionParseError
            If the decision text cannot be parsed.
        """

        first_word = decision_text.strip().split(" ")[0]
        try:
            decision_value = float(first_word)
        except ValueError:
            raise VerifierDecisionParseError(decision_text) from None
        if not (0 <= decision_value <= 100):
            raise VerifierDecisionParseError(decision_text)

        if decision_value < 50:
            discrete_decision = 0
        elif decision_value == 50:
            discrete_decision = 3
        else:
            discrete_decision = 1
        continuous_decision = (decision_value / 100) * 2 - 1

        return discrete_decision, continuous_decision
