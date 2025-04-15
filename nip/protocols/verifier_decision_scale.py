"""Classes for handling the decision scale of a verifier.

A verifier scale allows for more nuanced decisions than just "accept" or "reject". This
module contains the classes and functions necessary to handle the verifier decision
scale in a protocol.

Verifier scales are only relevant for text-based protocols.
"""

from abc import ABC, abstractmethod
from typing import Literal
from functools import cached_property
import re

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
    def decision_texts_and_outcomes(self) -> list[tuple[str, Literal[0, 1, 3], float]]:
        """The possible decision texts and discrete and continuous outcomes, in order.

        This is a list of tuples, where each tuple contains the following:

        - The decision text (str)
        - The discrete outcome (int): 0 for reject, 1 for accept, 3 for neither accept
          nor reject.
        - The continuous outcome (float): between -1 and 1, where -1 is reject and 1 is
          accept.

        The decision texts should be ordered from reject to accept (i.e. the continuous
        outcome value should be increasing)
        """

    @cached_property
    def possible_decision_texts(self) -> list[str]:
        """The possible decision texts from the verifier model in order."""
        return [
            decision_text for decision_text, _, _ in self.decision_texts_and_outcomes
        ]

    def extract_decision(
        self, decision_text: str
    ) -> tuple[Literal[0, 1, 3], float, str]:
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
            The continuous decision from the verifier model. This is a number between -1
            and 1, where -1 is "reject" and 1 is "accept". This is a more fine-grained
            version of ``discrete_decision``.
        raw_decision_text : str
            The raw decision text from the verifier model, which should be an element of
            ``self.possible_decision_texts``.

        Raises
        ------
        VerifierDecisionParseError
            If the decision text cannot be parsed.
        """

        decision_text_normalised = decision_text.strip().lower()

        for (
            possible_decision_text,
            discrete_decision,
            continuous_decision,
        ) in self.decision_texts_and_outcomes:
            if decision_text_normalised.startswith(possible_decision_text):
                return (
                    discrete_decision,
                    continuous_decision,
                    possible_decision_text,
                )

        raise VerifierDecisionParseError(decision_text)


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

    decision_texts_and_outcomes = [
        ("reject", 0, -1.0),
        ("accept", 1, 1.0),
    ]


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
    def decision_texts_and_outcomes(self) -> list[tuple[str, Literal[0, 1, 3], float]]:
        """The possible decision texts and discrete and continuous outcomes, in order.

        This is a list of tuples, where each tuple contains the following:

        - The decision text (str)
        - The discrete outcome (int): 0 for reject, 1 for accept, 3 for neither accept
          nor reject.
        - The continuous outcome (float): between -1 and 1, where -1 is reject and 1 is
          accept.

        The decision texts should be ordered from reject to accept (i.e. the continuous
        outcome value should be increasing)
        """
        if self.hyper_params.protocol_common.verifier_decision_scale == "likert_scale":
            return [
                ("strongly disagree", 0, -1.0),
                ("disagree", 0, -0.5),
                ("neither agree nor disagree", 3, 0.0),
                ("agree", 1, 0.5),
                ("strongly agree", 1, 1.0),
            ]
        else:
            return [
                ("strongly disagree", 0, -1.0),
                ("disagree", 0, -0.5),
                ("agree", 1, 0.5),
                ("strongly agree", 1, 1.0),
            ]


@register_verifier_decision_scale_handler("out_of_10")
class OutOf10VerifierDecisionScaleHandler(VerifierDecisionScaleHandler):
    """Handler for the out of 10 verifier decision scale.

    The decision text is expected to be a number between 0 and 10.
    """

    decision_texts_and_outcomes = []

    for decision_value in range(11):
        if decision_value < 5:
            discrete_decision = 0
        elif decision_value == 5:
            discrete_decision = 3
        else:
            discrete_decision = 1
        continuous_decision = (decision_value / 10) * 2 - 1
        decision_texts_and_outcomes.append(
            (str(decision_value), discrete_decision, continuous_decision)
        )


@register_verifier_decision_scale_handler("out_of_100")
class OutOf100VerifierDecisionScaleHandler(VerifierDecisionScaleHandler):
    """Handler for the out of 100 verifier decision scale.

    The decision text is expected to be a number between 0 and 100.
    """

    decision_texts_and_outcomes = []

    for decision_value in range(101):
        if decision_value < 50:
            discrete_decision = 0
        elif decision_value == 50:
            discrete_decision = 3
        else:
            discrete_decision = 1
        continuous_decision = (decision_value / 100) * 2 - 1
        decision_texts_and_outcomes.append(
            (str(decision_value), discrete_decision, continuous_decision)
        )
