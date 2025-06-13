"""Classes for handling the decision spectrum of a verifier.

A verifier decision spectrum allows for more nuanced decisions than just "accept" or
"reject". This module contains the classes and functions necessary to handle the
verifier decision spectrum in a protocol.

Verifier decision spectra are only relevant for text-based protocols.
"""

from abc import ABC, abstractmethod
from typing import Literal
from functools import cached_property
import re

from nip.parameters import HyperParameters
from nip.parameters.types import VerifierDecisionSpectrumType


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


class VerifierDecisionSpectrumHandler(ABC):
    """Base class for handling the verifier decision spectrum.

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

    @cached_property
    def strongest_reject_decision_text(self) -> str:
        """The strongest reject decision text from the verifier model."""
        return self.possible_decision_texts[0]

    @cached_property
    def strongest_accept_decision_text(self) -> str:
        """The strongest accept decision text from the verifier model."""
        return self.possible_decision_texts[-1]

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

        match = None
        for (
            possible_decision_text,
            discrete_decision,
            continuous_decision,
        ) in self.decision_texts_and_outcomes:

            # Set `match` if it is the first match or if it is longer than the previous
            # match. This is to avoid matching "1" with "10" or similar cases.
            if decision_text_normalised.startswith(possible_decision_text) and (
                match is None or len(possible_decision_text) > len(match[2])
            ):
                match = (
                    discrete_decision,
                    continuous_decision,
                    possible_decision_text,
                )

        if match is not None:
            return match

        raise VerifierDecisionParseError(decision_text)


VERIFIER_DECISION_SPECTRUM_HANDLERS: dict[
    VerifierDecisionSpectrumType, type[VerifierDecisionSpectrumHandler]
] = {}


def register_verifier_decision_spectrum_handler(
    decision_spectrum_type: VerifierDecisionSpectrumType,
) -> type[VerifierDecisionSpectrumHandler]:
    """Register a verifier decision spectrum handler.

    Parameters
    ----------
    decision_spectrum_type : VerifierDecisionSpectrumType
        The decision spectrum type to register the handler for.

    Returns
    -------
    handler_class : type[VerifierDecisionSpectrumHandler]
        The class of the handler.
    """

    def decorator(
        handler_class: type[VerifierDecisionSpectrumHandler],
    ) -> type[VerifierDecisionSpectrumHandler]:
        VERIFIER_DECISION_SPECTRUM_HANDLERS[decision_spectrum_type] = handler_class
        return handler_class

    return decorator


def build_verifier_decision_spectrum_handler(
    hyper_params: HyperParameters,
) -> VerifierDecisionSpectrumHandler:
    """Build the verifier decision spectrum handler.

    Parameters
    ----------
    hyper_params : HyperParameters
        The hyperparameters for the experiment.

    Returns
    -------
    handler : VerifierDecisionSpectrumHandler
        The verifier decision spectrum handler.
    """

    decision_spectrum_type = hyper_params.protocol_common.verifier_decision_spectrum
    handler_class = VERIFIER_DECISION_SPECTRUM_HANDLERS[decision_spectrum_type]
    return handler_class(hyper_params)


@register_verifier_decision_spectrum_handler("accept_reject")
class AcceptRejectVerifierDecisionSpectrumHandler(VerifierDecisionSpectrumHandler):
    """Handler for the accept/reject verifier decision spectrum.

    The decision text is expected to be either "accept" or "reject".
    """

    decision_texts_and_outcomes = [
        ("reject", 0, -1.0),
        ("accept", 1, 1.0),
    ]


@register_verifier_decision_spectrum_handler("likert_int_scale_11")
class LikertIntScale11VerifierDecisionSpectrumHandler(VerifierDecisionSpectrumHandler):
    """Handler for the 11-point Likert integer scale verifier decision spectrum.

    Decisions are specified as integers between 0 and 10, where some integers have
    special names given by the Likert scale, as follows:

    - 0: strongly disagree
    - 5: neither agree nor disagree
    - 10: strongly agree
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


@register_verifier_decision_spectrum_handler("likert_scale_7")
class LikertScale7VerifierDecisionSpectrumHandler(VerifierDecisionSpectrumHandler):
    """Handler for the 7-point Likert scale verifier decision spectrum."""

    decision_texts_and_outcomes = [
        ("strongly disagree", 0, -1.0),
        ("disagree", 0, -0.667),
        ("somewhat disagree", 0, -0.333),
        ("neither agree nor disagree", 3, 0.0),
        ("somewhat agree", 1, 0.333),
        ("agree", 1, 0.667),
        ("strongly agree", 1, 1.0),
    ]


@register_verifier_decision_spectrum_handler("likert_scale_6")
class LikertScale6VerifierDecisionSpectrumHandler(VerifierDecisionSpectrumHandler):
    """Handler for the 6-point Likert scale verifier decision spectrum."""

    decision_texts_and_outcomes = [
        ("strongly disagree", 0, -1.0),
        ("disagree", 0, -0.667),
        ("somewhat disagree", 0, -0.333),
        ("somewhat agree", 1, 0.333),
        ("agree", 1, 0.667),
        ("strongly agree", 1, 1.0),
    ]


@register_verifier_decision_spectrum_handler("likert_scale_5")
class LikertScale5VerifierDecisionSpectrumHandler(VerifierDecisionSpectrumHandler):
    """Handler for the 5-point Likert scale verifier decision spectrum."""

    decision_texts_and_outcomes = [
        ("strongly disagree", 0, -1.0),
        ("disagree", 0, -0.5),
        ("neither agree nor disagree", 3, 0.0),
        ("agree", 1, 0.5),
        ("strongly agree", 1, 1.0),
    ]


@register_verifier_decision_spectrum_handler("likert_scale_4")
class LikertScale4VerifierDecisionSpectrumHandler(VerifierDecisionSpectrumHandler):
    """Handler for the 4-point Likert scale verifier decision spectrum."""

    decision_texts_and_outcomes = [
        ("strongly disagree", 0, -1.0),
        ("disagree", 0, -0.5),
        ("agree", 1, 0.5),
        ("strongly agree", 1, 1.0),
    ]


@register_verifier_decision_spectrum_handler("out_of_10")
class OutOf10VerifierDecisionSpectrumHandler(VerifierDecisionSpectrumHandler):
    """Handler for the out of 10 verifier decision spectrum.

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


@register_verifier_decision_spectrum_handler("out_of_100")
class OutOf100VerifierDecisionSpectrumHandler(VerifierDecisionSpectrumHandler):
    """Handler for the out of 100 verifier decision spectrum.

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
