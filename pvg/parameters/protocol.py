"""Parameters for the interaction protocols."""

from abc import ABC
from dataclasses import dataclass
from typing import Optional

from pvg.parameters.base import SubParameters
from pvg.parameters.types import Guess, MinMessageRoundsSchedulerType


@dataclass
class CommonProtocolParameters(SubParameters):
    """Common additional parameters for the interaction protocol.

    Parameters
    ----------
    verifier_first : bool
        Whether the verifier goes first in the protocol.
    prover_reward : float
        The reward given to the prover when the verifier guesses "accept".
    verifier_reward : float
        The reward given to the verifier when it guesses correctly.
    verifier_incorrect_penalty : float
        The penalty given to the verifier when it guesses incorrectly.
    verifier_terminated_penalty : float
        The reward given to the verifier if the episode terminates before it guesses.
    verifier_no_guess_reward : float
        The reward given to the verifier if it does not make a guess in a round.
    shared_reward : bool
        Whether to use a shared reward function, where the prover gets the same reward
        as the verifier. This overrides `prover_reward`.
    force_guess: Guess, optional
        The guess to force the verifier to make. If not provided, the verifier makes a
        guess using its policy.
    """

    verifier_first: bool = True

    prover_reward: float = 1.0
    verifier_reward: float = 1.0
    verifier_incorrect_penalty: float = -1.0
    verifier_terminated_penalty: float = -1.0
    verifier_no_guess_reward: float = 0.0
    shared_reward: bool = False

    force_guess: Optional[Guess] = None


@dataclass
class LongProtocolParameters(SubParameters, ABC):
    """Additional parameters for interaction protocols with multiple rounds.

    Parameters
    ----------
    max_message_rounds : int
        The maximum number of rounds of the game. Each round corresponds to one move by
        one or more agents.
    min_message_rounds : int
        The minimum number of rounds of messages. Before this point, the verifier's
        guesses are not registered.
    min_message_rounds_scheduler : MinMessageRoundsScheduler
        The scheduler to use for the minimum number of message rounds, allowing it to
        change over time. TODO: not currently implemented.
    """

    max_message_rounds: int = 8
    min_message_rounds: int = 0
    min_message_rounds_scheduler: MinMessageRoundsSchedulerType = (
        MinMessageRoundsSchedulerType.CONSTANT
    )

    def __post_init__(self):
        super().__post_init__()

        # Convert the scheduler to an enum type
        if not isinstance(
            self.min_message_rounds_scheduler, MinMessageRoundsSchedulerType
        ):
            self.min_message_rounds_scheduler = MinMessageRoundsSchedulerType[
                self.min_message_rounds_scheduler
            ]


@dataclass
class PvgProtocolParameters(LongProtocolParameters):
    """Additional parameters for the PVG interaction protocol.

    Parameters
    ----------
    max_message_rounds : int
        The maximum number of rounds of the game. Each round corresponds to one move by
        one or more agents.
    min_message_rounds : int
        The minimum number of rounds of messages. Before this point, the verifier's
        guesses are not registered.
    min_message_rounds_scheduler : MinMessageRoundsScheduler
        The scheduler to use for the minimum number of message rounds, allowing it to
        change over time. TODO: not currently implemented.
    """


@dataclass
class DebateProtocolParameters(LongProtocolParameters):
    """Additional parameters for the debate interaction protocol.

    Parameters
    ----------
    max_message_rounds : int
        The maximum number of rounds of the game. Each round corresponds to one move by
        one or more agents.
    min_message_rounds : int
        The minimum number of rounds of messages. Before this point, the verifier's
        guesses are not registered.
    min_message_rounds_scheduler : MinMessageRoundsScheduler
        The scheduler to use for the minimum number of message rounds, allowing it to
        change over time. TODO: not currently implemented.
    """
