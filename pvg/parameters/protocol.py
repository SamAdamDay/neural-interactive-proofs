"""Parameters for the interaction protocols."""

from abc import ABC
from dataclasses import dataclass, field
from typing import Optional

from pvg.parameters.parameters_base import SubParameters, register_parameter_class
from pvg.parameters.types import Guess, MinMessageRoundsSchedulerType


@register_parameter_class
@dataclass
class CommonProtocolParameters(SubParameters):
    """Common additional parameters for the interaction protocol.

    Parameters
    ----------
    verifier_first : bool
        Whether the verifier goes first in the protocol.
    randomize_prover_stance : bool
        Whether, for each datapoint, the verdict the prover arguing for is randomized.
        This is only relevant when there is a single prover, and when using a text-based
        protocol.
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
    zero_knowledge: bool
        Whether to use a zero-knowledge version of the protocol.
    """

    verifier_first: bool = True
    randomize_prover_stance: bool = False

    prover_reward: float = 1.0
    verifier_reward: float = 1.0
    verifier_incorrect_penalty: float = -1.0
    verifier_terminated_penalty: float = -1.0
    verifier_no_guess_reward: float = 0.0
    shared_reward: bool = False

    force_guess: Optional[Guess] = None

    zero_knowledge: bool = False


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
                self.min_message_rounds_scheduler.upper()
            ]


@register_parameter_class
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


@register_parameter_class
@dataclass
class DebateProtocolParameters(LongProtocolParameters):
    """Additional parameters for the Debate interaction protocol.

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
    sequential : bool
        Whether the provers send messages one after the other, or both simultaneously.
    prover0_first : bool
        When the provers send messages sequentially, whether prover 0 goes first.
    randomize_channel_order : bool
        Whether to randomize the order of the channels when prompting the verifier. Only
        relevant in text-based protocols.
    """

    sequential: bool = False
    prover0_first: bool = True

    randomize_channel_order: bool = True


@register_parameter_class
@dataclass
class MnipProtocolParameters(LongProtocolParameters):
    """Additional parameters for the MNIP interaction protocol.

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
    sequential : bool
        Whether the provers send messages one after the other, or both simultaneously.
    prover0_first : bool
        When the provers send messages sequentially, whether prover 0 goes first.
    randomize_channel_order : bool
        Whether to randomize the order of the channels when prompting the verifier. Only
        relevant in text-based protocols.
    """

    sequential: bool = False
    prover0_first: bool = True

    randomize_channel_order: bool = True


@dataclass
class ZkProtocolParameters(SubParameters):
    """Additional parameters for zero-knowledge versions of the interaction protocols.

    Parameters
    ----------
    simulator_reward_coefficient : float
        The coefficient to multiply the logit closeness by to get the simulator reward.
    aux_prover_reward_coefficient : float
        The coefficient of the simulator reward in the prover reward.
    """

    simulator_reward_coefficient: float = 1.0
    aux_prover_reward_coefficient: float = 1.0  # We may want to change this eventually
    use_multiple_simulators: bool = True
    use_mixed_sl_and_rl: bool = False
    distance_function: str = "kl_divergence"
    use_dists_in_simulator_losses: bool = True
    use_dists_in_other_losses: bool = True