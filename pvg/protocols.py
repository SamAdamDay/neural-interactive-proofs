"""Implementations of interaction protocols.

A protocol is implemented by a protocol handler, which specifies the agents present, how
they interact, and how the environment is updated.

Every protocol handler is a subclass of `ProtocolHandler` and registers itself with the
use of the `register_protocol_handler` decorator. The `build_protocol_handler` factory
function can then be used to build a protocol handler from parameters.
"""

from abc import ABC, abstractmethod
from functools import cached_property

import torch
from torch import Tensor
from typing import TypeVar

from tensordict.tensordict import TensorDictBase

from jaxtyping import Int, Bool, Float

from pvg.parameters import Parameters, InteractionProtocolType, Guess


class ProtocolHandler(ABC):
    """Base class for protocol handlers.

    A protocol handler gives the implementation of an exchange protocol, specifying what
    agents are present, how they interact, and how the environment is updated.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    """

    def __init__(self, params: Parameters):
        self.params = params

    @property
    @abstractmethod
    def agent_names(self) -> list[str]:
        """The names of the agents in the protocol in turn order.

        Returns
        -------
        agent_names : list[str]
            The names of the agents in the protocol.
        """

    @property
    @abstractmethod
    def prover_names(self) -> list[str]:
        """The names of the provers in the protocol.

        Returns
        -------
        prover_names : list[str]
            The names of the provers in the protocol.
        """

    @property
    @abstractmethod
    def max_message_rounds(self) -> int:
        """The maximum number of rounds in the protocol.

        Returns
        -------
        max_message_rounds : int
            The maximum number of rounds in the protocol.
        """

    @property
    @abstractmethod
    def min_message_rounds(self) -> int:
        """The minimum number of rounds in the protocol.

        Returns
        -------
        min_message_rounds : int
            The minimum number of rounds in the protocol.
        """

    @abstractmethod
    def get_active_agents_mask(
        self, round: Int[Tensor, "..."]
    ) -> Bool[Tensor, "... num_agents"]:
        """Get a boolean mask indicating which agents are active in a given round.

        Parameters
        ----------
        round : Int[Tensor, "..."]
            The round of the protocol.

        Returns
        -------
        active_agents : Bool[Tensor, "... num_agents"]
            A boolean mask indicating which agents are active in the given round.
        """

    def get_verifier_turn_mask(self, round: Int[Tensor, "..."]) -> Bool[Tensor, "..."]:
        """Get a boolean mask indicating whether it's the verifier's turn.

        Parameters
        ----------
        round : Int[Tensor, "..."]
            The round of the protocol.

        Returns
        -------
        verifier_turn : Bool[Tensor, "..."]
            A boolean mask indicating whether it is the verifier's turn in the given
            round.
        """
        return self.get_active_agents_mask(round)[
            ..., self.agent_names.index("verifier")
        ]

    @cached_property
    def agent_turn_names(self) -> list[list[str]]:
        """A list of which agent names are active in each round.

        This is a list of lists of agent names, where each inner list contains the agent
        names that are active in the corresponding round.

        Returns
        -------
        agent_turn_names : list[list[str]]
            A list of which agent names are active in each round.
        """

        active_agents_mask = self.get_active_agents_mask(
            torch.arange(self.max_message_rounds)
        )

        agent_turn_names = []
        for round in range(self.max_message_rounds):
            active_agent_names = []
            for i, agent_name in enumerate(self.agent_names):
                if active_agents_mask[round, i]:
                    active_agent_names.append(agent_name)
            agent_turn_names.append(active_agent_names)

        return agent_turn_names

    @abstractmethod
    def _include_prover_rewards(
        self,
        verifier_decision_made: Bool[Tensor, "..."],
        verifier_decision: Int[Tensor, "..."],
        reward: Float[Tensor, "... num_agents"],
    ):
        """Compute the rewards for the provers and add them to the current reward.

        The `reward` tensor is updated in place, adding in the rewards for the provers
        at the appropriate indices.

        Parameters
        ----------
        verifier_decision_made : Bool[Tensor, "..."]
            A boolean mask indicating whether the verifier has made a decision.
        verifier_decision : Int[Tensor, "..."]
            The verifier's decision.
        reward : Float[Tensor, "... num_agents"]
            The currently computed reward, which should include the reward for the
            verifier.
        """

    def step_interaction_protocol(
        self,
        env_td: TensorDictBase,
    ) -> tuple[Bool[Tensor, "..."], Float[Tensor, "... num_agents"]]:
        """Take a step in the interaction protocol.

        Computes the done signal, reward and next decision restriction.

        Used in the `_step` method of the environment.

        Parameters
        ----------
        env_td : TensorDictBase
            The current observation and state.

        Returns
        -------
        done : Bool[Tensor, "..."]
            A boolean mask indicating whether the episode is done.
        reward : Float[Tensor, "... num_agents"]
            The reward for the agents.
        """

        protocol_params = self.params.protocol_common

        y: Int[Tensor, "... 1"] = env_td["y"]
        round: Int[Tensor, "..."] = env_td["round"]
        decision: Int[Tensor, "... agent"] = env_td["agents", "decision"]
        done: Bool[Tensor, "..."] = env_td["done"]

        # Get the mask of the batch items where it is the verifier's turn
        verifier_turn_mask = self.get_verifier_turn_mask(round)

        # If the verifier has made a guess we terminate the episode
        verifier_index = (..., self.agent_names.index("verifier"))

        if self.params.protocol_common.force_guess == Guess.ONE:
            decision[verifier_index] = torch.ones_like(decision[verifier_index])
        elif self.params.protocol_common.force_guess == Guess.ZERO:
            decision[verifier_index] = torch.zeros_like(decision[verifier_index])
        elif self.params.protocol_common.force_guess == Guess.Y:
            decision[verifier_index] = env_td["y"].squeeze()

        verifier_decision_made = verifier_turn_mask & (decision[verifier_index] != 2)
        verifier_decision_made = verifier_decision_made & (
            round >= self.min_message_rounds
        )
        done = done | verifier_decision_made

        # Compute the reward for the verifier when they make a guess
        reward = torch.empty(
            (*done.shape, len(self.agent_names)), dtype=torch.float, device=done.device
        )
        reward[verifier_index] = torch.zeros_like(done, dtype=torch.float)
        reward[verifier_index][
            verifier_decision_made & (decision[verifier_index] == y.squeeze())
        ] = protocol_params.verifier_reward
        reward[verifier_index][
            verifier_decision_made & (decision[verifier_index] != y.squeeze())
        ] = protocol_params.verifier_incorrect_penalty

        # If we reach the end of the episode and the verifier has not made a guess,
        # terminate it with a negative reward for the verifier
        done = done | (round >= self.max_message_rounds - 1)
        reward[verifier_index][
            (round >= self.max_message_rounds - 1) & ~verifier_decision_made
        ] = protocol_params.verifier_terminated_penalty

        # If the verifier has not made a guess and it's their turn, given them a small
        # reward for continuing
        reward[verifier_index][
            verifier_turn_mask & ~done
        ] = protocol_params.verifier_no_guess_reward

        # Compute the rewards for the provers and add them
        self._include_prover_rewards(
            verifier_decision_made, decision[verifier_index], reward
        )

        return done, reward


PROTOCOL_HANDLER_REGISTRY: dict[InteractionProtocolType, type[ProtocolHandler]] = {}

P = TypeVar("P", bound=ProtocolHandler)


def register_protocol_handler(protocol_handler: InteractionProtocolType):
    """Decorator to register a protocol handler."""

    def decorator(cls: type[P]) -> type[P]:
        PROTOCOL_HANDLER_REGISTRY[protocol_handler] = cls
        return cls

    return decorator


def build_protocol_handler(
    params: Parameters,
) -> ProtocolHandler:
    """Factory function for building a trainer from parameters.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    """
    return PROTOCOL_HANDLER_REGISTRY[params.interaction_protocol](params)


@register_protocol_handler(InteractionProtocolType.PVG)
class PvgProtocol(ProtocolHandler):
    """Handler for the PVG protocol.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    """

    prover_names = ["prover"]

    @property
    def agent_names(self) -> list[str]:
        if self.params.pvg_protocol.verifier_first:
            return ["verifier", "prover"]
        else:
            return ["prover", "verifier"]

    @property
    def max_message_rounds(self) -> int:
        return self.params.pvg_protocol.max_message_rounds

    @property
    def min_message_rounds(self) -> int:
        return self.params.pvg_protocol.min_message_rounds

    def get_active_agents_mask(
        self, round: Int[Tensor, "..."]
    ) -> Bool[Tensor, "... 2"]:
        """Get a boolean mask indicating which agents are active in a given round.

        The agents are active in alternating rounds.

        Parameters
        ----------
        round : Int[Tensor, "..."]
            The round of the protocol.

        Returns
        -------
        active_agents : Bool[Tensor, "... 2"]
            A boolean mask indicating which agents are active in the given round.
        """
        return torch.stack([round % 2 == 0, round % 2 == 1], dim=-1)

    def _include_prover_rewards(
        self,
        verifier_decision_made: Bool[Tensor, "..."],
        verifier_decision: Int[Tensor, "..."],
        reward: Float[Tensor, "... num_agents"],
    ):
        protocol_params = self.params.protocol_common
        verifier_index = (..., self.agent_names.index("verifier"))
        prover_index = (..., self.agent_names.index("prover"))

        if protocol_params.shared_reward:
            reward[prover_index] = reward[verifier_index]
        else:
            reward[prover_index] = (
                verifier_decision_made & (verifier_decision == 1)
            ).float() * protocol_params.prover_reward


@register_protocol_handler(InteractionProtocolType.ABSTRACT_DECISION_PROBLEM)
class AdpProtocol(PvgProtocol):
    """Implementation of the Abstract Decision Problem protocol.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    """

    agent_names = ["prover", "verifier"]
    max_message_rounds = 2
    min_message_rounds = 2


class TwoProverProtocol(PvgProtocol, ABC):
    """Base class for protocols with two provers.

    The first prover tries to convince the verifier that the label is 0, and the second
    tries to convince them that the label is 1.
    """

    agent_names = ["prover0", "prover1", "verifier"]
    prover_names = ["prover0", "prover1"]


@register_protocol_handler(InteractionProtocolType.DEBATE)
class DebateProtocol(TwoProverProtocol):
    """Implementation of the Debate protocol.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    """

    @property
    def max_message_rounds(self) -> int:
        return self.params.debate_protocol.max_message_rounds

    @property
    def min_message_rounds(self) -> int:
        return self.params.debate_protocol.min_message_rounds

    def get_active_agents_mask(
        self, round: Int[Tensor, "..."]
    ) -> Bool[Tensor, "... 3"]:
        """Get a boolean mask indicating which agents are active in a given round.

        The two provers play simultaneously, and the verifier plays after them.

        Parameters
        ----------
        round : Int[Tensor, "..."]
            The round of the protocol.

        Returns
        -------
        active_agents : Bool[Tensor, "... 2"]
            A boolean mask indicating which agents are active in the given round.
        """
        return torch.stack([round % 2 == 0, round % 2 == 0, round % 2 == 1], dim=-1)

    def _include_prover_rewards(
        self,
        verifier_decision_made: Bool[Tensor, "..."],
        verifier_decision: Int[Tensor, "..."],
        reward: Float[Tensor, "... num_agents"],
    ):
        protocol_params = self.params.protocol_common

        if protocol_params.shared_reward:
            reward[..., 0] = reward[..., 1] = reward[..., 2]
        else:
            for prover_num in range(2):
                reward[..., prover_num] = (
                    verifier_decision_made & (verifier_decision == prover_num)
                ).float() * protocol_params.prover_reward


@register_protocol_handler(InteractionProtocolType.MERLIN_ARTHUR)
class MerlinArthurProtocol(TwoProverProtocol):
    """Implementation of the Merlin-Arthur protocol.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    """

    max_message_rounds = 2
    min_message_rounds = 2

    def get_active_agents_mask(
        self, round: Int[Tensor, "..."]
    ) -> Bool[Tensor, "... 3"]:
        """Get a boolean mask indicating which agents are active in a given round.

        A random one of the two provers goes first, and the verifier goes second.

        Parameters
        ----------
        round : Int[Tensor, "..."]
            The round of the protocol.

        Returns
        -------
        active_agents : Bool[Tensor, "... 2"]
            A boolean mask indicating which agents are active in the given round.
        """
        prover1_first = torch.randint_like(round, 2).bool()
        return torch.stack(
            [
                (round % 2 == 0) & prover1_first,
                (round % 2 == 0) & ~prover1_first,
                round % 2 == 1,
            ],
            dim=-1,
        )

    def _include_prover_rewards(
        self,
        verifier_decision_made: Bool[Tensor, "..."],
        verifier_decision: Int[Tensor, "..."],
        reward: Float[Tensor, "... num_agents"],
    ):
        protocol_params = self.params.protocol_common

        if protocol_params.shared_reward:
            reward[..., 0] = reward[..., 1] = reward[..., 2]
        else:
            for prover_num in range(2):
                reward[..., prover_num] = (
                    verifier_decision_made & (verifier_decision == prover_num)
                ).float() * protocol_params.prover_reward
