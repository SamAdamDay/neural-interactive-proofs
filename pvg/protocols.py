"""Implementations of interaction protocols.

A protocol is implemented by a protocol handler, which specifies the agents present, how
they interact, and how the environment is updated.

Every protocol handler is a subclass of `ProtocolHandler` and registers itself with the
use of the `register_protocol_handler` decorator. The `build_protocol_handler` factory
function can then be used to build a protocol handler from parameters.
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from tensordict.tensordict import TensorDict, TensorDictBase

from typing import Optional
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
    def verifier_names(self) -> list[str]:
        """The names of the verifiers in the protocol.

        Returns
        -------
        verifier_names : list[str]
            The names of the verifiers in the protocol.
        """

    @property
    @abstractmethod
    def conversations(self) -> list[list[str]]:
        """The set of conversations that the different agents are engaging in.

        Returns
        -------
        conversations : list[list[str]]
            The set of conversations that the different agents are engaging in.
        """

    @property
    def num_conversations(self) -> int:
        """Returns the number of conversations that the different agents are engaging in.

        Returns
        -------
        int
            The number of conversations that the different agents are engaging in.
        """
        return len(self.conversations)

    def get_conversation_indices(self, name: str) -> list[int]:
        """Returns the number of conversations that the different agents are engaging in.

        Returns
        -------
        int
            The number of conversations that the different agents are engaging in.
        """
        return [i for i, conv in enumerate(self.conversations) if name in conv]

    @property
    def zk(self) -> bool:
        """Whether the protocol is zero-knowledge or not.

        Returns
        -------
        zk : bool
            Whether the protocol is zero-knowledge or not. Default is False.
        """
        return False

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

    @property
    def verifier_starts(self) -> bool:
        """Checks if the verifier starts the protocol.

        Returns
        -------
        bool
            True if the verifier starts the protocol, False otherwise.
        """

    def get_active_agents_mask(
        self, round: Int[Tensor, "..."], c: int
    ) -> Bool[Tensor, "... num_agents"]:
        """Get a boolean mask indicating which agents are active in a given round.

        Parameters
        ----------
        round : Int[Tensor, "..."]
            The round of the protocol.
        c : int
            The conversation index.

        Returns
        -------
        active_agents : Bool[Tensor, "... num_agents"]
            A boolean mask indicating which agents are active in the given round.
        """
        conversation = self.conversations[c]
        m = [torch.zeros_like(round, dtype=torch.bool) for _ in self.agent_names]
        if conversation == ["simulator"]:
            m[self.agent_names.index("simulator")] = torch.ones_like(
                round, dtype=torch.bool
            )
        else:
            for agent in conversation:
                if "verifier" in agent:
                    m[self.agent_names.index(agent)] = round % 2 == 1 - int(
                        self.verifier_starts
                    )
                elif "prover" in agent:
                    m[self.agent_names.index(agent)] = round % 2 == int(
                        self.verifier_starts
                    )
                else:
                    raise ValueError(f"Unknown agent: {agent}")

        return torch.stack(m, dim=-1)

    def get_agent_turn_mask(
        self, name: str, round: Int[Tensor, "..."], conversation: str
    ) -> Bool[Tensor, "..."]:
        """Get a boolean mask indicating whether it's the named agent's turn.

        Parameters
        ----------
        round : Int[Tensor, "..."]
            The round of the protocol.

        Returns
        -------
        agent_turn : Bool[Tensor, "..."]
            A boolean mask indicating whether it is the named agent's turn in the given
            round.
        """
        return self.get_active_agents_mask(round, conversation)[
            ..., self.agent_names.index(name)
        ]

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

        for c in range(self.num_conversations):

            # TODO LH from here

            conversation = self.conversations[c]

            y: Int[Tensor, "... 1"] = env_td["y"]
            round: Int[Tensor, "..."] = env_td["round"]
            decision: Int[Tensor, "... agent"] = env_td["agents", "decision"]
            done: Bool[Tensor, "..."] = env_td["done"]

            # Get the mask of the batch items where it is the verifier's turn
            verifier_turn_mask = self.get_verifier_turn_mask(round)

            verifier_index = (..., self.agent_names.index("verifier"))

            if self.params.protocol_common.force_guess == Guess.ONE:
                decision[verifier_index] = torch.ones_like(decision[verifier_index])
            elif self.params.protocol_common.force_guess == Guess.ZERO:
                decision[verifier_index] = torch.zeros_like(decision[verifier_index])
            elif self.params.protocol_common.force_guess == Guess.Y:
                decision[verifier_index] = env_td["y"].squeeze()

            # If the verifier has made a guess we terminate the episode
            verifier_decision_made = verifier_turn_mask & (
                decision[verifier_index] != 2
            )
            verifier_decision_made = verifier_decision_made & (
                round >= self.min_message_rounds
            )
            done = done | verifier_decision_made

            # Compute the reward for the verifier when they make a guess
            reward = torch.empty(
                (*done.shape, len(self.agent_names)),
                dtype=torch.float,
                device=done.device,
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

            # TODO LH revisit

        return done, reward


PROTOCOL_HANDLER_REGISTRY: dict[InteractionProtocolType, type[ProtocolHandler]] = {}


def register_protocol_handler(protocol_handler: InteractionProtocolType):
    """Decorator to register a protocol handler."""

    def decorator(cls: type[ProtocolHandler]):
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


class ZkProtocol(ProtocolHandler, ABC):
    """Base class for zero-knowledge protocols.

    Introduces a second verifier and a simulator. The simulator tries to mimic the interaction between the second verifier and the prover, and the second verifier tries to prevent this. The prover tries to make sure the simulator can succeed (which implies that it is not `leaking` knowledge).
    """

    zk = True

    def _include_simulator_rewards(
        self,
        verifier_a_probs: Float[Tensor, "..."],
        simulator_probs: Float[Tensor, "..."],
        reward: Float[Tensor, "... num_agents"],
    ):
        protocol_params = self.params.protocol_common

        # Simulator reward
        reward[..., -1] = (
            1 - abs(verifier_a_probs - simulator_probs).mean(dim=-1)
        ) * protocol_params.simulator_reward

        # Adversarial verifier reward
        reward[..., -2] = -reward[..., -1]


class TwoProverProtocol(ProtocolHandler, ABC):
    """Base class for protocols with two provers.

    The provers answer the verifier's questions and try to either convince the prover of different (if "adversarial") or the same (if not) answers.
    """

    agent_names = ["prover0", "prover1", "verifier"]
    prover_names = ["prover0", "prover1"]
    verifier_names = ["verifier"]
    adversarial: bool

    def _include_prover_rewards(
        self,
        verifier_decision_made: Bool[Tensor, "..."],
        verifier_decision: Int[Tensor, "..."],
        reward: Float[Tensor, "... num_agents"],
    ):
        protocol_params = self.params.protocol_common

        if protocol_params.shared_reward:
            reward[..., 0] = reward[..., 1] = reward[..., 2]
        elif self.adversarial:
            for prover_num in range(2):
                reward[..., prover_num] = (
                    verifier_decision_made & (verifier_decision == prover_num)
                ).float() * protocol_params.prover_reward
        else:
            reward[..., 0] = reward[..., 1] = (
                verifier_decision_made & (verifier_decision == 1)
            ).float() * protocol_params.prover_reward


@register_protocol_handler(InteractionProtocolType.NIP)
class NipProtocol(ProtocolHandler):
    """Handler for the NIP protocol.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    """

    prover_names = ["prover"]
    verifier_names = ["verifier"]
    agent_names = ["verifier", "prover"]
    conversations = [
        ["verifier", "prover"],
    ]
    verifier_starts = True

    @property
    def max_message_rounds(self) -> int:
        return self.params.nip_protocol.max_message_rounds

    @property
    def min_message_rounds(self) -> int:
        return self.params.nip_protocol.min_message_rounds

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
class AdpProtocol(NipProtocol):
    """Implementation of the Abstract Decision Problem protocol.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    """

    agent_names = ["prover", "verifier"]
    max_message_rounds = 2
    min_message_rounds = 2
    conversations = [
        ["prover", "verifier"],
    ]
    verifier_starts = False


@register_protocol_handler(InteractionProtocolType.DEBATE)
class DebateProtocol(TwoProverProtocol):
    """Implementation of the Debate protocol.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    """

    adversarial = True
    conversations = [
        ["prover0", "prover1", "verifier"],
    ]
    verifier_starts = False

    @property
    def max_message_rounds(self) -> int:
        return self.params.debate_protocol.max_message_rounds

    @property
    def min_message_rounds(self) -> int:
        return self.params.debate_protocol.min_message_rounds


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
    adversarial = True
    conversations = [
        ["prover0", "prover1", "verifier"],
    ]
    verifier_starts = False

    def get_active_agents_mask(
        self, round: Int[Tensor, "..."], c: int
    ) -> Bool[Tensor, "... num_agents"]:
        """Get a boolean mask indicating which agents are active in a given round.

        Parameters
        ----------
        round : Int[Tensor, "..."]
            The round of the protocol.
        c : int
            The conversation index.

        Returns
        -------
        active_agents : Bool[Tensor, "... num_agents"]
            A boolean mask indicating which agents are active in the given round.
        """
        conversation = self.conversations[c]
        m = [torch.zeros_like(round, dtype=torch.bool) for _ in self.agent_names]
        if conversation == ["simulator"]:
            m[self.agent_names.index("simulator")] = torch.ones_like(
                round, dtype=torch.bool
            )
        else:
            m[self.agent_names.index("verifier")] = round % 2 == 1
            prover0_first = torch.randint_like(round, 2).bool()
            m[self.agent_names.index("prover0")] = (round % 2 == 0) & prover0_first
            m[self.agent_names.index("prover1")] = (round % 2 == 0) & ~prover0_first

        return torch.stack(m, dim=-1)


@register_protocol_handler(InteractionProtocolType.MNIP)
class MnipProtocol(TwoProverProtocol):
    """Implementation of the MNIP protocol.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    """

    adversarial = False
    conversations = [["prover0", "verifier"], ["prover1", "verifier"]]
    verifier_starts = True

    @property
    def max_message_rounds(self) -> int:
        return self.params.mnip_protocol.max_message_rounds

    @property
    def min_message_rounds(self) -> int:
        return self.params.mnip_protocol.min_message_rounds

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
        pass
        # return torch.stack([round % 2 == 0, round % 2 == 0, round % 2 == 1], dim=-1)


@register_protocol_handler(InteractionProtocolType.ZKNIP)
class ZknipProtocol(NipProtocol, ZkProtocol):
    """Implementation of the ZKNIP protocol.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    """

    verifier_names = ["verifier0", "verifier1"]
    agent_names = ["verifier0", "prover", "verifier1", "simulator"]
    conversations = [["verifier0", "prover"], ["verifier1", "prover"], ["simulator"]]

    @property
    def max_message_rounds(self) -> int:
        return self.params.zknip_protocol.max_message_rounds

    @property
    def min_message_rounds(self) -> int:
        return self.params.zknip_protocol.min_message_rounds
