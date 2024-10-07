"""Implementations of interaction protocols.

A protocol is implemented by a protocol handler, which specifies the agents present, how
they interact, and how the environment is updated.

Every protocol handler is a subclass of `ProtocolHandler` and registers itself with the
use of the `register_protocol_handler` decorator. The `build_protocol_handler` factory
function can then be used to build a protocol handler from parameters.
"""

from abc import ABC, abstractmethod
from functools import cached_property
from itertools import product
from math import ceil, floor
from typing import TypeVar, Generic, Callable

import torch
from torch import Tensor
import torch.nn.functional as F

from tensordict.tensordict import TensorDictBase

from einops import rearrange

from jaxtyping import Int, Bool, Float

from pvg.parameters import Parameters, InteractionProtocolType, Guess, ScenarioType
from pvg.experiment_settings import ExperimentSettings
from pvg.utils.nested_array_dict import NestedArrayDict
from pvg.utils.maths import logit_or_2, logit_or_n


class ProtocolHandler(ABC):
    """Base class for protocol handlers.

    A protocol handler gives the implementation of an exchange protocol, specifying what
    agents are present, how they interact, and how the environment is updated.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    """

    def __init__(self, params: Parameters, settings: ExperimentSettings):
        self.params = params
        self.settings = settings

        self._validate_active_agents()

    @property
    @abstractmethod
    def agent_names(self) -> list[str]:
        """The names of the agents in the protocol in turn order."""

    @cached_property
    def prover_names(self) -> list[str]:
        """The names of the provers in the protocol."""
        return [agent_name for agent_name in self.agent_names if "prover" in agent_name]

    @property
    def verifier_names(self) -> list[str]:
        """The names of the verifiers in the protocol."""
        return [
            agent_name for agent_name in self.agent_names if "verifier" in agent_name
        ]

    @property
    @abstractmethod
    def max_message_rounds(self) -> int:
        """The maximum number of rounds in the protocol."""

    @property
    @abstractmethod
    def min_message_rounds(self) -> int:
        """The minimum number of rounds in the protocol."""

    @property
    @abstractmethod
    def max_verifier_turns(self) -> int:
        """The maximum number of turns the verifier can take."""

    @property
    @abstractmethod
    def message_channel_names(self) -> list[str]:
        """The names of the message channels in the protocol.

        A message channel is a communication channel visible to selected agents

        Returns
        -------
        message_channel_names : list[str]
            The names of the message channels in the protocol.
        """

    @property
    def num_message_channels(self) -> int:
        """The number of message channels in the protocol.

        A message channel is a communication channel visible to selected agents

        Returns
        -------
        num_message_channels : int
            The number of message channels in the protocol.
        """
        return len(self.message_channel_names)

    @property
    @abstractmethod
    def agent_channel_visibility(self) -> list[tuple[str, str]]:
        """A specification of which agents can see which message channels.

        Returns
        -------
        agent_channel_visibility : list[tuple[str, str]]
            A list of tuples of the form `(agent_name, channel_name)`. A tuple indicates
            that the agent can see the channel.
        """

    @cached_property
    def agent_channel_visibility_mask(self) -> Bool[Tensor, "agent channel"]:
        """A boolean mask indicating which agents can see which message channels.

        Returns
        -------
        agent_channel_visibility : Bool[Tensor, "agent channel"]
            The agent-channel visibility mask, where `agent_channel_visibility[agent,
            channel]` is `True` if the agent can see the channel.
        """

        agent_channel_visibility_mask = torch.zeros(
            (len(self.agent_names), len(self.message_channel_names)),
            dtype=torch.bool,
            device=self.settings.device,
        )

        for agent_name, channel_name in self.agent_channel_visibility:
            agent_index = self.agent_names.index(agent_name)
            channel_index = self.message_channel_names.index(channel_name)
            agent_channel_visibility_mask[agent_index, channel_index] = True

        return agent_channel_visibility_mask

    def get_agent_visible_channels(self, agent_name: str) -> list[str]:
        """Get the names of the channels visible to an agent.

        Parameters
        ----------
        agent_name : str
            The name of the agent.

        Returns
        -------
        visible_channels : list[str]
            The names of the channels visible to the agent.
        """

        visible_channels = []
        for channel_agent_name, channel in self.agent_channel_visibility:
            if channel_agent_name == agent_name:
                visible_channels.append(channel)

        return visible_channels

    def can_agent_see_channel(self, agent_name: str, channel_name: str) -> bool:
        """Determine whether an agent can see a channel.

        Returns
        -------
        can_see_channel : bool
            Whether the agent can see the channel.
        """

        return (agent_name, channel_name) in self.agent_channel_visibility

    @abstractmethod
    def get_active_agents_mask_from_rounds(
        self, round: Int[Tensor, "..."]
    ) -> Bool[Tensor, "... agent channel"]:
        """Get a boolean mask of active agents for a batch of rounds.

        Given a batch or rounds, returns a boolean mask indicating which agents are
        sending messages in each round and channel.

        Parameters
        ----------
        round : Int[Tensor, "..."]
            The round of the protocol.

        Returns
        -------
        active_agents : Bool[Tensor, "... agent channel"]
            The boolean mask. `active_agents[*batch, agent, channel]` is `True` if the
            agent sends a message in the channel in round `round[*batch]`.
        """

    @abstractmethod
    def can_agent_be_active(
        self, agent_name: str, round: int, channel_name: str
    ) -> bool:
        """Specifies whether an agent can be active in a given round and channel.

        For non-deterministic protocols, this is true if the agent has some probability
        of being active.

        Returns
        -------
        can_be_active : bool
            Whether the agent can be active in the given round and channel.
        """

    def can_agent_be_active_any_channel(self, agent_name: str, round: int) -> bool:
        """Specifies whether an agent can be active in any channel in a given round.

        For non-deterministic protocols, this is true if the agent has some probability
        of being active.

        Returns
        -------
        can_be_active : bool
            Whether the agent can be active in the given round.
        """

        return any(
            self.can_agent_be_active(agent_name, round, channel_name)
            for channel_name in self.message_channel_names
        )

    def get_verifier_guess_mask_from_rounds(
        self, round: Int[Tensor, "..."], adversarial: bool = False
    ) -> Bool[Tensor, "..."]:
        """Get a boolean mask indicating when the verifiers can make a guess.

        Takes as input a tensor of rounds and returns a boolean mask indicating when the
        verifiers can make a guess for each element in the batch.

        Parameters
        ----------
        round : Int[Tensor, "..."]
            The batch of rounds.
        adversarial : bool, optional
            Whether to get the mask for the adversarial verifier, by default False

        Returns
        -------
        verifier_turn : Bool[Tensor, "..."]
            Which batch items the verifiers can make a guess in.
        """
        verifier_name = "adversarial_verifier" if adversarial else "verifier"
        active_agents_mask = self.get_active_agents_mask_from_rounds(round)
        verifier_active_mask = active_agents_mask[
            ..., self.agent_names.index(verifier_name), :
        ]
        return verifier_active_mask.any(dim=-1)

    @cached_property
    def agent_first_active_round(self) -> dict[str, int]:
        """The first round in which each agent is or can be active.

        For non-deterministic protocols, this is the first round in which the agent has
        some probability of being active.

        Returns
        -------
        agents_first_active_rounds : dict[str, int]
            The first round in which each agent is active
        """

        agents_first_active_rounds = {}
        for round in range(100):
            for agent_name in set(self.agent_names) - set(
                agents_first_active_rounds.keys()
            ):
                if self.can_agent_be_active_any_channel(agent_name, round):
                    agents_first_active_rounds[agent_name] = round
            if len(agents_first_active_rounds) == len(self.agent_names):
                break
        else:
            raise ValueError(
                "Could not determine the first active round for all agents."
            )

    @abstractmethod
    def _include_non_verifier_rewards(
        self,
        env_td: TensorDictBase | NestedArrayDict,
        verifier_decision_made: Bool[Tensor, "..."],
        verifier_decision: Int[Tensor, "..."],
        reward: Float[Tensor, "... agent"],
        round: Int[Tensor, "..."],
    ):
        """Compute the rewards for the other agents and add them to the current reward.

        The `reward` tensor is updated in place, adding in the rewards for the agents
        at the appropriate indices.

        Parameters
        ----------
        env_td : TensorDictBase | NestedArrayDict
            The current observation and state. If a `NestedArrayDict`, it is converted
            to a `TensorDictBase`.
        verifier_decision_made : Bool[Tensor, "..."]
            A boolean mask indicating whether the verifier has made a decision.
        verifier_decision : Int[Tensor, "..."]
            The verifier's decision.
        reward : Float[Tensor, "... agent"]
            The currently computed reward, which should include the reward for the
            verifier.
        """

    def step_interaction_protocol(
        self,
        env_td: TensorDictBase | NestedArrayDict,
    ) -> tuple[Bool[Tensor, "..."], Bool[Tensor, "..."], Float[Tensor, "... agent"]]:
        """Take a step in the interaction protocol.

        Computes the done signal, reward and next decision restriction.

        Used in the `_step` method of the environment.

        Parameters
        ----------
        env_td : TensorDictBase | NestedArrayDict
            The current observation and state. If a `NestedArrayDict`, it is converted
            to a `TensorDictBase`.

        Returns
        -------
        done : Bool[Tensor, "..."]
            A boolean mask indicating whether the episode is done because the verifier
            has guessed.
        terminated : Bool[Tensor, "..."]
            A boolean mask indicating whether the episode has been terminated because
            the max number of rounds has been reached and the verifier has not guessed.
        reward : Float[Tensor, "... agent"]
            The reward for the agents.
        """

        protocol_params = self.params.protocol_common

        y: Int[Tensor, "... 1"] = env_td["y"]
        round: Int[Tensor, "..."] = env_td["round"]
        decision: Int[Tensor, "... agent"] = env_td["agents", "decision"]
        done: Bool[Tensor, "..."] = env_td["done"]
        terminated: Bool[Tensor, "..."] = env_td["terminated"]

        if isinstance(env_td, NestedArrayDict):
            y = torch.from_numpy(y)
            round = torch.from_numpy(round)
            decision = torch.from_numpy(decision)
            done = torch.from_numpy(done)
            terminated = torch.from_numpy(terminated)

        # Get the mask of the batch items where the (non-adversarial) verifier can make a guess
        verifier_guess_mask = self.get_verifier_guess_mask_from_rounds(round)

        # If the verifier has made a guess we terminate the episode. For now we assume that the primary verifier controls when an episode is done, though eventually we should generalise this to allow for different dones per agent so that channels can be terminated independently.
        verifier_indices = (
            ...,
            self.agent_names.index("verifier"),
        )

        if self.params.protocol_common.force_guess == Guess.ONE:
            decision[verifier_indices] = torch.ones_like(decision[verifier_indices])
        elif self.params.protocol_common.force_guess == Guess.ZERO:
            decision[verifier_indices] = torch.zeros_like(decision[verifier_indices])
        elif self.params.protocol_common.force_guess == Guess.Y:
            decision[verifier_indices] = env_td["y"].squeeze()

        verifier_decision_made = verifier_guess_mask & (decision[verifier_indices] != 2)
        verifier_decision_made = verifier_decision_made & (
            round >= self.min_message_rounds
        )
        done = done | verifier_decision_made

        # Compute the reward for the verifier when they make a guess
        verifier_index = (..., self.agent_names.index("verifier"))
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
        terminated = terminated | (
            (round >= self.max_message_rounds - 1) & ~verifier_decision_made
        )
        reward[verifier_index][
            (round >= self.max_message_rounds - 1) & ~verifier_decision_made
        ] = protocol_params.verifier_terminated_penalty

        # If the verifier has not made a guess and it's their turn, given them a small
        # reward for continuing
        reward[verifier_index][
            verifier_guess_mask & ~done & ~terminated
        ] = protocol_params.verifier_no_guess_reward

        # Compute the rewards for the other agents and add them
        self._include_non_verifier_rewards(
            env_td, verifier_decision_made, decision[verifier_index], reward, round
        )

        return done, terminated, reward

    def _validate_active_agents(self):
        """Make sure that agents are only active in channels they can see."""

        iterator = product(
            range(self.max_message_rounds),
            self.agent_names,
            self.message_channel_names,
        )
        for round, agent_name, channel_name in iterator:
            if agent_name in self.active_agents_by_round[round][channel_name]:
                assert (agent_name, channel_name) in self.agent_channel_visibility, (
                    f"Protocol specification error: Agent {agent_name!r} is active "
                    f"in round {round} and channel {channel_name!r} but cannot see it."
                )


class DeterministicProtocolHandler(ProtocolHandler, ABC):
    """Base class for protocol handlers of deterministic protocols.

    A protocol handler gives the implementation of an exchange protocol, specifying what
    agents are present, how they interact, and how the environment is updated.

    An exchange protocol is deterministic if the agents' which agents are active in each
    round and channel is determined by the round and channel alone.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    """

    @abstractmethod
    def is_agent_active(self, agent_name: str, round: int, channel_name: str) -> bool:
        """Specifies whether an agent is active in a given round and channel.

        An agent must only send a message in a channel which is visible to them.

        Returns
        -------
        is_active : bool
            Whether the agent is active in the given round and channel.
        """

    def can_agent_be_active(
        self, agent_name: str, round: int, channel_name: str
    ) -> bool:
        """Specifies whether an agent can be active in a given round and channel.

        For deterministic protocols, this is the same as `is_agent_active`.

        Returns
        -------
        can_be_active : bool
            Whether the agent can be active in the given round and channel.
        """

        return self.is_agent_active(agent_name, round, channel_name)

    @cached_property
    def active_agents_by_round(self) -> list[dict[str, list[str]]]:
        """A list of which agent names are active in each round and channel.

        This specifies the channels to which agents can send messages in each round.

        Returns
        -------
        active_agents_by_round : list[dict[str, list[str]]]
            The agent names active in each round and channel.
            `agent_turn_names[round][channel_name]` is a list of the agent names active
            in round `round` and channel `channel_name`.
        """

        active_agents_by_round = []
        for round in range(self.max_message_rounds):
            agents_per_channel = {}
            for channel_name in self.message_channel_names:
                active_agent_names = []
                for agent_name in self.agent_names:
                    if self.is_agent_active(agent_name, round, channel_name):
                        active_agent_names.append(agent_name)
                agents_per_channel[channel_name] = active_agent_names
            active_agents_by_round.append(agents_per_channel)

        return active_agents_by_round

    @cached_property
    def active_agents_mask(self) -> Bool[Tensor, "... agent channel"]:
        """A boolean mask indicating which agents are active in each round and channel.

        Returns
        -------
        active_agents : Bool[Tensor, "... agent channel"]
            A boolean mask indicating which agents are active in each round and channel.
        """

        active_agents = torch.zeros(
            (self.max_message_rounds, len(self.agent_names), self.num_message_channels),
            dtype=torch.bool,
            device=self.settings.device,
        )

        iterator = product(
            range(self.max_message_rounds),
            enumerate(self.agent_names),
            enumerate(self.message_channel_names),
        )
        for round, (agent_index, agent_name), (channel_index, channel_name) in iterator:
            if agent_name in self.active_agents_by_round[round][channel_name]:
                active_agents[round, agent_index, channel_index] = True

        return active_agents

    def get_active_agents_mask_from_rounds(
        self, round: Int[Tensor, "..."]
    ) -> Bool[Tensor, "... agent channel"]:
        """Get a boolean mask of active agents for a batch of rounds.

        Given a batch or rounds, returns a boolean mask indicating which agents are
        active in each round and channel.

        Parameters
        ----------
        round : Int[Tensor, "..."]
            The round of the protocol.

        Returns
        -------
        active_agents : Bool[Tensor, "... agent channel"]
            The boolean mask. `active_agents[*batch, agent, channel]` is `True` if the
            agent sends a message in the channel in round `round[*batch]`.
        """

        return self.active_agents_mask[round, :, :]

    def get_verifier_guess_mask_from_rounds(
        self, round: Int[Tensor, "..."], adversarial: bool = False
    ) -> Bool[Tensor, "..."]:
        """Get a boolean mask indicating when the verifier can make a guess.

        Takes as input a tensor of rounds and returns a boolean mask indicating when the
        verifier can make a guess for each element in the batch.

        Parameters
        ----------
        round : Int[Tensor, "..."]
            The batch of rounds.
        adversarial : bool, optional
            Whether to get the mask for the adversarial verifier, by default False

        Returns
        -------
        verifier_turn : Bool[Tensor, "..."]
            Which batch items the verifier can make a guess in.
        """
        verifier_name = "adversarial_verifier" if adversarial else "verifier"
        active_agents_mask = self.get_active_agents_mask_from_rounds(round)
        verifier_active_mask = active_agents_mask[
            ..., self.agent_names.index(verifier_name), :
        ]
        return verifier_active_mask.any(dim=-1)

    @cached_property
    def agent_first_active_round(self) -> dict[str, int]:
        """The first round in which each agent is or can be active.

        For deterministic protocols, this is the first round in which the agent is active.

        Returns
        -------
        agents_first_active_rounds : dict[str, int]
            The first round in which each agent is active
        """

        agents_first_active_rounds = {}
        for round, active_agents_by_channel in enumerate(self.active_agents_by_round):
            for active_agent_names in active_agents_by_channel.values():
                for agent_name in active_agent_names:
                    if agent_name not in agents_first_active_rounds:
                        agents_first_active_rounds[agent_name] = round

        return agents_first_active_rounds


PROTOCOL_HANDLER_REGISTRY: dict[
    tuple[InteractionProtocolType, bool], type[ProtocolHandler]
] = {}

P = TypeVar("P", bound=ProtocolHandler)


def register_protocol_handler(
    protocol_handler: InteractionProtocolType,
    zero_knowledge: bool = False,
) -> Callable[[type[P]], type[P]]:
    """Decorator to register a protocol handler."""

    def decorator(cls: type[P]) -> type[P]:
        PROTOCOL_HANDLER_REGISTRY[protocol_handler, zero_knowledge] = cls
        return cls

    return decorator


def build_protocol_handler(
    params: Parameters, settings: ExperimentSettings
) -> ProtocolHandler:
    """Factory function for building a trainer from parameters.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    """
    return PROTOCOL_HANDLER_REGISTRY[
        params.interaction_protocol, params.protocol_common.zero_knowledge
    ](params, settings)


@register_protocol_handler(InteractionProtocolType.PVG)
class PvgProtocol(DeterministicProtocolHandler):
    """Handler for the PVG protocol.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    """

    message_channel_names = ["main"]
    agent_channel_visibility = [("prover", "main"), ("verifier", "main")]
    agent_names = ["prover", "verifier"]

    @property
    def verifier_first(self) -> bool:
        return self.params.protocol_common.verifier_first

    @property
    def max_message_rounds(self) -> int:
        return self.params.pvg_protocol.max_message_rounds

    @property
    def max_verifier_turns(self) -> int:
        """The maximum number of turns the verifier can take."""
        if self.params.protocol_common.verifier_first:
            return ceil(self.max_message_rounds / 2)
        else:
            return floor(self.max_message_rounds / 2)

    @property
    def min_message_rounds(self) -> int:
        return self.params.pvg_protocol.min_message_rounds

    def is_agent_active(self, agent_name: str, round: int, channel_name: str) -> bool:
        """Specifies whether an agent is active in a given round and channel.

        An agent must only send a message in a channel which is visible to them.

        Returns
        -------
        is_active : bool
            Whether the agent is active in the given round and channel.
        """

        if self.verifier_first:
            if agent_name == "verifier":
                return round % 2 == 0
            elif agent_name == "prover":
                return round % 2 == 1
        else:
            if agent_name == "prover":
                return round % 2 == 0
            elif agent_name == "verifier":
                return round % 2 == 1

    def _include_non_verifier_rewards(
        self,
        env_td: TensorDictBase | NestedArrayDict,
        verifier_decision_made: Bool[Tensor, "..."],
        verifier_decision: Int[Tensor, "..."],
        reward: Float[Tensor, "... agent"],
        round: Int[Tensor, "..."],
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
    min_message_rounds = 0
    verifier_first = False


@register_protocol_handler(InteractionProtocolType.DEBATE)
class DebateProtocol(PvgProtocol):
    """Implementation of the Debate protocol.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    """

    agent_names = ["prover0", "prover1", "verifier"]
    message_channel_names = ["prover0_channel", "prover1_channel"]
    agent_channel_visibility = [
        ("prover0", "prover0_channel"),
        ("prover0", "prover1_channel"),
        ("prover1", "prover0_channel"),
        ("prover1", "prover1_channel"),
        ("verifier", "prover0_channel"),
        ("verifier", "prover1_channel"),
    ]

    def is_agent_active(self, agent_name: str, round: int, channel_name: str) -> bool:
        """Specifies whether an agent is active in a given round and channel.

        In sequential MNIP with verifier first, the order is:

        - Verifier in both channels
        - First prover (determined by `prover0_first`) in their respective channel
        - Second prover in their respective channel

        In simultaneous MNIP with verifier first, the order is:

        - Verifier in both channels
        - Provers in their respective channels at the same time

        Returns
        -------
        is_active : bool
            Whether the agent is active in the given round and channel.
        """

        if self.params.debate_protocol.prover0_first:
            first_prover = "prover0"
            second_prover = "prover1"
        else:
            first_prover = "prover1"
            second_prover = "prover0"

        if self.verifier_first:

            # Verifier first, sequential
            if self.params.debate_protocol.sequential:
                if agent_name == "verifier":
                    return round % 3 == 0
                elif agent_name == first_prover:
                    if channel_name.startswith(f"{agent_name}_channel"):
                        return round % 3 == 1
                    else:
                        return False
                elif agent_name == second_prover:
                    if channel_name.startswith(f"{agent_name}_channel"):
                        return round % 3 == 2
                    else:
                        return False

            # Verifier first, simultaneous
            else:
                if agent_name in ["prover0", "prover1"]:
                    if channel_name.startswith(f"{agent_name}_channel"):
                        return round % 2 == 1 and channel_name.startswith(
                            f"{agent_name}_channel"
                        )
                    else:
                        return False
                elif agent_name == "verifier":
                    return round % 2 == 0

        else:

            # Provers first, sequential
            if self.params.debate_protocol.sequential:
                if agent_name == first_prover:
                    if channel_name.startswith(f"{agent_name}_channel"):
                        return round % 3 == 0
                    else:
                        return False
                elif agent_name == second_prover:
                    if channel_name.startswith(f"{agent_name}_channel"):
                        return round % 3 == 1
                    else:
                        return False
                elif agent_name == "verifier":
                    return round % 3 == 2

            # Provers first, simultaneous
            else:
                if agent_name in ["prover0", "prover1"]:
                    if channel_name.startswith(f"{agent_name}_channel"):
                        return round % 2 == 0
                    else:
                        return False
                elif agent_name == "verifier":
                    return round % 2 == 1

    @property
    def max_message_rounds(self) -> int:
        return self.params.debate_protocol.max_message_rounds

    @property
    def min_message_rounds(self) -> int:
        return self.params.debate_protocol.min_message_rounds

    def _include_non_verifier_rewards(
        self,
        env_td: TensorDictBase | NestedArrayDict,
        verifier_decision_made: Bool[Tensor, "..."],
        verifier_decision: Int[Tensor, "..."],
        reward: Float[Tensor, "... agent"],
        round: Int[Tensor, "..."],
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
class MerlinArthurProtocol(ProtocolHandler):
    """Implementation of the Merlin-Arthur protocol.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    """

    agent_names = ["prover0", "prover1", "verifier"]
    message_channel_names = ["main"]
    agent_channel_visibility = [
        ("prover0", "main"),
        ("prover1", "main"),
        ("verifier", "main"),
    ]

    max_message_rounds = 2
    min_message_rounds = 0
    max_verifier_turns = 1

    def get_active_agents_mask_from_rounds(
        self, round: Int[Tensor, "..."]
    ) -> Bool[Tensor, "... agent channel"]:
        """Get a boolean mask indicating which agents are active in a given round.

        A random one of the two provers goes first, and the verifier goes second.

        Parameters
        ----------
        round : Int[Tensor, "..."]
            The round of the protocol.

        Returns
        -------
        active_agents : Bool[Tensor, "... agent channel"]
            A boolean mask indicating which agents are active in the given round.
        """
        prover1_first = torch.randint_like(round, 2).bool()
        return rearrange(
            [
                (round % 2 == 0) & prover1_first,
                (round % 2 == 0) & ~prover1_first,
                round % 2 == 1,
            ],
            "agent ... -> ... agent 1",
        )

    def can_agent_be_active(
        self, agent_name: str, round: int, channel_name: str
    ) -> bool:
        """Specifies whether an agent can be active in a given round.

        When the verifier goes second, both provers can be active in (zero-based) even
        rounds, and the verifier is active in odd rounds.

        Returns
        -------
        can_be_active : bool
            Whether the agent can be active in the given round.
        """

        if self.params.protocol_common.verifier_first:
            if agent_name in ["prover0", "prover1"]:
                return round % 2 == 1
            elif agent_name == "verifier":
                return round % 2 == 0
        else:
            if agent_name in ["prover0", "prover1"]:
                return round % 2 == 0
            elif agent_name == "verifier":
                return round % 2 == 1

    def _include_non_verifier_rewards(
        self,
        env_td: TensorDictBase | NestedArrayDict,
        verifier_decision_made: Bool[Tensor, "..."],
        verifier_decision: Int[Tensor, "..."],
        reward: Float[Tensor, "... agent"],
        round: Int[Tensor, "..."],
    ):
        protocol_params = self.params.protocol_common

        if protocol_params.shared_reward:
            reward[..., 0] = reward[..., 1] = reward[..., 2]
        else:
            for prover_num in range(2):
                reward[..., prover_num] = (
                    verifier_decision_made & (verifier_decision == prover_num)
                ).float() * protocol_params.prover_reward

    def _validate_active_agents(self):
        # Since the Arthur-Merlin protocol is non-deterministic, we cannot validate the
        # active agents in the same way as for deterministic protocols
        pass


@register_protocol_handler(InteractionProtocolType.MNIP)
class MnipProtocol(PvgProtocol):
    """Implementation of the MNIP protocol.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    """

    agent_names = ["prover0", "prover1", "verifier"]
    message_channel_names = ["prover0_channel", "prover1_channel"]
    agent_channel_visibility = [
        ("prover0", "prover0_channel"),
        ("prover1", "prover1_channel"),
        ("verifier", "prover0_channel"),
        ("verifier", "prover1_channel"),
    ]

    def is_agent_active(self, agent_name: str, round: int, channel_name: str) -> bool:
        """Specifies whether an agent is active in a given round and channel.

        In sequential MNIP with verifier first, the order is:

        - Verifier in both channels
        - First prover (determined by `prover0_first`) in their respective channel
        - Second prover in their respective channel

        In simultaneous MNIP with verifier first, the order is:

        - Verifier in both channels
        - Provers in their respective channels at the same time

        Returns
        -------
        is_active : bool
            Whether the agent is active in the given round and channel.
        """

        if self.params.mnip_protocol.prover0_first:
            first_prover = "prover0"
            second_prover = "prover1"
        else:
            first_prover = "prover1"
            second_prover = "prover0"

        if self.params.protocol_common.verifier_first:

            # Verifier first, sequential
            if self.params.mnip_protocol.sequential:
                if agent_name == "verifier":
                    return round % 3 == 0
                elif agent_name == first_prover:
                    if channel_name == f"{agent_name}_channel":
                        return round % 3 == 1
                    else:
                        return False
                elif agent_name == second_prover:
                    if channel_name == f"{agent_name}_channel":
                        return round % 3 == 2
                    else:
                        return False

            # Verifier first, simultaneous
            else:
                if agent_name in ["prover0", "prover1"]:
                    if channel_name == f"{agent_name}_channel":
                        return (
                            round % 2 == 1 and channel_name == f"{agent_name}_channel"
                        )
                    else:
                        return False
                elif agent_name == "verifier":
                    return round % 2 == 0

        else:

            # Provers first, sequential
            if self.params.mnip_protocol.sequential:
                if agent_name == first_prover:
                    if channel_name == f"{agent_name}_channel":
                        return round % 3 == 0
                    else:
                        return False
                elif agent_name == second_prover:
                    if channel_name == f"{agent_name}_channel":
                        return round % 3 == 1
                    else:
                        return False
                elif agent_name == "verifier":
                    return round % 3 == 2

            # Provers first, simultaneous
            else:
                if agent_name in ["prover0", "prover1"]:
                    if channel_name == f"{agent_name}_channel":
                        return round % 2 == 0
                    else:
                        return False
                elif agent_name == "verifier":
                    return round % 2 == 1

    def _include_non_verifier_rewards(
        self,
        env_td: TensorDictBase | NestedArrayDict,
        verifier_decision_made: Bool[Tensor, "..."],
        verifier_decision: Int[Tensor, "..."],
        reward: Float[Tensor, "... agent"],
        round: Int[Tensor, "..."],
    ):
        protocol_params = self.params.protocol_common

        if protocol_params.shared_reward:
            reward[..., 0] = reward[..., 1] = reward[..., 2]
        else:

            for prover_num in range(2):
                reward[..., prover_num] = (
                    verifier_decision_made & (verifier_decision == 1)
                ).float() * protocol_params.prover_reward


@register_protocol_handler(InteractionProtocolType.PVG, zero_knowledge=True)
class ZeroKnowledgePvgProtocol(PvgProtocol, ABC):
    """Base class for zero-knowledge protocols.

    Introduces an adversarial verifier and a simulator. The simulator tries to mimic the
    interaction between the adversarial verifier and the prover, and the adversarial
    verifier tries to prevent this. The prover tries to make sure the simulator can
    succeed (which implies that it is not `leaking` knowledge).
    """

    @property
    def agent_names(self) -> list[str]:
        # Keep the same agents apart from adding the second verifier and the simulator
        return super().agent_names + ["adversarial_verifier", "simulator"]

    @cached_property
    def message_channel_names(self) -> list[str]:

        # Duplicate the existing channels (one per verifier) and add the simulator
        # channel
        message_channel_names = []
        for channel_name in super().message_channel_names:
            message_channel_names.append(channel_name + "_standard")
            message_channel_names.append(channel_name + "_adversarial")
            message_channel_names.append(channel_name + "_simulator")

        return message_channel_names

    @property
    def agent_channel_visibility(self) -> list[tuple[str, str]]:

        agent_channel_visibility = []

        # Make the adversarial channels visible to the all non-verifier agents and the
        # adversarial verifier
        for agent_name, channel_name in super().agent_channel_visibility:
            if agent_name == "verifier":
                agent_channel_visibility.append(
                    ("verifier", channel_name + "_standard")
                )
                agent_channel_visibility.append(
                    ("adversarial_verifier", channel_name + "_adversarial")
                )
                agent_channel_visibility.append(
                    ("simulator", channel_name + "_simulator")
                )
            else:
                agent_channel_visibility.append(
                    (agent_name, channel_name + "_standard")
                )
                agent_channel_visibility.append(
                    (agent_name, channel_name + "_adversarial")
                )

        return agent_channel_visibility

    def __init__(self, params: Parameters, settings: ExperimentSettings):
        super().__init__(params, settings)

        # We rely on the message logits, so we need to make sure they are available
        if params.scenario not in [
            ScenarioType.GRAPH_ISOMORPHISM,
            ScenarioType.IMAGE_CLASSIFICATION,
        ]:
            raise NotImplementedError(
                "Zero-knowledge protocols are only supported for graph isomorphism and "
                "image classification scenarios."
            )

    def is_agent_active(self, agent_name: str, round: int, channel_name: str) -> bool:

        # Simulators are the only agents active in the simulator channel, and are always active
        if agent_name == "simulator":
            return channel_name.endswith("_simulator")
        if channel_name.endswith("_simulator"):
            return False

        # Verifiers are active in their corresponding channels
        if agent_name in self.verifier_names:
            if channel_name.endswith("_adversarial"):
                if agent_name == "verifier":
                    return False
                return super().is_agent_active("verifier", round, channel_name[:-12])
            elif channel_name.endswith("_standard"):
                if agent_name == "adversarial_verifier":
                    return False
                return super().is_agent_active("verifier", round, channel_name[:-9])

        return super().is_agent_active(agent_name, round, channel_name[:-2])

    def _include_non_verifier_rewards(
        self,
        env_td: TensorDictBase | NestedArrayDict,
        verifier_decision_made: Bool[Tensor, "..."],
        verifier_decision: Int[Tensor, "..."],
        reward: Float[Tensor, "... agent"],
        round: Int[Tensor, "..."],
    ):

        protocol_params = self.params.protocol_common
        prover_index = self.agent_names.index("prover")
        verifier_index = self.agent_names.index("verifier")
        adversarial_verifier_index = self.agent_names.index("adversarial_verifier")
        simulator_index = self.agent_names.index("simulator")

        # Get the main message logits and flatten the channel and position dimensions
        main_message_logits: Float[Tensor, "... agent channel position logit"] = env_td[
            "agents", "main_message_logits"
        ]
        adversarial_verifier_logits = main_message_logits[
            ..., adversarial_verifier_index, :, :, :
        ]
        adversarial_verifier_logits = rearrange(
            adversarial_verifier_logits,
            "... channel position logit -> ... (channel position logit)",
        )
        simulator_logits = main_message_logits[..., simulator_index, :, :, :]
        simulator_logits = rearrange(
            simulator_logits,
            "... channel position logit -> ... (channel position logit)",
        )  # TODO-LH

        # Simulator reward, which is the cosine similarity between the adversarial_verifier and
        # simulator logits, multiplied by a coefficient
        simulator_reward = (
            F.cosine_similarity(adversarial_verifier_logits, simulator_logits, dim=-1)
            * self.params.zk_protocol.simulator_reward_coefficient
        )

        # Adversarial verifier reward
        reward[..., adversarial_verifier_index] = -simulator_reward

        # Prover reward
        if protocol_params.shared_reward:
            prover_reward = reward[..., verifier_index]
        else:
            prover_reward = (verifier_decision_made & (verifier_decision == 1)).float()
            prover_reward *= protocol_params.prover_reward
            prover_reward += (
                self.params.zk_protocol.aux_prover_reward_coefficient * simulator_reward
            )

        reward[..., prover_index] = prover_reward
        reward[..., simulator_index] = simulator_reward


class ZeroKnowledgeProtocol(ABC):
    """Mixin for zero-knowledge protocols.

    Introduces a second verifier and a simulator. The simulator tries to mimic the
    interaction between the second verifier and the prover(s), and the second verifier
    tries to prevent this. The prover(s) tries to make sure the simulator can succeed
    (which implies that it is not `leaking` knowledge).
    """

    verifier_names = ["verifier", "adversarial_verifier"]

    @property
    def agent_names(self):
        # Keep the same agents apart from adding the second verifier and the simulator
        return super().agent_names + ["adversarial_verifier", "simulator"]

    @cached_property
    def message_channel_names(self):

        # Duplicate the existing channels (one per verifier) and add the simulator
        # channel
        message_channel_names = []
        for channel_name in super().message_channel_names:
            message_channel_names.append(channel_name + "_standard")
            message_channel_names.append(channel_name + "_adversarial")
            message_channel_names.append(channel_name + "_simulator")

        return message_channel_names

    @property
    def agent_channel_visibility(self):

        agent_channel_visibility = []

        # Clone the existing visibility settings and add a separate channel for the
        # simulator
        for agent_name, channel_name in super().agent_channel_visibility:
            if agent_name == "verifier":
                agent_channel_visibility.append(
                    ("verifier", channel_name + "_standard")
                )
                agent_channel_visibility.append(
                    ("adversarial_verifier", channel_name + "_adversarial")
                )
                agent_channel_visibility.append(
                    ("simulator", channel_name + "_simulator")
                )
            else:
                agent_channel_visibility.append(
                    (agent_name, channel_name + "_standard")
                )
                agent_channel_visibility.append(
                    (agent_name, channel_name + "_adversarial")
                )

        return agent_channel_visibility

    def __init__(self, params: Parameters, settings: ExperimentSettings):
        super().__init__(params, settings)

        # We rely on the message logits, so we need to make sure they are available
        if params.scenario not in [
            ScenarioType.GRAPH_ISOMORPHISM,
            ScenarioType.IMAGE_CLASSIFICATION,
        ]:
            raise NotImplementedError(
                "Zero-knowledge protocols are only supported for graph isomorphism and "
                "image classification scenarios."
            )

    def is_agent_active(self, agent_name: str, round: int, channel_name: str) -> bool:

        # Simulators are the only agents active in the simulator channel, and are always active
        if agent_name == "simulator":
            return channel_name.endswith("_simulator")
        if channel_name.endswith("_simulator"):
            return False

        # Verifiers are active in their corresponding channels
        if agent_name in self.verifier_names:
            if channel_name.endswith("_adversarial"):
                if agent_name == "verifier":
                    return False
                return super().is_agent_active("verifier", round, channel_name[:-12])
            elif channel_name.endswith("_standard"):
                if agent_name == "adversarial_verifier":
                    return False
                return super().is_agent_active("verifier", round, channel_name[:-9])

        return super().is_agent_active(agent_name, round, channel_name[:-2])

    def _include_non_verifier_rewards(
        self,
        env_td: TensorDictBase | NestedArrayDict,
        verifier_decision_made: Bool[Tensor, "..."],
        verifier_decision: Int[Tensor, "..."],
        reward: Float[Tensor, "... agent"],
        round: Int[Tensor, "..."],
    ):

        # Get the main message logits and decision logits. Possibly we should also include the additional linear meassage logits? #TODO-LH
        main_message_logits: Float[Tensor, "... agent channel position logit"] = env_td[
            "agents", "main_message_logits"
        ]
        decision_logits: Float[Tensor, "... agent 3"] = env_td[
            "agents", "decision_logits"
        ]

        # Get necessary masks and indices
        active_agents_mask = self.get_active_agents_mask_from_rounds(round)
        adversarial_verifier_guess_mask = self.get_verifier_guess_mask_from_rounds(
            round, adversarial=True
        )
        prover_indices = [self.agent_names.index(p_n) for p_n in self.prover_names]
        adversarial_verifier_index = self.agent_names.index("adversarial_verifier")
        adversarial_channel_indices = [
            self.message_channel_names.index(m_c_n)
            for m_c_n in self.message_channel_names
            if m_c_n.endswith("_adversarial")
        ]
        simulator_index = self.agent_names.index("simulator")
        simulator_channel_indices = [
            self.message_channel_names.index(m_c_n)
            for m_c_n in self.message_channel_names
            if m_c_n.endswith("_simulator")
        ]

        # Get the message logits for the adversarial channels. Note that this currently assumes that in any given channel, only one agent is active. This assumoption can be removed by using utils.math.logit_or_n instead of simply summing the logits across the agent dimension.
        adversarial_channel_logits = main_message_logits[
            ..., adversarial_channel_indices, :, :
        ]
        adversarial_channel_mask = (
            active_agents_mask[..., adversarial_channel_indices]
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand_as(adversarial_channel_logits)
        )
        logits_per_adversarial_channel = (
            adversarial_channel_logits * adversarial_channel_mask
        ).sum(dim=-4)
        logits_per_adversarial_channel = rearrange(
            logits_per_adversarial_channel,
            "... channel position logit -> ... (channel position logit)",
        )

        # Get the message logits for the simulator channels
        logits_per_simulator_channel = main_message_logits[
            ..., simulator_index, simulator_channel_indices, :, :
        ]
        logits_per_simulator_channel = rearrange(
            logits_per_simulator_channel,
            "... channel position logit -> ... (channel position logit)",
        )

        # Simulator reward, which is the cosine similarity between the adversarial_verifier and
        # simulator logits, multiplied by a coefficient. At the moment we sum the similarities for the message logits and decision logits (because decisons are not necessarily allowed in every round, unlike messages, but we may want to do something esle here). Moreover, this does not currently take into account the possibility that the adversarial verifier has guessed (and therefore ended the trajectory) before the non-adversarial verifier, which means that for some timesteps we will be comparing logits that don't correspond to any action that makes any difference. #TODO-LH
        simulator_reward = (
            F.cosine_similarity(
                logits_per_adversarial_channel, logits_per_simulator_channel, dim=-1
            )
            * self.params.zk_protocol.simulator_reward_coefficient
        )
        if adversarial_verifier_guess_mask.any():
            # Get the adversarial verifier and simulator decision logits
            adversarial_verifier_decision_logits = decision_logits[
                adversarial_verifier_guess_mask, adversarial_verifier_index
            ]
            simulator_decision_logits = decision_logits[
                adversarial_verifier_guess_mask, simulator_index
            ]
            # Add the cosine similarity to the reward
            simulator_reward += (
                F.cosine_similarity(
                    adversarial_verifier_decision_logits, simulator_decision_logits
                )
                * self.params.zk_protocol.simulator_reward_coefficient
            )

        reward[..., simulator_index] = simulator_reward

        # Adversarial verifier reward
        reward[..., adversarial_verifier_index] = -simulator_reward

        # Prover rewards
        super()._include_non_verifier_rewards(
            env_td, verifier_decision_made, verifier_decision, reward, round
        )
        reward[..., prover_indices] += (
            self._prover_zk_loss_coefficient() * simulator_reward
        )

    # Eventually this should dynamically update the coefficient to allow for lexicographic optimisation, as in https://www.ijcai.org/proceedings/2022/0476.pdf
    def _prover_zk_loss_coefficient(self) -> float:

        return self.params.zk_protocol.aux_prover_reward_coefficient


# @register_protocol_handler(InteractionProtocolType.PVG, zero_knowledge=True)
class AltZeroKnowledgePvgProtocol(PvgProtocol, ZeroKnowledgeProtocol):
    pass


@register_protocol_handler(
    InteractionProtocolType.ABSTRACT_DECISION_PROBLEM, zero_knowledge=True
)
class ZeroKnowledgeAdpProtocol(AdpProtocol, ZeroKnowledgeProtocol):
    pass


@register_protocol_handler(InteractionProtocolType.MERLIN_ARTHUR, zero_knowledge=True)
class ZeroKnowledgeMerlinArthurProtocol(MerlinArthurProtocol, ZeroKnowledgeProtocol):
    pass


@register_protocol_handler(InteractionProtocolType.MNIP, zero_knowledge=True)
class ZeroKnowledgeMnipProtocol(MnipProtocol, ZeroKnowledgeProtocol):
    pass


@register_protocol_handler(InteractionProtocolType.DEBATE, zero_knowledge=True)
class ZeroKnowledgeDebateProtocol(ZeroKnowledgeProtocol, DebateProtocol):
    pass


@register_protocol_handler(InteractionProtocolType.MULTI_CHANNEL_TEST)
class MultiChannelTestProtocol(DeterministicProtocolHandler):
    """A protocol for testing multi-channel communication between agents."""

    agent_names = ["prover0", "prover1", "prover2", "verifier"]
    message_channel_names = ["main", "prover0_verifier", "prover_chat"]
    agent_channel_visibility = [
        ("prover0", "main"),
        ("prover1", "main"),
        ("prover2", "main"),
        ("verifier", "main"),
        ("prover0", "prover0_verifier"),
        ("verifier", "prover0_verifier"),
        ("prover0", "prover_chat"),
        ("prover1", "prover_chat"),
        ("prover2", "prover_chat"),
    ]

    max_message_rounds = 8
    min_message_rounds = 2
    max_verifier_turns = 4

    def is_agent_active(self, agent_name: str, round: int, channel_name: str):
        if channel_name == "main":
            if round % 3 == 0:
                return agent_name == "prover1"
            elif round % 3 == 1:
                return agent_name == "prover2"
            elif round % 3 == 2:
                return agent_name == "verifier"
        elif channel_name == "prover0_verifier":
            if round % 3 == 1:
                return agent_name == "prover0"
            elif round % 3 == 2:
                return agent_name == "verifier"
        elif channel_name == "prover_chat":
            if round % 3 == 0:
                return agent_name == "prover0"
            elif round % 3 == 1:
                return agent_name == "prover1"
            else:
                return agent_name == "prover2"

    def _include_non_verifier_rewards(
        self,
        env_td: TensorDictBase | NestedArrayDict,
        verifier_decision_made: Bool[Tensor, "..."],
        verifier_decision: Int[Tensor, "..."],
        reward: Float[Tensor, "... agent"],
        round: Int[Tensor, "..."],
    ):
        pass
