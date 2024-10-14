"""Base classes for protocol handlers.

These define the standard interface which protocol handlers must implement, and give
a base implementation for deterministic protocols with a single verifier.
"""

from abc import ABC, abstractmethod
from functools import cached_property
from itertools import product
from typing import ClassVar, Optional

import torch
from torch import Tensor

from tensordict.tensordict import TensorDictBase

from jaxtyping import Int, Bool, Float

from pvg.parameters import Parameters, Guess
from pvg.experiment_settings import ExperimentSettings
from pvg.utils.nested_array_dict import NestedArrayDict


class ProtocolHandler(ABC):
    """Base class for protocol handlers.

    A protocol handler gives the implementation of an exchange protocol, specifying what
    agents are present, how they interact, and how the environment is updated.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    """

    can_be_zero_knowledge: ClassVar[bool] = True

    def __init__(self, params: Parameters, settings: ExperimentSettings):
        self.params = params
        self.settings = settings

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
    def num_agents(self) -> int:
        """The number of agents in the protocol."""
        return len(self.agent_names)

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
    def get_active_agents_mask_from_rounds_and_seed(
        self, round: Int[Tensor, "..."], seed: Int[Tensor, "..."]
    ) -> Bool[Tensor, "... agent channel"]:
        """Get a boolean mask of active agents for a batch of rounds.

        Given a batch or rounds, returns a boolean mask indicating which agents are
        sending messages in each round and channel.

        Parameters
        ----------
        round : Int[Tensor, "..."]
            The round of the protocol.
        seed : Int[Tensor, "..."]
            The per-environment seed.

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

    @abstractmethod
    def get_verifier_guess_mask_from_rounds_and_seed(
        self, round: Int[Tensor, "..."], seed: Int[Tensor, "..."]
    ) -> Bool[Tensor, "..."]:
        """Get a boolean mask indicating when the verifiers can make a guess/decision.

        Takes as input a tensor of rounds and returns a boolean mask indicating when the
        verifiers can make a guess for each element in the batch.

        Parameters
        ----------
        round : Int[Tensor, "..."]
            The batch of rounds.
        seed : Int[Tensor, "..."]
            The per-environment seed.

        Returns
        -------
        verifier_turn : Bool[Tensor, "..."]
            Which batch items the verifiers can make a guess in.
        """

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
    def step_interaction_protocol(
        self,
        env_td: TensorDictBase | NestedArrayDict,
    ) -> tuple[
        Bool[Tensor, "..."],
        Bool[Tensor, "... agent"],
        Bool[Tensor, "..."],
        Float[Tensor, "... agent"],
    ]:
        """Take a step in the interaction protocol.

        Computes the done signals and reward.

        Used in the `_step` method of the environment.

        Parameters
        ----------
        env_td : TensorDictBase | NestedArrayDict
            The current observation and state. If a `NestedArrayDict`, it is converted
            to a `TensorDictBase`.

        Returns
        -------
        shared_done : Bool[Tensor, "..."]
            A boolean mask indicating whether the episode is done because all relevant
            agents have made a decision.
        agent_done : Bool[Tensor, "... agent"]
            A boolean mask indicating whether each agent is done, because they have made
            a decision. This is the same as `shared_done` for agents which don't make
            decisions.
        terminated : Bool[Tensor, "..."]
            A boolean mask indicating whether the episode has been terminated because
            the max number of rounds has been reached and the verifier has not guessed.
        reward : Float[Tensor, "... agent"]
            The reward for the agents.
        """

    def _get_agent_decision_made_mask(
        self,
        round: Int[Tensor, "..."],
        y: Int[Tensor, "... 1"],
        guess_mask: Bool[Tensor, "..."],
        decision: Int[Tensor, "..."],
        *,
        follow_force_guess: bool = True,
    ) -> Bool[Tensor, "..."]:
        """Get a mask indicating whether an agent has made a decision

        Parameters
        ----------
        round : Int[Tensor, "..."]
            The round number.
        y : Int[Tensor, "... 1"]
            The target value.
        guess_mask : Bool[Tensor, "..."]
            A mask indicating whether the agent is allowed to make a guess.
        decision : Int[Tensor, "..."]
            The decision output of the agent.
        follow_force_guess : bool, default=True
            Whether to follow the `force_guess` parameter, which forces the agent to
            make a certain decision.
        """

        if follow_force_guess:
            if self.params.protocol_common.force_guess == Guess.ONE:
                decision = torch.ones_like(decision)
            elif self.params.protocol_common.force_guess == Guess.ZERO:
                decision = torch.zeros_like(decision)
            elif self.params.protocol_common.force_guess == Guess.Y:
                decision = y.squeeze(-1)

        verifier_decision_made = guess_mask & (decision != 2)
        verifier_decision_made = verifier_decision_made & (
            round >= self.min_message_rounds
        )

        return verifier_decision_made


class SingleVerifierProtocolHandler(ProtocolHandler, ABC):
    """Base class for protocol handlers with a single verifier.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The settings of the experiment.
    verifier_name : str, default="verifier"
        The name of the verifier.
    """

    def __init__(
        self,
        params: Parameters,
        settings: ExperimentSettings,
        *,
        verifier_name: str = "verifier",
    ):
        super().__init__(params, settings)

        self.verifier_name = verifier_name

    @property
    def verifier_index(self) -> int:
        """The index of the verifier in the list of agent names."""
        return self.agent_names.index(self.verifier_name)

    @property
    def verifier_names(self) -> list[str]:
        """The names of the verifiers in the protocol."""
        return [self.verifier_name]

    def get_verifier_guess_mask_from_rounds_and_seed(
        self, round: Int[Tensor, "..."], seed: Int[Tensor, "..."]
    ) -> Bool[Tensor, "..."]:
        """Get a boolean mask indicating when the verifier can make a guess.

        Takes as input a tensor of rounds and returns a boolean mask indicating when the
        verifier can make a guess for each element in the batch.

        Parameters
        ----------
        round : Int[Tensor, "..."]
            The batch of rounds.
        seed : Int[Tensor, "..."]
            The per-environment seed.

        Returns
        -------
        verifier_turn : Bool[Tensor, "..."]
            Which batch items the verifiers can make a guess in.
        """
        active_agents_mask = self.get_active_agents_mask_from_rounds_and_seed(
            round, seed
        )
        verifier_active_mask = active_agents_mask[
            ..., self.agent_names.index(self.verifier_name), :
        ]
        return verifier_active_mask.any(dim=-1)

    def step_interaction_protocol(
        self,
        env_td: TensorDictBase | NestedArrayDict,
    ) -> tuple[
        Bool[Tensor, "..."],
        Bool[Tensor, "... agent"],
        Bool[Tensor, "..."],
        Float[Tensor, "... agent"],
    ]:
        """Take a step in the interaction protocol.

        Computes the done signals and reward.

        Used in the `_step` method of the environment.

        Parameters
        ----------
        env_td : TensorDictBase | NestedArrayDict
            The current observation and state. If a `NestedArrayDict`, it is converted
            to a `TensorDictBase`. Has keys:

            - "y" (... 1): The target value.
            - "round" (...): The current round.
            - ("agents", "decision") (... agent): The decision of each agent.
            - "done" (...): A boolean mask indicating whether the episode is done.
            - ("agents", "done") (... agent): A boolean mask indicating whether each
                agent is done.
            - "terminated" (...): A boolean mask indicating whether the episode has been
                terminated.

        Returns
        -------
        shared_done : Bool[Tensor, "..."]
            A boolean mask indicating whether the episode is done because all relevant
            agents have made a decision.
        agent_done : Bool[Tensor, "... agent"]
            A boolean mask indicating whether each agent is done, because they have made
            a decision. This is the same as `shared_done` for agents which don't make
            decisions.
        terminated : Bool[Tensor, "..."]
            A boolean mask indicating whether the episode has been terminated because
            the max number of rounds has been reached and the verifier has not guessed.
        reward : Float[Tensor, "... agent"]
            The reward for the agents.
        """

        protocol_params = self.params.protocol_common

        y: Int[Tensor, "... 1"] = env_td["y"]
        round: Int[Tensor, "..."] = env_td["round"]
        seed: Int[Tensor, "..."] = env_td["seed"]
        decision: Int[Tensor, "... agent"] = env_td["agents", "decision"]
        shared_done: Bool[Tensor, "..."] = env_td["done"]
        agent_done: Bool[Tensor, "... agent"] = env_td["agents", "done"]
        terminated: Bool[Tensor, "..."] = env_td["terminated"]

        if isinstance(env_td, NestedArrayDict):
            y = torch.from_numpy(y)
            round = torch.from_numpy(round)
            seed = torch.from_numpy(seed)
            decision = torch.from_numpy(decision)
            shared_done = torch.from_numpy(shared_done)
            terminated = torch.from_numpy(terminated)

        # Get the mask of the batch items where the verifier can make a guess
        verifier_guess_mask = self.get_verifier_guess_mask_from_rounds_and_seed(
            round, seed
        )

        # Determine if the verifier has made a decision
        verifier_decision_made = self._get_agent_decision_made_mask(
            round=round,
            y=y,
            guess_mask=verifier_guess_mask,
            decision=decision[..., self.verifier_index],
        )

        # When the verifier has made a decision, the shared done is set to `True`.
        shared_done = shared_done | verifier_decision_made

        # Compute the reward for the verifier when they make a guess
        verifier_idx = (..., self.verifier_index)
        reward = torch.empty(
            (*shared_done.shape, len(self.agent_names)),
            dtype=torch.float,
            device=shared_done.device,
        )
        reward[verifier_idx] = torch.zeros_like(shared_done, dtype=torch.float)
        reward[verifier_idx][
            verifier_decision_made & (decision[verifier_idx] == y.squeeze(-1))
        ] = protocol_params.verifier_reward
        reward[verifier_idx][
            verifier_decision_made & (decision[verifier_idx] != y.squeeze(-1))
        ] = protocol_params.verifier_incorrect_penalty

        # If we reach the end of the episode and the verifier has not made a guess,
        # terminate it with a negative reward for the verifier
        terminated = terminated | (
            (round >= self.max_message_rounds - 1) & ~verifier_decision_made
        )
        reward[verifier_idx][
            (round >= self.max_message_rounds - 1) & ~verifier_decision_made
        ] = protocol_params.verifier_terminated_penalty

        # If the verifier has not made a guess and it's their turn, given them a small
        # reward for continuing
        reward[verifier_idx][
            verifier_guess_mask & ~shared_done & ~terminated
        ] = protocol_params.verifier_no_guess_reward

        # Compute the rewards for the other agents and add them
        self._include_prover_rewards(
            verifier_decision_made, decision[verifier_idx], reward
        )

        # The agent-specific done signal is the same as the shared done signal
        agent_done = agent_done | shared_done[..., None]

        return shared_done, agent_done, terminated, reward

    @abstractmethod
    def _include_prover_rewards(
        self,
        verifier_decision_made: Bool[Tensor, "..."],
        verifier_decision: Int[Tensor, "..."],
        reward: Float[Tensor, "... agent"],
    ):
        """Compute the rewards for the other agents and add them to the current reward.

        The `reward` tensor is updated in place, adding in the rewards for the agents
        at the appropriate indices.

        Parameters
        ----------
        verifier_decision_made : Bool[Tensor, "..."]
            A boolean mask indicating whether the verifier has made a decision.
        verifier_decision : Int[Tensor, "..."]
            The verifier's decision.
        reward : Float[Tensor, "... agent"]
            The currently computed reward, which should include the reward for the
            verifier.
        """


class DeterministicSingleVerifierProtocolHandler(SingleVerifierProtocolHandler, ABC):
    """Base class for handlers of deterministic protocols with a single verifier.

    A protocol handler gives the implementation of an exchange protocol, specifying what
    agents are present, how they interact, and how the environment is updated.

    An exchange protocol is deterministic if the agents' which agents are active in each
    round and channel is determined by the round and channel alone.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The settings of the experiment.
    """

    def __init__(
        self,
        params: Parameters,
        settings: ExperimentSettings,
        *,
        verifier_name: str = "verifier",
    ):
        super().__init__(params, settings, verifier_name=verifier_name)

        self._validate_active_agents()

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

    def get_active_agents_mask_from_rounds_and_seed(
        self, round: Int[Tensor, "..."], seed: Int[Tensor, "..."] | None
    ) -> Bool[Tensor, "... agent channel"]:
        """Get a boolean mask of active agents for a batch of rounds.

        Given a batch or rounds, returns a boolean mask indicating which agents are
        active in each round and channel.

        Parameters
        ----------
        round : Int[Tensor, "..."]
            The round of the protocol.
        seed : Int[Tensor, "..."] | None
            The per-environment seed. This is ignored for deterministic protocols, so it
            can be `None`.

        Returns
        -------
        active_agents : Bool[Tensor, "... agent channel"]
            The boolean mask. `active_agents[*batch, agent, channel]` is `True` if the
            agent sends a message in the channel in round `round[*batch]`.
        """

        return self.active_agents_mask[round, :, :]

    def get_verifier_guess_mask_from_rounds_and_seed(
        self, round: Int[Tensor, "..."], seed: Int[Tensor, "..."] | None
    ) -> Bool[Tensor, "..."]:
        """Get a boolean mask indicating when the verifier can make a guess.

        Takes as input a tensor of rounds and returns a boolean mask indicating when the
        verifier can make a guess for each element in the batch.

        Parameters
        ----------
        round : Int[Tensor, "..."]
            The batch of rounds.
        seed : Int[Tensor, "..."] | None
            The per-environment seed. This is ignored for deterministic protocols, so it
            can be `None`.

        Returns
        -------
        verifier_turn : Bool[Tensor, "..."]
            Which batch items the verifier can make a guess in.
        """
        active_agents_mask = self.get_active_agents_mask_from_rounds_and_seed(
            round, seed
        )
        verifier_active_mask = active_agents_mask[..., self.verifier_index, :]
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
