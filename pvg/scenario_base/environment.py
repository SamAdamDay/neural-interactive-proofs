"""Base class for the RL environment."""

from abc import ABC, abstractmethod
from typing import Optional, Any, ClassVar
from operator import mul
from functools import reduce, cached_property
from itertools import chain
from math import prod

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.data.tensor_specs import (
    CompositeSpec,
    BinaryDiscreteTensorSpec,
    DiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs import EnvBase

import numpy as np
from numpy.typing import NDArray

import einops

from jaxtyping import Float, Int

from pvg.scenario_base import (
    TensorDictDataLoader,
    NestedArrayDictDataLoader,
    Dataset,
    TensorDictDataset,
    NestedArrayDictDataset,
)
from pvg.protocols import ProtocolHandler
from pvg.parameters import Parameters
from pvg.experiment_settings import ExperimentSettings
from pvg.utils.data import VariableDataCycler
from pvg.utils.nested_array_dict import (
    NestedArrayDict,
    NumpySpec,
    CompositeSpec as CompositeArraySpec,
    IntArraySpec,
    StringArraySpec,
    FloatArraySpec,
    BoolArraySpec,
)


class Environment(ABC):
    """The base class for all Prover-Verifier RL environments.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The settings of the experiment.
    dataset : Dataset
        The dataset for the environment.
    protocol_handler : ProtocolHandler
        The protocol handler for the environment.
    train : bool, optional
        Whether the environment is used for training or evaluation.
    """

    @cached_property
    def steps_per_env_per_iteration(self) -> int:
        """The number of steps per batched environment in each iteration."""
        # The number of environments is the number of episodes we can fit in a batch
        if self.params.rl.steps_per_env_per_iteration is not None:
            steps_per_env_per_iteration = self.params.rl.steps_per_env_per_iteration
            if (
                self.params.rl.frames_per_batch is not None
                and self.params.rl.frames_per_batch % steps_per_env_per_iteration != 0
            ):
                raise ValueError(
                    f"The parameter `rl.steps_per_env_per_iteration` must divide "
                    f"`rl.frames_per_batch` without remainder, but got "
                    f"{steps_per_env_per_iteration} and "
                    f"{self.params.rl.frames_per_batch}."
                )
        else:
            steps_per_env_per_iteration = self.protocol_handler.max_message_rounds
            if (
                self.params.rl.frames_per_batch is not None
                and self.params.rl.frames_per_batch % steps_per_env_per_iteration != 0
            ):
                raise ValueError(
                    f"The maximum number of message rounds must divide "
                    f"`rl.frames_per_batch` without remainder, but got "
                    f"{steps_per_env_per_iteration} and "
                    f"{self.params.rl.frames_per_batch}."
                )
        return steps_per_env_per_iteration

    @property
    def frames_per_batch(self) -> int:
        if self.params.rl.frames_per_batch is not None:
            return self.params.rl.frames_per_batch
        else:
            return len(self.dataset) * self.steps_per_env_per_iteration

    @property
    def num_envs(self) -> int:
        """The number of batched environments."""
        return self.frames_per_batch // self.steps_per_env_per_iteration

    @property
    def batch_size(self) -> tuple[int, ...]:
        """The batch size of the environment."""
        return (self.num_envs,)

    @property
    @abstractmethod
    def observation_spec(self) -> TensorSpec | NumpySpec:
        """The specification for the observation keys."""

    @property
    @abstractmethod
    def action_spec(self) -> TensorSpec | NumpySpec:
        """The specification for the action keys."""

    @property
    @abstractmethod
    def state_spec(self) -> TensorSpec | NumpySpec:
        """The specification for the state keys."""

    @property
    @abstractmethod
    def reward_spec(self) -> TensorSpec | NumpySpec:
        """The specification for the agent reward keys."""

    @property
    @abstractmethod
    def done_spec(self) -> TensorSpec | NumpySpec:
        """The specification for the done keys (done and terminated)."""

    def __init__(
        self,
        params: Parameters,
        settings: ExperimentSettings,
        dataset: Dataset,
        protocol_handler: ProtocolHandler,
        *,
        train: bool = True,
    ):
        self.params = params
        self.settings = settings
        self.protocol_handler = protocol_handler
        self.train = train
        self.dataset = dataset
        self.data_cycler: Optional[VariableDataCycler] = None

        self.num_agents = len(self.protocol_handler.agent_names)

        self.data_cycler: Optional[VariableDataCycler] = None

    @abstractmethod
    def step(self, env_state: Any) -> Any:
        """Perform a step in the environment.

        Parameters
        ----------
        env_state : Any
            The current environment state.

        Returns
        -------
        next_env_state : Any
            The next environment state.
        """

    @abstractmethod
    def reset(self, env_state: Optional[Any] = None, **kwargs) -> Any:
        """Reset the environment.

        Parameters
        ----------
        env_state : Optional[Any]
            The current environment state.

        Returns
        -------
        next_env_state : Any
            The reset environment state.
        """


class TensorDictEnvironment(EnvBase, Environment, ABC):
    """The base class for all Prover-Verifier RL environments which use tensordicts.

    To implement a new environment, subclass this class and implement the following
    attribute and methods:

    - `_message_history_shape`: The shape of the message history and 'x' tensors.
    - `_get_observation_spec`: The specification of the agent observations.
    - `_get_action_spec`: The specification of the agent actions.
    - `_get_state_spec` (optional): The specification of the states space.
    - `_get_reward_spec` (optional): The specification of the agent rewards.
    - `_get_done_spec` (optional): The specification of the agent done signals.
    - `_step`: Perform a step in the environment.
    - `_compute_message_history`: Compute the new message history and next message.
    - `_masked_reset`: Reset the environment for a subset of the episodes.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The settings of the experiment.
    dataset : TensorDictDataset
        The dataset for the environment.
    protocol_handler : ProtocolHandler
        The protocol handler for the environment.
    train : bool, optional
        Whether the environment is used for training or evaluation.
    """

    _int_dtype: torch.dtype = torch.int

    def __init__(
        self,
        params: Parameters,
        settings: ExperimentSettings,
        dataset: TensorDictDataset,
        protocol_handler: ProtocolHandler,
        *,
        train: bool = True,
    ):
        super().__init__(device=settings.device)

        # Call the `EnvBase` constructor
        super(nn.Module, self).__init__(
            params=params,
            settings=settings,
            protocol_handler=protocol_handler,
            dataset=dataset,
            train=train,
        )

        # Call the batch size property of `Environment` to set the batch size
        self.batch_size = super(nn.Module, self).batch_size

        # Create environment specs
        self.observation_spec = self._get_observation_spec()
        self.action_spec = self._get_action_spec()
        self.state_spec = self._get_state_spec()
        self.reward_spec = self._get_reward_spec()
        self.done_spec = self._get_done_spec()

        # Create a random number generator
        self.rng = torch.Generator()

    @property
    @abstractmethod
    def main_message_space_shape(self) -> tuple:
        """The shape of the main message space used by the agents to communicate

        This is typically informed by the type of data in the environment. E.g. for
        images this might have shape "height width"
        """

    @property
    @abstractmethod
    def main_message_out_key(self) -> str:
        """The tensordict key which contains the main message sent by each agent

        This key is the main output of an agent. Typically, the agent's forward pass
        will output logits for this key, which will then be selected from.
        """

    @property
    def message_history_shape(self) -> tuple:
        """The shape of the message history and 'x' tensors

        This is used to make the specification for these.

        Returns
        -------
        message_history_shape: tuple
            The common shape of the message history and 'x' tensors.
        """
        return (
            self.num_envs,
            self.protocol_handler.max_message_rounds,
            self.protocol_handler.num_message_channels,
            self.params.message_size,
            *self.main_message_space_shape,
        )

    @abstractmethod
    def _get_observation_spec(self) -> TensorSpec:
        """Get the specification of the agent observations.

        The observation space has the following elements:

        - `round`: The current round of the interaction.
        - `decision_restriction`: The restriction on what the verifier can decide.
            * 0: The verifier can decide anything.
            * 1: The verifier can only decide to continue interacting.
            * 2: The verifier can only make a guess.
        - `x`: The message history.
        - `message`: The next message.
        - `pretrained_embeddings`: The pretrained embeddings, if any. This is a nested
          specification, where the sub-keys are the pretrained model names.
        - `linear_message_history`: The linear message history, if it is included.

        Returns
        -------
        observation_spec : TensorSpec
            The observation specification.
        """

        observation_spec = CompositeSpec(
            round=DiscreteTensorSpec(
                self.protocol_handler.max_message_rounds,
                shape=(self.num_envs,),
                dtype=torch.long,
                device=self.device,
            ),
            decision_restriction=DiscreteTensorSpec(
                3,
                shape=(self.num_envs,),
                dtype=self._int_dtype,
                device=self.device,
            ),
            x=UnboundedContinuousTensorSpec(
                shape=self.message_history_shape,
                dtype=torch.float,
                device=self.device,
            ),
            message_history=BinaryDiscreteTensorSpec(
                self.message_history_shape[-1],
                shape=self.message_history_shape,
                dtype=torch.float,
                device=self.device,
            ),
            shape=(self.num_envs,),
            device=self.device,
        )

        # Add specifications for the pretrained embeddings, if any
        pretrained_model_names = self.dataset.pretrained_model_names
        for model_name in pretrained_model_names:
            observation_spec["pretrained_embeddings", model_name] = (
                UnboundedContinuousTensorSpec(
                    shape=(
                        self.num_envs,
                        *self.dataset.get_pretrained_embedding_feature_shape(
                            model_name
                        ),
                    ),
                    dtype=self.dataset.get_pretrained_embedding_dtype(model_name),
                    device=self.device,
                )
            )

        # Add the linear message history, if it is included
        if self.params.include_linear_message_space:
            observation_spec["linear_message_history"] = BinaryDiscreteTensorSpec(
                self.params.d_linear_message_space,
                shape=(
                    self.num_envs,
                    self.protocol_handler.max_message_rounds,
                    self.protocol_handler.num_message_channels,
                    self.params.message_size,
                    self.params.d_linear_message_space,
                ),
                dtype=torch.float,
                device=self.device,
            )

        return observation_spec

    @abstractmethod
    def _get_action_spec(self) -> TensorSpec:
        """Get the specification of the agent actions.

        Subclasses should call this method and add any additional action spaces.

        Returns
        -------
        action_spec : TensorSpec
            The action specification.
        """

        action_spec = CompositeSpec(
            agents=CompositeSpec(
                decision=DiscreteTensorSpec(
                    3,
                    shape=(self.num_envs, self.num_agents),
                    dtype=self._int_dtype,
                    device=self.device,
                ),
                main_message_logits=UnboundedContinuousTensorSpec(
                    shape=(
                        self.num_envs,
                        self.num_agents,
                        self.protocol_handler.num_message_channels,
                        self.params.message_size,
                        prod(self.main_message_space_shape),
                    ),
                    device=self.device,
                ),
                decision_logits=UnboundedContinuousTensorSpec(
                    shape=(
                        self.num_envs,
                        self.num_agents,
                        3,
                    ),
                    device=self.device,
                ),
                shape=(self.num_envs, self.num_agents),
                device=self.device,
            ),
            shape=(self.num_envs,),
            device=self.device,
        )

        if self.params.include_linear_message_space:
            action_spec["agents"]["linear_message_selected"] = DiscreteTensorSpec(
                self.params.d_linear_message_space,
                shape=(
                    self.num_envs,
                    self.num_agents,
                    self.protocol_handler.num_message_channels,
                    self.params.message_size,
                ),
                dtype=torch.long,
                device=self.device,
            )

        return action_spec

    def _get_state_spec(self) -> TensorSpec:
        """Get the specification of the states space.

        Defaults to the true label.

        Returns
        -------
        state_spec : TensorSpec
            The state specification.
        """
        return CompositeSpec(
            y=BinaryDiscreteTensorSpec(
                1,
                shape=(self.num_envs, 1),
                dtype=torch.long,
                device=self.device,
            ),
            shape=(self.num_envs,),
            device=self.device,
        )

    def _get_reward_spec(self) -> TensorSpec:
        """Get the specification of the agent rewards.

        Returns
        -------
        reward_spec : TensorSpec
            The reward specification.
        """
        return CompositeSpec(
            agents=CompositeSpec(
                reward=UnboundedContinuousTensorSpec(
                    shape=(self.num_envs, self.num_agents),
                    device=self.device,
                ),
                shape=(self.num_envs, self.num_agents),
                device=self.device,
            ),
            shape=(self.num_envs,),
            device=self.device,
        )

    def _get_done_spec(self) -> TensorSpec:
        """Get the specification of the agent done signals.

        Returns
        -------
        done_spec : TensorSpec
            The done specification.
        """
        return CompositeSpec(
            done=BinaryDiscreteTensorSpec(
                self.num_envs,
                shape=(self.num_envs,),
                dtype=torch.bool,
                device=self.device,
            ),
            terminated=BinaryDiscreteTensorSpec(
                self.num_envs,
                shape=(self.num_envs,),
                dtype=torch.bool,
                device=self.device,
            ),
            shape=(self.num_envs,),
            device=self.device,
        )

    def _step(self, env_td: TensorDictBase) -> TensorDictBase:
        """Perform a step in the environment.

        Parameters
        ----------
        env_td : TensorDictBase
            The current observation and state.

        Returns
        -------
        next_td : TensorDictBase
            The next observation, state, reward, and done signal.
        """

        # Create an initial next tensordict, which will be updated with the message
        # history, next message, done signal and reward
        next_td = TensorDict(
            dict(
                round=env_td["round"] + 1,
                agents=TensorDict(
                    {},
                    batch_size=(*self.batch_size, self.num_agents),
                ),
            ),
            batch_size=self.batch_size,
            device=self.device,
        )

        # The observations are passed through unchanged
        for key in self.observation_spec.keys():
            if key not in [
                "x",
                "message_history",
                "message",
                "linear_message_history",
                "linear_message",
                "round",
                "decision_restriction",
            ]:
                next_td[key] = env_td[key]

        # Compute the message history and next message in the main message space
        next_td = self._compute_message_history_and_next_message(
            env_td,
            next_td,
            message_out_key=self.main_message_out_key,
            message_in_key="message",
            message_history_key="message_history",
            message_shape=self.main_message_space_shape,
        )

        # Do the same for the linear message space, if it is included
        if self.params.include_linear_message_space:
            next_td = self._compute_message_history_and_next_message(
                env_td,
                next_td,
                message_out_key="linear_message_selected",
                message_in_key="linear_message",
                message_shape=(self.params.d_linear_message_space,),
                message_history_key="linear_message_history",
            )

        # Clone the message history to the 'x' feature tensor
        next_td["x"] = next_td["message_history"].clone()

        # Compute the done signal and reward
        done, terminated, reward = self.protocol_handler.step_interaction_protocol(
            env_td
        )
        done = done | terminated  # TODO: Improve handling of terminated
        next_td.set("done", done)
        next_td.set("terminated", terminated)
        next_td.set(("agents", "reward"), reward)
        next_td.set(
            "decision_restriction", torch.zeros_like(done, dtype=self._int_dtype)
        )

        return next_td

    def _compute_message_history_and_next_message(
        self,
        env_td: TensorDictBase,
        next_td: TensorDictBase,
        *,
        message_out_key: str,
        message_in_key: str,
        message_history_key: str,
        message_shape: tuple[int, ...],
    ) -> TensorDictBase:
        """Compute the new message history and next message for given keys

        This is a generic method for updating one-hot encoded next message and message
        history tensors given a choice of message for each agent.

        Used in the `_step` method of the environment.

        Parameters
        ----------
        env_td : TensorDictBase
            The current observation and state.
        next_td : TensorDictBase
            The 'next' tensordict, to be updated with the message history and next
            message.
        message_out_key : str
            The key in the 'agents' sub-tensordict which contains the message selected
            by each agent. This results from the output of the agent's forward pass.
        message_in_key : str
            The key which contains the next message to be sent, which is used as input
            to each agent.
        message_history_key : str
            The key which contains the message history tensor.
        message_shape : tuple[int, ...]
            The shape of the message space.

        Returns
        -------
        next_td : TensorDictBase
            The updated 'next' tensordict.
        """

        # Get a string representation of the message space dims, for type annotation.
        # dim_1 dim_2 etc.
        message_shape_str = " ".join(f"dim_{i}" for i in range(len(message_shape)))
        message_shape_ones = " ".join(["1"] * len(message_shape))

        # ... round channel position {message_shape_str}
        message_history = env_td.get(message_history_key)
        round: Int[Tensor, "..."] = env_td.get("round")
        message_selected: Int[Tensor, "... agent channel position"] = env_td.get(
            ("agents", message_out_key)
        )

        # Get the mask for the active agents per channel in the current round
        # (... agent channel)
        active_agents_mask = self.protocol_handler.get_active_agents_mask_from_rounds(
            round
        )

        # Sum up the messages from the agents whose turn it is. If two agents select the
        # same message number, the message will be 2.
        # (... channel position {message_shape_str})
        message = F.one_hot(message_selected, reduce(mul, message_shape)).float()
        message = torch.where(active_agents_mask[..., None, None], message, 0)
        message = einops.reduce(
            message,
            f"... agent channel position ({message_shape_str}) "
            f"-> ... channel position {message_shape_str}",
            reduction="sum",
            **{f"dim_{i}": dim for i, dim in enumerate(message_shape)},
        )

        # Get a mask for which round it is
        round_mask = F.one_hot(round, self.protocol_handler.max_message_rounds).bool()

        # Reshape it so that it looks like the message history with 1's for the message
        # space dims and channel and position dim
        round_mask = einops.rearrange(
            round_mask, f"... round -> ... round 1 1 {message_shape_ones}"
        )

        # Insert the message into the message history at the current round
        message_history = message_history.masked_scatter(round_mask, message)

        # Add the message history and next message to the next tensordict
        next_td.set(message_history_key, message_history)
        next_td.set(message_in_key, message)

        return next_td

    def _reset(self, env_td: Optional[TensorDictBase] = None) -> TensorDictBase:
        """(Partially) reset the environment.

        For each episode which is done, takes a new sample from the dataset and resets
        the episode.

        Parameters
        ----------
        env_td : Optional[TensorDictBase]
            The current observation, state and done signal.

        Returns
        -------
        env_td : TensorDictBase
            The reset environment tensordict.
        """

        # If no tensordict is given, we're starting afresh
        if env_td is None or "done" not in env_td.keys():
            observation_zeros = self.observation_spec.zero()
            state_zeros = self.state_spec.zero()
            done_zeros = self.done_spec.zero()
            env_td = observation_zeros.update(state_zeros).update(done_zeros)
            new_mask = torch.ones(
                *self.batch_size, dtype=torch.bool, device=self.device
            )

        else:
            new_mask = env_td["done"]
            env_td = env_td.clone()

        # If we don't have a data cycler yet, create one
        if self.data_cycler is None:
            dataloader = TensorDictDataLoader(
                self.dataset,
                batch_size=self.num_envs,
                shuffle=True,
                generator=self.rng,
                pin_memory=self.settings.pin_memory,
                pin_memory_device=(
                    str(self.device)
                    if self.settings.pin_memory and str(self.device) != "cpu"
                    else ""
                ),
            )
            self.data_cycler = VariableDataCycler(
                dataloader, device=self.device, non_blocking=self.settings.pin_memory
            )

        # Sample a new batch of data for the episodes that are done
        batch = self.data_cycler.get_batch(new_mask.sum().item())

        # Reset the episodes that are done
        env_td = self._masked_reset(env_td, new_mask, batch)

        return env_td

    @abstractmethod
    def _masked_reset(
        self, env_td: TensorDictBase, mask: torch.Tensor, data_batch: TensorDict
    ) -> TensorDictBase:
        """Reset the environment for a subset of the episodes.

        Takes a new sample from the dataset and inserts it into the given episodes. Also
        resets the other elements of the episodes.

        Parameters
        ----------
        env_td : TensorDictBase
            The current observation, state and done signal.
        mask : torch.Tensor
            A boolean mask of the episodes to reset.
        data_batch : TensorDict
            The data batch to insert into the episodes.

        Returns
        -------
        env_td : TensorDictBase
            The reset environment tensordict.
        """

        env_td["y"][mask] = data_batch["y"].unsqueeze(-1)
        env_td["message_history"][mask] = torch.zeros_like(
            env_td["message_history"][mask]
        )
        env_td["x"][mask] = torch.zeros_like(env_td["x"][mask])
        env_td["message"][mask] = 0
        env_td["round"][mask] = 0
        env_td["done"][mask] = False
        env_td["terminated"][mask] = False
        env_td["decision_restriction"][mask] = 0

        pretrained_model_names = self.dataset.pretrained_model_names
        for model_name in pretrained_model_names:
            env_td["pretrained_embeddings", model_name][mask] = data_batch[
                "pretrained_embeddings", model_name
            ]

        return env_td

    def _set_seed(self, seed: int | None):
        self.rng = torch.manual_seed(seed)


class PureTextEnvironment(Environment, ABC):
    """Base for environments which handle non-tokenised text with nested array dicts."""

    dataset: NestedArrayDictDataset

    @property
    @abstractmethod
    def observation_spec(self) -> CompositeArraySpec:
        """The specification for the observation keys."""
        return CompositeArraySpec(
            round=IntArraySpec(*self.batch_size, "batch"),
            message_history=StringArraySpec(
                (
                    *self.batch_size,
                    self.protocol_handler.max_message_rounds,
                    self.protocol_handler.num_message_channels,
                ),
                "batch round channel",
            ),
            shape=self.batch_size,
            dim_names="batch",
        )

    @cached_property
    def action_spec(self) -> CompositeArraySpec:
        """The specification for the action keys."""
        return CompositeArraySpec(
            agents=CompositeArraySpec(
                decision=IntArraySpec(
                    (*self.batch_size, self.num_agents), "batch agent"
                ),
                message=StringArraySpec(
                    (
                        *self.batch_size,
                        self.num_agents,
                        self.protocol_handler.num_message_channels,
                    ),
                    "batch agent channel",
                ),
                retry_count=IntArraySpec(
                    (
                        *self.batch_size,
                        self.num_agents,
                        self.protocol_handler.num_message_channels,
                    ),
                    "batch agent channel",
                ),
                token_limit=BoolArraySpec(
                    (
                        *self.batch_size,
                        self.num_agents,
                        self.protocol_handler.num_message_channels,
                    ),
                    "batch agent channel",
                ),
                shape=(*self.batch_size, self.num_agents),
                dim_names="batch agent",
            ),
            shape=self.batch_size,
            dim_names="batch",
        )

    @cached_property
    def state_spec(self) -> CompositeArraySpec:
        """The specification for the state keys."""
        return CompositeArraySpec(
            y=IntArraySpec(*self.batch_size, "batch"),
            shape=self.batch_size,
            dim_names="batch",
        )

    @cached_property
    def reward_spec(self) -> CompositeArraySpec:
        """The specification for the agent reward keys."""
        return CompositeArraySpec(
            agents=CompositeArraySpec(
                reward=FloatArraySpec(
                    (*self.batch_size, self.num_agents), "batch agent"
                ),
                shape=(*self.batch_size, self.num_agents),
                dim_names="batch agent",
            ),
            shape=self.batch_size,
            dim_names="batch",
        )

    @cached_property
    def done_spec(self) -> CompositeArraySpec:
        """The specification for the done keys (done and terminated)."""
        return CompositeArraySpec(
            done=BoolArraySpec(*self.batch_size, "batch"),
            terminated=BoolArraySpec(*self.batch_size, "batch"),
            shape=self.batch_size,
            dim_names="batch",
        )

    def step(self, env_state: NestedArrayDict) -> NestedArrayDict:
        """Take a step in the environment.

        Parameters
        ----------
        env_state : NestedArrayDict
            The current observation, state and done signal.

        Returns
        -------
        env_state : NestedArrayDict
            The input dict with a "next" sub-dict with the next observation, state,
            reward, and done signal.
        """

        next_state = NestedArrayDict({}, batch_size=self.batch_size)

        # Copy the (references to the) observations into the next state
        for key in chain(
            self.observation_spec.keys(recurse=True),
            self.state_spec.keys(recurse=True),
        ):
            if key not in [("round",), ("message_history",)]:
                next_state[key] = env_state[key]

        round = env_state["round"]
        next_state["round"] = round + 1

        # Add the latest messages to the message history
        message_history = env_state["message_history"].copy()
        for channel_id, channel_name in enumerate(
            self.protocol_handler.message_channel_names
        ):

            who_messaged = None
            for agent_id, agent_name in enumerate(self.protocol_handler.agent_names):
                message = env_state["agents", "message"][0, agent_id, channel_id]
                if message is not None:
                    if who_messaged is not None:
                        raise RuntimeError(
                            f"Agents {who_messaged} and {agent_name} both messaged on "
                            f"channel {channel_name}. "
                        )
                    who_messaged = agent_name
                    message_history[0, round, channel_id] = message

        next_state["message_history"] = message_history

        # Step the interaction protocol to obtain the next done and reward signals
        done, terminated, reward = self.protocol_handler.step_interaction_protocol(
            env_state
        )
        next_state["done"] = done.numpy()
        next_state["terminated"] = terminated.numpy()
        next_state["agents", "reward"] = reward.numpy()

        # Add the next state as a sub-dictionary
        env_state["next"] = next_state

        return env_state

    def reset(
        self,
        env_state: Optional[NestedArrayDict] = None,
        data_batch: Optional[NestedArrayDict] = None,
    ) -> NestedArrayDict:

        # If no state is provided, create a new one
        if env_state is None or "done" not in env_state.keys():
            env_state = self.zero()
            new_mask = np.ones(*self.batch_size, dtype=bool)

        else:
            new_mask = env_state["done"]
            env_state = env_state.clone()

        if data_batch is None:

            # If we don't have a data cycler yet, create one
            if self.data_cycler is None:
                dataloader = NestedArrayDictDataLoader(
                    self.dataset,
                    batch_size=self.batch_size[0],
                    shuffle=True,
                )
                self.data_cycler = VariableDataCycler(dataloader)

            # Sample a new batch of data for the episodes that are done
            data_batch = self.data_cycler.get_batch(new_mask.sum().item())

        # Reset the episodes that are done
        env_state = self._masked_reset(env_state, new_mask, data_batch)

        return env_state

    def zero(self) -> NestedArrayDict:
        """Return a zeroed environment state.

        Returns
        -------
        env_state : NestedArrayDict
            The zeroed environment state.
        """

        observation_zeros = self.observation_spec.zero()
        state_zeros = self.state_spec.zero()
        done_zeros = self.done_spec.zero()

        return observation_zeros.update(state_zeros).update(done_zeros)

    def get_next_state_from_state(self, state_env: NestedArrayDict) -> NestedArrayDict:
        """Get the next state environment from the current state environment.

        The current state environment should contain the "next" sub-dictionary, which
        contains the next observation, state, reward, and done signal.

        Parameters
        ----------
        state_env : NestedArrayDict
            The current state environment, which should contain the "next"
            sub-dictionary.

        Returns
        -------
        next_state_env : NestedArrayDict
            The next state environment.
        """

        next_state = NestedArrayDict({}, batch_size=self.batch_size)

        for key in chain(
            self.observation_spec.keys(recurse=True),
            self.state_spec.keys(recurse=True),
            self.done_spec.keys(recurse=True),
        ):
            next_state[key] = state_env["next", key]

        return next_state

    def add_dummy_actions_and_next_to_state(
        self, state_env: NestedArrayDict
    ) -> NestedArrayDict:
        """Complete a done state with dummy actions and dummy next state.

        This method adds dummy actions and copies the current state to the next state.

        It is used to complete the state when the environment is done.

        Parameters
        ----------
        state_env : NestedArrayDict
            The current state environment. This is modified in place.

        Returns
        -------
        state_env : NestedArrayDict
            The modified state environment.
        """

        state_env.update(self.action_spec.zero())

        next_state = NestedArrayDict({}, batch_size=self.batch_size)

        for key in chain(
            self.observation_spec.keys(recurse=True),
            self.state_spec.keys(recurse=True),
            self.done_spec.keys(recurse=True),
        ):
            next_state[key] = state_env[key]

        next_state.update(self.reward_spec.zero())

        state_env.update({"next": next_state})

        return state_env

    def _masked_reset(
        self,
        env_state: NestedArrayDict,
        mask: NDArray[np.bool_],
        data_batch: NestedArrayDict,
    ) -> NestedArrayDict:
        """Reset the environment for a subset of the episodes.

        Takes a new sample from the dataset and inserts it into the given episodes. Also
        resets the other elements of the episodes.

        Parameters
        ----------
        env_state : NestedArrayDict
            The current observation, state and done signal.
        mask : NDArray[np.bool_]
            A boolean mask of the episodes to reset.
        data_batch : NestedArrayDict
            The data batch to insert into the episodes.

        Returns
        -------
        env_state : NestedArrayDict
            The reset environment tensordict.
        """

        env_state["y"][mask] = data_batch["y"]
        env_state["message_history"][mask] = None
        env_state["round"][mask] = 0
        env_state["done"][mask] = False
        env_state["terminated"][mask] = False

        return env_state
