"""Base class for the RL environment."""

from abc import ABC, abstractmethod
from typing import Optional, Any, Literal
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
from pvg.parameters import HyperParameters
from pvg.experiment_settings import ExperimentSettings
from pvg.utils.data import VariableDataCycler
from pvg.utils.types import NumpyStringDtype, String
from pvg.utils.nested_array_dict import (
    NestedArrayDict,
    NumpySpec,
    CompositeSpec as CompositeArraySpec,
    IntArraySpec,
    StringArraySpec,
    FloatArraySpec,
    BoolArraySpec,
)
from pvg.utils.future import TypedDict, NotRequired


class Environment(ABC):
    """The base class for all Prover-Verifier RL environments.

    Parameters
    ----------
    hyper_params : HyperParameters
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
        if self.hyper_params.rl.steps_per_env_per_iteration is not None:
            steps_per_env_per_iteration = (
                self.hyper_params.rl.steps_per_env_per_iteration
            )
            if (
                self.hyper_params.rl.frames_per_batch is not None
                and self.hyper_params.rl.frames_per_batch % steps_per_env_per_iteration
                != 0
            ):
                raise ValueError(
                    f"The parameter `rl.steps_per_env_per_iteration` must divide "
                    f"`rl.frames_per_batch` without remainder, but got "
                    f"{steps_per_env_per_iteration} and "
                    f"{self.hyper_params.rl.frames_per_batch}."
                )
        else:
            steps_per_env_per_iteration = self.protocol_handler.max_message_rounds
            if (
                self.hyper_params.rl.frames_per_batch is not None
                and self.hyper_params.rl.frames_per_batch % steps_per_env_per_iteration
                != 0
            ):
                raise ValueError(
                    f"The maximum number of message rounds must divide "
                    f"`rl.frames_per_batch` without remainder, but got "
                    f"{steps_per_env_per_iteration} and "
                    f"{self.hyper_params.rl.frames_per_batch}."
                )
        return steps_per_env_per_iteration

    @property
    def frames_per_batch(self) -> int:
        """The number of frames to sample per training iteration.

        This can be set directly with `rl.frames_per_batch`, or it can be determined by
        `rl.rollouts_per_iteration` and `steps_per_env_per_iteration`.
        """
        if self.hyper_params.rl.frames_per_batch is not None:
            return self.hyper_params.rl.frames_per_batch
        else:
            if self.hyper_params.rl.rollouts_per_iteration is not None:
                return (
                    self.hyper_params.rl.rollouts_per_iteration
                    * self.steps_per_env_per_iteration
                )
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
        hyper_params: HyperParameters,
        settings: ExperimentSettings,
        dataset: Dataset,
        protocol_handler: ProtocolHandler,
        *,
        train: bool = True,
    ):
        self.hyper_params = hyper_params
        self.settings = settings
        self.protocol_handler = protocol_handler
        self.train = train
        self.dataset = dataset

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
    hyper_params : HyperParameters
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

    dataset: TensorDictDataset

    _int_dtype: torch.dtype = torch.int

    def __init__(
        self,
        hyper_params: HyperParameters,
        settings: ExperimentSettings,
        dataset: TensorDictDataset,
        protocol_handler: ProtocolHandler,
        *,
        train: bool = True,
    ):
        super().__init__(device=settings.device)

        # Call the `EnvBase` constructor
        super(nn.Module, self).__init__(
            hyper_params=hyper_params,
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
        """The shape of the main message space used by the agents to communicate.

        This is typically informed by the type of data in the environment. E.g. for
        images this might have shape "height width"
        """

    @property
    @abstractmethod
    def main_message_out_key(self) -> str:
        """The tensordict key which contains the main message sent by each agent.

        This key is the main output of an agent. Typically, the agent's forward pass
        will output logits for this key, which will then be selected from.
        """

    @property
    def message_history_shape(self) -> tuple:
        """The shape of the message history and 'x' tensors.

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
            self.hyper_params.message_size,
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
        - `seed`: A shared seed for the environment.
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
            seed=DiscreteTensorSpec(
                2**16,
                shape=(self.num_envs,),
                dtype=torch.long,
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
        if self.hyper_params.include_linear_message_space:
            observation_spec["linear_message_history"] = BinaryDiscreteTensorSpec(
                self.hyper_params.d_linear_message_space,
                shape=(
                    self.num_envs,
                    self.protocol_handler.max_message_rounds,
                    self.protocol_handler.num_message_channels,
                    self.hyper_params.message_size,
                    self.hyper_params.d_linear_message_space,
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
                        self.hyper_params.message_size,
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

        if self.hyper_params.include_linear_message_space:
            action_spec["agents"]["linear_message_selected"] = DiscreteTensorSpec(
                self.hyper_params.d_linear_message_space,
                shape=(
                    self.num_envs,
                    self.num_agents,
                    self.protocol_handler.num_message_channels,
                    self.hyper_params.message_size,
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
            datapoint_id=DiscreteTensorSpec(
                len(self.dataset),
                shape=(self.num_envs,),
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

        We have both shared and agent-specific done signals. This is for convenience,
        where the shared done signal indicates that all relevant agents are done and so
        the environment should be reset.

        Returns
        -------
        done_spec : TensorSpec
            The done specification.
        """
        return CompositeSpec(
            # TODO: This leads to issues because TorchRL calls `any` on the done signal
            agents=CompositeSpec(
                done=BinaryDiscreteTensorSpec(
                    self.num_agents,
                    shape=(self.num_envs, self.num_agents),
                    dtype=torch.bool,
                    device=self.device,
                ),
                shape=(self.num_envs, self.num_agents),
                device=self.device,
            ),
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
        if self.hyper_params.include_linear_message_space:
            next_td = self._compute_message_history_and_next_message(
                env_td,
                next_td,
                message_out_key="linear_message_selected",
                message_in_key="linear_message",
                message_shape=(self.hyper_params.d_linear_message_space,),
                message_history_key="linear_message_history",
            )

        # Clone the message history to the 'x' feature tensor
        next_td["x"] = next_td["message_history"].clone()

        # Compute the done signal and reward
        shared_done, agent_done, terminated, reward = (
            self.protocol_handler.step_interaction_protocol(env_td)
        )
        shared_done = shared_done | terminated  # TODO: Improve handling of terminated
        next_td.set("done", shared_done)
        next_td.set(("agents", "done"), agent_done)
        next_td.set("terminated", terminated)
        next_td.set(("agents", "reward"), reward)
        next_td.set(
            "decision_restriction", torch.zeros_like(shared_done, dtype=self._int_dtype)
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
        """Compute the new message history and next message for given keys.

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
        round_id: Int[Tensor, "..."] = env_td.get("round")
        seed: Int[Tensor, "..."] = env_td.get("seed")
        message_selected: Int[Tensor, "... agent channel position"] = env_td.get(
            ("agents", message_out_key)
        )

        # Get the mask for the active agents per channel in the current round
        # (... agent channel)
        active_agents_mask = (
            self.protocol_handler.get_active_agents_mask_from_rounds_and_seed(
                round_id, seed
            )
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
        round_mask = F.one_hot(
            round_id, self.protocol_handler.max_message_rounds
        ).bool()

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
        """Reset the environment (partially).

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

        env_td["seed"][mask] = torch.randint(
            0, self.observation_spec["seed"].n, (mask.sum().item(),), device=self.device
        )
        env_td["y"][mask] = data_batch["y"].unsqueeze(-1)
        env_td["datapoint_id"][mask] = data_batch["id"]
        env_td["message_history"][mask] = torch.zeros_like(
            env_td["message_history"][mask]
        )
        env_td["x"][mask] = torch.zeros_like(env_td["x"][mask])
        env_td["message"][mask] = 0
        env_td["round"][mask] = 0
        env_td["done"][mask] = False
        env_td["agents", "done"][mask] = False
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


class PromptMessage(TypedDict):
    """A message in the prompt for a language model API.

    The prompt is a list of messages, where each message is a dictionary with keys as
    follows.

    Attributes
    ----------
    role : Literal["system", "assistant", "user"]
        The role of the message sender.
    content : str
        The content of the message.
    name : str, optional
        The name of the message sender.
    """

    role: Literal["system", "assistant", "user"]
    content: str
    name: NotRequired[str]


class PureTextEnvironment(Environment, ABC):
    """Base for environments which handle non-tokenised text with nested array dicts."""

    dataset: NestedArrayDictDataset

    @property
    def max_prompt_messages(self) -> int:
        """The maximum number messages which can be sent in a prompt to an agent.

        The prompt for the agent is constructed from the message history, but may be longer
        than the number of rounds because messages can be split by channel, and system
        messages can be included.

        This gives a rough upper bound on the number of messages which can be sent in a
        prompt. Hopefully this is enough to cover all cases.
        """

        return 2 * (
            self.protocol_handler.max_message_rounds
            * self.protocol_handler.num_message_channels
            + 10
        )

    @property
    @abstractmethod
    def observation_spec(self) -> CompositeArraySpec:
        """The specification for the observation keys."""
        return CompositeArraySpec(
            round=IntArraySpec(*self.batch_size, "batch"),
            seed=IntArraySpec(*self.batch_size, "batch"),
            message_history=StringArraySpec(
                (
                    *self.batch_size,
                    self.protocol_handler.max_message_rounds,
                    self.protocol_handler.num_message_channels,
                ),
                "batch round channel",
            ),
            message_agent_id=IntArraySpec(
                (
                    *self.batch_size,
                    self.protocol_handler.max_message_rounds,
                    self.protocol_handler.num_message_channels,
                ),
                "batch round channel",
            ),
            raw_message_history=StringArraySpec(
                (
                    *self.batch_size,
                    self.protocol_handler.max_message_rounds,
                    self.protocol_handler.num_agents,
                ),
                "batch round agent",
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
                raw_message=StringArraySpec(
                    (
                        *self.batch_size,
                        self.num_agents,
                    ),
                    "batch agent",
                ),
                prompt=StringArraySpec(
                    (
                        *self.batch_size,
                        self.protocol_handler.num_agents,
                        self.max_prompt_messages,
                        len(PromptMessage.__annotations__),
                    ),
                    "batch agent message field",
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
            datapoint_id=IntArraySpec(*self.batch_size, "batch"),
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
        """The specification for the done keys (done and terminated).

        We have both shared and agent-specific done signals. This is for convenience,
        where the shared done signal indicates that all relevant agents are done and so
        the environment should be reset.
        """
        return CompositeArraySpec(
            agents=CompositeArraySpec(
                done=BoolArraySpec((*self.batch_size, self.num_agents), "batch agent"),
                shape=(*self.batch_size, self.num_agents),
                dim_names="batch agent",
            ),
            done=BoolArraySpec(*self.batch_size, "batch"),
            terminated=BoolArraySpec(*self.batch_size, "batch"),
            padding=BoolArraySpec(*self.batch_size, "batch"),
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

        round_id = env_state["round"]
        next_state["round"] = round_id + 1

        # Add the latest messages to the message history
        message_history = env_state["message_history"].copy()
        message_agent_id = env_state["message_agent_id"].copy()
        for channel_id, channel_name in enumerate(
            self.protocol_handler.message_channel_names
        ):

            who_messaged = None
            for agent_id, agent_name in enumerate(self.protocol_handler.agent_names):
                message = env_state["agents", "message"][0, agent_id, channel_id]
                if message is not None:
                    if who_messaged is not None:
                        raise RuntimeError(
                            f"Agents {who_messaged!r} and {agent_name!r} both messaged "
                            f"on channel {channel_name!r}. "
                        )
                    who_messaged = agent_name
                    message_history[0, round_id.item(), channel_id] = message
                    message_agent_id[0, round_id.item(), channel_id] = agent_id

        next_state["message_history"] = message_history
        next_state["message_agent_id"] = message_agent_id

        # Add the raw messages to the raw message history
        raw_message_history = env_state["raw_message_history"].copy()
        raw_message_history[0, round_id.item()] = env_state["agents", "raw_message"][0]
        next_state["raw_message_history"] = raw_message_history

        # Step the interaction protocol to obtain the next done and reward signals
        shared_done, agent_done, terminated, reward = (
            self.protocol_handler.step_interaction_protocol(env_state)
        )
        next_state["done"] = shared_done.numpy()
        next_state["agents", "done"] = agent_done.numpy()
        next_state["terminated"] = terminated.numpy()
        next_state["padding"] = np.zeros(*self.batch_size, dtype=bool)
        next_state["agents", "reward"] = reward.numpy()

        # Add the next state as a sub-dictionary
        env_state["next"] = next_state

        return env_state

    def reset(
        self,
        env_state: Optional[NestedArrayDict] = None,
        data_batch: Optional[NestedArrayDict] = None,
    ) -> NestedArrayDict:
        """Reset the pure text environment.

        This method resets the environment for the episodes which are done. It samples a
        new batch of data for these episodes and calls `_masked_reset` to reset the
        episodes.

        Parameters
        ----------
        env_state : Optional[NestedArrayDict]
            The current environment state.
        data_batch : Optional[NestedArrayDict]
            The data batch to use for the episodes that are done.

        Returns
        -------
        env_state : NestedArrayDict
            The reset environment state.
        """

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
            next_state[key] = state_env[("next", *key)]

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

    @abstractmethod
    def get_datapoint_from_env_state_as_dict(self, env_state: NestedArrayDict) -> dict:
        """Get the datapoint from a single-element environment state as a dictionary.

        This returns a dictionary which specifies the datapoint for the environment
        state.

        This method should be extended by base classes to include whatever additional
        fields consistute the datapoint.

        Parameters
        ----------
        env_state : NestedArrayDict
            The environment state.

        Returns
        -------
        datapoint : dict
            The datapoint as a dictionary.
        """

        return dict(y=int(env_state["y"]))

    def prompt_list_to_array(
        self, prompt_list: list[PromptMessage]
    ) -> String[NDArray, "message field"]:
        """Convert a prompt in the form of a list of dictionaries to a numpy array.

        Each element of the list is a dictionary with keys defined in `PromptMessage`.
        We convert this to a numpy array with columns corresponding to the keys in
        `PromptMessage`.

        Parameters
        ----------
        prompt_list : list[PromptMessage]
            The list of prompts to convert.
        """

        required_keys = sorted(PromptMessage.__required_keys__)
        optional_keys = sorted(PromptMessage.__optional_keys__)

        prompt_array = np.full(
            (
                self.max_prompt_messages,
                len(required_keys) + len(optional_keys),
            ),
            None,
            dtype=NumpyStringDtype,
        )

        for i, prompt in enumerate(prompt_list):
            for j, key in enumerate(required_keys):
                prompt_array[i, j] = prompt[key]
            for j, key in enumerate(optional_keys):
                if key in prompt:
                    prompt_array[i, j + len(required_keys)] = prompt[key]

        return prompt_array

    def prompt_array_to_list(
        self, prompt_array: String[NDArray, "message field"]
    ) -> list[PromptMessage]:
        """Convert a prompt in the form of a numpy array to a list of dictionaries.

        Each row of the numpy array corresponds to a message in the prompt, and each
        column corresponds to a field of the message.

        The prompt array has a fixed number of rows, but the prompt may be shorter. If
        any required field is None in a row, we take that to indicate the end of the
        prompt.

        Parameters
        ----------
        prompt_array : String[NDArray, "message field"]
            The numpy array to convert.

        Returns
        -------
        prompt_list : list[PromptMessage]
            The list of prompts.
        """

        required_keys = sorted(PromptMessage.__required_keys__)
        optional_keys = sorted(PromptMessage.__optional_keys__)

        prompt_list = []
        for row in prompt_array:
            prompt = {}

            any_none = False
            for key, value in zip(required_keys, row[: len(required_keys)]):
                prompt[key] = value
                if value is None:
                    any_none = True
                    break

            # If any of the required keys are None, we have reached the end of the
            # prompt messages
            if any_none:
                break

            for key, value in zip(optional_keys, row[len(required_keys) :]):
                if value is not None:
                    prompt[key] = value

            prompt_list.append(prompt)

        return prompt_list

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
        env_state["datapoint_id"][mask] = data_batch["id"]
        env_state["seed"][mask] = np.random.randint(0, 2**16, mask.sum())
        env_state["message_history"][mask] = None
        env_state["message_agent_id"][mask] = -1
        env_state["raw_message_history"][mask] = None
        env_state["round"][mask] = 0
        env_state["done"][mask] = False
        env_state["agents", "done"][mask] = False
        env_state["terminated"][mask] = False
        env_state["padding"][mask] = False

        return env_state
