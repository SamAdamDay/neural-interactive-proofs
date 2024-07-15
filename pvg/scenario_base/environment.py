"""Base class for the RL environment."""

from abc import ABC, abstractmethod
from typing import Optional
from operator import mul
from functools import reduce

import torch
from torch import Tensor
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

import einops

from jaxtyping import Float, Int

from pvg.scenario_base import DataLoader, Dataset
from pvg.protocols import ProtocolHandler
from pvg.parameters import Parameters
from pvg.experiment_settings import ExperimentSettings
from pvg.utils.data import VariableDataCycler
from pvg.utils.maths import manual_seed


class Environment(EnvBase, ABC):
    """The base class for all Prover-Verifier RL environments.

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
    dataset : Dataset
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
        dataset: Dataset,
        protocol_handler: ProtocolHandler,
        *,
        train: bool = True,
    ):
        super().__init__(device=settings.device)
        self.params = params
        self.settings = settings
        self.protocol_handler = protocol_handler
        self.train = train

        self.num_agents = len(protocol_handler.agent_names)

        # Create a random number generator
        self.rng = torch.Generator()

        # Load the dataset
        self.dataset = dataset
        self.data_cycler: Optional[VariableDataCycler] = None

        # The number of environments is the number of episodes we can fit in a batch
        if params.rl.steps_per_env_per_iteration is not None:
            steps_per_env_per_iteration = params.rl.steps_per_env_per_iteration
            if params.rl.frames_per_batch % steps_per_env_per_iteration != 0:
                raise ValueError(
                    f"The parameter `rl.steps_per_env_per_iteration` must divide "
                    f"`rl.frames_per_batch` without remainder, but got "
                    f"{steps_per_env_per_iteration} and {params.rl.frames_per_batch} "
                )
        else:
            steps_per_env_per_iteration = self.protocol_handler.max_message_rounds
            if params.rl.frames_per_batch % steps_per_env_per_iteration != 0:
                raise ValueError(
                    f"The maximum number of message rounds must divide "
                    f"`rl.frames_per_batch` without remainder, but got "
                    f"{steps_per_env_per_iteration} and {params.rl.frames_per_batch} "
                )
        self.num_envs = params.rl.frames_per_batch // steps_per_env_per_iteration
        self.batch_size = (self.num_envs,)

        # Create environment specs
        self.observation_spec = self._get_observation_spec()
        self.action_spec = self._get_action_spec()
        self.state_spec = self._get_state_spec()
        self.reward_spec = self._get_reward_spec()
        self.done_spec = self._get_done_spec()

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
    def round_last_in_main_message_history(self) -> bool:
        """Whether the round dim comes last in the main message history

        For image datasets it makes sense to have the round come before the height and
        width dims because it acts as a channel.
        """
        return True

    @property
    def message_history_shape(self) -> tuple:
        """The shape of the message history and 'x' tensors

        This is used to make the specification for these.

        Returns
        -------
        message_history_shape: tuple
            The common shape of the message history and 'x' tensors.
        """

        if self.round_last_in_main_message_history:
            return (
                self.num_envs,
                *self.main_message_space_shape,
                self.protocol_handler.max_message_rounds,
            )
        else:
            return (
                self.num_envs,
                self.protocol_handler.max_message_rounds,
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


        Subclasses should call this method and add at least:

        - `x`: The message history.
        - `message`: The next message.

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
                shape=(self.num_envs,),
                device=self.device,
            ),
            shape=(self.num_envs,),
            device=self.device,
        )

        if self.params.include_linear_message_space:
            action_spec["agents"]["linear_message_selected"] = DiscreteTensorSpec(
                self.params.d_linear_message_space,
                shape=(self.num_envs, self.num_agents),
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
                    device=self.device,  # TODO Ask Sam about this
                ),
                shape=(self.num_envs,),
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
            round_last_in_message_history=self.round_last_in_main_message_history,
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
                round_last_in_message_history=False,
            )

        # Clone the message history to the 'x' feature tensor
        next_td["x"] = next_td["message_history"].clone()

        # Compute the done signal and reward
        done, reward = self.protocol_handler.step_interaction_protocol(env_td)
        next_td.set("done", done)
        next_td.set("terminated", done)
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
        round_last_in_message_history: bool = True,
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
        round_last_in_message_history : bool, default=True
            Whether the round dim comes last in the message history shape.

        Returns
        -------
        next_td : TensorDictBase
            The updated 'next' tensordict.
        """

        # Get a string representation of the message space dims, for type annotation.
        # dim_1 dim_2 etc.
        message_shape_str = " ".join(f"dim_{i}" for i in range(len(message_shape)))

        message_history: (
            Float[Tensor, f"... round {message_shape_str}"]
            | Float[Tensor, f"... {message_shape_str} round"]
        ) = env_td.get(message_history_key)
        round: Int[Tensor, "..."] = env_td.get("round")
        message_selected: Int[Tensor, "... agent"] = env_td.get(
            ("agents", message_out_key)
        )

        # Compute index of the agent(s) whose turn it is.
        # (... agent)
        active_agents_mask = self.protocol_handler.get_active_agents_mask(round)

        # Sum up the messages from the agents whose turn it is. If two agents select the
        # same message number, the message will be 2.
        # (... {message_shape_str})
        message = F.one_hot(message_selected, reduce(mul, message_shape)).float()
        message = torch.where(active_agents_mask[..., None], message, 0)
        message = einops.reduce(
            message,
            f"... agent ({message_shape_str}) -> ... {message_shape_str}",
            reduction="sum",
            **{f"dim_{i}": dim for i, dim in enumerate(message_shape)},
        )

        # Get a mask for which round it is
        round_mask = F.one_hot(round, self.protocol_handler.max_message_rounds).bool()

        # Reshape it so that it looks like the message history with 1's for the message
        # space dims
        message_shape_ones = " ".join(["1"] * len(message_shape))
        if round_last_in_message_history:
            round_mask_shape = f"{message_shape_ones} round"
        else:
            round_mask_shape = f"round {message_shape_ones}"
        round_mask = einops.rearrange(
            round_mask, f"... round -> ... {round_mask_shape}"
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
        if env_td is None or not "done" in env_td.keys():
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
            dataloader = DataLoader(
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
        self.rng = manual_seed(seed)
