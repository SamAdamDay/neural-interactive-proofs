"""Base class for the RL environment."""

from abc import ABC, abstractmethod
from typing import Optional

import torch

from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.data.tensor_specs import (
    CompositeSpec,
    BinaryDiscreteTensorSpec,
    DiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs import EnvBase

from pvg.scenario_base import DataLoader, Dataset
from pvg.protocols import ProtocolHandler
from pvg.parameters import Parameters
from pvg.experiment_settings import ExperimentSettings
from pvg.utils.data import VariableDataCycler


class Environment(EnvBase, ABC):
    """The base class for all Prover-Verifier RL environments.

    To implement a new environment, subclass this class and implement the following
    attribute and methods:

    - `_get_observation_spec`: The specification of the agent observations.
    - `_get_action_spec`: The specification of the agent actions.
    - `_get_state_spec` (optional): The specification of the states space.
    - `_get_reward_spec` (optional): The specification of the agent rewards.
    - `_get_done_spec` (optional): The specification of the agent done signals.
    - `_step`: Perform a step in the environment.
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
    def _message_history_shape(self) -> tuple:
        """Get the shape of the message history and 'x' tensors

        This is used to make the specification for these.

        Returns
        -------
        message_history_shape: tuple
            The common shape of the message history and 'x' tensors.
        """

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


        Subclasses should call this method and add at least:

        - `x`: The message history.
        - `message`: The next message.

        Returns
        -------
        observation_spec : TensorSpec
            The observation specification.
        """
        return CompositeSpec(
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
                shape=self._message_history_shape,
                dtype=torch.float,
                device=self.device,
            ),
            message_history=BinaryDiscreteTensorSpec(
                self._message_history_shape[-1],
                shape=self._message_history_shape,
                dtype=torch.float,
                device=self.device,
            ),
            shape=(self.num_envs,),
            device=self.device,
        )

    @abstractmethod
    def _get_action_spec(self) -> TensorSpec:
        """Get the specification of the agent actions.

        Subclasses should call this method and add any additional action spaces.

        Returns
        -------
        action_spec : TensorSpec
            The action specification.
        """
        return CompositeSpec(
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
                "round",
                "decision_restriction",
            ]:
                next_td[key] = env_td[key]

        # Compute the message history and next message
        next_td = self._compute_message_history(env_td, next_td)

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

    @abstractmethod
    def _compute_message_history(
        self,
        env_td: TensorDictBase,
        next_td: TensorDictBase,
    ) -> TensorDictBase:
        """Compute the new message history and next message.

        Used in the `_step` method of the environment.

        Parameters
        ----------
        env_td : TensorDictBase
            The current observation and state.
        next_td : TensorDictBase
            The 'next' tensordict, to be updated with the message history and next
            message.

        Returns
        -------
        next_td : TensorDictBase
            The updated 'next' tensordict.
        """

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
                pin_memory_device=str(self.device) if self.settings.pin_memory else "",
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
        pass

    def _set_seed(self, seed: int | None):
        self.rng = torch.manual_seed(seed)
