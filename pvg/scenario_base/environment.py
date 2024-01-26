"""Base class for the RL environment."""

from abc import ABC, abstractmethod
from typing import Optional

import torch

from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.data.tensor_specs import (
    CompositeSpec,
    BinaryDiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs import EnvBase

from pvg.scenario_base import DataLoader, Dataset
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
    device : TorchDevice
        The device on which to run the environment.
    dataset : Dataset
        The dataset for the environment.
    """

    _int_dtype: torch.dtype = (torch.int,)

    def __init__(
        self,
        params: Parameters,
        settings: ExperimentSettings,
        dataset: Dataset,
        train: bool = True,
    ):
        super().__init__(device=settings.device)
        self.params = params
        self.settings = settings
        self.train = train

        self.agent_names = list(params.agents.keys())

        # Create a random number generator
        self.rng = torch.Generator()

        # Load the dataset
        self.dataset = dataset
        self.data_cycler: Optional[VariableDataCycler] = None

        # The number of environments is the number of episodes we can fit in a batch
        self.num_envs = params.ppo.frames_per_batch // params.max_message_rounds
        self.batch_size = (self.num_envs,)

        # Create environment specs
        self.observation_spec = self._get_observation_spec()
        self.action_spec = self._get_action_spec()
        self.state_spec = self._get_state_spec()
        self.reward_spec = self._get_reward_spec()
        self.done_spec = self._get_done_spec()

    @abstractmethod
    def _get_observation_spec(self) -> TensorSpec:
        """Get the specification of the agent observations.

        Returns
        -------
        observation_spec : TensorSpec
            The observation specification.
        """
        pass

    @abstractmethod
    def _get_action_spec(self) -> TensorSpec:
        """Get the specification of the agent actions.

        Returns
        -------
        action_spec : TensorSpec
            The action specification.
        """
        pass

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
            ),
            shape=(self.num_envs,),
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
                reward=UnboundedContinuousTensorSpec(shape=(self.num_envs, 2)),
                shape=(self.num_envs,),
            ),
            shape=(self.num_envs,),
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
                self.num_envs, shape=(self.num_envs,), dtype=torch.bool
            ),
            shape=(self.num_envs,),
        )

    @abstractmethod
    def _step(self, env_td: TensorDictBase) -> TensorDictBase:
        """Perform a step in the environment.

        Parameters
        ----------
        env_td : TensorDictBase
            The current observation and state.

        Returns
        -------
        next : TensorDictBase
            The next observation, state, reward, and done signal.
        """
        pass

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
        if env_td is None:
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
            )
            self.data_cycler = VariableDataCycler(dataloader)

        # Sample a new batch of data for the episodes that are done
        batch = self.data_cycler.get_batch(new_mask.sum().item()).to(self.device)

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
