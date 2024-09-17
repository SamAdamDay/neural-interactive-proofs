"""The RL environment for text classification."""

from typing import Optional, ClassVar
from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
from numpy.typing import NDArray, DTypeLike

from pvg.scenario_base import Environment, DataLoader, Dataset
from pvg.protocols import ProtocolHandler
from pvg.parameters import Parameters
from pvg.experiment_settings import ExperimentSettings
from pvg.utils.data import VariableDataCycler
from pvg.utils.nested_array_dict import (
    NestedArrayDict,
    CompositeSpec,
    IntArraySpec,
    StringArraySpec,
    FloatArraySpec,
    BoolArraySpec,
)


class TextClassificationEnvironment(Environment):
    """The RL environment for text classification."""

    def __init__(
        self,
        params: Parameters,
        settings: ExperimentSettings,
        dataset: Dataset,
        protocol_handler: ProtocolHandler,
        *,
        train: bool = True,
    ):
        super().__init__(params, settings, dataset, protocol_handler, train=train)

    @cached_property
    def observation_spec(self) -> CompositeSpec:
        """The specification for the observation keys."""
        return CompositeSpec(
            round=IntArraySpec(self.num_envs, "batch"),
            problem=StringArraySpec(self.num_envs, "batch"),
            solution=StringArraySpec(self.num_envs, "batch"),
            message_history=StringArraySpec(
                (
                    self.num_envs,
                    self.protocol_handler.max_message_rounds,
                    self.protocol_handler.num_message_channels,
                ),
                "batch round channel",
            ),
            shape=self.batch_size,
            dim_names="batch",
        )

    @cached_property
    def action_spec(self) -> CompositeSpec:
        """The specification for the action keys."""
        return CompositeSpec(
            agents=CompositeSpec(
                decision=IntArraySpec(
                    (self.batch_size, self.num_agents), "batch agent"
                ),
                message=StringArraySpec(
                    (
                        self.batch_size,
                        self.num_agents,
                        self.protocol_handler.num_message_channels,
                    ),
                    "batch agent channel",
                ),
                shape=(self.batch_size, self.num_agents),
                dim_names="batch agent",
            ),
            shape=self.batch_size,
            dim_names="batch",
        )

    @cached_property
    def state_spec(self) -> CompositeSpec:
        """The specification for the state keys."""
        return CompositeSpec(
            y=IntArraySpec(self.num_envs, "batch"),
            shape=self.batch_size,
            dim_names="batch",
        )

    @cached_property
    def reward_spec(self) -> CompositeSpec:
        """The specification for the agent reward keys."""
        return CompositeSpec(
            agents=CompositeSpec(
                reward=FloatArraySpec(
                    (self.batch_size, self.num_agents), "batch agent"
                ),
                shape=(self.batch_size, self.num_agents),
                dim_names="batch agent",
            ),
            shape=self.batch_size,
            dim_names="batch",
        )

    @cached_property
    def done_spec(self) -> CompositeSpec:
        """The specification for the done keys (done and terminated)."""
        return CompositeSpec(
            done=BoolArraySpec(self.num_envs, "batch"),
            terminated=BoolArraySpec(self.num_envs, "batch"),
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
        next_state : NestedArrayDict
            The next observation, state, reward, and done signal.
        """

        next_state = NestedArrayDict(
            dict(
                round=env_state["round"] + 1,
            ),
            batch_size=self.batch_size,
        )

    def reset(self, env_state: Optional[NestedArrayDict] = None) -> NestedArrayDict:

        # If no state is provided, create a new one
        if env_state is None or not "done" in env_state.keys():
            observation_zeros = self.observation_spec.zero()
            state_zeros = self.state_spec.zero()
            done_zeros = self.done_spec.zero()
            env_state = observation_zeros.update(state_zeros).update(done_zeros)
            new_mask = np.ones(*self.batch_size, dtype=bool)

        else:
            new_mask = env_state["done"]
            env_state = env_state.clone()

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
        batch = self.data_cycler.get_batch(new_mask.sum().item())

        # Reset the episodes that are done
        env_state = self._masked_reset(env_state, new_mask, batch)

        return env_state

    def _masked_reset(
        self,
        env_state: NestedArrayDict,
        mask: NDArray[np.bool_],
        data_batch: NDArray[np.str_],
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
        data_batch : NDArray[np.str_]
            The data batch to insert into the episodes.

        Returns
        -------
        env_state : NestedArrayDict
            The reset environment tensordict.
        """

        env_state["y"][mask] = data_batch["y"].unsqueeze(-1)
        env_state["message_history"][mask] = np.zeros_like(
            env_state["message_history"][mask]
        )
        env_state["round"][mask] = 0
        env_state["done"][mask] = False
        env_state["terminated"][mask] = False
