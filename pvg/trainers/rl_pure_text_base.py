"""Base classes for RL trainers for text-based environments which only use APIs."""

from abc import ABC, abstractmethod
from typing import Optional
from itertools import repeat
from multiprocessing import Pool

import torch

import numpy as np

from tqdm import tqdm

from pvg.parameters import Parameters
from pvg.factory import ScenarioInstance
from pvg.experiment_settings import ExperimentSettings
from pvg.scenario_base.data import NestedArrayDictDataLoader, NestedArrayDictDataset
from pvg.scenario_base.environment import PureTextEnvironment
from pvg.code_validation.agents import CodeValidationCombinedWholeAgent
from pvg.trainers.base import Trainer
from pvg.utils.data import VariableDataCycler, truncated_iterator
from pvg.utils.nested_array_dict import (
    NestedArrayDict,
    stack_nested_array_dicts,
    concatenate_nested_array_dicts,
)


class PureTextRlTrainer(Trainer, ABC):
    """Base class for RL trainers for text-based environments which only use APIs.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    scenario_instance : ScenarioInstance
        The components of the experiment.
    settings : ExperimentSettings
        The instance-specific settings of the experiment, like device, logging, etc.
    """

    def __init__(
        self,
        params: Parameters,
        scenario_instance: ScenarioInstance,
        settings: ExperimentSettings,
    ):
        super().__init__(params, scenario_instance, settings)

        # Create a random number generator
        self.rng = torch.Generator()

    @property
    def combined_agent(self) -> CodeValidationCombinedWholeAgent:
        """The agents combined into a single operator."""
        return self.scenario_instance.combined_whole

    def sample_rollouts(
        self,
        environment: PureTextEnvironment,
        use_tqdm: bool = False,
    ) -> NestedArrayDict:
        """Sample rollouts in the environment.

        Parameters
        ----------
        dataset : NestedArrayDictDataset
            The dataset of task instances.
        environment : PureTextEnvironment
            The environment to sample rollouts in.
        use_tqdm : bool
            Whether to create a tqdm progress bar for the rollouts.
        """

        dataloader = NestedArrayDictDataLoader(
            environment.dataset,
            batch_size=environment.batch_size[0],
            shuffle=True,
            generator=self.rng,
        )
        data_cycler = VariableDataCycler(
            dataloader, default_batch_size=environment.batch_size[0]
        )

        with Pool(self.settings.num_rollout_workers) as pool:
            arg_iterator = (
                (environment, self.max_message_rounds, self.combined_agent, data_batch)
                for data_batch in truncated_iterator(data_cycler, environment.num_envs)
            )

            rollout_iterator = pool.imap_unordered(_sample_single_rollout, arg_iterator)

            if use_tqdm:
                rollout_iterator = tqdm(rollout_iterator, total=environment.num_envs)

            rollout_list: list[NestedArrayDict] = list(rollout_iterator)

        return stack_nested_array_dicts(rollout_list, dim=0)


def _sample_single_rollout(
    args: tuple[
        PureTextEnvironment,
        int,
        CodeValidationCombinedWholeAgent,
        Optional[NestedArrayDict],
    ]
) -> NestedArrayDict:
    """Sample a single rollout in an the environment.

    This function steps the environment until it is done and then pads the rollout
    with zero states up to the maximum number of message rounds.

    This function is intended to be applied by a pool of workers. As such it lives in
    the module scope and takes all trainer attributes required as arguments.

    Parameters
    ----------
    environment : PureTextEnvironment
        The environment to sample a rollout in.
    data_batch : NestedArrayDict, optional
        The data batch to use for the rollout. If None, the data batch will be
        sampled from the dataset.
    pbar : tqdm, optional
        The progress bar to update, if any.

    Returns
    -------
    NestedArrayDict
        The rollout in the environment. Has batch size (max_message_rounds, )
    """

    environment, max_message_rounds, combined_agent, data_batch = args

    done = False
    env_state = environment.reset(data_batch=data_batch)
    env_states = []

    for _ in range(max_message_rounds):
        if not done:

            # Run the forward pass on all agents to sample actions
            env_state = combined_agent.forward(env_state)

            # Step the environment to get the next state. This writes the next state
            # in the "next" sub-dictionary.
            env_state = environment.step(env_state)

            # Check if the environment is done. The state has batch size 1, so we
            # only need to check the first element.
            done = env_state["next", "done"][0]

            # Append the current state to the environment states
            env_states.append(env_state)

            # Update the current state to the next state
            env_state = environment.get_next_state_from_state(env_state)

        # If we are done, we need to pad the rollout with zero actions
        else:
            if "next" not in env_state.keys():
                env_state = environment.add_dummy_actions_and_next_to_state(env_state)
            env_states.append(env_state)

    return concatenate_nested_array_dicts(env_states, dim=0)
