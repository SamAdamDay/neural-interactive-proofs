"""Base classes for RL trainers for text-based environments which only use APIs."""

from abc import ABC, abstractmethod
from typing import Optional, Literal, Iterable
from itertools import repeat
from multiprocessing import Pool
from functools import cached_property
from pathlib import Path
import pickle
from dataclasses import dataclass
import json

import yaml

import torch

import numpy as np

from wandb import Artifact
import wandb

from tqdm import tqdm
import wandb.errors

from pvg.parameters import Parameters
from pvg.factory import ScenarioInstance
from pvg.experiment_settings import ExperimentSettings
from pvg.scenario_base.data import NestedArrayDictDataLoader, NestedArrayDictDataset
from pvg.scenario_base.environment import PureTextEnvironment
from pvg.scenario_base.agents import PureTextWholeAgent
from pvg.trainers.trainer_base import Trainer
from pvg.utils.data import VariableDataCycler, truncated_iterator
from pvg.utils.nested_array_dict import (
    NestedArrayDict,
    stack_nested_array_dicts,
    concatenate_nested_array_dicts,
)
from pvg.constants import (
    ROLLOUTS_ARTIFACT_PREFIX,
    ROLLOUTS_ARTIFACT_TYPE,
    RAW_TRANSCRIPT_ARTIFACT_PREFIX,
    RAW_TRANSCRIPT_ARTIFACT_TYPE,
    PROCESSED_TRANSCRIPT_ARTIFACT_PREFIX,
    PROCESSED_TRANSCRIPT_ARTIFACT_TYPE,
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

    @dataclass
    class State(Trainer.State):
        pass

    _state: State

    @property
    def state(self) -> State:

        if not hasattr(self, "_state"):
            self._state = self.State()

        # Get the state of the agents to fill out the `agents` field
        for agent_name, agent_whole in self.scenario_instance.agents.items():
            self._state.agents[agent_name] = agent_whole.get_state()

        return self._state

    @state.setter
    def state(self, state: State):
        self._state = state

        for agent_name, agent in self.scenario_instance.agents.items():
            if agent_name not in state.agents:
                raise ValueError(f"Agent {agent_name!r} not found in state.")
            agent.set_state(state.agents[agent_name])

    @property
    def train_environment(self) -> PureTextEnvironment:
        """The training environment."""
        return self.scenario_instance.train_environment

    @property
    def test_environment(self) -> PureTextEnvironment:
        """The test environment."""
        return self.scenario_instance.test_environment

    @cached_property
    def agent_wholes(self) -> dict[str, PureTextWholeAgent]:
        """The 'whole' part of each agent.

        Agents are not split into parts, so an agent consists of only a 'whole' part.
        """
        return {
            agent_name: agent.whole
            for agent_name, agent in self.scenario_instance.agents.items()
        }

    @property
    def combined_agent(self) -> PureTextWholeAgent:
        """The agents combined into a single operator."""
        return self.scenario_instance.combined_whole

    @property
    def checkpoint_rollouts_dir(self) -> Path:
        """The directory to save the rollouts to."""
        return self.checkpoint_base_dir.joinpath("rollouts")

    @property
    def raw_transcripts_dir(self) -> Path:
        """The directory to save the raw transcripts to."""
        return self.checkpoint_base_dir.joinpath("raw_transcripts")

    @property
    def processed_transcripts_dir(self) -> Path:
        """The directory to save the processed transcripts to."""
        return self.checkpoint_base_dir.joinpath("processed_transcripts")

    @property
    def checkpoint_analysis_dir(self) -> Path:
        """The directory to save the rollout analysis to."""
        return self.checkpoint_base_dir.joinpath("analysis")

    def sample_rollouts(
        self,
        environment: PureTextEnvironment,
        iteration: int | Literal["test"],
        use_tqdm: bool = False,
        tqdm_desc: str = "Sampling rollouts",
    ) -> NestedArrayDict:
        """Sample rollouts in the environment.

        We sample `environment.num_envs` rollouts from the environment. A rollout is a
        sequence of length `max_message_rounds` of states in the environment. The
        sampled rollout nested array dict thus has shape (num_envs, max_message_rounds).

        Parameters
        ----------
        dataset : NestedArrayDictDataset
            The dataset of task instances.
        iteration : int | Literal["test"]
            The iteration number, or "test" if the rollouts are from the test set.
        environment : PureTextEnvironment
            The environment to sample rollouts in.
        use_tqdm : bool
            Whether to create a tqdm progress bar for the rollouts.
        tqdm_desc : str
            The description to use for the tqdm progress bar.

        Returns
        -------
        rollouts : NestedArrayDict
            The rollouts in the environment. Has batch size (num_envs,
            max_message_rounds)
        """

        generator = torch.Generator()
        generator.manual_seed(self.params.seed)
        dataloader = NestedArrayDictDataLoader(
            environment.dataset,
            batch_size=environment.batch_size[0],
            shuffle=True,
            generator=generator,
            initial_skip=environment.num_envs * iteration,
        )
        data_cycler = VariableDataCycler(
            dataloader, default_batch_size=environment.batch_size[0]
        )

        arg_iterator = (
            (environment, self.max_message_rounds, self.combined_agent, data_batch)
            for data_batch in truncated_iterator(data_cycler, environment.num_envs)
        )

        def get_rollouts(rollout_iterator: Iterable) -> list[NestedArrayDict]:

            if use_tqdm:
                rollout_iterator = tqdm(
                    rollout_iterator,
                    total=environment.num_envs,
                    desc=tqdm_desc,
                )

            return list(rollout_iterator)

        # When the number of rollout workers is set to 0, we sample the rollouts
        # sequentially, without using a pool
        if self.settings.num_rollout_workers == 0:
            rollout_iterator = map(_sample_single_rollout, arg_iterator)
            rollout_list = get_rollouts(rollout_iterator)

        # If we have multiple workers, we can use a pool to parallelize the rollouts
        else:
            with Pool(self.settings.num_rollout_workers) as pool:
                rollout_iterator = pool.imap_unordered(
                    _sample_single_rollout, arg_iterator
                )
                rollout_list = get_rollouts(rollout_iterator)

        return stack_nested_array_dicts(rollout_list, dim=0)

    def save_rollouts(
        self, rollouts: NestedArrayDict, iteration: int | Literal["test"]
    ):
        """Save the rollouts to the checkpoint directory.

        Parameters
        ----------
        rollouts : NestedArrayDict
            The rollouts to save.
        iteration : int | Literal["test"]
            The iteration number, or "test" if the rollouts are from the test set.
        """

        if self.params.text_rl.save_transcripts:
            raw_transcripts, processed_transcripts = self._extract_transcripts(rollouts)

        # If we are running a test run, we don't want to save the rollouts
        if self.settings.test_run:
            return

        self.checkpoint_rollouts_dir.mkdir(parents=True, exist_ok=True)

        rollout_path = self.checkpoint_rollouts_dir.joinpath(f"{iteration}.pt")

        with open(rollout_path, "wb") as f:
            pickle.dump(rollouts, f)

        # If using W&B, also log the rollouts as an artifact
        if self.settings.wandb_run is not None:
            self._add_file_to_wandb_artifact(
                f"{ROLLOUTS_ARTIFACT_PREFIX}{self.settings.wandb_run.name}",
                ROLLOUTS_ARTIFACT_TYPE,
                rollout_path,
            )

        # Save the raw and processed transcripts
        if self.params.text_rl.save_transcripts:

            self.raw_transcripts_dir.mkdir(parents=True, exist_ok=True)
            self.processed_transcripts_dir.mkdir(parents=True, exist_ok=True)

            if self.params.text_rl.transcript_format == "yaml":
                file_extension = "yaml"
            elif self.params.text_rl.transcript_format == "json":
                file_extension = "json"
            else:
                raise NotImplementedError(
                    f"Invalid transcript format "
                    f"{self.params.text_rl.transcript_format!r}"
                )

            raw_transcript_path = self.raw_transcripts_dir.joinpath(
                f"raw_{iteration}.{file_extension}"
            )
            processed_transcript_path = self.processed_transcripts_dir.joinpath(
                f"processed_{iteration}.{file_extension}"
            )

            with open(raw_transcript_path, "w") as f:
                if self.params.text_rl.transcript_format == "yaml":
                    yaml.dump(raw_transcripts, f)
                elif self.params.text_rl.transcript_format == "json":
                    json.dump(raw_transcripts, f, indent=4)

            with open(processed_transcript_path, "w") as f:
                if self.params.text_rl.transcript_format == "yaml":
                    yaml.dump(processed_transcripts, f)
                elif self.params.text_rl.transcript_format == "json":
                    json.dump(processed_transcripts, f, indent=4)

            # If using W&B, also log the transcripts as artifacts
            if self.settings.wandb_run is not None:
                self._add_file_to_wandb_artifact(
                    f"{RAW_TRANSCRIPT_ARTIFACT_PREFIX}{self.settings.wandb_run.name}",
                    RAW_TRANSCRIPT_ARTIFACT_TYPE,
                    raw_transcript_path,
                )
                self._add_file_to_wandb_artifact(
                    f"{PROCESSED_TRANSCRIPT_ARTIFACT_PREFIX}"
                    f"{self.settings.wandb_run.name}",
                    PROCESSED_TRANSCRIPT_ARTIFACT_TYPE,
                    processed_transcript_path,
                )

    def load_rollouts(self, iteration: int) -> NestedArrayDict:
        """Load the rollouts from the checkpoint directory.

        Parameters
        ----------
        iteration : int
            The iteration number.

        Returns
        -------
        NestedArrayDict
            The rollouts.
        """

        # If we are running a test run, we shouldn't be loading rollouts
        if self.settings.test_run:
            raise RuntimeError("Attempted to load rollouts in test run.")

        self.checkpoint_rollouts_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_filepath = self.checkpoint_rollouts_dir.joinpath(f"{iteration}.pt")

        # If using W&B, try to download the rollouts from the artifact first
        if self.settings.wandb_run is not None and not checkpoint_filepath.is_file():
            artifact_name = (
                f"{ROLLOUTS_ARTIFACT_PREFIX}{self.settings.wandb_run.name}:latest"
            )
            try:
                artifact: Artifact = self.settings.wandb_run.use_artifact(
                    artifact_name,
                    type=ROLLOUTS_ARTIFACT_TYPE,
                )
                artifact.download(self.checkpoint_rollouts_dir)
            except wandb.errors.CommError as e:
                # W&B doesn't use subclasses for errors, so we have to check the
                # message. If the error was not that the artifact was not found, we
                # re-raise it.
                if f"artifact '{artifact_name}' not found in" not in e.message:
                    raise e

        with open(checkpoint_filepath, "rb") as f:
            return pickle.load(f)

    def _extract_transcripts(
        self, rollouts: NestedArrayDict
    ) -> tuple[list[dict], list[dict]]:
        """Extract the raw and processed transcripts from the rollouts.

        The raw transcript is the sequence of outputs generated by the models, per
        agent, while the processed transcript is the result of processing these and
        extracting the message per channel.

        Note that in the raw transcripts the messages are per agent, while in the
        processed transcripts the messages are per channel.

        The transcripts have variable length, where if a round has no messages from any
        agent, we declare that the end of the transcript.

        Parameters
        ----------
        rollouts : NestedArrayDict
            The rollouts to extract the transcripts from. A NestedArrayDict with keys:

            - "message_history" (batch round round channel) : The message history for
              each rollout. In each timestep this gives the history of all messages
              generated up to that point.
            - "message_agent_id" (batch round round channel) : The ID of the agent that
              generated each message in the message history.
            - ("agents", "raw_message") (batch round agent) : The raw message generated
              by each model in each timestep.
            - ("agents", "decision") (batch round agent) : The decision made by each
              agent in each timestep.

            The nested array dict also contains keys which specify the datapoint for
            each rollout, as extracted by
            `environment.get_datapoint_from_env_state_as_dict`.

        Returns
        -------
        raw_transcripts : list[dict]
            The raw transcripts. This is a list of transcripts, where each transcript is
            dictionary containing meta data and a "transcript" key. The value at
            "transcript" is a list of dictionaries whose keys are the agent names and
            values are the messages generated by the agents.
        processed_transcripts : list[dict]
            The processed transcripts. This is a list of transcripts, where each
            transcript is dictionary containing meta data and a "transcript" key. The
            value at "transcript" is a list of dictionaries whose keys are
            `f"{active_agent_name}@{channel_name}"` and values are the messages in each
            channel.
        """

        message_history = rollouts["message_history"]
        message_agent_id = rollouts["message_agent_id"]
        raw_message = rollouts["agents", "raw_message"]
        decision = rollouts["agents", "decision"]
        num_rollouts = rollouts.batch_size[0]

        protocol_handler = self.scenario_instance.protocol_handler
        channel_names = protocol_handler.message_channel_names
        agent_names = protocol_handler.agent_names

        raw_transcripts = []
        processed_transcripts = []

        for rollout_id in range(num_rollouts):

            raw_transcript = []
            processed_transcript = []

            for round_id in range(self.max_message_rounds):

                raw_transcript_round = {}
                for agent_id, agent_name in enumerate(agent_names):
                    if raw_message[rollout_id, round_id, agent_id] is not None:
                        raw_transcript_round[agent_name] = raw_message[
                            rollout_id, round_id, agent_id
                        ]

                # If we ever have a round where no agent messaged, we are done for the
                # whole rollout
                if not raw_transcript_round:
                    break

                raw_transcript.append(raw_transcript_round)

                processed_transcript_round = {}

                # We first check the decision made by a verifier, and if it is made, we
                # set the processed transcript to "Accept" or "Reject" based on the
                # decision.
                for verifier_name in protocol_handler.verifier_names:
                    key = f"{verifier_name}.decision"
                    verifier_index = agent_names.index(verifier_name)
                    if decision[rollout_id, round_id, verifier_index] == 0:
                        processed_transcript_round[key] = "Reject"
                        break
                    elif decision[rollout_id, round_id, verifier_index] == 1:
                        processed_transcript_round[key] = "Accept"
                        break

                # Otherwise, we look at the last message history in the rollout. The key
                # is the active agent name and channel name, with an "@" in between.
                else:
                    for channel_id, channel_name in enumerate(channel_names):

                        # Get the id of the agent who messaged in this channel
                        agent_id = message_agent_id[
                            rollout_id, -1, round_id, channel_id
                        ]

                        # If the agent id is -1, it means no agent messaged in this
                        # channel in this round
                        if agent_id == -1:
                            continue

                        agent_name = agent_names[
                            message_agent_id[rollout_id, -1, round_id, channel_id]
                        ]

                        # Add the message to the processed transcript with the key
                        # "{agent_name}@{channel_name}"
                        key = f"{agent_name}@{channel_name}"
                        processed_transcript_round[key] = message_history[
                            rollout_id, -1, round_id, channel_id
                        ]

                if processed_transcript_round:
                    processed_transcript.append(processed_transcript_round)

            metadata = self.train_environment.get_datapoint_from_env_state_as_dict(
                rollouts[rollout_id, 0]
            )

            raw_transcripts.append(dict(transcript=raw_transcript, **metadata))
            processed_transcripts.append(
                dict(transcript=processed_transcript, **metadata)
            )

        return raw_transcripts, processed_transcripts


def _sample_single_rollout(
    args: tuple[
        PureTextEnvironment,
        int,
        PureTextWholeAgent,
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
    max_message_rounds : int
        The maximum number of message rounds in the rollout.
    combined_agent : PureTextWholeAgent
        The combined agent to use for the rollout.
    data_batch : NestedArrayDict, optional
        The data batch to use for the rollout. If None, the data batch will be
        sampled from the dataset.

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
            env_state["padding"] = np.ones(*environment.batch_size, dtype=bool)
            if "next" not in env_state.keys():
                env_state = environment.add_dummy_actions_and_next_to_state(env_state)
            env_states.append(env_state)

    return concatenate_nested_array_dicts(env_states, dim=0)
