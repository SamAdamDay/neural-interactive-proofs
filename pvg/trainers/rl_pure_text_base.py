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
from time import sleep

import yaml

import torch

import numpy as np

from jaxtyping import Bool

from wandb import Artifact
import wandb

from tqdm import tqdm
import wandb.errors

from pvg.parameters import HyperParameters
from pvg.factory import ScenarioInstance
from pvg.experiment_settings import ExperimentSettings
from pvg.scenario_base.data import NestedArrayDictDataLoader, NestedArrayDictDataset
from pvg.scenario_base.environment import PureTextEnvironment
from pvg.scenario_base.agents import PureTextWholeAgent
from pvg.scenario_base.rollout_analysis import (
    PureTextRolloutAnalyser,
    ROLLOUT_ANALYSERS,
)
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
    hyper_params : HyperParameters
        The parameters of the experiment.
    scenario_instance : ScenarioInstance
        The components of the experiment.
    settings : ExperimentSettings
        The instance-specific settings of the experiment, like device, logging, etc.
    """

    @dataclass
    class State(Trainer.State):
        """The state of the experiment."""

        train_loop_stage: Literal[
            "sample_rollouts",
            "log_stats",
            "create_fine_tune_jobs",
            "await_fine_tune_jobs",
            "test",
            "done",
        ] = "sample_rollouts"

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

    def train(self):

        rollouts: Optional[NestedArrayDict] = None

        while self.state.iteration < self.hyper_params.rl.num_iterations:

            self.settings.logger.info(
                f"[{self.state.iteration+1}/{self.hyper_params.rl.num_iterations}] "
                f"Iteration begins."
            )

            # Sample rollouts from the training environment
            if self.state.train_loop_stage == "sample_rollouts":

                rollouts = self._stage_sample_rollouts()

                # Advance to the next stage
                self.state.train_loop_stage = "log_stats"

                self.save_checkpoint()

            # Log the statistics of the rollouts
            if self.state.train_loop_stage == "log_stats":

                # Load the rollouts if they are not already set (i.e. if we are resuming
                # this stage)
                if rollouts is None:
                    rollouts = self._load_rollouts(self.state.iteration)

                self._stage_log_stats(rollouts)

                # Advance to the next stage
                self.state.train_loop_stage = "create_fine_tune_jobs"

                self.save_checkpoint()

            # We don't fine-tune on the last iteration
            if self.state.iteration == self.hyper_params.rl.num_iterations - 1:

                # Advance to the test stage
                self.state.iteration = self.hyper_params.rl.num_iterations
                self.state.train_loop_stage = "test"

                self.save_checkpoint()

                break

            # Create fine-tune jobs for each agent
            if self.state.train_loop_stage == "create_fine_tune_jobs":

                # Load all the rollouts if we are fine-tuning on all previous rollouts
                if self.hyper_params.text_rl.fine_tune_on_all_previous_rollouts:
                    rollouts = self._load_rollouts(range(self.state.iteration + 1))

                # Load the rollouts if they are not already set (i.e. if we are resuming
                # this stage)
                elif rollouts is None:
                    rollouts = self._load_rollouts(self.state.iteration)

                self._stage_create_fine_tune_jobs(rollouts)

                # Advance to the next stage
                self.state.train_loop_stage = "await_fine_tune_jobs"

                self.save_checkpoint()

            # Await the completion of the fine-tune jobs
            if self.state.train_loop_stage == "await_fine_tune_jobs":

                self._stage_await_fine_tune_jobs()

                # Advance to the next iteration and stage
                self.state.train_loop_stage = "sample_rollouts"
                self.state.iteration += 1

                self.save_checkpoint()

        if self.state.train_loop_stage == "test":

            if self.hyper_params.text_rl.run_test_loop:

                self._stage_run_test_loop()

            # Mark the experiment as done
            self.state.train_loop_stage = "done"

            # Save the final checkpoint
            self.save_checkpoint()

        self.settings.logger.info("Training complete.")

    def run_analysers(
        self,
        analysers: list[str | type[PureTextRolloutAnalyser]],
        model_name: str,
        *,
        overwrite=False,
        use_tqdm=True,
        dry_run=False,
    ):
        """Run the given analysers on the rollouts of the experiment.

        This method can only be called after the experiment has finished.

        Parameters
        ----------
        analysers : list[str | type[PureTextRolloutAnalyser]]
            The analysers to run. Either the name of the analyser or the analyser class
            itself.
        model_name : str
            The name of the model to use for the analysis.
        overwrite : bool, default=False
            Whether to overwrite the existing analysis files, if they exist.
        use_tqdm : bool, default=True
            Whether create a progress bar for the analysis.
        dry_run : bool, default=False
            Whether to do a dry run using a dummy API, not saving the results.
        """

        for analyser_cls in analysers:

            if isinstance(analyser_cls, str):
                try:
                    analyser_cls: type[PureTextRolloutAnalyser] = ROLLOUT_ANALYSERS[
                        self.hyper_params.scenario, analyser_cls
                    ]
                except KeyError:
                    raise ValueError(
                        f"Analyser {analyser_cls!r} not found in list of analysers."
                    )

            analyser = analyser_cls(
                hyper_params=self.hyper_params,
                settings=self.settings,
                protocol_handler=self.scenario_instance.protocol_handler,
                model_name=model_name,
                use_dummy_api=dry_run,
            )

            analysis_dir = self.checkpoint_analysis_dir.joinpath(analyser_cls.name)
            analysis_dir.mkdir(parents=True, exist_ok=True)

            for iteration in range(self.hyper_params.rl.num_iterations):

                print(  # noqa: T201
                    f"Running analyser {analyser_cls.name!r} on iteration "
                    f"{iteration+1}/{self.hyper_params.rl.num_iterations}"
                )

                analysis_file = analysis_dir.joinpath(f"{iteration}.pt")

                if analysis_file.exists():
                    if not overwrite:
                        self.settings.logger.warning(
                            f"Analysis file {analysis_file!r} already exists. Skipping."
                        )
                        continue
                    else:
                        self.settings.logger.warning(
                            f"Overwriting existing analysis file {analysis_file!r}"
                        )
                    if not dry_run:
                        analysis_file.unlink()

                try:
                    rollouts = self._load_rollouts(iteration)
                except FileNotFoundError:
                    self.settings.logger.warning(
                        f"No rollouts found for iteration {iteration+1}. Skipping."
                    )
                    continue

                evaluations = analyser.forward(rollouts, use_tqdm=use_tqdm)

                if not dry_run:
                    with open(analysis_file, "wb") as f:
                        pickle.dump(evaluations, f)

    def _stage_sample_rollouts(self) -> NestedArrayDict:
        """Training stage: sample rollouts from the training environment.

        Returns
        -------
        rollouts : NestedArrayDict
            The sampled rollouts.
        """

        # Sample rollouts
        rollouts = self._sample_rollouts(
            self.train_environment,
            self.state.iteration,
            use_tqdm=not self.settings.test_run,
        )

        # Save the rollouts to the checkpoint directory
        self._save_rollouts(rollouts, self.state.iteration)

        return rollouts

    def _stage_log_stats(self, rollouts: NestedArrayDict):
        """Training stage: log the statistics of the rollouts.

        Parameters
        ----------
        rollouts : NestedArrayDict
            The rollouts sampled in this iteration.
        """

        log_stats = self._get_log_stats(rollouts, train=True)
        self.settings.stat_logger.log(log_stats, self.state.iteration)

    @abstractmethod
    def _stage_create_fine_tune_jobs(self, rollouts: NestedArrayDict):
        """Training stage: create fine-tune jobs for each agent.

        Parameters
        ----------
        rollouts : NestedArrayDict, optional
            The rollouts sampled in this iteration.
        """

    def _stage_await_fine_tune_jobs(self):
        """Training stage: await the completion of the fine-tune jobs."""

        self.settings.logger.info("Awaiting completion of fine-tune jobs...")

        while True:

            num_successful_jobs = 0
            for agent_name, agent_whole in self.agent_wholes.items():
                if agent_whole.get_fine_tune_job_status() == "succeeded":
                    num_successful_jobs += 1
                elif agent_whole.get_fine_tune_job_status() == "failed":
                    raise RuntimeError(
                        f"Fine-tune job for {agent_name!r} failed. "
                        f"{agent_whole.get_fine_tune_job_error_repr()}"
                    )

            if num_successful_jobs == len(self.agent_wholes):
                self.settings.logger.info("All fine-tune jobs succeeded")
                break

            # Wait for a minute before checking again
            sleep(60)

        # Make all the agents use the new, fine-tuned models
        for agent_name, agent_whole in self.agent_wholes.items():
            agent_whole.switch_to_next_model()

    def _stage_run_test_loop(self):
        """Training stage: run the test loop."""

        # Sample rollouts from the test environment
        rollouts = self._sample_rollouts(
            self.test_environment, "test", use_tqdm=True, tqdm_desc="Testing"
        )

        # Log the statistics of the rollouts
        log_stats = self._get_log_stats(rollouts, train=False)
        self.settings.stat_logger.log(log_stats)

        # Save the rollouts to the checkpoint directory
        self._save_rollouts(rollouts, "test")

    def _sample_rollouts(
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
        generator.manual_seed(self.hyper_params.seed)
        if iteration == "test":
            initial_skip = 0
        else:
            initial_skip = environment.num_envs * iteration
        dataloader = NestedArrayDictDataLoader(
            environment.dataset,
            batch_size=environment.batch_size[0],
            shuffle=True,
            generator=generator,
            initial_skip=initial_skip,
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

    def _save_rollouts(
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

        if self.hyper_params.text_rl.save_transcripts:
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
        if self.hyper_params.text_rl.save_transcripts:

            self.raw_transcripts_dir.mkdir(parents=True, exist_ok=True)
            self.processed_transcripts_dir.mkdir(parents=True, exist_ok=True)

            if self.hyper_params.text_rl.transcript_format == "yaml":
                file_extension = "yaml"
            elif self.hyper_params.text_rl.transcript_format == "json":
                file_extension = "json"
            else:
                raise NotImplementedError(
                    f"Invalid transcript format "
                    f"{self.hyper_params.text_rl.transcript_format!r}"
                )

            raw_transcript_path = self.raw_transcripts_dir.joinpath(
                f"raw_{iteration}.{file_extension}"
            )
            processed_transcript_path = self.processed_transcripts_dir.joinpath(
                f"processed_{iteration}.{file_extension}"
            )

            with open(raw_transcript_path, "w") as f:
                if self.hyper_params.text_rl.transcript_format == "yaml":
                    yaml.dump(raw_transcripts, f)
                elif self.hyper_params.text_rl.transcript_format == "json":
                    json.dump(raw_transcripts, f, indent=4)

            with open(processed_transcript_path, "w") as f:
                if self.hyper_params.text_rl.transcript_format == "yaml":
                    yaml.dump(processed_transcripts, f)
                elif self.hyper_params.text_rl.transcript_format == "json":
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

    def _load_rollouts(self, iterations: int | Iterable[int]) -> NestedArrayDict:
        """Load the rollouts from the checkpoint directory.

        Parameters
        ----------
        iterations : int | Iterable[int]
            The iteration numbers to load the rollouts for. These will be concatenated
            into a single NestedArrayDict.

        Returns
        -------
        NestedArrayDict
            The concatenated rollouts for each iteration requested.
        """

        # If we are running a test run, we shouldn't be loading rollouts
        if self.settings.test_run:
            raise RuntimeError("Attempted to load rollouts in test run.")

        if isinstance(iterations, int):
            iterations = [iterations]

        self.checkpoint_rollouts_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_filepaths = [
            self.checkpoint_rollouts_dir.joinpath(f"{iteration}.pt")
            for iteration in iterations
        ]

        # If using W&B, try to download the rollouts from the artifact first
        if self.settings.wandb_run is not None and not all(
            filepath.is_file() for filepath in checkpoint_filepaths
        ):
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

        checkpoints = []
        for iteration, checkpoint_filepath in zip(iterations, checkpoint_filepaths):
            if not checkpoint_filepath.is_file():
                raise FileNotFoundError(
                    f"Attempted to load rollouts for iteration {iteration}, but "
                    f"file {checkpoint_filepath!r} not found."
                )
            with open(checkpoint_filepath, "rb") as f:
                checkpoints.append(pickle.load(f))

        return concatenate_nested_array_dicts(checkpoints)

    def _get_log_stats(
        self,
        rollouts: NestedArrayDict,
        *,
        train=True,
    ) -> dict:
        """Get the statistics to log for the given rollouts.

        Parameters
        ----------
        rollouts : NestedArrayDict
            The rollouts to get the statistics for.

        Returns
        -------
        stats : dict
            The statistics to log.
        """

        if train:
            prefix = ""
        else:
            prefix = "test_"

        done: Bool[np.ndarray, "..."] = rollouts["done"]
        next_done: Bool[np.ndarray, "..."] = rollouts["next", "done"]
        next_terminated: Bool[np.ndarray, "..."] = rollouts["next", "terminated"]
        padding: Bool[np.ndarray, "..."] = rollouts["padding"]

        last_timestep = (next_done | next_terminated) & ~padding

        log_stats = {}

        for agent_index, agent_name in enumerate(self.agent_names):

            # Get the total episode reward for each agent
            episode_reward = rollouts["next", "agents", "reward"][..., agent_index].sum(
                axis=-1
            )
            log_stats[f"{agent_name}.{prefix}mean_episode_reward"] = (
                episode_reward.mean().item()
            )
            log_stats[f"{agent_name}.{prefix}std_episode_reward"] = (
                episode_reward.std().item()
            )

            # The proportion of messages that were retried or hit the token limit
            log_stats[f"{agent_name}.{prefix}retry_proportion"] = (
                rollouts["agents", "retry_count"][..., agent_index, :][~done]
                .mean()
                .item()
            )
            log_stats[f"{agent_name}.{prefix}token_limit_proportion"] = (
                rollouts["agents", "token_limit"][..., agent_index, :][~done]
                .mean()
                .item()
            )

        episode_length = (
            rollouts["message_history"][..., -1, :, 0] != None  # noqa: E711
        )
        log_stats[f"{prefix}mean_episode_length"] = (
            episode_length.sum(axis=-1).mean().item()
        )
        log_stats[f"{prefix}std_episode_length"] = (
            episode_length.sum(axis=-1).std().item()
        )

        # Get the mean and std accuracy of the verifier
        verifier_decision = rollouts["agents", "decision"][
            ..., self.agent_names.index("verifier")
        ]
        accuracy = verifier_decision[last_timestep] == rollouts["y"][last_timestep]
        log_stats[f"{prefix}mean_accuracy"] = accuracy.mean().item()
        log_stats[f"{prefix}std_accuracy"] = accuracy.std().item()

        # Get the mean and accuracy of the verifier by class
        for class_value in [0, 1]:
            class_mask = rollouts["y"][last_timestep] == class_value
            class_accuracy = verifier_decision[last_timestep][class_mask] == class_value
            log_stats[f"{prefix}mean_{class_value}_accuracy"] = (
                class_accuracy.mean().item()
            )
            log_stats[f"{prefix}std_{class_value}_accuracy"] = (
                class_accuracy.std().item()
            )

        # Get the mean and std verifier decision
        log_stats[f"{prefix}mean_decision"] = (
            verifier_decision[last_timestep].mean().item()
        )
        log_stats[f"{prefix}std_decision"] = (
            verifier_decision[last_timestep].std().item()
        )

        # Get the precision and recall of the verifier
        true_positives = (
            (verifier_decision[last_timestep] == 1)
            & (rollouts["y"][last_timestep] == 1)
        ).sum()
        false_positives = (
            (verifier_decision[last_timestep] == 1)
            & (rollouts["y"][last_timestep] == 0)
        ).sum()
        false_negatives = (
            (verifier_decision[last_timestep] == 0)
            & (rollouts["y"][last_timestep] == 1)
        ).sum()
        log_stats[f"{prefix}precision"] = true_positives / (
            true_positives + false_positives
        )
        log_stats[f"{prefix}recall"] = true_positives / (
            true_positives + false_negatives
        )

        return log_stats

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
