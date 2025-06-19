"""Base classes for RL trainers for text-based environments that only use APIs."""

from abc import ABC, abstractmethod
from typing import Optional, Literal, Iterable
from multiprocessing import Pool
from functools import cached_property
from itertools import chain
from pathlib import Path
import pickle
import dataclasses
from dataclasses import dataclass
import json
from warnings import warn
from asyncio import TaskGroup, Queue
import asyncio
import logging

import yaml

import torch

import numpy as np
from numpy.typing import NDArray

from einops import reduce

from jaxtyping import Bool, Int, Float

from wandb import Artifact
import wandb

from tqdm import tqdm
import wandb.errors

from nip.scenario_base.data import NestedArrayDictDataLoader
from nip.scenario_base.environment import PureTextEnvironment
from nip.scenario_base.agents import (
    PureTextWholeAgent,
    PureTextCombinedWhole,
    PureTextSharedModelGroup,
    PureTextSharedModelGroupState,
)
from nip.scenario_base.rollout_analysis import (
    PureTextRolloutAnalyser,
    ROLLOUT_ANALYSERS,
)
from nip.trainers.trainer_base import Trainer, CheckPointNotFoundError
from nip.utils.maths import aggregate_mean_grouped_by_class, entropy_numpy
from nip.utils.data import VariableDataCycler, truncated_iterator
from nip.utils.nested_array_dict import (
    NestedArrayDict,
    stack_nested_array_dicts,
    concatenate_nested_array_dicts,
)
from nip.utils.rollouts import get_pretty_pure_text_round_message
from nip.utils.types import String, PromptMessage
from nip.utils.io import yes_no_user_prompt
from nip.constants import (
    ROLLOUTS_ARTIFACT_PREFIX,
    ROLLOUTS_ARTIFACT_TYPE,
    RAW_TRANSCRIPT_ARTIFACT_PREFIX,
    RAW_TRANSCRIPT_ARTIFACT_TYPE,
    PROCESSED_TRANSCRIPT_ARTIFACT_PREFIX,
    PROCESSED_TRANSCRIPT_ARTIFACT_TYPE,
    PROMPTS_ARTIFACT_PREFIX,
    PROMPTS_ARTIFACT_TYPE,
)

logger = logging.getLogger(__name__)


class FineTuneJobError(Exception):
    """Exception raised when a fine-tune job fails."""


class PureTextRlTrainer(Trainer, ABC):
    """Base class for RL trainers for text-based environments that only use APIs.

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
        """The state of the experiment.

        Parameters
        ----------
        iteration : int
            The current iteration number.
        agents : dict[str, AgentCheckpoint]
            The checkpoints of the agents.
        train_loop_stage : str
            The current stage of the training loop. One of:

            - "sample_rollouts": Sample rollouts from the training environment.
            - "log_stats": Log the statistics of the sampled rollouts.
            - "create_fine_tune_jobs": Create fine-tune jobs for each shared agent
              group.
            - "await_fine_tune_jobs": Await the completion of the fine-tune jobs.
            - "recreate_fine_tune_jobs": Create fine-tune jobs for each shared agent
              group whose previous fine-tune job failed or was cancelled. This stage is
              only used when one of the fine-tune jobs fails or is cancelled.
            - "test_during_training": Run the test loop during training.
            - "test": Run the test loop after training.
            - "done": The training is complete.

        shared_model_groups : dict[str, PureTextSharedModelGroupState]
            The state of each shared model group.
        base_run_state_artifact_version : int
            When rerunning tests, we step through the states of the base run in order.
            This is the version of the base run state artifact that we're on.
        """

        train_loop_stage: Literal[
            "sample_rollouts",
            "log_stats",
            "create_fine_tune_jobs",
            "await_fine_tune_jobs",
            "test_during_training",
            "test",
            "done",
        ] = "sample_rollouts"
        shared_model_groups: dict[str, PureTextSharedModelGroupState] = (
            dataclasses.field(default_factory=dict)
        )
        base_run_state_artifact_version: int = 0

    _state: State

    @property
    def state(self) -> State:
        """The state of the experiment."""

        if not hasattr(self, "_state"):
            self._state = self.State()

        # Get the state of the agents to fill out the ``shared_model_groups`` field
        for group_name, shared_model_group in self.shared_model_groups.items():
            self._state.shared_model_groups[group_name] = shared_model_group.get_state()

        return self._state

    @state.setter
    def state(self, state: State):
        self._state = state

        for group_name, shared_model_group in self.shared_model_groups.items():

            if group_name in state.shared_model_groups:
                shared_model_group.set_state(state.shared_model_groups[group_name])

            elif group_name in state.agents:
                shared_model_group.set_state(state.agents[group_name])
                warn(
                    "Experiment state does not contain shared model group state. Using "
                    "agent state instead."
                )

            else:
                raise ValueError(
                    f"Shared model group {group_name!r} not found in state."
                )

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

    @cached_property
    def shared_model_groups(self) -> dict[str, PureTextSharedModelGroup]:
        """The agents grouped by having a shared model."""
        return self.scenario_instance.shared_model_groups

    @property
    def combined_agent(self) -> PureTextCombinedWhole:
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
    def prompts_dir(self) -> Path:
        """The directory to save the prompts to."""
        return self.checkpoint_base_dir.joinpath("prompts")

    @property
    def checkpoint_analysis_dir(self) -> Path:
        """The directory to save the rollout analysis to."""
        return self.checkpoint_base_dir.joinpath("analysis")

    def train(self):
        """Train the agents in the environment.

        Runs the training loop for the specified number of iterations. The training loop
        consists of the following stages:

        1. Sample rollouts from the training environment.
        2. Log the statistics of the rollouts.
        3. Run the test loop during training.
        4. Create fine-tune jobs for each agent.
        5. Await the completion of the fine-tune jobs.

        The training loop can be resumed from a previous checkpoint. If the training
        loop is resumed, the state of the experiment is loaded from the checkpoint, and
        the training loop is resumed from the last stage.
        """

        asyncio.run(self._train())

    async def _train(self):
        """Run the actual training loop implementation, which is asynchronous."""

        rerun_tests = self.hyper_params.base_run.base_run_type == "rerun_tests"

        if rerun_tests:
            logger.info(
                f"Rerunning tests from base run {self.hyper_params.base_run.run_id!r}. "
                f"Loading the state from the base run."
            )
            base_run_state_artifact_version = self.state.base_run_state_artifact_version

        # Should we test during the "log_stats" stage instead of the
        # "test_during_training" stage? This is a bit of a hack, to allow testing during
        # training when rerunning older runs which didn't have a "test_during_training"
        # stage
        test_during_log_stats_stage = (
            rerun_tests
            and self.hyper_params.base_run.rerun_tests_force_test_during_training_state
        )

        # This condition happens when resuming a previously completed run but with more
        # iterations
        if (
            self.state.iteration < self.hyper_params.rl.num_iterations
            and self.state.train_loop_stage == "done"
        ):
            if self.settings.force_more_iterations:
                logger.info(
                    "Forcing more iterations to be run, even though the state "
                    "indicates that the training is done."
                )
                self.state.train_loop_stage = "sample_rollouts"
                self.save_checkpoint()
            else:
                logger.info(
                    "Training is already done. If you want to run more iterations, "
                    "set `force_more_iterations=True` in the experiment settings."
                )
                return

        async with TaskGroup() as task_group:
            # Make sure all the shared model groups are ready
            for shared_model_group in self.shared_model_groups.values():
                task_group.create_task(shared_model_group.wait_for_ready())

        rollouts: Optional[NestedArrayDict] = None

        while self.state.iteration < self.hyper_params.rl.num_iterations:

            if rerun_tests:
                try:
                    self.load_and_set_state_from_checkpoint(
                        from_base_run=True,
                        version=f"v{base_run_state_artifact_version}",
                    )
                except CheckPointNotFoundError:
                    logger.info(
                        f"Reached the end of the base run. Iteration: "
                        f"{self.state.iteration}, stage: "
                        f"{self.state.train_loop_stage!r}."
                    )
                    break

                self.state.base_run_state_artifact_version = (
                    base_run_state_artifact_version
                )

                logger.info(
                    f"Loaded state artifact version "
                    f"'v{self.state.base_run_state_artifact_version}'. Iteration: "
                    f"{self.state.iteration}, stage: "
                    f"{self.state.train_loop_stage!r}."
                )

                self.save_checkpoint()

            # Sample rollouts from the training environment
            if self.state.train_loop_stage == "sample_rollouts" and not rerun_tests:

                logger.info(
                    f"[{self.state.iteration+1}/{self.hyper_params.rl.num_iterations}] "
                    f"{self._get_iteration_begin_message()}"
                )

                # Make sure all the shared model groups are in evaluation mode. We can
                # do this concurrently for each group
                async with TaskGroup() as task_group:
                    for shared_model_group in self.shared_model_groups.values():
                        task_group.create_task(shared_model_group.eval())

                rollouts = await self._stage_sample_rollouts()

                # Advance to the next stage
                self.state.train_loop_stage = "log_stats"

                self.save_checkpoint()

            # Log the statistics of the rollouts
            elif self.state.train_loop_stage == "log_stats" and not rerun_tests:

                # Load the rollouts if they are not already set (i.e. if we are resuming
                # this stage)
                if rollouts is None:
                    rollouts = self._load_rollouts(self.state.iteration)

                self._stage_log_stats(rollouts)

                # Advance to the next stage
                self.state.train_loop_stage = "test_during_training"

                self.save_checkpoint()

            # Run the test loop during training. This may happen during the "log_stats"
            # stage. See the comment above.
            elif (
                self.state.train_loop_stage == "test_during_training"
                and not test_during_log_stats_stage
            ) or (
                self.state.train_loop_stage == "log_stats"
                and test_during_log_stats_stage
            ):

                if test_during_log_stats_stage:
                    logger.info(
                        "Testing during 'log_stats' stage for compatibility with older "
                        "runs without a 'test_during_training' stage."
                    )

                # Run the test loop if we're doing that this iteration
                if self._check_if_run_test_loop():

                    # Make sure all the shared model groups are in evaluation mode. We
                    # can do this concurrently for each group
                    async with TaskGroup() as task_group:
                        for shared_model_group in self.shared_model_groups.values():
                            task_group.create_task(shared_model_group.eval())

                    await self._stage_run_test_loop()

                # Advance to the next stage
                self.state.train_loop_stage = "create_fine_tune_jobs"

                self.save_checkpoint()

            # If we've done the above stages and we're at the last iteration, we don't
            # need to create fine-tune jobs, so we can advance to the test stage
            elif (
                self.state.iteration == self.hyper_params.rl.num_iterations - 1
                and not rerun_tests
            ):

                # Advance to the test stage
                self.state.iteration = self.hyper_params.rl.num_iterations
                self.state.train_loop_stage = "test"

                self.save_checkpoint()

                break

            # Create fine-tune jobs for each agent
            elif (
                self.state.train_loop_stage == "create_fine_tune_jobs"
                or self.state.train_loop_stage == "recreate_fine_tune_jobs"
            ) and not rerun_tests:

                # Load all the rollouts if we are fine-tuning on all previous rollouts
                if self.hyper_params.text_rl.fine_tune_on_all_previous_rollouts:
                    rollouts = self._load_rollouts(
                        chain(
                            self._previous_compatible_iterations(),
                            (self.state.iteration,),
                        )
                    )

                # Load the rollouts if they are not already set (i.e. if we are resuming
                # this stage)
                elif rollouts is None:
                    rollouts = self._load_rollouts(self.state.iteration)

                # Make sure all the shared model groups are in training mode. We can do
                # this concurrently for each group. When recreating fine-tune jobs, we
                # only do this for groups that have failed.
                async with TaskGroup() as task_group:
                    for shared_model_group in self.shared_model_groups.values():
                        if (
                            self.state.train_loop_stage == "create_fine_tune_jobs"
                            or await shared_model_group.fine_tune_job_failed()
                        ):
                            task_group.create_task(shared_model_group.train())

                await self._stage_create_fine_tune_jobs(
                    rollouts,
                    only_failed=(
                        self.state.train_loop_stage == "recreate_fine_tune_jobs"
                    ),
                )

                # Advance to the next stage
                self.state.train_loop_stage = "await_fine_tune_jobs"

                self.save_checkpoint()

            # Await the completion of the fine-tune jobs
            elif (
                self.state.train_loop_stage == "await_fine_tune_jobs"
                and not rerun_tests
            ):

                try:
                    await self._stage_await_fine_tune_jobs()

                except* FineTuneJobError as exception_group:
                    if yes_no_user_prompt(
                        "Do you want to re-submit all failed fine-tune jobs? (If any "
                        "other fine-tune jobs fail between now and the resubmission "
                        "stage, they will be re-submitted as well.)",
                        initial_message="\n".join(
                            [str(exception) for exception in exception_group.exceptions]
                        ),
                        default_answer="n",
                    ):
                        self.state.train_loop_stage = "recreate_fine_tune_jobs"
                        self.save_checkpoint()
                    else:
                        raise exception_group

                else:
                    # Advance to the next iteration and stage
                    self.state.train_loop_stage = "sample_rollouts"
                    self.state.iteration += 1

                    self.save_checkpoint()

            # If we're rerunning tests, step the state artifact version number so that
            # we get the next state in the base run
            if rerun_tests:
                base_run_state_artifact_version += 1

        # Mark the experiment as done
        self.state.train_loop_stage = "done"

        # Save the final checkpoint
        self.save_checkpoint()

        logger.info("Training complete.")

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
                protocol_handler=self.protocol_handler,
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
                        logger.warning(
                            f"Analysis file {analysis_file!r} already exists. Skipping."
                        )
                        continue
                    else:
                        logger.warning(
                            f"Overwriting existing analysis file {analysis_file!r}"
                        )
                    if not dry_run:
                        analysis_file.unlink()

                try:
                    rollouts = self._load_rollouts(iteration)
                except FileNotFoundError:
                    logger.warning(
                        f"No rollouts found for iteration {iteration+1}. Skipping."
                    )
                    continue

                evaluations = analyser.forward(rollouts, use_tqdm=use_tqdm)

                if not dry_run:
                    with open(analysis_file, "wb") as f:
                        pickle.dump(evaluations, f)

    def _get_iteration_begin_message(self) -> str:
        """Get the message to log at the beginning of each iteration.

        Returns
        -------
        message : str
            The message to log at the beginning of each iteration.
        """
        return "Iteration begins."

    async def _stage_sample_rollouts(self) -> NestedArrayDict:
        """Training stage: sample rollouts from the training environment.

        Returns
        -------
        rollouts : NestedArrayDict
            The sampled rollouts.
        """

        rollouts = await self._sample_rollouts(
            self.train_environment,
            self.state.iteration,
            use_tqdm=not self.settings.test_run,
        )

        # Save the rollouts to the checkpoint directory
        self._save_rollouts(rollouts, self.train_environment)

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
    async def _stage_create_fine_tune_jobs(
        self, rollouts: NestedArrayDict, only_failed: bool = False
    ):
        """Training stage: create fine-tune jobs for each agent.

        Parameters
        ----------
        rollouts : NestedArrayDict, optional
            The rollouts sampled in this iteration.
        only_failed : bool, default=False
            Whether to only create fine-tune jobs for shared model groups whose previous
            fine-tune job failed or was cancelled. If False, fine-tune jobs are created
            for all shared model groups.
        """

    async def _stage_await_fine_tune_jobs(self):
        """Training stage: await the completion of the fine-tune jobs.

        Raises
        ------
        ExceptionGroup[FineTuneJobError]
            If any of the fine-tune jobs fail or are cancelled. Note that since we use a
            task group to await the fine-tune jobs, the exceptions are raised as an
            :py:class:`ExceptionGroup`. This can be caught using an ``except*``
            statement. If ``exception_group`` is caught, then
            ``exception_group.exceptions`` will contain the individual exceptions for
            each fine-tune job that failed.

        Example
        -------
        >>> try:
        >>>     await trainer._stage_await_fine_tune_jobs()
        >>> except* FineTuneJobError as exception_group:
        >>>     for exception in exception_group.exceptions:
        >>>         logger.error(exception)
        """

        logger.info("Awaiting completion of fine-tune jobs...")

        async def wait_for_fine_tune_job(
            group_name: str, shared_model_group: PureTextSharedModelGroup
        ):
            """Wait for a fine-tune job to complete for a single shared model group.

            Once the fine-tune job is complete, switch to the fine-tuned model.
            """

            while True:

                status = await shared_model_group.get_fine_tune_job_status()

                if status == "succeeded":
                    logger.info(
                        f"Fine-tune job for group {group_name!r} succeeded. Switching "
                        f"to next model."
                    )
                    await shared_model_group.switch_to_next_model()
                    return

                elif status == "failed":
                    error_repr = await shared_model_group.get_fine_tune_job_error_repr()
                    message = f"Fine-tune job for group {group_name!r} failed."
                    if error_repr != "":
                        message += f" Error: {error_repr}"
                    raise FineTuneJobError(message)

                elif status == "cancelled":
                    message = f"Fine-tune job for group {group_name!r} was cancelled."
                    raise FineTuneJobError(message)

                elif status == "not_found":
                    message = (
                        f"Fine-tune job for group {group_name!r} not found. This may "
                        "happen if the job was never created or if it was deleted."
                    )
                    raise FineTuneJobError(message)

                # Wait for a minute before checking again
                await asyncio.sleep(60)

        async with TaskGroup() as task_group:
            for group_name, shared_model_group in self.shared_model_groups.items():
                if shared_model_group.shared_agent_params.freeze_agent:
                    continue
                task_group.create_task(
                    wait_for_fine_tune_job(group_name, shared_model_group)
                )

        logger.info("All fine-tune jobs succeeded.")

    async def _stage_run_test_loop(self):
        """Training stage: run the test loop."""

        # Sample rollouts from the test environment
        rollouts = await self._sample_rollouts(
            self.test_environment, "test", use_tqdm=True, tqdm_desc="Testing"
        )

        # Log the statistics of the rollouts
        log_stats = self._get_log_stats(rollouts, train=False)
        self.settings.stat_logger.log(log_stats)

        # Save the rollouts to the checkpoint directory
        self._save_rollouts(rollouts, self.test_environment)

    def _check_if_run_test_loop(self) -> bool:
        """Check if the test loop should be run in the current iteration.

        Returns
        -------
        run_test_loop : bool
            Whether the test loop should be run.
        """

        if self.hyper_params.text_rl.test_scheme == "none":
            return False
        elif self.hyper_params.text_rl.test_scheme == "all":
            return True
        elif self.hyper_params.text_rl.test_scheme == "last":
            return self.state.iteration == self.hyper_params.rl.num_iterations - 1
        elif self.hyper_params.text_rl.test_scheme == "first_and_last":
            return (
                self.state.iteration == 0
                or self.state.iteration == self.hyper_params.rl.num_iterations - 1
            )
        else:
            raise ValueError(
                f"Invalid test scheme {self.hyper_params.text_rl.test_scheme!r}"
            )

    async def _sample_rollouts(
        self,
        environment: PureTextEnvironment,
        iteration: int | Literal["test"],
        use_tqdm: bool = False,
        tqdm_desc: str = "Sampling rollouts",
    ) -> NestedArrayDict:
        """Sample rollouts in the environment.

        We sample ``environment.num_envs`` rollouts from the environment. A rollout is a
        sequence of length ``max_message_rounds`` of states in the environment. The
        sampled rollout nested array dict thus has shape (num_envs, max_message_rounds).

        Parameters
        ----------
        environment : PureTextEnvironment
            The environment to sample rollouts in.
        iteration : int | Literal["test"]
            The iteration number, or "test" if the rollouts are from the test set.
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

        if iteration == "test":
            if self.hyper_params.text_rl.test_on_whole_dataset:
                num_rollouts = (
                    len(environment.dataset) * self.hyper_params.rl.num_test_iterations
                )
            else:
                num_rollouts = (
                    environment.num_envs * self.hyper_params.rl.num_test_iterations
                )
        else:
            num_rollouts = environment.num_envs

        sample_queue = Queue()
        if use_tqdm:
            progress_bar = tqdm(total=num_rollouts, desc=tqdm_desc)

        async def sample_task(
            data_batch: Optional[NestedArrayDict],
        ):
            sample = await self._sample_rollouts_for_single_environment(
                environment, data_batch
            )
            await sample_queue.put(sample)
            if use_tqdm:
                progress_bar.update(1)

        async with TaskGroup() as task_group:
            for data_batch in truncated_iterator(data_cycler, num_rollouts):
                task_group.create_task(sample_task(data_batch))

        rollout_list = []
        while not sample_queue.empty():
            sample = await sample_queue.get()
            rollout_list.extend(sample)

        rollouts_stacked = stack_nested_array_dicts(rollout_list, dim=0)

        return rollouts_stacked

    async def _sample_rollouts_for_single_environment(
        self,
        environment: PureTextEnvironment,
        data_batch: Optional[NestedArrayDict] = None,
    ) -> list[NestedArrayDict]:
        """Sample rollouts for a single environment.

        A single environment is associated with a single datapoint. This method samples
        rollouts from it. It is intended that subclasses are able reimplement this if
        they need to sample rollouts in a different way.

        In this default implementation, we sample a single rollout by stepping the
        environment until it is done, and then padding the rollout with zero states up
        to the maximum number of message rounds.

        Parameters
        ----------
        environment : PureTextEnvironment
            The environment to sample rollouts in.
        data_batch : NestedArrayDict, optional
            The data batch to use for the rollout. If None, the data batch will be
            sampled from the dataset.

        Returns
        -------
        list[NestedArrayDict]
            The a single-element list containing the rollout in the environment. Has
            batch size (max_message_rounds, )
        """

        ended = False
        env_state = environment.reset(data_batch=data_batch)
        env_states = []

        for _ in range(self.max_message_rounds):
            if not ended:

                # Run the forward pass on all agents to sample actions
                env_state = await self.combined_agent(env_state, environment)

                # Step the environment to get the next state. This writes the next state
                # in the "next" sub-dictionary.
                env_state = environment.step(env_state)

                # Check if the environment is done or terminated. The state has batch
                # size 1, so we only need to check the first element.
                ended = (
                    env_state["next", "done"][0] or env_state["next", "terminated"][0]
                )

                # Append the current state to the environment states
                env_states.append(env_state)

                # Update the current state to the next state
                env_state = environment.get_next_state_from_state(env_state)

            # If we are done, we need to pad the rollout with zero actions
            else:
                env_state["padding"] = np.ones(*environment.batch_size, dtype=bool)
                if "next" not in env_state.keys():
                    env_state = environment.add_dummy_actions_and_next_to_state(
                        env_state
                    )
                env_states.append(env_state)

        sampled_rollout = concatenate_nested_array_dicts(env_states, dim=0)

        return [sampled_rollout]

    def _previous_compatible_iterations(self) -> Iterable[int]:
        """Get the previous iterations which are combinable with the current iteration.

        The method is used when combining rollouts from different iterations, and
        returns an iterable of the previous iteration numbers which are able to be
        combined with the current iteration.

        Returns
        -------
        previous_iterations : Iterable[int]
            The previous iterations which are combinable with the current iteration.
        """

        return range(self.state.iteration)

    def _get_verifier_guess_replacement_proportion(self, iteration: int) -> float:
        """Get the proportion of rollouts to replace the guess with the true label.

        For this proportion of the sampled rollouts, we replace the verifier guess with
        either "Decision: accept" or "Decision: reject" based on the true label.

        This value can be annealed over the course of the training.

        Parameters
        ----------
        iteration : int
            The current iteration number.

        Returns
        -------
        proportion : float
            The proportion of rollouts where we replace the guess with the true label.

        Raises
        ------
        ValueError
            If the annealing type is invalid.
        """

        anneal_type = self.hyper_params.text_rl.verifier_guess_replacement_annealing
        initial_proportion = (
            self.hyper_params.text_rl.verifier_guess_replacement_proportion
        )
        rate = self.hyper_params.text_rl.verifier_guess_replacement_annealing_rate

        if anneal_type == "none":
            return initial_proportion
        elif anneal_type == "linear":
            return max(initial_proportion - iteration * rate, 0)
        elif anneal_type == "exponential":
            return initial_proportion * (1 - rate) ** iteration
        else:
            raise ValueError(
                f"Invalid annealing type {anneal_type!r} for verifier guess "
                f"replacement."
            )

    def _save_rollouts(
        self,
        rollouts: NestedArrayDict,
        environment: PureTextEnvironment,
        iteration: Optional[int] = None,
    ):
        """Save the rollouts to the checkpoint directory.

        Parameters
        ----------
        rollouts : NestedArrayDict
            The rollouts to save.
        environment : PureTextEnvironment
            The environment the rollouts were sampled in.
        iteration : int, optional
            The iteration number. If not provided, the current iteration number is used.
        """

        if iteration is None:
            iteration = self.state.iteration

        if environment.split == "train":
            base_name = f"{iteration}"
        elif environment.split == "validation":
            base_name = f"validation_{iteration}"
        else:
            base_name = f"test_{iteration}"

        if self.hyper_params.text_rl.save_transcripts:
            raw_transcripts, processed_transcripts, prompts = (
                self._extract_transcripts_and_prompts(rollouts, environment)
            )

        # If we are running a test run, we don't want to save the rollouts
        if self.settings.test_run:
            return

        self.checkpoint_rollouts_dir.mkdir(parents=True, exist_ok=True)

        rollout_path = self.checkpoint_rollouts_dir.joinpath(f"{base_name}.pt")

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
            self.prompts_dir.mkdir(parents=True, exist_ok=True)

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
                f"raw_{base_name}.{file_extension}"
            )
            processed_transcript_path = self.processed_transcripts_dir.joinpath(
                f"processed_{base_name}.{file_extension}"
            )
            prompts_path = self.prompts_dir.joinpath(
                f"prompts_{base_name}.{file_extension}"
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

            with open(prompts_path, "w") as f:
                if self.hyper_params.text_rl.transcript_format == "yaml":
                    yaml.dump(prompts, f)
                elif self.hyper_params.text_rl.transcript_format == "json":
                    json.dump(prompts, f, indent=4)

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
                self._add_file_to_wandb_artifact(
                    f"{PROMPTS_ARTIFACT_PREFIX}{self.settings.wandb_run.name}",
                    PROMPTS_ARTIFACT_TYPE,
                    prompts_path,
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
        else:
            iterations = list(iterations)

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
            except wandb.errors.CommError as e:
                # W&B doesn't use subclasses for errors, so we have to check the
                # message. If the error was not that the artifact was not found, we
                # re-raise it.
                if f"artifact '{artifact_name}' not found in" not in e.message:
                    raise e
            else:
                artifact.download(self.checkpoint_rollouts_dir)

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
        train : bool, default=True
            Whether the rollouts are from the training environment.

        Returns
        -------
        stats : dict
            The statistics to log.
        """

        if train:
            prefix = ""
        else:
            prefix = f"{self.test_environment.split}_"

        reward: Float[np.ndarray, "rollout round agent"] = rollouts[
            "next", "agents", "reward"
        ]
        done: Bool[np.ndarray, "rollout round"] = rollouts["done"]
        next_done: Bool[np.ndarray, "rollout round"] = rollouts["next", "done"]
        next_terminated: Bool[np.ndarray, "rollout round"] = rollouts[
            "next", "terminated"
        ]
        padding: Bool[np.ndarray, "rollout round"] = rollouts["padding"]
        datapoint_id: Int[np.ndarray, "rollout"] = rollouts["datapoint_id"][..., 0]
        verifier_decision = rollouts["agents", "decision"][
            ..., self.agent_names.index("verifier")
        ]
        verifier_continuous_decision = rollouts["agents", "continuous_decision"][
            ..., self.agent_names.index("verifier")
        ]

        last_timestep = (next_done | next_terminated) & ~padding

        log_stats = {}

        for agent_index, agent_name in enumerate(self.agent_names):

            # Get the total episode reward for each agent
            episode_reward = reward[..., agent_index].sum(axis=-1)
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
        accuracy = verifier_decision[last_timestep] == rollouts["y"][last_timestep]
        log_stats[f"{prefix}mean_accuracy"] = accuracy.mean().item()
        log_stats[f"{prefix}std_accuracy"] = accuracy.std().item()

        # Get the min mean accuracy over the datapoints.
        mean_accuracy_per_datapoint = aggregate_mean_grouped_by_class(
            accuracy.astype(float), datapoint_id
        )
        mean_accuracy_per_datapoint = mean_accuracy_per_datapoint[
            ~np.isnan(mean_accuracy_per_datapoint)
        ]
        log_stats[f"{prefix}worst_datapoint_accuracy"] = (
            mean_accuracy_per_datapoint.min().item()
        )

        # Get the mean and std accuracy of the verifier by class
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
        verifier_last_decision = verifier_decision[last_timestep][
            verifier_decision[last_timestep] != 2
        ]
        log_stats[f"{prefix}mean_decision"] = verifier_last_decision.mean().item()
        log_stats[f"{prefix}std_decision"] = verifier_last_decision.std().item()

        # Get the proportion of rollouts where the verifier does not make a decision
        log_stats[f"{prefix}no_decision_proportion"] = (
            (verifier_decision[last_timestep] == 2).mean().item()
        )

        # Get the proportion of rollouts where the verifier decides to neither accept
        # nor reject
        log_stats[f"{prefix}neither_agree_nor_disagree_proportion"] = (
            (verifier_decision[last_timestep] == 3).mean().item()
        )

        # Get Shannon entropy of the verifier decision
        log_stats[f"{prefix}verifier_decision_entropy"] = entropy_numpy(
            verifier_continuous_decision
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

    def _extract_transcripts_and_prompts(
        self, rollouts: NestedArrayDict, environment: PureTextEnvironment
    ) -> tuple[list[dict], list[dict], list[list[dict[str, list[PromptMessage]]]]]:
        """Extract the raw and processed transcripts, and prompts, from the rollouts.

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

            - ("agents", "message") (batch round agent channel) : The processed message
              sent by each agent to each channel in each timestep.
            - ("agents", "raw_message") (batch round agent) : The raw message generated
              by each model in each timestep.
            - ("agents", "prompt") (batch round agent message field) : The prompt used
              by to generate the message for each agent in each timestep.
            - ("agents", "decision") (batch round agent) : The decision made by each
              agent in each timestep.
            - ("agents", "continuous_decision") (batch round agent) : A float version of
              the decision made by each agent at each timestep, which is a value between
              -1 and 1.
            - ("agents", "raw_decision") (batch round agent) : The raw decision text
              sent by each agent in each timestep.
            - ("agents", "reward") (batch round agent) : The reward received by each
              agent in each timestep.

            The nested array dict also contains keys which specify the datapoint for
            each rollout, as extracted by
            ``environment.get_datapoint_from_env_state_as_dict``.

        environment : PureTextEnvironment
            The environment the rollouts were sampled in.

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
            ``f"{active_agent_name}@{channel_name}"`` and values are the messages in
            each channel.
        prompts : list[list[dict[str, list[PromptMessage]]]]
            The prompts used to generate the messages at each timestep. This is a list
            containing for each batch item a list of dictionaries, one for each round.
            Each dictionary has the agent names as keys and the prompts used by the
            agents the as values. The prompts are a list of dictionaries, whose type is
            specified by the ``PromptMessage`` class.
        """

        message: String[NDArray, "batch round agent channel"] = rollouts[
            "agents", "message"
        ]
        raw_message: String[NDArray, "batch round agent"] = rollouts[
            "agents", "raw_message"
        ]
        prompt: String[NDArray, "batch round agent message field"] = rollouts[
            "agents", "prompt"
        ]
        decision: Int[NDArray, "batch round agent"] = rollouts["agents", "decision"]
        continuous_decision: Float[NDArray, "batch round agent"] = rollouts[
            "agents", "continuous_decision"
        ]
        raw_decision: String[NDArray, "batch round agent"] = rollouts[
            "agents", "raw_decision"
        ]
        reward = reduce(
            rollouts["next", "agents", "reward"],
            "batch round agent -> batch agent",
            "sum",
        )
        num_rollouts = rollouts.batch_size[0]

        agent_names = self.protocol_handler.agent_names

        raw_transcripts = []
        processed_transcripts = []
        prompts = []

        for rollout_id in range(num_rollouts):

            raw_transcript = []
            processed_transcript = []
            prompts_by_round = []

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

                processed_transcript_round = get_pretty_pure_text_round_message(
                    protocol_handler=self.protocol_handler,
                    decision=decision[rollout_id, round_id],
                    raw_decision=raw_decision[rollout_id, round_id],
                    continuous_decision=continuous_decision[rollout_id, round_id],
                    message=message[rollout_id, round_id],
                )

                if processed_transcript_round:
                    processed_transcript.append(processed_transcript_round)

                # Get the prompts used by each agent in this round. If this is empty for
                # an agent, it means the agent did not message in this round.
                prompt_round = {}
                for agent_id, agent_name in enumerate(agent_names):
                    agent_prompt = environment.prompt_array_to_list(
                        prompt[rollout_id, round_id, agent_id]
                    )
                    if len(agent_prompt) > 0:
                        prompt_round[agent_name] = agent_prompt
                prompts_by_round.append(prompt_round)

            metadata = self.train_environment.get_datapoint_from_env_state_as_dict(
                rollouts[rollout_id, 0]
            )
            metadata["reward"] = {
                agent_name: reward[rollout_id, agent_id].item()
                for agent_id, agent_name in enumerate(agent_names)
            }

            raw_transcripts.append(dict(transcript=raw_transcript, **metadata))
            processed_transcripts.append(
                dict(transcript=processed_transcript, **metadata)
            )
            prompts.append(prompts_by_round)

        return raw_transcripts, processed_transcripts, prompts

    def _get_fine_tune_job_name(
        self,
        shared_model_group: PureTextSharedModelGroup,
    ) -> str:
        """Get a name for the fine-tune job for the given shared model group.

        This name is generated from the run id, the iteration number, and the shared
        model group name, and is used make the job more easily identifiable.

        Parameters
        ----------
        shared_model_group : PureTextSharedModelGroup
            The shared model group to create the fine-tune job for.

        Returns
        -------
        job_name : str
            The name of the fine-tune job.
        """

        if self.settings.run_id is not None:
            job_name = self.settings.run_id
        else:
            job_name = "NO_RUN_ID"
        job_name += f"_iter_{self.state.iteration}_{shared_model_group.group_name}"

        return job_name
