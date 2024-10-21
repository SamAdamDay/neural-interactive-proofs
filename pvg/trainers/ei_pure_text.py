"""Expert Iteration (EI) trainer for text-based environments which only use APIs."""

from dataclasses import dataclass
from typing import Literal, Optional
from time import sleep
import pickle

import numpy as np

from jaxtyping import Bool

from pvg.trainers.rl_pure_text_base import PureTextRlTrainer
from pvg.trainers.registry import register_trainer
from pvg.parameters import TrainerType
from pvg.scenario_base.rollout_analysis import (
    PureTextRolloutAnalyser,
    ROLLOUT_ANALYSERS,
)
from pvg.utils.nested_array_dict import NestedArrayDict


@register_trainer(TrainerType.PURE_TEXT_EI)
class PureTextEiTrainer(PureTextRlTrainer):
    """Expert Iteration (EI) trainer for text-based environments which only use APIs.

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
    class State(PureTextRlTrainer.State):
        """The state of the experiment."""

        train_loop_stage: Literal[
            "sample_rollouts",
            "log_stats",
            "create_fine_tune_jobs",
            "await_fine_tune_jobs",
            "test",
            "done",
        ] = "sample_rollouts"

    state: State

    def train(self):

        rollouts: Optional[NestedArrayDict] = None

        while self.state.iteration < self.params.rl.num_iterations:

            self.settings.logger.info(
                f"[{self.state.iteration+1}/{self.params.rl.num_iterations}] Iteration "
                f"begins."
            )

            # Sample rollouts from the training environment
            if self.state.train_loop_stage == "sample_rollouts":

                # Sample rollouts
                rollouts = self.sample_rollouts(
                    self.train_environment, use_tqdm=not self.settings.test_run
                )

                # Save the rollouts to the checkpoint directory
                self.save_rollouts(rollouts, self.state.iteration)

                # Advance to the next stage
                self.state.train_loop_stage = "log_stats"

                self.save_checkpoint()

            # Log the statistics of the rollouts
            if self.state.train_loop_stage == "log_stats":

                # Load the rollouts if they are not already set (i.e. if we are resuming
                # this stage)
                if rollouts is None:
                    rollouts = self.load_rollouts(self.state.iteration)

                log_stats = self._get_log_stats(rollouts, train=True)
                self.settings.stat_logger.log(log_stats, self.state.iteration)

                # Advance to the next stage
                self.state.train_loop_stage = "create_fine_tune_jobs"

                self.save_checkpoint()

            # We don't fine-tune on the last iteration
            if self.state.iteration == self.params.rl.num_iterations - 1:

                # Advance to the test stage
                self.state.iteration = self.params.rl.num_iterations
                self.state.train_loop_stage = "test"

                self.save_checkpoint()

                break

            # Create fine-tune jobs for each agent
            if self.state.train_loop_stage == "create_fine_tune_jobs":

                # Load the rollouts if they are not already set (i.e. if we are resuming
                # this stage)
                if rollouts is None:
                    rollouts = self.load_rollouts(self.state.iteration)

                for agent_name, agent_whole in self.agent_wholes.items():

                    # Select the rollouts with a high reward for the given agent
                    selected_rollouts = self._select_good_rollouts(rollouts, agent_name)

                    # Create a fine-tune job for these rollouts
                    self.settings.logger.info(
                        f"Creating fine-tune job for {agent_name!r}"
                    )
                    agent_whole.create_fine_tune_job(selected_rollouts)

                # Advance to the next stage
                self.state.train_loop_stage = "await_fine_tune_jobs"

                self.save_checkpoint()

            # Await the completion of the fine-tune jobs
            if self.state.train_loop_stage == "await_fine_tune_jobs":

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

                # Advance to the next iteration and stage
                self.state.train_loop_stage = "sample_rollouts"
                self.state.iteration += 1

                self.save_checkpoint()

        if self.state.train_loop_stage == "test":

            if self.params.pure_text_ei.run_test_loop:

                # Sample rollouts from the test environment
                rollouts = self.sample_rollouts(
                    self.test_environment, use_tqdm=True, tqdm_desc="Testing"
                )

                # Log the statistics of the rollouts
                log_stats = self._get_log_stats(rollouts, train=False)
                self.settings.stat_logger.log(log_stats)

                # Save the rollouts to the checkpoint directory
                self.save_rollouts(rollouts, "test")

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
                        self.params.scenario, analyser_cls
                    ]
                except KeyError:
                    raise ValueError(
                        f"Analyser {analyser_cls!r} not found in list of analysers."
                    )

            analyser = analyser_cls(
                params=self.params,
                settings=self.settings,
                protocol_handler=self.scenario_instance.protocol_handler,
                model_name=model_name,
                use_dummy_api=dry_run,
            )

            analysis_dir = self.checkpoint_analysis_dir.joinpath(analyser_cls.name)
            analysis_dir.mkdir(parents=True, exist_ok=True)

            for iteration in range(self.params.rl.num_iterations):

                print(  # noqa: T201
                    f"Running analyser {analyser_cls.name!r} on iteration "
                    f"{iteration+1}/{self.params.rl.num_iterations}"
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
                    rollouts = self.load_rollouts(iteration)
                except FileNotFoundError:
                    self.settings.logger.warning(
                        f"No rollouts found for iteration {iteration+1}. Skipping."
                    )
                    continue

                evaluations = analyser.forward(rollouts, use_tqdm=use_tqdm)

                if not dry_run:
                    with open(analysis_file, "wb") as f:
                        pickle.dump(evaluations, f)

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

        log_stats = {}

        for agent_index, agent_name in enumerate(self.agent_names):

            # Get the total episode reward for each agent
            log_stats[f"{agent_name}.{prefix}mean_episode_reward"] = (
                rollouts["next", "agents", "reward"][..., agent_index]
                .sum(axis=-1)
                .mean()
                .item()
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

        log_stats[f"{prefix}mean_episode_length"] = (
            (rollouts["message_history"][..., -1, :, 0] != None)  # noqa: E711
            .sum(axis=-1)
            .mean()
            .item()
        )

        # Get the mean accuracy of the verifier
        verifier_decision = rollouts["agents", "decision"][
            ..., self.agent_names.index("verifier")
        ]
        log_stats[f"{prefix}mean_accuracy"] = (
            (verifier_decision[next_done] == rollouts["y"][next_done]).mean().item()
        )

        return log_stats

    def _select_good_rollouts(
        self, rollouts: NestedArrayDict, agent_name: str
    ) -> NestedArrayDict:
        """Select the rollouts with a high reward for the given agent, for fine-tuning.

        Parameters
        ----------
        rollouts : NestedArrayDict
            The rollouts to select from.
        agent_name : str
            The name of the agent for which to select the rollouts.

        Returns
        -------
        selected_rollouts : NestedArrayDict
            The selected rollouts.
        """

        agent_index = self.agent_names.index(agent_name)

        # Select the rollouts with a high reward for the given agent
        good_mask = (
            rollouts["next", "agents", "reward"][..., agent_index].sum(axis=-1)
            >= self.params.pure_text_ei.reward_threshold
        )
        return rollouts[good_mask]
