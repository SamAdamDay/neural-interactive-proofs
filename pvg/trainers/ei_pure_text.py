"""Expert Iteration (EI) trainer for text-based environments which only use APIs."""

from dataclasses import dataclass
from typing import Literal, Optional
from time import sleep

import numpy as np
from numpy import ma

from jaxtyping import Bool

from pvg.trainers.rl_pure_text_base import PureTextRlTrainer
from pvg.trainers.registry import register_trainer
from pvg.parameters import TrainerType
from pvg.utils.nested_array_dict import NestedArrayDict

# TODO: Abstract this
from pvg.code_validation.prover_watchdog import CodeValidationProverWatchdog


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
        ] = "sample_rollouts"

    def __init__(self, params, scenario_instance, settings):
        super().__init__(params, scenario_instance, settings)

        if self.params.ei.use_prover_watchdog:
            self.prover_watchdog = CodeValidationProverWatchdog(
                params=self.params,
                settings=self.settings,
                protocol_handler=self.scenario_instance.protocol_handler,
            )

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
                rollouts = self.sample_rollouts(self.train_environment, use_tqdm=True)

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

            # We don't fine-tune on the last iteration
            if self.state.iteration == self.params.rl.num_iterations - 1:
                self.state.iteration = self.params.rl.num_iterations
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

        self.settings.logger.info("Training complete.")

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

        # If the prover watchdog is enabled, run it to get the evaluations
        if self.params.ei.use_prover_watchdog:
            watchdog_evaluations = self.prover_watchdog.forward(rollouts, use_tqdm=True)

            for (agent_name, channel_name), evaluation in watchdog_evaluations.items():
                log_stats[f"{agent_name}.{channel_name}.{prefix}mean_watchdog_eval"] = (
                    evaluation.mean().item()
                )

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
            (rollouts["message_history"][..., -1, :, 0] != None)
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
            >= self.params.ei.reward_threshold
        )
        return rollouts[good_mask]
