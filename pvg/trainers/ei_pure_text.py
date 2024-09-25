"""Expert Iteration (EI) trainer for text-based environments which only use APIs."""

from dataclasses import dataclass
from typing import Literal, Optional
from time import sleep

from tqdm import tqdm

from pvg.trainers.rl_pure_text_base import PureTextRlTrainer
from pvg.trainers.registry import register_trainer
from pvg.parameters import TrainerType
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
            "sample_rollouts", "create_fine_tune_jobs", "await_fine_tune_jobs"
        ] = "sample_rollouts"

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
                self.state.train_loop_stage = "create_fine_tune_jobs"

                self.save_checkpoint()

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

        agent_index = list(self.scenario_instance.agents.keys()).index(agent_name)

        # Select the rollouts with a high reward for the given agent
        good_mask = (
            rollouts["next", "agents", "reward"][..., :, agent_index].sum(axis=-1)
            >= self.params.ei.reward_threshold
        )
        return rollouts[good_mask]
