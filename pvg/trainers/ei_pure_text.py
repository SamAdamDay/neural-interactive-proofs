"""Expert Iteration (EI) trainer for text-based environments which only use APIs."""

import numpy as np

from pvg.trainers.rl_pure_text_base import PureTextRlTrainer
from pvg.trainers.registry import register_trainer
from pvg.parameters import TrainerType
from pvg.utils.nested_array_dict import NestedArrayDict


@register_trainer(TrainerType.PURE_TEXT_EI)
class PureTextEiTrainer(PureTextRlTrainer):
    """Expert Iteration (EI) trainer for text-based environments which only use APIs.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    scenario_instance : ScenarioInstance
        The components of the experiment.
    settings : ExperimentSettings
        The instance-specific settings of the experiment, like device, logging, etc.
    """

    def _stage_create_fine_tune_jobs(self, rollouts: NestedArrayDict):
        """Training stage: create fine-tune jobs for each agent.

        Parameters
        ----------
        rollouts : NestedArrayDict, optional
            The rollouts sampled in this iteration.
        """

        for agent_name, agent_whole in self.agent_wholes.items():

            # Select the rollouts to fine-tune on
            selected_rollouts = self._select_rollouts_for_fine_tuning(
                rollouts, agent_name
            )

            # Create a fine-tune job for these rollouts
            self.settings.logger.info(f"Creating fine-tune job for {agent_name!r}")
            agent_whole.create_fine_tune_job(selected_rollouts)

    def _select_rollouts_for_fine_tuning(
        self, rollouts: NestedArrayDict, agent_name: str
    ) -> NestedArrayDict:
        """Select rollouts to fine-tune on, based on the reward.

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
        agent_episode_reward = rollouts["next", "agents", "reward"][
            ..., agent_index
        ].sum(axis=-1)

        num_rollouts = agent_episode_reward.shape[0]

        if self.hyper_params.pure_text_ei.rollout_selection_method == "threshold":

            # Select the rollouts with a high reward for the given agent
            good_mask = (
                agent_episode_reward >= self.hyper_params.pure_text_ei.reward_threshold
            )
            return rollouts[good_mask]

        elif (
            self.hyper_params.pure_text_ei.rollout_selection_method
            == "weighted_sampling"
        ):

            # Compute the weights by normalizing the rewards
            if self.hyper_params.pure_text_ei.weighting_minimum is not None:
                weights = np.maximum(
                    agent_episode_reward,
                    self.hyper_params.pure_text_ei.weighting_minimum,
                )
                weights = weights - self.hyper_params.pure_text_ei.weighting_minimum
                weights = weights / weights.sum()
            else:
                weights = agent_episode_reward - agent_episode_reward.min()
                weights = weights / weights.sum()

            # Add a small constant to the weights to avoid zero weights
            weights += self.hyper_params.pure_text_ei.weighting_epsilon / num_rollouts
            weights /= weights.sum()

            sample_size = round(
                num_rollouts
                * self.hyper_params.pure_text_ei.weighting_sample_size_factor
            )

            index = np.random.choice(
                num_rollouts,
                size=sample_size,
                p=weights,
                replace=self.hyper_params.pure_text_ei.weighting_use_replacement,
            )

            return rollouts[index]

        else:
            raise ValueError(
                f"Unknown rollout selection method: "
                f"{self.hyper_params.pure_text_ei.rollout_selection_method!r}"
            )
