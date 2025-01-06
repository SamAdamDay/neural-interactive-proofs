"""REINFORCE policy gradient RL trainer."""

from typing import Optional

from torchrl.objectives.value import GAE
from torchrl.objectives import ValueEstimators
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs.transforms import Transform, Compose as ComposeTransform

from pvg.utils.bugfix import Reward2GoTransform

from pvg.trainers.rl_tensordict_base import ReinforcementLearningTrainer
from pvg.trainers.registry import register_trainer
from pvg.parameters import TrainerType
from pvg.rl_objectives import ReinforceLossImproved


@register_trainer(TrainerType.REINFORCE)
class ReinforceTrainer(ReinforcementLearningTrainer):
    """Policy gradient trainer using the REINFORCE algorithm.

    Can use the generalized advantage estimator and a critic network, or just the
    reward-to-go.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    scenario_instance : ScenarioInstance
        The components of the experiment.
    device : TorchDevice
        The device to use for training.
    """

    def _get_replay_buffer(self, transform: Optional[Transform] = None) -> ReplayBuffer:
        """Construct the replay buffer, which will store the rollouts.

        When not using the advantage, the reward-to-go transform is added to the list of
        transforms.

        Parameters
        ----------
        transform : Transform, optional
            The transform to apply to the data before storing it in the replay buffer,
            if any. The reward-to-go transform is added after this.

        Returns
        -------
        ReplayBuffer
            The replay buffer.
        """

        # Add the reward-to-go transform if not using the advantage
        if not self.hyper_params.reinforce.use_advantage_and_critic:
            if transform is None:
                transforms = []
            else:
                transforms = [transform]
            transforms.append(
                Reward2GoTransform(
                    gamma=self.hyper_params.rl.gamma,
                    done_key=("agents", "done"),
                    in_keys=("next", "agents", "reward"),
                    out_keys=("agents", "reward_to_go"),
                )
            )
            transform = ComposeTransform(*transforms)

        return super()._get_replay_buffer(transform=transform)

    def _get_loss_module_and_gae(self) -> tuple[ReinforceLossImproved, GAE | None]:
        """Construct the loss module and the generalized advantage estimator.

        Returns
        -------
        loss_module : ReinforceLossImproved
            The loss module.
        gae : GAE | None
            The generalized advantage estimator, if using advantage and critic, None
            otherwise.
        """

        if self.hyper_params.reinforce.use_advantage_and_critic:
            loss_weighting_type = "advantage"
        else:
            loss_weighting_type = "reward_to_go"
        loss_module = ReinforceLossImproved(
            actor_network=self.policy_operator,
            critic_network=self.value_operator,
            loss_weighting_type=loss_weighting_type,
            gamma=self.hyper_params.rl.gamma,
            functional=self.hyper_params.functionalize_modules,
            loss_critic_type=self.hyper_params.rl.loss_critic_type,
            clip_value=self.clip_value,
        )
        loss_module.set_keys(
            reward=self.train_environment.reward_key,
            action=self.train_environment.action_keys,
            sample_log_prob=("agents", "sample_log_prob"),
            value=("agents", "value"),
            done=("agents", "done"),
            terminated=("agents", "terminated"),
        )

        # Make the generalized advantage estimator
        if self.hyper_params.reinforce.use_advantage_and_critic:
            loss_module.make_value_estimator(
                ValueEstimators.GAE,
                gamma=self.hyper_params.rl.gamma,
                lmbda=self.hyper_params.rl.lmbda,
            )
            gae = loss_module.value_estimator
        else:
            gae = None

        return loss_module, gae
