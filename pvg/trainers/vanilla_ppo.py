"""Vanilla PPO RL trainer."""

from torchrl.objectives.value import GAE
from torchrl.objectives import ValueEstimators

from pvg.rl_objectives import ClipPPOLossImproved, KLPENPPOLossImproved
from pvg.trainers.rl_tensordict_base import ReinforcementLearningTrainer
from pvg.trainers.registry import register_trainer
from pvg.parameters import TrainerType, PpoLossType


@register_trainer(TrainerType.VANILLA_PPO)
class VanillaPpoTrainer(ReinforcementLearningTrainer):
    """Vanilla Proximal Policy Optimization trainer.

    Implements a multi-agent PPO algorithm, specifically IPPO, since the value estimator
    is not shared between agents.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    scenario_instance : ScenarioInstance
        The components of the experiment.
    device : TorchDevice
        The device to use for training.
    """

    def _get_loss_module_and_gae(self) -> tuple[ClipPPOLossImproved, GAE]:
        """Construct the loss module and the generalized advantage estimator

        Returns
        -------
        loss_module : ClipPPOLossMultipleActions
            The loss module.
        gae : GAE
            The generalized advantage estimator.
        """

        # Construct the loss module
        if self.hyper_params.ppo.loss_type == PpoLossType.CLIP:
            loss_module = ClipPPOLossImproved(
                actor=self.policy_operator,
                critic=self.value_operator,
                clip_epsilon=self.hyper_params.ppo.clip_epsilon,
                entropy_coef=self.hyper_params.ppo.entropy_eps,
                critic_coef=self.hyper_params.ppo.critic_coef,
                normalize_advantage=self.hyper_params.ppo.normalize_advantage,
                functional=self.hyper_params.functionalize_modules,
                loss_critic_type=self.hyper_params.rl.loss_critic_type,
                clip_value=self.clip_value,
            )
        elif self.hyper_params.ppo.loss_type == PpoLossType.KL_PENALTY:
            loss_module = KLPENPPOLossImproved(
                actor=self.policy_operator,
                critic=self.value_operator,
                dtarg=self.hyper_params.ppo.kl_target,
                beta=self.hyper_params.ppo.kl_beta,
                decrement=self.hyper_params.ppo.kl_decrement,
                increment=self.hyper_params.ppo.kl_increment,
                entropy_coef=self.hyper_params.ppo.entropy_eps,
                critic_coef=self.hyper_params.ppo.critic_coef,
                normalize_advantage=self.hyper_params.ppo.normalize_advantage,
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
        loss_module.make_value_estimator(
            ValueEstimators.GAE,
            gamma=self.hyper_params.rl.gamma,
            lmbda=self.hyper_params.rl.lmbda,
        )
        gae = loss_module.value_estimator

        return loss_module, gae
