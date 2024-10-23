"""Stackelberg Policy Gradient RL trainer."""

from torchrl.objectives.value import GAE
from torchrl.objectives import ValueEstimators

from pvg.rl_objectives import SpgLoss
from pvg.trainers.rl_tensordict_base import ReinforcementLearningTrainer
from pvg.trainers.registry import register_trainer
from pvg.parameters import TrainerType


@register_trainer(TrainerType.SPG)
class SpgTrainer(ReinforcementLearningTrainer):
    """Stackelberg Policy Gradient trainer

    Implements an n-player version of Stackelberg Policy Gradient / Opponent-Shaping

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    device : TorchDevice
        The device to use for training.
    """

    def _get_loss_module_and_gae(self) -> tuple[SpgLoss, GAE]:
        """Construct the loss module and the generalized advantage estimator

        Returns
        -------
        loss_module : SpgLoss
            The loss module.
        gae : GAE
            The generalized advantage estimator.
        """

        # Construct the loss module
        stackelberg_sequence_int = [
            tuple(self.agent_names.index(name) for name in group)
            for group in self.hyper_params.spg.stackelberg_sequence
        ]
        loss_module = SpgLoss(
            actor=self.policy_operator,
            critic=self.value_operator,
            variant=self.hyper_params.spg.variant,
            stackelberg_sequence=stackelberg_sequence_int,
            names=self.agent_names,
            ihvp={
                "variant": self.hyper_params.spg.ihvp_variant,
                "num_iterations": self.hyper_params.spg.ihvp_num_iterations,
                "rank": self.hyper_params.spg.ihvp_rank,
                "rho": self.hyper_params.spg.ihvp_rho,
            },
            additional_lola_term=self.hyper_params.spg.additional_lola_term,
            sos_params=self.hyper_params.spg.sos_params,
            agent_lr_factors=[
                agent_params.agent_lr_factor
                for agent_params in self.hyper_params.agents.values()
            ],
            lr=self.hyper_params.rl.lr,
            clip_epsilon=self.hyper_params.ppo.clip_epsilon,
            entropy_coef=self.hyper_params.ppo.entropy_eps,
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
