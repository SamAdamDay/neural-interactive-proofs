"""Stackelberg Policy Gradient :cite:p:`Fiez2020` RL trainer."""

from torchrl.objectives.value import GAE
from torchrl.objectives import ValueEstimators

from nip.rl_objectives import SpgLoss
from nip.trainers.rl_tensordict_base import TensorDictRlTrainer
from nip.trainers.registry import register_trainer


@register_trainer("spg")
class SpgTrainer(TensorDictRlTrainer):
    """Stackelberg Policy Gradient :cite:p:`Fiez2020` trainer.

    Implements an n-player version of Stackelberg Policy Gradient / Opponent-Shaping

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    device : TorchDevice
        The device to use for training.
    """

    def _get_loss_module_and_gae(self) -> tuple[SpgLoss, GAE]:
        """Construct the loss module and the generalized advantage estimator.

        Returns
        -------
        loss_module : SpgLoss
            The loss module.
        gae : GAE
            The generalized advantage estimator.
        """

        # Construct the loss module
        loss_module = SpgLoss(
            actor=self.policy_operator,
            critic=self.value_operator,
            variant=self.hyper_params.spg.variant,
            stackelberg_sequence=self.protocol_handler.stackelberg_sequence,
            agent_names=self.agent_names,
            agents=self.scenario_instance.agents,
            ihvp_arguments={
                "variant": self.hyper_params.spg.ihvp_variant,
                "num_iterations": self.hyper_params.spg.ihvp_num_iterations,
                "rank": self.hyper_params.spg.ihvp_rank,
                "rho": self.hyper_params.spg.ihvp_rho,
            },
            additional_lola_term=self.hyper_params.spg.additional_lola_term,
            sos_scaling_factor=self.hyper_params.spg.sos_scaling_factor,
            sos_threshold_factor=self.hyper_params.spg.sos_threshold_factor,
            agent_lr_factors={
                name: self.hyper_params.agents[name].agent_lr_factor
                for name in self.protocol_handler.agent_names
            },
            lr=self.hyper_params.rl.lr,
            clip_epsilon=self.hyper_params.ppo.clip_epsilon,
            entropy_coef=self.hyper_params.ppo.entropy_eps,
            normalize_advantage=self.hyper_params.ppo.normalize_advantage,
            functional=self.hyper_params.functionalize_modules,
            loss_critic_type=self.hyper_params.rl.loss_critic_type,
            clip_value=self.clip_value,
            device=self.device,
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
