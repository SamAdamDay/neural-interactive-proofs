"""Stackelberg Policy Gradient RL trainer."""

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.objectives.value import GAE
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ValueEstimators

from pvg.rl_objectives import SpgLoss
from pvg.trainers.rl_trainer_base import ReinforcementLearningTrainer
from pvg.trainers.registry import register_trainer
from pvg.parameters import TrainerType


@register_trainer(TrainerType.SPG)
class SpgTrainer(ReinforcementLearningTrainer):
    """Stackelberg Policy Gradient trainer

    Implements an n-player version of Stackelberg Policy Gradient / Opponent-Shaping

    Parameters
    ----------
    params : Parameters
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
            tuple(self._agent_names.index(name) for name in group)
            for group in self.params.spg.stackelberg_sequence
        ]
        loss_module = SpgLoss(
            actor=self.policy_operator,
            critic=self.value_operator,
            variant=self.params.spg.variant,
            stackelberg_sequence=stackelberg_sequence_int,
            names=self._agent_names,
            ihvp={
                "variant": self.params.spg.ihvp_variant,
                "num_iterations": self.params.spg.ihvp_num_iterations,
                "rank": self.params.spg.ihvp_rank,
                "rho": self.params.spg.ihvp_rho,
            },
            additional_lola_term=self.params.spg.additional_lola_term,
            sos_params=self.params.spg.sos_params,
            agent_lr_factors=[
                agent_params.agent_lr_factor
                for agent_params in self.params.agents.values()
            ],
            lr=self.params.rl.lr,
            clip_epsilon=self.params.ppo.clip_epsilon,
            entropy_coef=self.params.ppo.entropy_eps,
            normalize_advantage=self.params.ppo.normalize_advantage,
            functional=self.params.functionalize_modules,
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
            gamma=self.params.rl.gamma,
            lmbda=self.params.rl.lmbda,
        )
        gae = loss_module.value_estimator

        return loss_module, gae
