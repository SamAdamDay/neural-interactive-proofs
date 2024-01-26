"""Stackelberg Policy Gradient RL trainer."""

import torch

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.objectives import LossModule
from torchrl.objectives.value import GAE
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import ProbabilisticActor, ActorValueOperator
from torchrl.objectives import ValueEstimators

from pvg.utils.torchrl_objectives import SpgLoss
from pvg.trainers.base import ReinforcementLearningTrainer
from pvg.utils.distributions import CompositeCategoricalDistribution


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

    def _train_setup(self):
        """Some setup before the training loop"""

        # Get some components, for convenient access
        combined_policy_head = self.scenario_instance.combined_policy_head
        body = self.scenario_instance.combined_body
        value_head = self.scenario_instance.combined_value_head

        # Create the policy head, which samples actions from the policy probability
        # distribution
        policy_head = ProbabilisticActor(
            combined_policy_head,
            spec=self.environment.action_spec,
            distribution_class=CompositeCategoricalDistribution,
            distribution_kwargs=dict(
                key_transform=lambda x: ("agents", x),
                log_prob_key=("agents", "sample_log_prob"),
            ),
            in_keys={out_key[1]: out_key for out_key in combined_policy_head.out_keys},
            out_keys=self.environment.action_keys,
            return_log_prob=True,
            log_prob_key=("agents", "sample_log_prob"),
        )

        # Combine the body, policy head and value head into a single model, and get the
        # full policy operator
        self._full_model = ActorValueOperator(body, policy_head, value_head)
        self._policy = self._full_model.get_policy_operator()

    def _get_data_collector(self) -> SyncDataCollector:
        """Construct the data collector, which generates rollouts from the environment

        Returns
        -------
        collector : SyncDataCollector
            The data collector.
        """
        return SyncDataCollector(
            self.environment,
            self._policy,
            device=self.device,
            storing_device=self.device,
            frames_per_batch=self.params.spg.frames_per_batch,
            total_frames=self.params.spg.frames_per_batch
            * self.params.spg.num_iterations,
        )

    def _get_replay_buffer(self) -> ReplayBuffer:
        """Construct the replay buffer, which will store the rollouts

        Returns
        -------
        ReplayBuffer
            The replay buffer.
        """
        return ReplayBuffer(
            storage=LazyTensorStorage(
                self.params.spg.frames_per_batch, device=self.device
            ),
            sampler=SamplerWithoutReplacement(),
            batch_size=self.params.spg.minibatch_size,
        )

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
        loss_module = SpgLoss(
            actor=self._full_model.get_policy_operator(),
            critic=self._full_model.get_value_operator(),
            variant=self.params.spg.variant,
            stackelberg_sequence=self.params.spg.stackelberg_sequence,
            names=self.params.spg.names,
            ihvp={
                "variant": self.params.spg.ihvp_variant,
                "num_iterations": self.params.spg.ihvp_num_iterations,
                "rank": self.params.spg.ihvp_rank,
                "rho": self.params.spg.ihvp_rho,
            },
            clip_epsilon=self.params.spg.clip_epsilon,
            entropy_coef=self.params.spg.entropy_eps,
            normalize_advantage=False,
        )
        loss_module.set_keys(
            reward=self.environment.reward_key,
            action=self.environment.action_keys,
            sample_log_prob=("agents", "sample_log_prob"),
            value=("agents", "value"),
            done=("agents", "done"),
            terminated=("agents", "terminated"),
        )

        # Make the generalized advantage estimator
        loss_module.make_value_estimator(
            ValueEstimators.GAE,
            gamma=self.params.spg.gamma,
            lmbda=self.params.spg.lmbda,
        )
        gae = loss_module.value_estimator

        return loss_module, gae

    def _get_optimizer(self, loss_module: LossModule) -> torch.optim.Adam:
        """Construct the optimizer for the loss module

        Parameters
        ----------
        loss_module : LossModule
            The loss module.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer.
        """
        return torch.optim.Adam(loss_module.parameters(), self.params.spg.lr)
