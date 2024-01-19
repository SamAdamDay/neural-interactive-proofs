"""A generic reinforcement learning trainer."""

from abc import ABC, abstractmethod

import numpy as np

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from tensordict import TensorDict

from torchrl.collectors import DataCollectorBase
from torchrl.objectives import LossModule
from torchrl.objectives.value import GAE
from torchrl.data.replay_buffers import ReplayBuffer

from pvg.parameters import Parameters
from pvg.scenario_base import ScenarioInstance, Environment, RolloutSampler
from pvg.experiment_settings import ExperimentSettings
from pvg.trainers.base import Trainer
from pvg.trainers.solo_agent import SoloAgentTrainer


class ReinforcementLearningTrainer(Trainer, ABC):
    """Base class for all reinforcement learning trainers.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    scenario_instance : ScenarioInstance
        The components of the experiment.
    device : TorchDevice
        The device to use for training.
    """

    def __init__(
        self,
        params: Parameters,
        scenario_instance: ScenarioInstance,
        settings: ExperimentSettings,
    ):
        super().__init__(params, scenario_instance, settings)

        self.environment = self.scenario_instance.environment

    def train(self):
        """Train the agents."""

        # Pretrain the agents first in isolation if requested
        if self.params.pretrain_agents:
            self._pretrain_agents()

        # Setup
        self._train_setup()

        # Build everything we need for training
        collector = self._get_data_collector()
        replay_buffer = self._get_replay_buffer()
        loss_module, gae = self._get_loss_module_and_gae()
        optimizer = self._get_optimizer(loss_module)

        # Run the training loop
        self._run_rl_training_loop(
            self.environment,
            collector,
            replay_buffer,
            loss_module,
            gae,
            optimizer,
        )

    def _pretrain_agents(self):
        """Pretrain the agent bodies in isolation.

        This just uses the SoloAgentTrainer class.
        """

        # Train the agents in isolation
        solo_agent_trainer = SoloAgentTrainer(
            self.params, self.scenario_instance, self.settings
        )
        solo_agent_trainer.train(as_pretraining=True)

        # Put the agents (back) in training mode
        for agent in self.scenario_instance.agents.values():
            agent.body.train()
            agent.policy_head.train()
            agent.value_head.train()

    def _train_setup(self):
        """Optional setup before training."""

    @abstractmethod
    def _get_data_collector(self) -> DataCollectorBase:
        """Construct the data collector, which generates rollouts from the environment

        Returns
        -------
        collector : DataCollectorBase
            The data collector.
        """
        pass

    @abstractmethod
    def _get_replay_buffer(self) -> ReplayBuffer:
        """Construct the replay buffer, which will store the rollouts

        Returns
        -------
        ReplayBuffer
            The replay buffer.
        """
        pass

    @abstractmethod
    def _get_loss_module_and_gae(self) -> tuple[LossModule, GAE]:
        """Construct the loss module and the generalized advantage estimator

        Returns
        -------
        loss_module : LossModule
            The loss module.
        gae : GAE
            The generalized advantage estimator.
        """
        pass

    @abstractmethod
    def _get_optimizer(self, loss_module: LossModule) -> torch.optim.Optimizer:
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
        pass

    def _run_rl_training_loop(
        self,
        environment: Environment,
        collector: DataCollectorBase,
        replay_buffer: ReplayBuffer,
        loss_module: LossModule,
        gae: GAE,
        optimizer: Optimizer,
    ):
        """Run a generic RL training loop.

        Parameters
        ----------
        environment : Environment
            The environment to train in.
        collector : DataCollectorBase
            The data collector to use for collecting data from the environment.
        replay_buffer : ReplayBuffer
            The replay buffer to use for storing the collected data.
        loss_module : LossModule
            The loss module.
        gae : GAE
            The generalized advantage estimator.
        optimizer : Optimizer
            The optimizer to use for optimizing the loss.
        """

        # Set the seed
        torch.manual_seed(self.params.seed)
        np.random.seed(self.params.seed)

        # Create the rollout sampler, which will sample rollouts from the environment
        # and save them to W&B
        if self.settings.wandb_run is not None:
            rollout_sampler = RolloutSampler(self.settings)

        # Create a progress bar
        pbar = self.settings.tqdm_func(
            total=self.params.ppo.num_iterations, desc="Training"
        )

        for iteration, tensordict_data in enumerate(collector):
            # Expand the done and terminated to match the reward shape (this is expected
            # by the value estimator)
            tensordict_data.set(
                ("next", "agents", "done"),
                tensordict_data.get(("next", "done"))
                .unsqueeze(-1)
                .expand(
                    tensordict_data.get_item_shape(("next", environment.reward_key))
                ),
            )
            tensordict_data.set(
                ("next", "agents", "terminated"),
                tensordict_data.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand(
                    tensordict_data.get_item_shape(("next", environment.reward_key))
                ),
            )

            # Compute the GAE
            with torch.no_grad():
                gae(
                    tensordict_data,
                    params=loss_module.critic_params,
                    target_params=loss_module.target_critic_params,
                )

            # Flatten the data and add it to the replay buffer
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view)

            for _ in range(self.params.ppo.num_epochs):
                for _ in range(
                    self.params.ppo.frames_per_batch // self.params.ppo.minibatch_size
                ):
                    # Sample a minibatch from the replay buffer
                    sub_data = replay_buffer.sample()

                    # Compute the loss
                    loss_vals: TensorDict = loss_module(sub_data)
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    # Take an optimization step
                    loss_value.backward()
                    clip_grad_norm_(
                        loss_module.parameters(), self.params.ppo.max_grad_norm
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                    # If we're in test mode, exit after one iteration
                    if self.settings.test_run:
                        break

                # If we're in test mode, exit after one iteration
                if self.settings.test_run:
                    break

            # Update the policy weights if the policy of the data collector and the
            # trained policy live on different devices.
            collector.update_policy_weights_()

            # Compute the mean rewards for the done episodes
            done = tensordict_data.get(("next", "agents", "done")).any(dim=-1)
            reward = tensordict_data.get(("next", "agents", "reward"))
            mean_rewards = {}
            for i, agent_name in enumerate(self.params.agents):
                mean_rewards[agent_name] = reward[..., i][done].mean().item()

            if self.settings.wandb_run is not None:
                # Compute the average episode length
                round = tensordict_data.get(("next", "round"))
                mean_episode_length = round[done].float().mean().item()

                # Log the mean episode length and mean rewards
                to_log = dict(mean_episode_length=mean_episode_length)
                for agent_name in self.params.agents:
                    to_log[f"{agent_name}.mean_reward"] = mean_rewards[agent_name]
                self.settings.wandb_run.log(to_log, step=iteration)

                # Sample rollouts from the data and save them to W&B
                if (iteration + 1) % self.settings.rollout_sample_period == 0:
                    rollout_sampler.sample_and_save_rollouts(tensordict_data, iteration)

            # If we're in test mode, exit after one iteration
            if self.settings.test_run:
                break

            # Update the progress bar
            pbar.update(1)

        # Close the progress bar
        pbar.close()
