"""A generic reinforcement learning trainer."""

from abc import ABC, abstractmethod
import re

import numpy as np

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from tensordict import TensorDict

from torchrl.collectors import DataCollectorBase
from torchrl.objectives.value import GAE
from torchrl.data.replay_buffers import ReplayBuffer

from pvg.parameters import Parameters
from pvg.scenario_base import Environment
from pvg.experiment_settings import ExperimentSettings
from pvg.trainers.base import Trainer
from pvg.trainers.solo_agent import SoloAgentTrainer
from pvg.model_cache import (
    cached_models_exist,
    save_model_state_dicts,
    load_cached_model_state_dicts,
)
from pvg.scenario_instance import ScenarioInstance
from pvg.artifact_logger import ArtifactLogger
from pvg.rl_objectives import Objective
from pvg.utils.maths import logit_entropy
from pvg.utils.torch import DummyOptimizer


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

    PRETRAINED_MODEL_CACHE_PARAM_KEYS = [
        "scenario",
        "dataset",
        "agents",
        "seed",
        "test_size",
        "d_representation",
        "solo_agent",
        "image_classification",
        "dataset_options",
        "pvg_protocol",
        "debate_protocol",
    ]

    def __init__(
        self,
        params: Parameters,
        scenario_instance: ScenarioInstance,
        settings: ExperimentSettings,
    ):
        super().__init__(params, scenario_instance, settings)

        self.train_environment = self.scenario_instance.train_environment
        self.test_environment = self.scenario_instance.test_environment

    def train(self):
        """Train the agents."""

        # Pretrain the agents first in isolation if requested
        if self.params.pretrain_agents:
            self._pretrain_agents()

        # Setup
        self._train_setup()

        # Build everything we need for training
        train_collector, test_collector = self._get_data_collectors()
        replay_buffer = self._get_replay_buffer()
        loss_module, gae = self._get_loss_module_and_gae()
        optimizer = self._get_optimizer(loss_module)

        # Run the training loop
        self._run_rl_training_loop(
            self.train_environment,
            self.test_environment,
            train_collector,
            test_collector,
            replay_buffer,
            loss_module,
            gae,
            optimizer,
        )

    def _pretrain_agents(self):
        """Pretrain the agent bodies in isolation.

        This just uses the SoloAgentTrainer class.
        """

        # The body models of the agents
        body_model_dict = {
            agent_name: agent.body
            for agent_name, agent in self.scenario_instance.agents.items()
        }

        # Get the parameters that define the model cache
        model_cache_params = self.params.to_dict()
        model_cache_params = dict(
            (key, value)
            for key, value in model_cache_params.items()
            if key in self.PRETRAINED_MODEL_CACHE_PARAM_KEYS
        )

        # Load the cached models if they exist
        if not self.settings.ignore_cache and cached_models_exist(
            model_cache_params, "solo_agents"
        ):
            load_cached_model_state_dicts(
                body_model_dict, model_cache_params, "solo_agents"
            )
        else:
            # Train the agents in isolation
            solo_agent_trainer = SoloAgentTrainer(
                self.params, self.scenario_instance, self.settings
            )
            solo_agent_trainer.train(as_pretraining=True)

            # Save the models
            save_model_state_dicts(
                body_model_dict, model_cache_params, "solo_agents", overwrite=True
            )

        # Put the agents (back) in training mode
        for agent in self.scenario_instance.agents.values():
            agent.body.train()
            agent.policy_head.train()
            agent.value_head.train()

    def _train_setup(self):
        """Optional setup before training."""

    @abstractmethod
    def _get_data_collectors(self) -> tuple[DataCollectorBase, DataCollectorBase]:
        """Construct the data collectors, which generate rollouts from the environment

        Constructs a collector for both the train and the test environment.

        Returns
        -------
        train_collector : SyncDataCollector
            The train data collector.
        test_collector : SyncDataCollector
            The test data collector.
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
    def _get_loss_module_and_gae(self) -> tuple[Objective, GAE]:
        """Construct the loss module and the generalized advantage estimator

        Returns
        -------
        loss_module : Objective
            The loss module.
        gae : GAE
            The generalized advantage estimator.
        """
        pass

    def _get_optimizer(self, loss_module: Objective) -> torch.optim.Adam:
        """Construct the optimizer for the loss module

        Parameters
        ----------
        loss_module : Objective
            The loss module.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer.
        """

        # Set the learning rate of the agent bodies to be a factor of the learning rate
        # of the loss module
        model_param_dict = []
        for agent_name, agent_params in self.params.agents.items():
            # The learning rate of the whole agent
            agent_lr = agent_params.agent_lr_factor * self.params.ppo.lr

            # Determine the learning rate of the body. If the LR factor is set in the
            # PPO parameters, use that. Otherwise, use the LR factor from the agent
            # parameters.
            if self.params.ppo.body_lr_factor is None:
                body_lr = agent_lr * agent_params.body_lr_factor
            else:
                body_lr = agent_lr * self.params.ppo.body_lr_factor

            # Set the learning rate for the body parameters, or freeze them if the LR is
            # 0
            body_params = [
                param
                for param_name, param in loss_module.named_parameters()
                if param_name.startswith(f"actor_network_params.module_0_{agent_name}")
            ]
            if body_lr == 0:
                for param in body_params:
                    param.requires_grad = False
            else:
                model_param_dict.append(dict(params=body_params, lr=body_lr))

            # Set the learning rate for the non-body parameters
            def is_non_body_param(param_name: str):
                if re.match(
                    f"actor_network_params.module_[1-9]_{agent_name}", param_name
                ):
                    return True
                if re.match(
                    f"critic_network_params.module_[0-9]_{agent_name}", param_name
                ):
                    return True
                return False

            # Set the learning rate for the non-body parameters, or freeze them if the
            # LR is 0
            non_body_params = [
                param
                for param_name, param in loss_module.named_parameters()
                if is_non_body_param(param_name)
            ]
            if agent_lr == 0:
                for param in non_body_params:
                    param.requires_grad = False
            else:
                model_param_dict.append(dict(params=non_body_params, lr=agent_lr))

        if len(model_param_dict) == 0:
            return DummyOptimizer()
        else:
            return torch.optim.Adam(model_param_dict)

    def _run_rl_training_loop(
        self,
        train_environment: Environment,
        test_environment: Environment,
        train_collector: DataCollectorBase,
        test_collector: DataCollectorBase,
        replay_buffer: ReplayBuffer,
        loss_module: Objective,
        gae: GAE,
        optimizer: Optimizer,
    ):
        """Run a generic RL training loop.

        Parameters
        ----------
        train_environment : Environment
            The environment to train in.
        test_environment : Environment
            The environment to test in.
        train_collector : DataCollectorBase
            The data collector to use for collecting data from the train environment.
        test_collector : DataCollectorBase
            The data collector to use for collecting data from the test environment.
        replay_buffer : ReplayBuffer
            The replay buffer to use for storing the collected data.
        loss_module : Objective
            The loss module.
        gae : GAE
            The generalized advantage estimator.
        optimizer : Optimizer
            The optimizer to use for optimizing the loss.
        """

        # Set the seed
        torch.manual_seed(self.params.seed)
        np.random.seed(self.params.seed)

        # Create the artifact logger, which will log things are various stages to W&B
        if self.settings.wandb_run is not None:
            artifact_logger = ArtifactLogger(
                self.settings, self.scenario_instance.agents
            )

        # Create a progress bar
        pbar = self.settings.tqdm_func(
            total=self.params.ppo.num_iterations, desc="Training"
        )

        for iteration, tensordict_data in enumerate(train_collector):
            # Expand the done and terminated to match the reward shape (this is expected
            # by the value estimator)
            tensordict_data.set(
                ("next", "agents", "done"),
                tensordict_data.get(("next", "done"))
                .unsqueeze(-1)
                .expand(
                    tensordict_data.get_item_shape(
                        ("next", train_environment.reward_key)
                    )
                ),
            )
            tensordict_data.set(
                ("next", "agents", "terminated"),
                tensordict_data.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand(
                    tensordict_data.get_item_shape(
                        ("next", train_environment.reward_key)
                    )
                ),
            )

            # Compute the GAE
            with torch.no_grad():
                gae(
                    tensordict_data,
                    params=loss_module.critic_network_params,
                    target_params=loss_module.target_critic_network_params,
                )

            # Flatten the data and add it to the replay buffer
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view)

            loss_outputs = {}

            total_steps = 0
            for _ in range(self.params.ppo.num_epochs):
                for _ in range(
                    self.params.ppo.frames_per_batch // self.params.ppo.minibatch_size
                ):
                    # Sample a minibatch from the replay buffer
                    sub_data = replay_buffer.sample()

                    # Compute the loss
                    loss_vals: TensorDict = loss_module(sub_data)

                    # Log the loss values
                    for key, val in loss_vals.items():
                        if key not in loss_outputs:
                            loss_outputs[key] = 0
                        loss_outputs[key] += val.mean().item()

                    # Only perform the optimization step if the loss values require
                    # gradients. This can be false for example if all agents are frozen
                    if loss_vals.requires_grad:
                        # Compute the gradients
                        loss_module.backward(loss_vals)

                        # Clip gradients and update parameters
                        clip_grad_norm_(
                            loss_module.parameters(), self.params.ppo.max_grad_norm
                        )
                        optimizer.step()
                        optimizer.zero_grad()

                    total_steps += 1

                    # If we're in test mode, exit after one iteration
                    if self.settings.test_run:
                        break

                # If we're in test mode, exit after one iteration
                if self.settings.test_run:
                    break

            # Update the policy weights if the policy of the data collector and the
            # trained policy live on different devices.
            train_collector.update_policy_weights_()

            # Take an average of the loss values
            for key in loss_outputs:
                loss_outputs[key] /= total_steps

            # Compute various statistics for the sampled episodes for logging
            done = tensordict_data.get(("next", "agents", "done")).any(dim=-1)
            reward = tensordict_data.get(("next", "agents", "reward"))
            value = tensordict_data.get(("agents", "value"))
            decision_logits = tensordict_data.get(("agents", "decision_logits"))
            mean_rewards = {}
            mean_values = {}
            mean_decision_entropy = {}
            for i, agent_name in enumerate(self._agent_names):
                mean_rewards[agent_name] = reward[..., i][done].mean().item()
                mean_values[agent_name] = value[..., i].mean().item()
                mean_decision_entropy[agent_name] = (
                    logit_entropy(decision_logits[..., i]).mean().item()
                )

            if self.settings.wandb_run is not None:
                # Compute the average episode length
                round = tensordict_data.get(("next", "round"))
                mean_episode_length = round[done].float().mean().item()

                # Log the various statistics
                to_log = dict(mean_episode_length=mean_episode_length)
                for agent_name in self._agent_names:
                    to_log[f"{agent_name}.mean_reward"] = mean_rewards[agent_name]
                    to_log[f"{agent_name}.mean_value"] = mean_values[agent_name]
                    to_log[
                        f"{agent_name}.mean_decision_entropy"
                    ] = mean_decision_entropy[agent_name]
                for key, val in loss_outputs.items():
                    to_log[key] = val
                self.settings.wandb_run.log(to_log, step=iteration)

                # Log artifacts to W&B if it's time to do so
                artifact_logger.log(tensordict_data, iteration)

            # If we're in test mode, exit after one iteration
            if self.settings.test_run:
                break

            # Update the progress bar
            pbar.update(1)

        # Close the progress bar
        pbar.close()

        # Create a progress bar
        pbar = self.settings.tqdm_func(
            total=self.params.ppo.num_test_iterations, desc="Testing"
        )

        # Run the test loop
        with torch.no_grad():
            mean_rewards = {agent_name: 0 for agent_name in self._agent_names}
            mean_episode_length = 0
            for iteration, tensordict_data in enumerate(test_collector):
                # Expand the done to match the reward shape
                tensordict_data.set(
                    ("next", "agents", "done"),
                    tensordict_data.get(("next", "done"))
                    .unsqueeze(-1)
                    .expand(
                        tensordict_data.get_item_shape(
                            ("next", test_environment.reward_key)
                        )
                    ),
                )

                # Compute the mean rewards for the done episodes
                done = tensordict_data.get(("next", "agents", "done")).any(dim=-1)
                reward = tensordict_data.get(("next", "agents", "reward"))
                for i, agent_name in enumerate(self._agent_names):
                    mean_rewards[agent_name] += reward[..., i][done].mean().item()

                # Compute the average episode length
                round = tensordict_data.get(("next", "round"))
                mean_episode_length += round[done].float().mean().item()

                # If we're in test mode, exit after one iteration
                if self.settings.test_run:
                    break

                # Update the progress bar
                pbar.update(1)

            # Compute the mean of the statistics
            for agent_name in self._agent_names:
                mean_rewards[agent_name] /= self.params.ppo.num_test_iterations
            mean_episode_length /= self.params.ppo.num_test_iterations

            if self.settings.wandb_run is not None:
                # Log the mean episode length and mean rewards
                to_log = dict(test_mean_episode_length=mean_episode_length)
                for agent_name in self._agent_names:
                    to_log[f"{agent_name}.test_mean_reward"] = mean_rewards[agent_name]
                self.settings.wandb_run.log(to_log)

        # Close the progress bar
        pbar.close()
