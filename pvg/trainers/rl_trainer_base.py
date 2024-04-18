"""A generic reinforcement learning trainer."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential

from torchrl.collectors import DataCollectorBase, SyncDataCollector
from torchrl.objectives.value import GAE
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.modules import ProbabilisticActor, ActorValueOperator
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs.transforms import Transform

from pvg.parameters import (
    Parameters,
    AgentUpdateSchedule,
    ConstantUpdateSchedule,
    ContiguousPeriodicUpdateSchedule,
)
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
from pvg.utils.training import ParamGroupFreezer
from pvg.utils.distributions import CompositeCategoricalDistribution


def update_schedule_iterator(schedule: AgentUpdateSchedule):
    """A True-False iterator which specifies on which iterations to update an agent.

    Parameters
    ----------
    schedule : AgentUpdateSchedule
        The update schedule.

    Yields
    ------
    bool
        Whether to update the agent on the current iteration.
    """

    if isinstance(schedule, ConstantUpdateSchedule):
        while True:
            yield True
    elif isinstance(schedule, ContiguousPeriodicUpdateSchedule):
        while True:
            for i in range(schedule.period):
                yield schedule.start <= i < schedule.stop
    else:
        raise ValueError(f"Unknown update schedule: {schedule}")


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

    policy_operator: TensorDictModuleBase
    value_operator: TensorDictModuleBase | None

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
        optimizer, param_group_freezer = self._get_optimizer_and_param_freezer(
            loss_module
        )

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
            param_group_freezer,
        )

    def _pretrain_agents(self):
        """Pretrain the agent bodies in isolation.

        This just uses the SoloAgentTrainer class.
        """

        # The body models of the agents. When the actor and critic don't share a body,
        # we use the policy body for pretraining, so this is the relevant body.
        if self.use_single_body:
            body_model_dict = {
                agent_name: agent.body
                for agent_name, agent in self.scenario_instance.agents.items()
            }
        else:
            body_model_dict = {
                agent_name: agent.policy_body
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
            agent.train()

    def _train_setup(self):
        """Some setup before the training loop"""

        # Build the policy and value operators differently depending on whether the body
        # is shared
        if self.use_single_body and self.use_critic:
            # Create the policy head, which samples actions from the policy probability
            # distribution
            combined_policy_head = self.scenario_instance.combined_policy_head
            combined_probabilistic_policy_head = ProbabilisticActor(
                combined_policy_head,
                spec=self.train_environment.action_spec,
                distribution_class=CompositeCategoricalDistribution,
                distribution_kwargs=dict(
                    key_transform=lambda x: ("agents", x),
                    log_prob_key=("agents", "sample_log_prob"),
                ),
                in_keys={
                    out_key[1]: out_key for out_key in combined_policy_head.out_keys
                },
                out_keys=self.train_environment.action_keys,
                return_log_prob=True,
                log_prob_key=("agents", "sample_log_prob"),
            )

            # Create the full model, which runs the combined body and heads
            full_model = ActorValueOperator(
                self.scenario_instance.combined_body,
                combined_probabilistic_policy_head,
                self.scenario_instance.combined_value_head,
            )
            self.policy_operator = full_model.get_policy_operator()
            self.value_operator = full_model.get_value_operator()

        else:
            # Create the policy operator, which runs the combined policy body and head
            # and samples actions from the policy probability distribution
            if self.use_critic:
                combined_policy_body = self.scenario_instance.combined_policy_body
            else:
                combined_policy_body = self.scenario_instance.combined_body
            combined_policy_head = self.scenario_instance.combined_policy_head
            self.policy_operator = ProbabilisticActor(
                TensorDictSequential(combined_policy_body, combined_policy_head),
                spec=self.train_environment.action_spec,
                distribution_class=CompositeCategoricalDistribution,
                distribution_kwargs=dict(
                    key_transform=lambda x: ("agents", x),
                    log_prob_key=("agents", "sample_log_prob"),
                ),
                in_keys={
                    out_key[1]: out_key for out_key in combined_policy_head.out_keys
                },
                out_keys=self.train_environment.action_keys,
                return_log_prob=True,
                log_prob_key=("agents", "sample_log_prob"),
            )

            # Create the value operator, which runs the combined value body and head
            if self.use_critic:
                self.value_operator = TensorDictSequential(
                    self.scenario_instance.combined_value_body,
                    self.scenario_instance.combined_value_head,
                )
            else:
                self.value_operator = None

    def _get_data_collectors(self) -> tuple[SyncDataCollector, SyncDataCollector]:
        """Construct the data collectors, which generate rollouts from the environment

        Constructs a collector for both the train and the test environment.

        Returns
        -------
        train_collector : SyncDataCollector
            The train data collector.
        test_collector : SyncDataCollector
            The test data collector.
        """

        train_collector = SyncDataCollector(
            self.train_environment,
            self.policy_operator,
            device=self.device,
            storing_device=self.device,
            frames_per_batch=self.params.rl.frames_per_batch,
            total_frames=self.params.rl.frames_per_batch
            * self.params.rl.num_iterations,
        )

        test_collector = SyncDataCollector(
            self.train_environment,
            self.policy_operator,
            device=self.device,
            storing_device=self.device,
            frames_per_batch=self.params.rl.frames_per_batch,
            total_frames=self.params.rl.frames_per_batch
            * self.params.rl.num_test_iterations,
        )

        return train_collector, test_collector

    def _get_replay_buffer(self, transform: Optional[Transform] = None) -> ReplayBuffer:
        """Construct the replay buffer, which will store the rollouts

        Parameters
        ----------
        transform : Transform, optional
            The transform to apply to the data before storing it in the replay buffer.

        Returns
        -------
        ReplayBuffer
            The replay buffer.
        """
        return ReplayBuffer(
            storage=LazyTensorStorage(
                self.params.rl.frames_per_batch, device=self.device
            ),
            sampler=SamplerWithoutReplacement(),
            batch_size=self.params.rl.minibatch_size,
            transform=transform,
        )

    @abstractmethod
    def _get_loss_module_and_gae(self) -> tuple[Objective, GAE | None]:
        """Construct the loss module and the generalized advantage estimator

        Returns
        -------
        loss_module : Objective
            The loss module.
        gae : GAE | None
            The generalized advantage estimator, or None if the loss module doesn't use
            one.
        """
        pass

    def _get_optimizer_and_param_freezer(
        self, loss_module: Objective
    ) -> tuple[torch.optim.Adam, ParamGroupFreezer]:
        """Construct the optimizer for the loss module and the model parameter freezer.

        Parameters
        ----------
        loss_module : Objective
            The loss module.

        Returns
        -------
        optimizer : torch.optim.Optimizer
            The optimizer.
        param_group_freezer : ParamGroupFreezer
            The parameter dictionaries for each agent.
        """

        # Get the learning rates and parameters for each of the agents
        all_param_dicts = []
        param_group_collections = {}
        for agent_name, agent in self.scenario_instance.agents.items():
            param_dict = agent.get_param_dicts(
                base_lr=self.params.rl.lr,
                named_parameters=loss_module.named_parameters(),
                body_lr_factor_override=self.params.rl.body_lr_factor,
            )
            all_param_dicts.extend(param_dict)
            param_group_collections[agent_name] = param_dict

        if len(all_param_dicts) == 0:
            optimizer = DummyOptimizer()
        else:
            optimizer = torch.optim.Adam(all_param_dicts)

        param_group_freezer = ParamGroupFreezer(optimizer, param_group_collections)

        return optimizer, param_group_freezer

    def _run_rl_training_loop(
        self,
        train_environment: Environment,
        test_environment: Environment,
        train_collector: DataCollectorBase,
        test_collector: DataCollectorBase,
        replay_buffer: ReplayBuffer,
        loss_module: Objective,
        gae: GAE | None,
        optimizer: Optimizer,
        param_group_freezer: ParamGroupFreezer,
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
        gae : GAE | None
            The generalized advantage estimator, or None if the loss module doesn't use
            one.
        optimizer : Optimizer
            The optimizer to use for optimizing the loss.
        param_group_freezer : ParamGroupFreezer
            The parameter group freezer to use for freezing the parameters of the
            agents' models.
        """

        agents = self.scenario_instance.agents
        agent_names = list(agents.keys())

        # Set the seed
        torch.manual_seed(self.params.seed)
        np.random.seed(self.params.seed)

        # Create the artifact logger, which will log things are various stages to W&B
        if self.settings.wandb_run is not None:
            artifact_logger = ArtifactLogger(self.settings, agents)

        # Create a progress bar
        pbar = self.settings.tqdm_func(
            total=self.params.rl.num_iterations, desc="Training"
        )

        # Create the update schedule iterators
        update_schedule_iterators = [
            update_schedule_iterator(self.params.agents[name].update_schedule)
            for name in agent_names
        ]

        iterator = zip(enumerate(train_collector), *update_schedule_iterators)
        for (iteration, tensordict_data), *agent_updates in iterator:
            # Step the profiler if it's being used
            if self.settings.profiler is not None:
                self.settings.profiler.step()

            # Freeze and unfreeze the parameters of the agents according to the update
            # schedule
            for agent_name, update in zip(agent_names, agent_updates):
                if update:
                    param_group_freezer.unfreeze(agent_name)
                else:
                    param_group_freezer.freeze(agent_name)

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
            if gae is not None:
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
            for _ in range(self.params.rl.num_epochs):
                for _ in range(
                    self.params.rl.frames_per_batch // self.params.rl.minibatch_size
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
                            loss_module.parameters(), self.params.rl.max_grad_norm
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
            done = tensordict_data.get(("next", "done"))
            reward = tensordict_data.get(("next", "agents", "reward"))
            value = tensordict_data.get(("agents", "value"), None)
            decision_logits = tensordict_data.get(("agents", "decision_logits"))
            mean_rewards = {}
            mean_values = {}
            mean_decision_entropy = {}
            for i, agent_name in enumerate(self._agent_names):
                mean_rewards[agent_name] = reward[..., i].mean().item()
                if value is not None:
                    mean_values[agent_name] = value[..., i].mean().item()
                mean_decision_entropy[agent_name] = (
                    logit_entropy(decision_logits[..., i]).mean().item()
                )
            verifier_decision = tensordict_data.get(("agents", "decision"))[
                ..., self._agent_names.index("verifier")
            ]
            mean_accuracy = (
                (verifier_decision[done] == tensordict_data["y"][done].squeeze())
                .float()
                .mean()
                .item()
            )

            if self.settings.wandb_run is not None:
                # Compute the average episode length
                round = tensordict_data.get(("next", "round"))
                mean_episode_length = round[done].float().mean().item()

                # Log the various statistics
                to_log = dict(mean_episode_length=mean_episode_length)
                for agent_name in self._agent_names:
                    to_log[f"{agent_name}.mean_step_reward"] = mean_rewards[agent_name]
                    to_log[f"{agent_name}.mean_episode_reward"] = (
                        mean_rewards[agent_name] * mean_episode_length
                    )
                    if agent_name in mean_values:
                        to_log[f"{agent_name}.mean_value"] = mean_values[agent_name]
                    to_log[
                        f"{agent_name}.mean_decision_entropy"
                    ] = mean_decision_entropy[agent_name]
                to_log["mean_accuracy"] = mean_accuracy
                for key, val in loss_outputs.items():
                    to_log[key] = val
                if "loss_critic" in loss_outputs:
                    to_log["loss_critic_unscaled"] = (
                        loss_outputs["loss_critic"] / loss_module.critic_coef.item()
                    )
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
            total=self.params.rl.num_test_iterations, desc="Testing"
        )

        # Run the test loop
        with torch.no_grad():
            # Put the agents in eval mode
            for agent in self.scenario_instance.agents.values():
                agent.eval()

            mean_rewards = {agent_name: 0 for agent_name in self._agent_names}
            mean_episode_length = 0
            mean_accuracy = 0
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

                # Compute the mean rewards
                done = tensordict_data.get(("next", "agents", "done")).any(dim=-1)
                reward = tensordict_data.get(("next", "agents", "reward"))
                for i, agent_name in enumerate(self._agent_names):
                    mean_rewards[agent_name] += reward[..., i].mean().item()

                # Compute the mean accuracy for the done episodes
                verifier_decision = tensordict_data.get(("agents", "decision"))[
                    ..., self._agent_names.index("verifier")
                ]

                mean_accuracy += (
                    (verifier_decision[done] == tensordict_data["y"][done].squeeze())
                    .float()
                    .mean()
                    .item()
                )

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
                mean_rewards[agent_name] /= self.params.rl.num_test_iterations
            mean_episode_length /= self.params.rl.num_test_iterations
            mean_accuracy /= self.params.rl.num_test_iterations

            if self.settings.wandb_run is not None:
                # Log the mean episode length and mean rewards
                to_log = dict(test_mean_episode_length=mean_episode_length)
                for agent_name in self._agent_names:
                    to_log[f"{agent_name}.test_mean_step_reward"] = mean_rewards[
                        agent_name
                    ]
                    to_log[f"{agent_name}.test_mean_episode_reward"] = (
                        mean_rewards[agent_name] * mean_episode_length
                    )
                to_log["test_mean_accuracy"] = mean_accuracy
                self.settings.wandb_run.log(to_log)

        # Close the progress bar
        pbar.close()
