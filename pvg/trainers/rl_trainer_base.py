"""A generic reinforcement learning trainer."""

from abc import ABC, abstractmethod
from typing import Optional
from contextlib import ExitStack

import numpy as np

import torch
from torch.nn.utils import clip_grad_norm_

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase, TensorDictSequential

from torchrl.collectors import SyncDataCollector
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
from pvg.utils.tensordict import tensordict_add, tensordict_scalar_multiply


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
        self.train_collector, self.test_collector = self._get_data_collectors()
        self.replay_buffer = self._get_replay_buffer()
        self.loss_module, self.gae = self._get_loss_module_and_gae()
        (
            self.optimizer,
            self.param_group_freezer,
        ) = self._get_optimizer_and_param_freezer(self.loss_module)

        # Run the training loop
        self._train_and_test()

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
            )
            all_param_dicts.extend(param_dict)
            param_group_collections[agent_name] = param_dict

        if len(all_param_dicts) == 0:
            optimizer = DummyOptimizer()
        else:
            optimizer = torch.optim.Adam(all_param_dicts,eps=1e-5)

        param_group_freezer = ParamGroupFreezer(
            optimizer,
            param_group_collections,
            use_required_grad=self.params.functionalize_modules,
        )

        return optimizer, param_group_freezer

    def _get_log_stats(
        self,
        tensordict_data: TensorDictBase,
        mean_loss_vals: Optional[TensorDictBase] = None,
        *,
        train=True,
    ) -> dict[str, float]:
        """Compute the statistics to log during training or testing.

        Parameters
        ----------
        tensordict_data : TensorDict
            The data sampled from the data collector.
        mean_loss_vals : TensorDict, optional
            The average loss values.
        train : bool, default=True
            Whether the statistics are for training or testing.

        Returns
        -------
        log_stats : dict[str, float]
            The statistics to log.
        """

        if train:
            prefix = ""
        else:
            prefix = "test_"

        round = tensordict_data.get(("next", "round"))
        done = tensordict_data.get(("next", "done"))
        reward = tensordict_data.get(("next", "agents", "reward"))
        advantage = tensordict_data.get(("advantage"))
        value = tensordict_data.get(("agents", "value"), None)
        value_target = tensordict_data.get(("agents", "value_target"), None)
        decision_logits = tensordict_data.get(("agents", "decision_logits"))
        message_logits_key = self.scenario_instance.agents[
            self._agent_names[0]
        ].message_logits_key
        message_logits = tensordict_data.get(("agents", message_logits_key))

        log_stats = {}

        # Compute the mean episode length
        mean_episode_length = round[done].float().mean().item()
        log_stats[f"{prefix}mean_episode_length"] = mean_episode_length

        for i, agent_name in enumerate(self._agent_names):
            # Compute the mean reward per step and per episode
            mean_reward = reward[..., i].mean().item()
            log_stats[f"{agent_name}.{prefix}mean_step_reward"] = mean_reward
            log_stats[f"{agent_name}.{prefix}mean_episode_reward"] = (
                mean_reward * mean_episode_length
            )

            # Compute the mean advantage
            if advantage is not None:
                log_stats[f"{agent_name}.{prefix}mean_advantage"] = (
                    advantage[..., i].mean().item()
                )

            # Compute the mean agent value
            if value is not None:
                log_stats[f"{agent_name}.{prefix}mean_value"] = (
                    value[..., i].mean().item()
                )

            # Compute the mean agent value target
            if value_target is not None:
                log_stats[f"{agent_name}.{prefix}mean_value_target"] = (
                    value_target[..., i].mean().item()
                )

            # Compute the residual critic variance
            if value is not None and value_target is not None:
                log_stats[f"{agent_name}.{prefix}residual_critic_variance"] = (
                    (value_target[..., i] - value[..., i]).var().item() / value_target[..., i].var().item()
                )

            # Compute the (normalised) agent decision entropy mean and standard deviation
            max_decision_ent = - np.log(1 / decision_logits.shape[-1])
            decision_logit_entropy = logit_entropy(decision_logits[..., i]) / max_decision_ent
            log_stats[f"{agent_name}.{prefix}mean_decision_entropy"] = (
                decision_logit_entropy.mean().item()
            )
            log_stats[f"{agent_name}.{prefix}std_decision_entropy"] = (
                decision_logit_entropy.std().item()
            )

            # Compute the (normalised) agent message entropy mean and standard deviation
            max_message_ent = - np.log(1 / message_logits.shape[-1])
            message_logit_entropy = logit_entropy(message_logits[..., i]) / max_message_ent
            log_stats[f"{agent_name}.{prefix}mean_message_entropy"] = (
                message_logit_entropy.mean().item()
            )
            log_stats[f"{agent_name}.{prefix}std_message_entropy"] = (
                message_logit_entropy.std().item()
            )

            # Compute the maximum message probability mean and standard deviation
            message_probs = torch.softmax(message_logits[..., i], dim=-1)
            max_message_probs = message_probs.max(dim=-1).values
            log_stats[f"{agent_name}.{prefix}mean_max_message_prob"] = (
                max_message_probs.mean().item()
            )
            log_stats[f"{agent_name}.{prefix}std_max_message_prob"] = (
                max_message_probs.std().item()
            )

        # Compute the mean accuracy for the done episodes
        verifier_decision = tensordict_data.get(("agents", "decision"))[
            ..., self._agent_names.index("verifier")
        ]
        log_stats[f"{prefix}mean_accuracy"] = (
            (verifier_decision[done] == tensordict_data["y"][done].squeeze())
            .float()
            .mean()
            .item()
        )

        # Log the loss values
        if mean_loss_vals is not None:

            def log_per_agent_losses(agent_losses: TensorDictBase):
                for key, val in agent_losses.items():
                    for i, agent_name in enumerate(self._agent_names):
                        log_stats[f"{prefix}{agent_name}.{key}"] = (
                            val[..., i].mean().item()
                        )

            for key, val in mean_loss_vals.items():
                if key == "agents":
                    log_per_agent_losses(val)
                else:
                    log_stats[f"{prefix}{key}"] = val.item()

            if "loss_critic" in mean_loss_vals.keys():
                log_stats[f"{prefix}loss_critic_unscaled"] = (
                    mean_loss_vals["loss_critic"] / self.loss_module.critic_coef
                ).item()

        return log_stats

    def _train_and_test(self):
        """Run generic RL training and test loops."""

        # Set the seed
        torch.manual_seed(self.params.seed)
        np.random.seed(self.params.seed)

        # Run the training loop with the appropriate context managers
        with ExitStack() as stack:
            self._build_train_context(stack)
            self._run_train_loop()

        # Run the test loop with the appropriate context managers
        with ExitStack() as stack:
            self._build_test_context(stack)
            self._run_test_loop()

    def _run_train_loop(self):
        agents = self.scenario_instance.agents

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
            for name in self._agent_names
        ]

        iterator = zip(enumerate(self.train_collector), *update_schedule_iterators)
        for (iteration, tensordict_data), *agent_updates in iterator:
            # Step the profiler if it's being used
            if self.settings.profiler is not None:
                self.settings.profiler.step()

            # Update the learning rate if annealing is enabled
            if self.params.rl.anneal_lr:   
                if iteration == 0:
                    for pg in self.optimizer.param_groups:
                        pg["original_lr"] = pg["lr"]
                for pg in self.optimizer.param_groups:
                    pg["lr"] = (1 - (iteration / self.params.rl.num_iterations)) * pg["original_lr"]

            # Freeze and unfreeze the parameters of the agents according to the update
            # schedule
            for agent_name, update in zip(self._agent_names, agent_updates):
                if update:
                    self.param_group_freezer.unfreeze(agent_name)
                else:
                    self.param_group_freezer.freeze(agent_name)

            # Expand the done and terminated to match the reward shape (this is expected
            # by the value estimator)
            tensordict_data.set(
                ("next", "agents", "done"),
                tensordict_data.get(("next", "done"))
                .unsqueeze(-1)
                .expand(
                    tensordict_data.get_item_shape(
                        ("next", self.train_environment.reward_key)
                    )
                ),
            )
            tensordict_data.set(
                ("next", "agents", "terminated"),
                tensordict_data.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand(
                    tensordict_data.get_item_shape(
                        ("next", self.train_environment.reward_key)
                    )
                ),
            )

            # Compute the GAE
            if self.gae is not None:
                with torch.no_grad():
                    if self.params.functionalize_modules:
                        self.gae(
                            tensordict_data,
                            params=self.loss_module.critic_network_params,
                            target_params=self.loss_module.target_critic_network_params,
                        )
                    else:
                        self.gae(tensordict_data)

            # Flatten the data and add it to the replay buffer
            data_view = tensordict_data.reshape(-1)
            self.replay_buffer.extend(data_view)

            # Train the agents on the replay buffer
            mean_loss_vals = self._train_on_replay_buffer()

            # Update the policy weights if the policy of the data collector and the
            # trained policy live on different devices.
            self.train_collector.update_policy_weights_()

            # Log statistics
            to_log = self._get_log_stats(tensordict_data, mean_loss_vals)
            self.settings.stat_logger.log(to_log, step=iteration)

            # Log artifacts to W&B
            if self.settings.wandb_run is not None:
                artifact_logger.log(tensordict_data, iteration)

            # If we're in test mode, exit after one iteration
            if self.settings.test_run:
                break

            # Update the progress bar
            pbar.update(1)

        # Close the progress bar
        pbar.close()

    def _train_on_replay_buffer(self) -> TensorDict:
        """Train the agents on data in the replay buffer.

        Returns
        -------
        mean_loss_vals : TensorDict
            The mean loss values over the training iterations.
        """

        mean_loss_vals = None

        total_steps = 0
        for _ in range(self.params.rl.num_epochs):
            for _ in range(
                self.params.rl.frames_per_batch // self.params.rl.minibatch_size
            ):
                # Sample a minibatch from the replay buffer
                sub_data = self.replay_buffer.sample()

                # Compute the loss
                loss_vals: TensorDict = self.loss_module(sub_data)

                # Log the loss values
                if mean_loss_vals is None:
                    mean_loss_vals = loss_vals.clone()
                else:
                    mean_loss_vals = tensordict_add(
                        mean_loss_vals, loss_vals, inplace=True
                    )

                # Only perform the optimization step if the loss values require
                # gradients. This can be false for example if all agents are frozen
                if loss_vals.requires_grad:
                    # Compute the gradients
                    self.loss_module.backward(loss_vals)

                    # Clip gradients and update parameters
                    clip_grad_norm_(
                        self.loss_module.parameters(), self.params.rl.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                total_steps += 1

                # If we're in test mode, exit after one iteration
                if self.settings.test_run:
                    break

            # If we're in test mode, exit after one iteration
            if self.settings.test_run:
                break

        # Take an average of the loss values
        mean_loss_vals = tensordict_scalar_multiply(
            mean_loss_vals, 1 / total_steps, inplace=True
        )

        return mean_loss_vals

    def _run_test_loop(self):
        # Create a progress bar
        pbar = self.settings.tqdm_func(
            total=self.params.rl.num_test_iterations, desc="Testing"
        )

        # Run the test loop
        with torch.no_grad():
            # Put the agents in eval mode
            for agent in self.scenario_instance.agents.values():
                agent.eval()

            aggregate_data = None

            for _, tensordict_data in enumerate(self.test_collector):
                # Expand the done to match the reward shape
                tensordict_data.set(
                    ("next", "agents", "done"),
                    tensordict_data.get(("next", "done"))
                    .unsqueeze(-1)
                    .expand(
                        tensordict_data.get_item_shape(
                            ("next", self.test_environment.reward_key)
                        )
                    ),
                )

                if aggregate_data is None:
                    aggregate_data = tensordict_data.cpu()
                else:
                    aggregate_data = torch.cat(
                        [aggregate_data, tensordict_data.cpu()], dim=0
                    )

                # If we're in test mode, exit after one iteration
                if self.settings.test_run:
                    break

                # Update the progress bar
                pbar.update(1)

            to_log = self._get_log_stats(tensordict_data, train=False)
            self.settings.stat_logger.log(to_log)

        # Close the progress bar
        pbar.close()
