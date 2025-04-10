"""Train agents in isolation, without any interaction with other agents.

This is useful for ensuring that the agents are able to learn the task in isolation.
"""

import logging
from contextlib import ExitStack
from typing import ClassVar, Literal

import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.optim import Adam, Optimizer

from tensordict import TensorDict
from tensordict.nn import TensorDictSequential

from nip.scenario_base.data import TensorDictDataLoader, Dataset
from nip.scenario_base.agents import Agent
from nip.trainers.trainer_base import (
    TensorDictTrainer,
    attach_progress_bar,
    IterationContext,
)
from nip.trainers.registry import register_trainer
from nip.parameters import AgentsParameters
from nip.utils.maths import set_seed


@register_trainer("solo_agent")
class SoloAgentTrainer(TensorDictTrainer):
    """Trainer for training tensordict agents in isolation.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    scenario_instance : ComponentHolder
        The components of the experiment.
    settings : ExperimentSettings
        The instance-specific settings of the experiment, like device, logging, etc.
    """

    trainer_type: ClassVar[Literal["rl", "solo_agent"]] = "solo_agent"

    def train(self, as_pretraining: bool = False):
        """Train the agents.

        Parameters
        ----------
        as_pretraining : bool, default=False
            Whether we're training the agents as a pretraining step. This affects the
            output and what we log to W&B.
        """

        set_seed(self.hyper_params.seed)
        torch_generator = torch.Generator().manual_seed(self.hyper_params.seed)

        if self.settings.logger is None:
            self.settings.logger = logging.getLogger(__name__)

        logger = self.settings.logger

        logger.info("Loading dataset and agents...")

        dataset = self.scenario_instance.train_dataset
        train_dataset, test_dataset = random_split(
            dataset,
            (1 - self.hyper_params.test_size, self.hyper_params.test_size),
        )

        # Select the non-random agents
        agents_params = AgentsParameters(
            **{
                name: self.hyper_params.agents[name]
                for name in self.protocol_handler.agent_names
                if not self.hyper_params.agents[name].is_random
            }
        )
        agents = {
            name: agent
            for name, agent in self.scenario_instance.agents.items()
            if name in agents_params
        }

        # Get the agent models, for convenience. When using a separate body for the
        # policy and value networks, we use the policy body.
        if self.use_single_body or not as_pretraining:
            agent_models = {
                name: TensorDictSequential(agent.body, agent.solo_head)
                for name, agent in agents.items()
            }
        else:
            agent_models = {
                name: TensorDictSequential(agent.policy_body, agent.solo_head)
                for name, agent in agents.items()
            }

        # Run the training loop in the appropriate context
        with ExitStack() as stack:
            self._build_train_context(stack)
            self._run_train_loop(
                train_dataset,
                agents_params,
                agents,
                agent_models,
                as_pretraining,
                torch_generator,
            )

        # Run the testing loop in the appropriate context
        with ExitStack() as stack:
            self._build_train_context(stack)
            self._run_test_loop(
                test_dataset,
                agents_params,
                agents,
                agent_models,
                as_pretraining,
                logger,
            )

    @attach_progress_bar(lambda self: self.hyper_params.solo_agent.num_epochs)
    def _run_train_loop(
        self,
        train_dataset: Dataset,
        agents_params: AgentsParameters,
        agents: dict[str, Agent],
        agent_models: dict[str, TensorDictSequential],
        as_pretraining: bool,
        torch_generator: torch.Generator,
        iteration_context: IterationContext,
    ):
        """Run the training loop.

        Parameters
        ----------
        train_dataset : Dataset
            The dataset to train on.
        agents_params : AgentsParameters
            The parameters of the agents.
        agents : dict[str, Agent]
            A dictionary of the classes which hold the agent components.
        agent_models : dict[str, TensorDictSequential]
            A dictionary of the actual models we're training.
        as_pretraining : bool
            Whether we're training the agents as a pretraining step.
        torch_generator : torch.Generator
            The random number generator to use.
        iteration_context : IterationContext
            The context to use for the training loop, which handles the progress bar.
        """

        # Create the optimizers, specifying the learning rates for the different parts
        # of the agent
        optimizers: dict[str, Optimizer] = {}
        for agent_name, agent in agents.items():
            model_param_dict = agent.get_model_parameter_dicts(
                base_lr=self.hyper_params.solo_agent.learning_rate,
                body_lr_factor_override=self.hyper_params.solo_agent.body_lr_factor_override,
            )
            optimizers[agent_name] = Adam(model_param_dict, eps=1e-5)

        # Create the data loaders
        train_dataloader = TensorDictDataLoader(
            train_dataset,
            batch_size=self.hyper_params.solo_agent.batch_size,
            shuffle=True,
            generator=torch_generator,
        )

        # Set the progress bar description
        desc = "Pretraining" if as_pretraining else "Training"
        iteration_context.set_description(desc)

        # Train the agents
        for epoch in range(self.hyper_params.solo_agent.num_epochs):
            # Step the profiler if it's being used
            if self.settings.profiler is not None:
                self.settings.profiler.step()

            total_loss = {agent_name: 0 for agent_name in agents_params}
            total_accuracy = {agent_name: 0 for agent_name in agents_params}

            for data in train_dataloader:
                data: TensorDict
                data = data.to(self.settings.device)

                # Train the agents on the batch
                for agent_name, agent_model in agent_models.items():
                    agents[agent_name].train()
                    optimizers[agent_name].zero_grad()

                    model_output = agent_model(data)
                    logits = model_output["decision_logits"]
                    loss = F.cross_entropy(logits, data["y"])

                    loss.backward()
                    optimizers[agent_name].step()

                    with torch.no_grad():
                        accuracy = (
                            (logits.argmax(dim=1) == data["y"]).float().mean().item()
                        )

                    total_loss[agent_name] += loss.item()
                    total_accuracy[agent_name] += accuracy

                # If we're in test mode, exit after one iteration
                if self.settings.test_run:
                    break

            # Log run statistics
            if not as_pretraining:
                to_log = {}
                for agent_name in agents_params:
                    train_loss = total_loss[agent_name] / len(train_dataloader)
                    train_accuracy = total_accuracy[agent_name] / len(train_dataloader)
                    to_log[f"{agent_name}.train_loss"] = train_loss
                    to_log[f"{agent_name}.train_accuracy"] = train_accuracy
                self.settings.stat_logger.log(to_log, step=epoch)

            # If we're in test mode, exit after one iteration
            if self.settings.test_run:
                break

            # Update the progress bar
            iteration_context.step()

    def _run_test_loop(
        self,
        test_dataset: Dataset,
        agents_params: AgentsParameters,
        agents: dict[str, Agent],
        agent_models: dict[str, TensorDictSequential],
        as_pretraining: bool,
        logger: logging.Logger,
    ):
        """Run the testing loop.

        Parameters
        ----------
        test_dataset : Dataset
            The dataset to test on.
        agents_params : AgentsParameters
            The parameters of the agents.
        agents : dict[str, Agent]
            A dictionary of the classes which hold the agent components.
        agent_models : dict[str, TensorDictSequential]
            A dictionary of the actual models we're testing.
        as_pretraining : bool
            Whether we're testing the agents as a pretraining step.
        logger : logging.Logger
            The logger to use.
        """

        test_loader = TensorDictDataLoader(
            test_dataset,
            batch_size=self.hyper_params.solo_agent.batch_size,
            shuffle=False,
        )

        total_loss = {agent_name: 0 for agent_name in agents_params}
        total_accuracy = {agent_name: 0 for agent_name in agents_params}

        # Test the agents
        logger.info("Testing...")
        for data in test_loader:
            data = data.to(self.settings.device)

            for agent_name, agent_model in agent_models.items():
                agents[agent_name].eval()

                with torch.no_grad():
                    model_output = agent_model(data)
                    logits = model_output["decision_logits"]
                    loss = F.cross_entropy(logits, data["y"])
                    accuracy = (logits.argmax(dim=1) == data["y"]).float().mean().item()

                total_loss[agent_name] += loss
                total_accuracy[agent_name] += accuracy

            # If we're in test mode, exit after one iteration
            if self.settings.test_run:
                break

        # Record the final results
        prefix = "pretrain_" if as_pretraining else ""
        to_log = {}
        for agent_name in agents_params:
            test_loss = total_loss[agent_name] / len(test_loader)
            test_accuracy = total_accuracy[agent_name] / len(test_loader)
            to_log[f"{agent_name}.{prefix}test_loss"] = test_loss
            to_log[f"{agent_name}.{prefix}test_accuracy"] = test_accuracy
        self.settings.stat_logger.log(to_log)
