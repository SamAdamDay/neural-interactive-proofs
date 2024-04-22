"""Train agents in isolation, without any interaction with other agents.

This is useful for ensuring that the agents are able to learn the task in isolation.
"""

import logging

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.optim import Adam, Optimizer

from tensordict import TensorDict
from tensordict.nn import TensorDictSequential

from pvg.scenario_base.data import DataLoader
from pvg.trainers.base import Trainer
from pvg.trainers.registry import register_trainer
from pvg.parameters import AgentsParameters, TrainerType


@register_trainer(TrainerType.SOLO_AGENT)
class SoloAgentTrainer(Trainer):
    """Trainer for training agents in isolation.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    scenario_instance : ComponentHolder
        The components of the experiment.
    settings : ExperimentSettings
        The instance-specific settings of the experiment, like device, logging, etc.
    """

    def train(self, as_pretraining: bool = False):
        """Train the agents.

        Parameters
        ----------
        as_pretraining : bool, default=False
            Whether we're training the agents as a pretraining step. This affects the
            output and what we log to W&B.
        """

        torch.manual_seed(self.params.seed)
        np.random.seed(self.params.seed)
        torch_generator = torch.Generator().manual_seed(self.params.seed)

        if self.settings.logger is None:
            self.settings.logger = logging.getLogger(__name__)

        logger = self.settings.logger

        logger.info("Loading dataset and agents...")

        dataset = self.scenario_instance.train_dataset
        train_dataset, test_dataset = random_split(
            dataset,
            (1 - self.params.test_size, self.params.test_size),
        )

        # Select the non-random agents
        agents_params = AgentsParameters(
            **{
                name: params
                for name, params in self.params.agents.items()
                if not params.is_random
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

        # Create the optimizers, specifying the learning rates for the different parts of
        # the agent
        optimizers: dict[str, Optimizer] = {}
        for agent_name, agent in agents.items():
            model_param_dict = agent.get_param_dicts(
                base_lr=self.params.solo_agent.learning_rate,
                body_lr_factor_override=self.params.solo_agent.body_lr_factor,
            )
            optimizers[agent_name] = Adam(model_param_dict)

        # Create the data loaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.params.solo_agent.batch_size,
            shuffle=True,
            generator=torch_generator,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.params.solo_agent.batch_size,
            shuffle=False,
        )

        # Create a progress bar
        desc = "Pretraining" if as_pretraining else "Training"
        pbar = self.settings.tqdm_func(
            total=self.params.solo_agent.num_epochs, desc=desc
        )

        # Train the agents
        for epoch in range(self.params.solo_agent.num_epochs):
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

            # Log to W&B if using
            if self.settings.wandb_run is not None and not as_pretraining:
                to_log = {}
                for agent_name in agents_params:
                    train_loss = total_loss[agent_name] / len(train_dataloader)
                    train_accuracy = total_accuracy[agent_name] / len(train_dataloader)
                    to_log[f"{agent_name}.train_loss"] = train_loss
                    to_log[f"{agent_name}.train_accuracy"] = train_accuracy
                self.settings.wandb_run.log(to_log, step=epoch)

            # If we're in test mode, exit after one iteration
            if self.settings.test_run:
                break

            # Update the progress bar
            pbar.update(1)

        # Close the progress bar
        pbar.close()

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

        # Record the final results with W&B if using
        if self.settings.wandb_run is not None:
            prefix = "pretrain_" if as_pretraining else ""
            to_log = {}
            for agent_name in agents_params:
                test_loss = total_loss[agent_name] / len(test_loader)
                test_accuracy = total_accuracy[agent_name] / len(test_loader)
                to_log[f"{agent_name}.{prefix}test_loss"] = test_loss
                to_log[f"{agent_name}.{prefix}test_accuracy"] = test_accuracy
            self.settings.wandb_run.log(to_log)
