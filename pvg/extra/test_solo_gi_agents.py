from abc import ABC
from typing import Optional, Callable
import logging

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.optim import Adam, Optimizer

from tensordict import TensorDict

from tqdm import tqdm

import wandb

from pvg.base import DataLoader
from pvg.graph_isomorphism import build_agents, GraphIsomorphismDataset
from pvg.parameters import Parameters


def train_and_test_solo_gi_agents(
    params: Parameters,
    test_size: float,
    device: str | torch.device,
    wandb_run: Optional[wandb.wandb_sdk.wandb_run.Run] = None,
    tqdm_func: Callable = tqdm,
    logger: Optional[logging.Logger | logging.LoggerAdapter] = None,
    ignore_cache: bool = False,
) -> tuple[nn.Module, nn.Module, dict]:
    """Train and test solo GI agents.

    Parameters
    ----------
    params : Parameters
        The parameters for the experiment.
    test_size : float
        The size of the test set as a fraction of the dataset size.
    device : str | torch.device
        The device to use.
    wandb_run : wandb.wandb_sdk.wandb_run.Run, optional
        The W&B run to log to, if any.
    tqdm_func : Callable, optional
        The tqdm function to use. Defaults to tqdm.
    logger : logging.Logger | logging.LoggerAdapter, optional
        The logger to log to. If None, creates a new logger.
    ignore_cache : bool, default=False
        If True, when the dataset is loaded, the cache is ignored and the dataset is
        rebuilt from the raw data.
    """

    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    torch_generator = torch.Generator().manual_seed(params.seed)

    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Loading dataset and agents...")

    dataset = GraphIsomorphismDataset(params, ignore_cache=ignore_cache)
    train_dataset, test_dataset = random_split(dataset, (1 - test_size, test_size))

    # Create the agents
    agent_names = ["prover", "verifier"]
    agents = build_agents(params, device)

    # Create the optimizers, specifying the learning rates for the different parts of
    # the agent
    optimizers: dict[str, Optimizer] = {}
    for agent_name in agent_names:
        model_param_dict = [
            {
                "params": agents[agent_name]["body"].parameters(),
                "lr": params.solo_agent.learning_rate
                * params.solo_agent.body_lr_factor,
            },
            {
                "params": agents[agent_name]["solo_head"].parameters(),
                "lr": params.solo_agent.learning_rate,
            },
        ]
        optimizers[agent_name] = Adam(model_param_dict)

    # Create the data loaders
    test_loader = DataLoader(
        train_dataset,
        batch_size=params.solo_agent.batch_size,
        shuffle=True,
        generator=torch_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=params.solo_agent.batch_size,
        shuffle=False,
    )

    # Train the prover and verifier
    with tqdm_func(total=params.solo_agent.num_epochs, desc="Training") as pbar:
        for epoch in range(params.solo_agent.num_epochs):
            total_loss = {agent_name: 0 for agent_name in agent_names}
            total_accuracy = {agent_name: 0 for agent_name in agent_names}

            for data in test_loader:
                data : TensorDict
                data = data.to(device)

                # Set the message to zero and ignore it. Needed because the solo agent
                # expects a message
                data["message"] = torch.zeros(data.batch_size, dtype=torch.long, device=device)
                data["ignore_message"] = torch.ones(
                    data.batch_size, device=device, dtype=torch.bool
                )

                # Train the agents on the batch
                for agent_name in agent_names:
                    agents[agent_name]["body"].train()
                    agents[agent_name]["solo_head"].train()
                    optimizers[agent_name].zero_grad()

                    body_output = agents[agent_name]["body"](data)
                    head_output = agents[agent_name]["solo_head"](body_output)
                    logits = head_output["decision_logits"]
                    loss = F.cross_entropy(logits, data["y"])

                    loss.backward()
                    optimizers[agent_name].step()

                    with torch.no_grad():
                        accuracy = (
                            (logits.argmax(dim=1) == data["y"]).float().mean().item()
                        )

                    total_loss[agent_name] += loss.item()
                    total_accuracy[agent_name] += accuracy

            # Log to W&B if using
            if wandb_run is not None:
                to_log = {}
                for agent_name in agent_names:
                    train_loss = total_loss[agent_name] / len(test_loader)
                    train_accuracy = total_accuracy[agent_name] / len(test_loader)
                    to_log[f"{agent_name}.train_loss"] = train_loss
                    to_log[f"{agent_name}.train_accuracy"] = train_accuracy
                wandb_run.log(to_log, step=epoch)

            # Update the progress bar
            pbar.update(1)

    total_loss = {agent_name: 0 for agent_name in agent_names}
    total_accuracy = {agent_name: 0 for agent_name in agent_names}

    # Test the prover and verifier
    logger.info("Testing...")
    for data in test_loader:
        data = data.to(device)

        for agent_name in agent_names:
            agents[agent_name]["body"].eval()
            agents[agent_name]["solo_head"].eval()

            with torch.no_grad():
                body_output = agents[agent_name]["body"](data)
                head_output = agents[agent_name]["solo_head"](body_output)
                logits = head_output["decision_logits"]
                loss = F.cross_entropy(logits, data["y"])
                accuracy = (logits.argmax(dim=1) == data["y"]).float().mean().item()

            total_loss[agent_name] += loss
            total_accuracy[agent_name] += accuracy

    # Record the final results with W&B if using
    if wandb_run is not None:
        to_log = {}
        for agent_name in agent_names:
            test_loss = total_loss[agent_name] / len(test_loader)
            test_accuracy = total_accuracy[agent_name] / len(test_loader)
            to_log[f"{agent_name}.test_loss"] = test_loss
            to_log[f"{agent_name}.test_accuracy"] = test_accuracy
        wandb_run.log(to_log)
