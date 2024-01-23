"""Test solo graph isomorphism agents using a grid of hyperparameters.

A solo agent is one which does not interact with any other agents, but instead tries to
solve the graph isomorphism problem on its own.
"""

from argparse import Namespace
import os
import json
from pathlib import Path
from typing import Callable
import logging

import numpy as np

import torch

import wandb

from pvg import (
    Parameters,
    AgentsParameters,
    GraphIsomorphismAgentParameters,
    SoloAgentParameters,
    ScenarioType,
    TrainerType,
    run_experiment,
    prepare_experiment,
)
from pvg.utils.experiments import (
    MultiprocessHyperparameterExperiment,
    SequentialHyperparameterExperiment,
)
from pvg.constants import WANDB_ENTITY, WANDB_PROJECT

MULTIPROCESS = True
TEST_SIZE = 0.2

param_grid = dict(
    dataset_name=["eru10000"],
    use_batch_norm=[True],
    use_pair_invariant_pooling=[True],
    num_epochs=[100],
    batch_size=[256],
    learning_rate=[0.001],
    learning_rate_scheduler=[None],
    freeze_body=[False],
    body_lr_factor=[0.01],
    prover_num_layers=[5],
    verifier_num_layers=[2],
    seed=[8144, 820, 4173, 3992],
)


def experiment_fn(
    combo: dict,
    run_id: str,
    cmd_args: Namespace,
    tqdm_func: Callable,
    logger: logging.Logger,
):
    logger.info(f"Starting run {run_id}")
    logger.debug(f"Combo: {combo}")

    device = torch.device(f"cuda:{cmd_args.gpu_num}")

    # Make sure W&B doesn't print anything when the logger level is higher than DEBUG
    if logger.level > logging.DEBUG:
        os.environ["WANDB_SILENT"] = "true"

    # Create the parameters object
    params = Parameters(
        scenario=ScenarioType.GRAPH_ISOMORPHISM,
        trainer=TrainerType.SOLO_AGENT,
        dataset=combo["dataset_name"],
        agents=AgentsParameters(
            [
                (
                    "prover",
                    GraphIsomorphismAgentParameters(
                        use_batch_norm=combo["use_batch_norm"],
                        use_pair_invariant_pooling=combo["use_pair_invariant_pooling"],
                        num_gnn_layers=combo["prover_num_layers"],
                        body_lr_factor=combo["body_lr_factor"],
                    ),
                ),
                (
                    "verifier",
                    GraphIsomorphismAgentParameters(
                        use_batch_norm=combo["use_batch_norm"],
                        use_pair_invariant_pooling=combo["use_pair_invariant_pooling"],
                        num_gnn_layers=combo["verifier_num_layers"],
                        body_lr_factor=combo["body_lr_factor"],
                    ),
                ),
            ]
        ),
        solo_agent=SoloAgentParameters(
            num_epochs=combo["num_epochs"],
            batch_size=combo["batch_size"],
            learning_rate=combo["learning_rate"],
        ),
        seed=combo["seed"],
    )

    if cmd_args.use_wandb:
        wandb_tags = [cmd_args.tag] if cmd_args.tag != "" else []
    else:
        wandb_tags = []

    # Train and test the agents
    run_experiment(
        params,
        device=device,
        logger=logger,
        tqdm_func=tqdm_func,
        ignore_cache=cmd_args.ignore_cache,
        use_wandb=cmd_args.use_wandb,
        wandb_project=cmd_args.wandb_project,
        wandb_entity=cmd_args.wandb_entity,
        run_id=run_id,
        wandb_tags=wandb_tags,
    )


def run_id_fn(combo_index: int, cmd_args: Namespace):
    return f"test_solo_gi_agents_{cmd_args.run_infix}_{combo_index}"


def run_preparer_fn(combo: dict, cmd_args: Namespace):
    params = Parameters(
        scenario=ScenarioType.GRAPH_ISOMORPHISM,
        trainer=TrainerType.SOLO_AGENT,
        dataset=combo["dataset_name"],
    )
    prepare_experiment(params=params, ignore_cache=cmd_args.ignore_cache)


if __name__ == "__main__":
    if MULTIPROCESS:
        experiment_class = MultiprocessHyperparameterExperiment
    else:
        experiment_class = SequentialHyperparameterExperiment

    experiment = experiment_class(
        param_grid=param_grid,
        experiment_fn=experiment_fn,
        run_id_fn=run_id_fn,
        run_preparer_fn=run_preparer_fn,
        experiment_name="TEST_SOLO_GI_AGENTS",
    )
    experiment.parser.add_argument(
        "--run-infix",
        type=str,
        help="The string to add in the middle of the run ID",
        default="a",
    )
    experiment.parser.add_argument(
        "--gpu-num", type=int, help="The (0-based) number of the GPU to use", default=0
    )
    experiment.parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Whether to use W&B to log the experiment",
    )
    experiment.parser.add_argument(
        "--wandb-project",
        type=str,
        help="The name of the W&B project to use",
        default=WANDB_PROJECT,
    )
    experiment.parser.add_argument(
        "--wandb-entity",
        type=str,
        help="The name of the W&B entity to use",
        default=WANDB_ENTITY,
    )
    experiment.parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="An optional tag for the W&B run",
    )
    experiment.parser.add_argument(
        "--ignore-cache",
        action="store_true",
        help="Ignore the cache and rebuild the dataset from the raw data",
    )
    experiment.run()
