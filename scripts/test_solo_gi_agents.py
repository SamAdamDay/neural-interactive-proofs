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

from pvg.parameters import (
    Parameters,
    GraphIsomorphismParameters,
    GraphIsomorphismAgentParameters,
    SoloAgentParameters,
)
from pvg.utils.experiments import (
    MultiprocessHyperparameterExperiment,
    SequentialHyperparameterExperiment,
)
from pvg.extra.test_solo_gi_agents import train_and_test_solo_gi_agents
from pvg.constants import GI_SOLO_AGENTS_RESULTS_DATA_DIR

MULTIPROCESS = False
TEST_SIZE = 0.2

param_grid = dict(
    dataset_name=["eru10000"],
    d_gnn=[16],
    d_decider=[16],
    use_batch_norm=[True],
    use_pair_invariant_pooling=[True],
    noise_sigma=[0.0],
    num_epochs=[500],
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
        scenario="graph_isomorphism",
        trainer="solo_agent",
        dataset=combo["dataset_name"],
        graph_isomorphism=GraphIsomorphismParameters(
            prover=GraphIsomorphismAgentParameters(
                d_gnn=combo["d_gnn"],
                d_decider=combo["d_decider"],
                use_batch_norm=combo["use_batch_norm"],
                noise_sigma=combo["noise_sigma"],
                use_pair_invariant_pooling=combo["use_pair_invariant_pooling"],
                num_decider_layers=combo["prover_num_layers"],
            ),
            verifier=GraphIsomorphismAgentParameters(
                d_gnn=combo["d_gnn"],
                d_decider=combo["d_decider"],
                use_batch_norm=combo["use_batch_norm"],
                noise_sigma=combo["noise_sigma"],
                use_pair_invariant_pooling=combo["use_pair_invariant_pooling"],
                num_decider_layers=combo["verifier_num_layers"],
            ),
        ),
        solo_agent=SoloAgentParameters(
            num_epochs=combo["num_epochs"],
            batch_size=combo["batch_size"],
            learning_rate=combo["learning_rate"],
            body_lr_factor=combo["body_lr_factor"],
        ),
    )

    # Set up W&B
    use_wandb = cmd_args.wandb_project != ""
    if use_wandb:
        wandb_tags = [cmd_args.tag] if cmd_args.tag != "" else []
        wandb_run = wandb.init(
            project=cmd_args.wandb_project, name=run_id, tags=wandb_tags
        )
        wandb_run.config.update(params.to_dict())

    # Train and test the agents to get the results
    _, _, results = train_and_test_solo_gi_agents(
        params=params,
        test_size=TEST_SIZE,
        wandb_run=wandb_run if use_wandb else None,
        device=device,
        tqdm_func=tqdm_func,
        logger=logger,
        ignore_cache=cmd_args.ignore_cache,
    )

    if use_wandb:
        wandb_run.finish()
    else:
        # Convert any numpy arrays to lists
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results[key] = value.tolist()

        results["run_id"] = run_id
        results["combo"] = combo

        # Save the results
        logger.info(f"Saving results locally...")
        filename = f"{run_id}.json"
        filepath = os.path.join(GI_SOLO_AGENTS_RESULTS_DATA_DIR, filename)
        with open(filepath, "w") as f:
            json.dump(results, f)


def run_id_fn(combo_index: int, cmd_args: Namespace):
    return f"test_solo_gi_agents_{cmd_args.run_infix}_{combo_index}"


if __name__ == "__main__":
    # Make sure the results directory exists
    Path(GI_SOLO_AGENTS_RESULTS_DATA_DIR).mkdir(parents=True, exist_ok=True)

    if MULTIPROCESS:
        experiment_class = MultiprocessHyperparameterExperiment
    else:
        experiment_class = SequentialHyperparameterExperiment

    experiment = experiment_class(
        param_grid=param_grid,
        experiment_fn=experiment_fn,
        run_id_fn=run_id_fn,
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
        "--wandb-project",
        type=str,
        help="The name of the W&B project to use. If not set saves the results locally",
        default="",
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
