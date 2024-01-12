"""Test solo graph isomorphism agents using a grid of hyperparameters.

A solo agent is one which does not interact with any other agents, but instead tries to
solve the graph isomorphism problem on its own.
"""

from argparse import Namespace
import os
import logging

import torch

from pvg import (
    Parameters,
    AgentsParameters,
    GraphIsomorphismAgentParameters,
    PpoParameters,
    ScenarioType,
    TrainerType,
    run_experiment,
)
from pvg.utils.experiments import (
    MultiprocessHyperparameterExperiment,
    SequentialHyperparameterExperiment,
)

MULTIPROCESS = True

param_grid = dict(
    dataset_name=["eru10000"],
    num_iterations=[1000],
    num_epochs=[4],
    minibatch_size=[256],
    gamma=[0.99],
    lmbda=[0.95],
    clip_epsilon=[0.2],
    entropy_eps=[0.0001],
    body_lr_factor=[0.01],
    prover_num_layers=[5],
    verifier_num_layers=[2],
    seed=[8144, 820, 4173, 3992],
)


def experiment_fn(
    combo: dict,
    run_id: str,
    cmd_args: Namespace,
    tqdm_func: callable,
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
        trainer=TrainerType.PPO,
        dataset=combo["dataset_name"],
        agents=AgentsParameters(
            [
                (
                    "prover",
                    GraphIsomorphismAgentParameters(
                        num_gnn_layers=combo["prover_num_layers"],
                    ),
                ),
                (
                    "verifier",
                    GraphIsomorphismAgentParameters(
                        num_gnn_layers=combo["verifier_num_layers"],
                    ),
                ),
            ]
        ),
        ppo=PpoParameters(
            num_iterations=combo["num_iterations"],
            num_epochs=combo["num_epochs"],
            minibatch_size=combo["minibatch_size"],
            gamma=combo["gamma"],
            lmbda=combo["lmbda"],
            clip_epsilon=combo["clip_epsilon"],
            entropy_eps=combo["entropy_eps"],
        ),
        seed=combo["seed"],
    )

    use_wandb = cmd_args.wandb_project != ""
    if use_wandb:
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
        use_wandb=use_wandb,
        wandb_project=cmd_args.wandb_project,
        run_id=run_id,
        wandb_tags=wandb_tags,
    )


def run_id_fn(combo_index: int, cmd_args: Namespace):
    return f"ppo_gi_{cmd_args.run_infix}_{combo_index}"


if __name__ == "__main__":
    if MULTIPROCESS:
        experiment_class = MultiprocessHyperparameterExperiment
    else:
        experiment_class = SequentialHyperparameterExperiment

    experiment = experiment_class(
        param_grid=param_grid,
        experiment_fn=experiment_fn,
        run_id_fn=run_id_fn,
        experiment_name="PPO_GI",
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
        help="The name of the W&B project to use",
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
