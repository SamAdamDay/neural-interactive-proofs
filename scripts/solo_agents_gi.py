"""Test solo graph isomorphism agents using a grid of hyperparameters.

A solo agent is one which does not interact with any other agents, but instead tries to
solve the graph isomorphism problem on its own. Solo agents are trained using supervised
learning.

This script runs through a grid of hyperparameters, specified in the ``param_grid``
dict. If the ``MULTIPROCESS`` variable is set to True, the experiments are run using a
pool of workers (specified by the ``--num-workers`` command line argument). Otherwise,
the experiments are run sequentially.

Additional settings, like whether to log to W&B can be set via command line arguments.
Run the script with the ``--help`` flag to see all available arguments.
"""

from argparse import Namespace
import os
import json
from pathlib import Path
from typing import Callable
import logging

import numpy as np

import torch

from nip import (
    HyperParameters,
    AgentsParameters,
    GraphIsomorphismAgentParameters,
    SoloAgentParameters,
    run_experiment,
    prepare_experiment,
    PreparedExperimentInfo,
)
from nip.utils.experiments import (
    MultiprocessHyperparameterExperiment,
    SequentialHyperparameterExperiment,
    ExperimentFunctionArguments,
)

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
    body_lr_factor=[{"actor": 0.01, "critic": 0.01}],
    prover_num_layers=[5],
    verifier_num_layers=[2],
    seed=[8144, 820, 4173, 3992],
)


def _construct_params(combo: dict, cmd_args: Namespace) -> HyperParameters:
    return HyperParameters(
        scenario="graph_isomorphism",
        trainer="solo_agent",
        dataset=combo["dataset_name"],
        agents=AgentsParameters(
            verifier=GraphIsomorphismAgentParameters(
                use_batch_norm=combo["use_batch_norm"],
                use_pair_invariant_pooling=combo["use_pair_invariant_pooling"],
                num_gnn_layers=combo["verifier_num_layers"],
                body_lr_factor=combo["body_lr_factor"],
            ),
            prover=GraphIsomorphismAgentParameters(
                use_batch_norm=combo["use_batch_norm"],
                use_pair_invariant_pooling=combo["use_pair_invariant_pooling"],
                num_gnn_layers=combo["prover_num_layers"],
                body_lr_factor=combo["body_lr_factor"],
            ),
        ),
        solo_agent=SoloAgentParameters(
            num_epochs=combo["num_epochs"],
            batch_size=combo["batch_size"],
            learning_rate=combo["learning_rate"],
        ),
        seed=combo["seed"],
    )


def experiment_fn(arguments: ExperimentFunctionArguments):
    """Run a single experiment.

    Parameters
    ----------
    arguments : ExperimentFunctionArguments
        The arguments for the experiment.
    """

    combo = arguments.combo
    cmd_args = arguments.cmd_args
    logger = arguments.child_logger_adapter

    logger.info(f"Starting run {arguments.run_id}")
    logger.debug(f"Combo: {combo}")

    device = torch.device(f"cuda:{cmd_args.gpu_num}")

    # Make sure W&B doesn't print anything when the logger level is higher than DEBUG
    if logger.level > logging.DEBUG:
        os.environ["WANDB_SILENT"] = "true"

    if cmd_args.use_wandb:
        wandb_tags = [cmd_args.tag] if cmd_args.tag != "" else []
    else:
        wandb_tags = []

    hyper_params = _construct_params(combo, cmd_args)

    # Train and test the agents
    run_experiment(
        hyper_params,
        device=device,
        logger=logger,
        tqdm_func=arguments.tqdm_func,
        ignore_cache=cmd_args.ignore_cache,
        use_wandb=cmd_args.use_wandb,
        wandb_project=cmd_args.wandb_project,
        wandb_entity=cmd_args.wandb_entity,
        run_id=arguments.run_id,
        wandb_tags=wandb_tags,
        global_tqdm_step_fn=arguments.global_tqdm_step_fn,
        wandb_group=arguments.common_run_name,
    )


def run_id_fn(combo_index: int | None, cmd_args: Namespace) -> str:
    """Generate the run ID for a given hyperparameter combination.

    Parameters
    ----------
    combo_index : int | None
        The index of the hyperparameter combination. If None, the run ID is for the
        entire experiment.
    cmd_args : Namespace
        The command line arguments.

    Returns
    -------
    run_id : str
        The run ID.
    """
    if combo_index is None:
        return f"test_solo_gi_agents_{cmd_args.run_infix}"
    return f"test_solo_gi_agents_{cmd_args.run_infix}_{combo_index}"


def run_preparer_fn(combo: dict, cmd_args: Namespace) -> PreparedExperimentInfo:
    """Prepare the experiment for a single run.

    Parameters
    ----------
    combo : dict
        The hyperparameter combination to use.
    cmd_args : Namespace
        The command line arguments.

    Returns
    -------
    prepared_experiment_info : PreparedExperimentInfo
        The prepared experiment data.
    """
    hyper_params = _construct_params(combo, cmd_args)
    return prepare_experiment(
        hyper_params=hyper_params, ignore_cache=cmd_args.ignore_cache
    )


if MULTIPROCESS:
    experiment_class = MultiprocessHyperparameterExperiment
    extra_args = dict(default_num_workers=4)
else:
    experiment_class = SequentialHyperparameterExperiment
    extra_args = dict()

experiment = experiment_class(
    param_grid=param_grid,
    experiment_fn=experiment_fn,
    run_id_fn=run_id_fn,
    run_preparer_fn=run_preparer_fn,
    experiment_name="TEST_SOLO_GI_AGENTS",
    **extra_args,
)

# Set the `parser` module attribute to enable the script auto-documented by Sphinx
parser = experiment.parser

if __name__ == "__main__":
    experiment.run()
