"""Test solo image classification agents using a grid of hyperparameters.

A solo agent is one which does not interact with any other agents, but instead tries to
solve the image classification problem on its own.
"""

from argparse import Namespace
import os
from typing import Callable
import logging

import torch

from nip import (
    HyperParameters,
    AgentsParameters,
    ImageClassificationAgentParameters,
    ImageClassificationParameters,
    SoloAgentParameters,
    DatasetParameters,
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
    dataset_name=["cifar10"],
    num_epochs=[50],
    batch_size=[256],
    learning_rate=[0.001],
    learning_rate_scheduler=[None],
    no_body_lr_factor=[True],
    prover_blocks_per_group=[4],
    prover_num_decider_layers=[3],
    prover_block_type=["conv2d"],
    prover_pretrained_embeddings_model=["resnet18"],
    verifier_blocks_per_group=[1],
    verifier_num_decider_layers=[2],
    verifier_block_type=["conv2d"],
    verifier_pretrained_embeddings_model=[None],
    num_block_groups=[1],
    initial_num_channels=[16],
    binarification_method=["select_two"],
    binarification_seed=[None],
    selected_classes=[None],
    seed=[8144, 820, 4173, 3992],
)


def _construct_params(combo: dict, cmd_args: Namespace) -> HyperParameters:
    return HyperParameters(
        scenario="image_classification",
        trainer="solo_agent",
        dataset=combo["dataset_name"],
        agents=AgentsParameters(
            verifier=ImageClassificationAgentParameters(
                num_blocks_per_group=combo["verifier_blocks_per_group"],
                num_decider_layers=combo["verifier_num_decider_layers"],
                building_block_type=combo["verifier_block_type"],
                pretrained_embeddings_model=combo[
                    "verifier_pretrained_embeddings_model"
                ],
            ),
            prover=ImageClassificationAgentParameters(
                num_blocks_per_group=combo["prover_blocks_per_group"],
                num_decider_layers=combo["prover_num_decider_layers"],
                building_block_type=combo["prover_block_type"],
                pretrained_embeddings_model=combo["prover_pretrained_embeddings_model"],
            ),
        ),
        solo_agent=SoloAgentParameters(
            num_epochs=combo["num_epochs"],
            batch_size=combo["batch_size"],
            learning_rate=combo["learning_rate"],
            body_lr_factor_override=combo["no_body_lr_factor"],
        ),
        image_classification=ImageClassificationParameters(
            num_block_groups=combo["num_block_groups"],
            initial_num_channels=combo["initial_num_channels"],
        ),
        dataset_options=DatasetParameters(
            binarification_method=combo["binarification_method"],
            binarification_seed=combo["binarification_seed"],
            selected_classes=combo["selected_classes"],
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
        dataset_on_device=cmd_args.dataset_on_device,
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
        return f"test_solo_ic_agents_{cmd_args.run_infix}"
    return f"test_solo_ic_agents_{cmd_args.run_infix}_{combo_index}"


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
    experiment_name="TEST_SOLO_IC_AGENTS",
    **extra_args,
)

experiment.parser.add_argument(
    "--dataset-on-device",
    action="store_true",
    dest="dataset_on_device",
    help="Store the whole dataset on the device (needs more GPU memory).",
)

# Set the `parser` module attribute to enable the script auto-documented by Sphinx
parser = experiment.parser

if __name__ == "__main__":
    experiment.run()
