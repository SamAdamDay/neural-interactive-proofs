"""Test solo image classification agents using a grid of hyperparameters.

A solo agent is one which does not interact with any other agents, but instead tries to
solve the image classification problem on its own.
"""

from argparse import Namespace
import os
from typing import Callable
import logging

import torch

from pvg import (
    Parameters,
    AgentsParameters,
    ImageClassificationAgentParameters,
    ImageClassificationParameters,
    SoloAgentParameters,
    DatasetParameters,
    ScenarioType,
    TrainerType,
    BinarificationMethodType,
    run_experiment,
    prepare_experiment,
    PreparedExperimentInfo,
)
from pvg.utils.experiments import (
    MultiprocessHyperparameterExperiment,
    SequentialHyperparameterExperiment,
    ExperimentFunctionArguments,
)

MULTIPROCESS = True
TEST_SIZE = 0.2

param_grid = dict(
    dataset_name=["svhn"],
    num_epochs=[50],
    batch_size=[2],
    learning_rate=[0.001],
    learning_rate_scheduler=[None],
    no_body_lr_factor=[True],
    prover_convs_per_group=[1],
    prover_num_decider_layers=[1],
    verifier_convs_per_group=[1],
    verifier_num_decider_layers=[1],
    num_conv_groups=[1],
    initial_num_channels=[1],
    binarification_method=[BinarificationMethodType.MERGE],
    binarification_seed=[None],
    selected_classes=[None],
    seed=[8144, 820, 4173, 3992],
)


def experiment_fn(arguments: ExperimentFunctionArguments):
    combo = arguments.combo
    cmd_args = arguments.cmd_args
    logger = arguments.child_logger_adapter

    logger.info(f"Starting run {arguments.run_id}")
    logger.debug(f"Combo: {combo}")

    device = torch.device(f"cuda:{cmd_args.gpu_num}")

    # Make sure W&B doesn't print anything when the logger level is higher than DEBUG
    if logger.level > logging.DEBUG:
        os.environ["WANDB_SILENT"] = "true"

    # Create the parameters object
    params = Parameters(
        scenario=ScenarioType.IMAGE_CLASSIFICATION,
        trainer=TrainerType.SOLO_AGENT,
        dataset=combo["dataset_name"],
        agents=AgentsParameters(
            verifier=ImageClassificationAgentParameters(
                num_convs_per_group=combo["verifier_convs_per_group"],
                num_decider_layers=combo["verifier_num_decider_layers"],
            ),
            prover=ImageClassificationAgentParameters(
                num_convs_per_group=combo["prover_convs_per_group"],
                num_decider_layers=combo["prover_num_decider_layers"],
            ),
        ),
        solo_agent=SoloAgentParameters(
            num_epochs=combo["num_epochs"],
            batch_size=combo["batch_size"],
            learning_rate=combo["learning_rate"],
            body_lr_factor_override=combo["no_body_lr_factor"],
        ),
        image_classification=ImageClassificationParameters(
            num_conv_groups=combo["num_conv_groups"],
            initial_num_channels=combo["initial_num_channels"],
        ),
        dataset_options=DatasetParameters(
            binarification_method=combo["binarification_method"],
            binarification_seed=combo["binarification_seed"],
            selected_classes=combo["selected_classes"],
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
        dataset_on_device=cmd_args.dataset_on_device,
        tqdm_func=arguments.tqdm_func,
        ignore_cache=cmd_args.ignore_cache,
        use_wandb=cmd_args.use_wandb,
        wandb_project=cmd_args.wandb_project,
        wandb_entity=cmd_args.wandb_entity,
        run_id=arguments.run_id,
        wandb_tags=wandb_tags,
        global_tqdm_step_fn=arguments.global_tqdm_step_fn,
    )


def run_id_fn(combo_index: int, cmd_args: Namespace) -> str:
    return f"test_solo_ic_agents_{cmd_args.run_infix}_{combo_index}"


def run_preparer_fn(combo: dict, cmd_args: Namespace) -> PreparedExperimentInfo:
    params = Parameters(
        scenario=ScenarioType.IMAGE_CLASSIFICATION,
        trainer=TrainerType.SOLO_AGENT,
        dataset=combo["dataset_name"],
    )
    return prepare_experiment(params=params, ignore_cache=cmd_args.ignore_cache)


if __name__ == "__main__":
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

    experiment.run()
