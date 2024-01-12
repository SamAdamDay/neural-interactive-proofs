"""Build and run an experiment.

This is the main entry point for running an experiment.

When adding a new scenario or a new trainer, add the scenario and trainer to the maps
below.
"""

import sys
import warnings
from typing import Optional

import wandb

from tqdm import tqdm

from pvg.parameters import Parameters, ScenarioType, TrainerType
from pvg.experiment_settings import ExperimentSettings
from pvg.scenario_base import ScenarioInstance
from pvg.graph_isomorphism import GraphIsomorphismScenarioInstance
from pvg.trainers import Trainer, SoloAgentTrainer, PpoTrainer
from pvg.utils.types import TorchDevice, LoggingType
from pvg.constants import WANDB_PROJECT, WANDB_ENTITY

SCENARIO_MAP: dict[ScenarioType, ScenarioInstance] = {
    ScenarioType.GRAPH_ISOMORPHISM: GraphIsomorphismScenarioInstance,
}

TRAINER_MAP: dict[TrainerType, Trainer] = {
    TrainerType.SOLO_AGENT: SoloAgentTrainer,
    TrainerType.PPO: PpoTrainer,
}


def run_experiment(
    params: Parameters,
    device: TorchDevice = "cpu",
    logger: LoggingType = None,
    tqdm_func: callable = tqdm,
    ignore_cache: bool = False,
    use_wandb: bool = False,
    wandb_project: str = WANDB_PROJECT,
    wandb_entity: str = WANDB_ENTITY,
    run_id: Optional[str] = None,
    wandb_tags: list = [],
    num_dataset_threads: int = 8,
    test_run: bool = False,
):
    """Build and run an experiment.

    Builds the experiment components according to the parameters and runs the
    experiment.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    device : TorchDevice, default="cpu"
        The device to use for training.
    logger : logging.Logger | logging.LoggerAdapter, optional
        The logger to log to. If None, the trainer will create a logger.
    tqdm_func : Callable, optional
        The tqdm function to use. Defaults to tqdm.
    ignore_cache : bool, default=False
        If True, when the dataset is loaded, the cache is ignored and the dataset is
        rebuilt from the raw data.
    use_wandb : bool, default=False
        If True, log the experiment to Weights & Biases.
    wandb_project : str, default=WANDB_PROJECT
        The name of the W&B project to log to.
    wandb_entity : str, default=WANDB_ENTITY
        The name of the W&B entity to log to.
    run_id : str, optional
        The ID of the run. Required if use_wandb is True.
    wandb_tags : list[str], default=[]
        The tags to add to the W&B run.
    num_dataset_threads : int, default=8
        The number of threads to use for saving the memory-mapped tensordict.
    test_run : bool, default=False
        If True, the experiment is run in test mode. This means we do the smallest
        number of iterations possible and then exit. This is useful for testing that
        the experiment runs without errors.
    """

    # Set up Weights & Biases.
    if use_wandb:
        if run_id is None:
            raise ValueError("run_id must be specified if use_wandb is True.")
        wandb_run = wandb.init(
            project=wandb_project, entity=wandb_entity, name=run_id, tags=wandb_tags
        )
        wandb_run.config.update(params.to_dict())
    else:
        wandb_run = None

    # Set up the experiment settings
    settings = ExperimentSettings(
        device=device,
        wandb_run=wandb_run,
        tqdm_func=tqdm_func,
        logger=logger,
        ignore_cache=ignore_cache,
        num_dataset_threads=num_dataset_threads,
        test_run=test_run,
    )

    # Build the scenario components of the experiment.
    if params.scenario in SCENARIO_MAP:
        scenario_instance = SCENARIO_MAP[params.scenario](params, settings)
    else:
        raise ValueError(f"Unknown scenario {params.scenario}")

    # Build the trainer.
    if params.trainer in TRAINER_MAP:
        trainer = TRAINER_MAP[params.trainer](params, scenario_instance, settings)
    else:
        raise ValueError(f"Unknown trainer {params.trainer}")

    # Suppress warnings about a batching rule not being implemented by PyTorch for
    # aten::_scaled_dot_product_efficient_attention. We can't do anything about this
    if not sys.warnoptions and not test_run:
        warnings.filterwarnings(
            "ignore",
            message=(
                "There is a performance drop because we have not yet implemented "
                "the batching rule for aten::_scaled_dot_product_efficient_attention"
            ),
            category=UserWarning,
        )

    # Run the experiment.
    trainer.train()

    # Close Weights & Biases.
    if use_wandb:
        wandb_run.finish()
