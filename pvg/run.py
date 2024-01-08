"""Build and run an experiment.

This is the main entry point for running an experiment.
"""

from typing import Optional

import wandb

from tqdm import tqdm

from pvg.parameters import Parameters, ScenarioType, TrainerType
from pvg.experiment_settings import ExperimentSettings
from pvg.graph_isomorphism import GraphIsomorphismComponentHolder
from pvg.trainers import SoloAgentTrainer
from pvg.utils.types import TorchDevice, LoggingType


def run_experiment(
    params: Parameters,
    device: TorchDevice = "cpu",
    logger: LoggingType = None,
    tqdm_func: callable = tqdm,
    ignore_cache: bool = False,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    run_id: Optional[str] = None,
    wandb_tags: list = [],
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
    wandb_project : str, optional
        The name of the W&B project to log to. Required if use_wandb is True.
    run_id : str, optional
        The ID of the run. Required if use_wandb is True.
    wandb_tags : list[str], default=[]
        The tags to add to the W&B run.
    """

    # Set up Weights & Biases.
    if use_wandb:
        if wandb_project is None:
            raise ValueError("wandb_project must be specified if use_wandb is True.")
        if run_id is None:
            raise ValueError("run_id must be specified if use_wandb is True.")
        wandb_run = wandb.init(
            project=wandb_project, name=run_id, tags=wandb_tags
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
    )

    # Build the scenario components of the experiment.
    if params.scenario == ScenarioType.GRAPH_ISOMORPHISM:
        component_holder = GraphIsomorphismComponentHolder(params, device)
    else:
        raise ValueError(f"Unknown scenario {params.scenario}")

    # Build the trainer.
    if params.trainer == TrainerType.SOLO_AGENT:
        trainer = SoloAgentTrainer(params, component_holder, settings)
    else:
        raise ValueError(f"Unknown trainer {params.trainer}")

    # Run the experiment.
    trainer.train()

    # Close Weights & Biases.
    if use_wandb:
        wandb_run.finish()
