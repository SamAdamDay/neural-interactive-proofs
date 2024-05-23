"""Build and run an experiment.

This is the main entry point for running an experiment.

When adding a new scenario or a new trainer, add the scenario and trainer to the maps
below.
"""

import sys
import warnings
from typing import Optional
from dataclasses import dataclass

import torch

import wandb

from tqdm import tqdm

from pvg.parameters import Parameters
from pvg.experiment_settings import ExperimentSettings
from pvg.scenario_instance import build_scenario_instance
from pvg.trainers import build_trainer
from pvg.utils.types import TorchDevice, LoggingType
from pvg.constants import WANDB_PROJECT, WANDB_ENTITY
from pvg.protocols import build_protocol_handler
from pvg.stat_logger import WandbStatLogger, DummyStatLogger
import pvg.graph_isomorphism
import pvg.image_classification


def run_experiment(
    params: Parameters,
    device: TorchDevice = "cpu",
    logger: Optional[LoggingType] = None,
    profiler: Optional[torch.profiler.profile] = None,
    tqdm_func: callable = tqdm,
    ignore_cache: bool = False,
    use_wandb: bool = False,
    wandb_project: str = WANDB_PROJECT,
    wandb_entity: str = WANDB_ENTITY,
    run_id: Optional[str] = None,
    allow_auto_generated_run_id: bool = False,
    print_wandb_run_url: bool = False,
    wandb_tags: list = [],
    num_dataset_threads: int = 8,
    pin_memory: bool = True,
    dataset_on_device: bool = False,
    enable_efficient_attention: bool = False,
    global_tqdm_step_fn: callable = lambda: ...,
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
    profiler : torch.profiler.profile, optional
        The PyTorch profiler being used to profile the training, if any.
    tqdm_func : Callable, optional
        The tqdm function to use. Defaults to tqdm.
    ignore_cache : bool, default=False
        If True, the dataset and model cache are ignored and rebuilt.
    use_wandb : bool, default=False
        If True, log the experiment to Weights & Biases.
    wandb_project : str, default=WANDB_PROJECT
        The name of the W&B project to log to.
    wandb_entity : str, default=WANDB_ENTITY
        The name of the W&B entity to log to.
    run_id : str, optional
        The ID of the run. Required if use_wandb is True and allow_auto_generated_run_id
        is False.
    allow_auto_generated_run_id : bool, default=False
        If True, the run ID can be auto-generated if not specified.
    print_wandb_run_url : bool, default=False
        If True, print the URL of the W&B run at the start of the experiment.
    wandb_tags : list[str], default=[]
        The tags to add to the W&B run.
    num_dataset_threads : int, default=8
        The number of threads to use for saving the memory-mapped tensordict.
    pin_memory : bool, default=True
        Whether to pin the memory of the tensors in the dataloader, and move them to the
        GPU with `non_blocking=True`. This can speed up training.
    dataset_on_device : bool, default=False
        Whether store the whole dataset on the device. This can speed up training but
        requires that the dataset fits on the device. This makes `pin_memory` redundant.
    enable_efficient_attention: bool, default=False
        Whether to enable the ' Memory-Efficient Attention' backend for the scaled
        dot-product attention. There may be a bug in this implementation which causes
        NaNs to appear in the backward pass. See
        https://github.com/pytorch/pytorch/issues/119320 for more information.
    global_tqdm_step_fn : Callable, default=lambda: ...
        A function to step the global tqdm progress bar. This is used when there are
        multiple processes running in parallel and each process needs to update the
        global progress bar.
    test_run : bool, default=False
        If True, the experiment is run in test mode. This means we do the smallest
        number of iterations possible and then exit. This is useful for testing that
        the experiment runs without errors.
    """

    # Set up Weights & Biases.
    if use_wandb:
        if run_id is None and not allow_auto_generated_run_id:
            raise ValueError(
                "run_id must be specified if use_wandb is True and "
                "allow_auto_generated_run_id is False."
            )
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_id,
            tags=wandb_tags,
            id=run_id,
            resume="never",
        )
        wandb_run.config.update(params.to_dict())
        if print_wandb_run_url:
            print(f"W&B run URL: {wandb_run.get_url()}")
        stat_logger = WandbStatLogger(wandb_run)
    else:
        wandb_run = None
        stat_logger = DummyStatLogger()

    # Set up the experiment settings
    settings = ExperimentSettings(
        device=device,
        wandb_run=wandb_run,
        stat_logger=stat_logger,
        tqdm_func=tqdm_func,
        logger=logger,
        profiler=profiler,
        ignore_cache=ignore_cache,
        num_dataset_threads=num_dataset_threads,
        pin_memory=pin_memory,
        dataset_on_device=dataset_on_device,
        enable_efficient_attention=enable_efficient_attention,
        global_tqdm_step_fn=global_tqdm_step_fn,
        test_run=test_run,
    )

    # Build the scenario components of the experiment.
    scenario_instance = build_scenario_instance(params, settings)

    # Build the trainer.
    trainer = build_trainer(params, scenario_instance, settings)

    # Suppress warnings about a batching rule not being implemented by PyTorch for
    # aten::_scaled_dot_product_efficient_attention and
    # aten::_scaled_dot_product_attention_math. We can't do anything about this
    if not sys.warnoptions and not test_run:
        warnings.filterwarnings(
            "ignore",
            message=(
                "There is a performance drop because we have not yet implemented "
                "the batching rule for aten::_scaled_dot_product_efficient_attention"
            ),
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=(
                "There is a performance drop because we have not yet implemented "
                "the batching rule for aten::_scaled_dot_product_attention_math"
            ),
            category=UserWarning,
        )

    # Run the experiment.
    trainer.train()

    # Close Weights & Biases.
    if use_wandb:
        wandb_run.finish()


@dataclass
class PreparedExperimentInfo:
    """Information about an experiment that has been prepared using `prepare_experiment`

    Parameters
    ----------
    total_num_iterations : int
        The total number of training iterations.
    """

    total_num_iterations: int


def prepare_experiment(
    params: Parameters,
    profiler: Optional[torch.profiler.profile] = None,
    ignore_cache: bool = False,
    num_dataset_threads: int = 8,
    test_run: bool = False,
):
    """Prepare for running an experiment.

    This is useful e.g. for downloading data before running an experiment. Without this,
    if running multiple experiments in parallel, the initial runs will all start
    downloading data at the same time, which can cause problems.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    profiler : torch.profiler.profile, optional
        The PyTorch profiler being used to profile the training, if any.
    ignore_cache : bool, default=False
        If True, when the dataset is loaded, the cache is ignored and the dataset is
        rebuilt from the raw data.
    num_dataset_threads : int, default=8
        The number of threads to use for saving the memory-mapped tensordict.
    test_run : bool, default=False
        If True, the experiment is run in test mode. This means we do the smallest
        number of iterations possible and then exit. This is useful for testing that
        the experiment runs without errors.

    Returns
    -------
    prepared_experiment_info : PreparedExperimentInfo
        Information about the prepared experiment.
    """

    settings = ExperimentSettings(
        device="cpu",
        wandb_run=None,
        logger=None,
        profiler=profiler,
        ignore_cache=ignore_cache,
        num_dataset_threads=num_dataset_threads,
        test_run=test_run,
    )

    # Build the scenario components of the experiment.
    scenario_instance = build_scenario_instance(params, settings)

    # Build the trainer.
    trainer = build_trainer(params, scenario_instance, settings)

    # Get the total number of training iterations.
    total_num_iterations = trainer.get_total_num_iterations()

    del scenario_instance
    del trainer

    return PreparedExperimentInfo(total_num_iterations=total_num_iterations)
