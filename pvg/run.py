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

from pvg.parameters import HyperParameters
from pvg.experiment_settings import ExperimentSettings
from pvg.factory import build_scenario_instance
from pvg.trainers import build_trainer
from pvg.utils.types import TorchDevice, LoggingType
from pvg.utils.env import get_env_var
from pvg.protocols import build_protocol_handler
from pvg.stat_logger import WandbStatLogger, DummyStatLogger
from pvg.base_run import get_base_wandb_run_and_new_hyper_params
import pvg.graph_isomorphism
import pvg.image_classification
import pvg.code_validation


def run_experiment(
    hyper_params: HyperParameters,
    device: TorchDevice = "cpu",
    logger: Optional[LoggingType] = None,
    profiler: Optional[torch.profiler.profile] = None,
    tqdm_func: callable = tqdm,
    ignore_cache: bool = False,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    run_id: Optional[str] = None,
    allow_auto_generated_run_id: bool = False,
    allow_resuming_wandb_run: bool = False,
    allow_overriding_wandb_config: bool = False,
    print_wandb_run_url: bool = False,
    wandb_tags: list = [],
    wandb_group: Optional[str] = None,
    num_dataset_threads: int = 8,
    num_rollout_workers: int = 4,
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
    hyper_params : HyperParameters
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
    wandb_project : str, optional
        The name of the W&B project to log to. If None, the default project is used.
    wandb_entity : str, optional
        The name of the W&B entity to log to. If None, the default entity is used.
    run_id : str, optional
        The ID of the run. Required if use_wandb is True and allow_auto_generated_run_id
        is False.
    allow_auto_generated_run_id : bool, default=False
        If True, the run ID can be auto-generated if not specified.
    allow_resuming_wandb_run : bool, default=False
        If True, the run can be resumed if the run ID is specified and the run exists.
    allow_overriding_wandb_config : bool, default=False
        If True, the W&B config can be overridden when resuming a run.
    print_wandb_run_url : bool, default=False
        If True, print the URL of the W&B run at the start of the experiment.
    wandb_tags : list[str], default=[]
        The tags to add to the W&B run.
    wandb_group : str, optional
        The name of the W&B group for the run. Runs with the same group are placed
        together in the UI. This is useful for doing multiple runs on the same machine.
    num_dataset_threads : int, default=8
        The number of threads to use for saving the memory-mapped tensordict.
    num_rollout_workers : int, default=4
        The number of workers to use for collecting rollout samples, when this is done
        in parallel.
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
        number of iterations possible and then exit. This is useful for testing that the
        experiment runs without errors.
    """

    # Get the base run and new hyper-parameters, if using a base run
    base_run, hyper_params = get_base_wandb_run_and_new_hyper_params(hyper_params)

    # Set up Weights & Biases.
    if use_wandb:
        if wandb_project is None:
            wandb_project = get_env_var("WANDB_PROJECT")
        if wandb_entity is None:
            wandb_entity = get_env_var("WANDB_ENTITY")
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
            group=wandb_group,
            id=run_id,
            resume="allow" if allow_resuming_wandb_run else "never",
        )
        wandb_run.config.update(
            hyper_params.to_dict(), allow_val_change=allow_overriding_wandb_config
        )
        if print_wandb_run_url:
            print(f"W&B run URL: {wandb_run.get_url()}")  # noqa: T201
        stat_logger = WandbStatLogger(wandb_run)
    else:
        wandb_run = None
        stat_logger = DummyStatLogger()

    # Set up the experiment settings
    settings = ExperimentSettings(
        device=device,
        run_id=run_id,
        wandb_run=wandb_run,
        stat_logger=stat_logger,
        tqdm_func=tqdm_func,
        logger=logger,
        profiler=profiler,
        ignore_cache=ignore_cache,
        base_wandb_run=base_run,
        num_dataset_threads=num_dataset_threads,
        num_rollout_workers=num_rollout_workers,
        pin_memory=pin_memory,
        dataset_on_device=dataset_on_device,
        enable_efficient_attention=enable_efficient_attention,
        global_tqdm_step_fn=global_tqdm_step_fn,
        test_run=test_run,
    )

    # Build the scenario components of the experiment.
    scenario_instance = build_scenario_instance(hyper_params, settings)

    # Build the trainer.
    trainer = build_trainer(hyper_params, scenario_instance, settings)

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
    """Information about an experiment prepared using `prepare_experiment`.

    Parameters
    ----------
    total_num_iterations : int
        The total number of training iterations.
    """

    total_num_iterations: int


def prepare_experiment(
    hyper_params: HyperParameters,
    profiler: Optional[torch.profiler.profile] = None,
    ignore_cache: bool = False,
    num_dataset_threads: int = 8,
    device: Optional[TorchDevice] = None,
    test_run: bool = False,
) -> PreparedExperimentInfo:
    """Prepare for running an experiment.

    This is useful e.g. for downloading data before running an experiment. Without this,
    if running multiple experiments in parallel, the initial runs will all start
    downloading data at the same time, which can cause problems.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    profiler : torch.profiler.profile, optional
        The PyTorch profiler being used to profile the training, if any.
    ignore_cache : bool, default=False
        If True, when the dataset is loaded, the cache is ignored and the dataset is
        rebuilt from the raw data.
    num_dataset_threads : int, default=8
        The number of threads to use for saving the memory-mapped tensordict.
    device : TorchDevice, optional
        The device to use for training. If None, the GPU is used if available, otherwise
        the CPU is used.
    test_run : bool, default=False
        If True, the experiment is run in test mode. This means we do the smallest
        number of iterations possible and then exit. This is useful for testing that
        the experiment runs without errors.

    Returns
    -------
    prepared_experiment_info : PreparedExperimentInfo
        Information about the prepared experiment.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    settings = ExperimentSettings(
        device=device,
        wandb_run=None,
        logger=None,
        profiler=profiler,
        ignore_cache=ignore_cache,
        num_dataset_threads=num_dataset_threads,
        test_run=test_run,
    )

    # Build the scenario components of the experiment.
    scenario_instance = build_scenario_instance(hyper_params, settings)

    # Build the trainer.
    trainer = build_trainer(hyper_params, scenario_instance, settings)

    # Get the total number of training iterations.
    total_num_iterations = trainer.get_total_num_iterations()

    del scenario_instance
    del trainer

    return PreparedExperimentInfo(total_num_iterations=total_num_iterations)
