"""Instance-specific settings for the experiment, like torch device, logging, etc.

Changing these settings should not effect the reproducibility of the experiment.
"""

from typing import Optional
from dataclasses import dataclass, field
import logging

import torch

import wandb

from tqdm import tqdm

from pvg.stat_logger import StatLogger, DummyStatLogger
from pvg.utils.types import TorchDevice, LoggingType


@dataclass
class ExperimentSettings:
    """Instance-specific settings for the experiment.

    Parameters
    ----------
    device : TorchDevice, default="cpu"
        The device to use for training.
    wandb_run : wandb.wandb_sdk.wandb_run.Run, optional
        The W&B run to log to, if any.
    silence_wandb : bool, default=True
        Whether to suppress W&B output.
    stat_logger : StatLogger, optional
        The logger to use for logging statistics. If not provided, a dummy logger is
        used, which does nothing.
    tqdm_func : Callable, optional
        The tqdm function to use. Defaults to tqdm.
    logger : logging.Logger | logging.LoggerAdapter, optional
        The logger to log to. If None, the trainer will create a logger.
    profiler : torch.profiler.profile, optional
        The PyTorch profiler being used to profile the training, if any.
    ignore_cache : bool, default=False
        If True, the dataset and model cache are ignored and rebuilt.
    num_rollout_samples : int, default=10
        The number of rollout samples to collect and save per iteration of RL training.
        These are useful to visualize the progress of the training.
    rollout_sample_period : int, default=1000
        The frequency with which to collect rollout samples. This is the number of
        iterations of RL training between each collection of rollout samples.
    save_models_period : int, default=1000
        The frequency with which to save the models. This is the number of iterations of
        RL training between each save of the models.
    num_dataset_threads : int, default=8
        The number of threads to use for saving the memory-mapped tensordict.
    pin_memory : bool, default=True
        Whether to pin the memory of the tensors in the dataloader, and move them to the
        GPU with `non_blocking=True`. This can speed up training.
    dataset_on_device : bool, default=False
        Whether store the whole dataset on the device. This can speed up training but
        requires that the dataset fits on the device. This makes `pin_memory` redundant.
    enable_efficient_attention: bool, default=False
        Whether to enable the 'Memory-Efficient Attention' backend for the scaled
        dot-product attention. There may be a bug in this implementation which causes
        NaNs to appear in the backward pass. See
        https://github.com/pytorch/pytorch/issues/119320 for more information.
    test_run : bool, default=False
        If True, the experiment is run in test mode. This means we do the smallest
        number of iterations possible and then exit. This is useful for testing that the
        experiment runs without errors. It doesn't make sense to use this with
        wandb_run.
    """

    device: TorchDevice = torch.device("cpu")
    wandb_run: Optional[wandb.wandb_sdk.wandb_run.Run] = None
    silence_wandb: bool = True
    stat_logger: Optional[StatLogger] = field(default_factory=DummyStatLogger)
    tqdm_func: callable = tqdm
    logger: Optional[LoggingType] = None
    profiler: Optional[torch.profiler.profile] = None
    ignore_cache: bool = False
    num_rollout_samples: int = 10
    rollout_sample_period: int = 1000
    checkpoint_period: int = 1000
    num_dataset_threads: int = 8
    pin_memory: bool = True
    dataset_on_device: bool = False
    enable_efficient_attention: bool = False
    test_run: bool = False

    def __post_init__(self):
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        if self.test_run and self.wandb_run is not None:
            raise ValueError("test_run cannot be True if wandb_run is not None.")
