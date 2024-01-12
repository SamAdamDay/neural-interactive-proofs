"""Instance-specific settings for the experiment, like torch device, logging, etc.

Changing these settings should not effect the reproducibility of the experiment.
"""

from typing import Optional
from dataclasses import dataclass
import logging

import wandb

from tqdm import tqdm

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
    tqdm_func : Callable, optional
        The tqdm function to use. Defaults to tqdm.
    logger : logging.Logger | logging.LoggerAdapter, optional
        The logger to log to. If None, the trainer will create a logger.
    ignore_cache : bool, default=False
        If True, when the dataset is loaded, the cache is ignored and the dataset is
        rebuilt from the raw data.
    num_dataset_threads : int, default=8
        The number of threads to use for saving the memory-mapped tensordict.
    test_run : bool, default=False
        If True, the experiment is run in test mode. This means we do the smallest
        number of iterations possible and then exit. This is useful for testing that
        the experiment runs without errors. It doesn't make sense to use this with
        wandb_run.
    """

    device: TorchDevice = "cpu"
    wandb_run: Optional[wandb.wandb_sdk.wandb_run.Run] = None
    tqdm_func: callable = tqdm
    logger: Optional[LoggingType] = None
    ignore_cache: bool = False
    num_dataset_threads: int = 8
    test_run: bool = False

    def __post_init__(self):
        if self.test_run and self.wandb_run is not None:
            raise ValueError("test_run cannot be True if wandb_run is not None.")
