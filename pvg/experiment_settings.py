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
    """

    device: TorchDevice = "cpu"
    wandb_run: Optional[wandb.wandb_sdk.wandb_run.Run] = None
    tqdm_func: callable = tqdm,
    logger: Optional[LoggingType] = None
    ignore_cache: bool = False