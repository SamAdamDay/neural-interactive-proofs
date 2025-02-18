"""Classes for logging experiment statistics during training.

Contains a dummy logger that does nothing, and a logger that logs statistics to Weights
& Biases.
"""

from abc import ABC, abstractmethod
from typing import Optional

import wandb


class StatLogger(ABC):
    """Base class for logging statistics during training."""

    @abstractmethod
    def log(self, to_log: dict, step: Optional[int] = None):
        """Log some statistics.

        Parameters
        ----------
        to_log : dict
            The statistics to log.
        step : int, optional
            The step at which the statistics are to be logged.
        """


class DummyStatLogger(StatLogger):
    """A dummy logger that does nothing."""

    def log(self, to_log: dict, step: Optional[int] = None):  # noqa: D102
        pass


class WandbStatLogger(StatLogger):
    """A logger that logs statistics to Weights & Biases."""

    def __init__(self, wandb_run: wandb.wandb_sdk.wandb_run.Run):
        self.wandb_run = wandb_run

    def log(self, to_log: dict, step: Optional[int] = None):
        """Log some statistics with Weights & Biases.

        Parameters
        ----------
        to_log : dict
            The statistics to log.
        step : int, optional
            The step at which the statistics are to be logged.
        """
        self.wandb_run.log(to_log, step=step)
