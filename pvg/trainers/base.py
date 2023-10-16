from abc import ABC, abstractmethod

import torch

from pvg.parameters import Parameters
from pvg.scenarios import Scenario
from pvg.data import Dataset


class Trainer(ABC):
    """Base class for all RL trainers.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    device : str | torch.device
        The device to use for training.
    """

    def __init__(self, params: Parameters, device: str | torch.device):
        self.params = params
        self.device = device

    @abstractmethod
    def train(self, scenario: Scenario, dataset: Dataset):
        """Train the agents."""
        pass
