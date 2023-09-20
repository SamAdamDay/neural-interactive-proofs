from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from pvg.parameters import Parameters


class Agent(nn.Module, ABC):
    """Base class for all agents."""

    def __init__(self, parameters: Parameters, device: Optional[str | torch.device]):
        super().__init__()
        self.parameters = parameters
        if device is None:
            device = "cpu"
        self.device = device

    @abstractmethod
    def to(device: str | torch.device):
        """Move the agent to the given device."""
        pass


class Prover(Agent, ABC):
    """Base class for all provers."""
    pass


class Verifier(Agent, ABC):
    """Base class for all verifier."""
    pass


class Scenario(ABC):
    """Base class for all scenarios: domain, task and agents.

    Parameters
    ----------
    parameters : Parameters
        The parameters of the experiment.
    device : str | torch.device
        The device to use for training.

    Class attributes
    ----------------
    name : str
        The name of the scenario.
    """

    name: str

    def __init__(self, parameters: Parameters, device: str | torch.device):
        self.parameters = parameters
        self.device = device
        self.prover: Prover
        self.verifier: Verifier
