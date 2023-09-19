from abc import ABC, abstractmethod

import torch

from pvg.parameters import Parameters


class Agent(ABC):
    """Base class for all agents."""

    def __init__(self, parameters: Parameters, device: str | torch.device):
        self.parameters = parameters
        self.device = device


class Prover(Agent, ABC):
    """Base class for all provers."""


class Verifier(Agent, ABC):
    """Base class for all verifier."""


class Scenario(ABC):
    """Base class for all scenarios: domain, task and agents.

    Parameters
    ----------
    parameters : Parameters
        The parameters of the experiment.
    device : str | torch.device
        The device to use for training.
    """

    prover: Prover
    verifier: Verifier

    def __init__(self, parameters: Parameters, device: str | torch.device):
        self.parameters = parameters
        self.device = device
