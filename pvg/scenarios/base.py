from abc import ABC, abstractmethod
from typing import Optional, Any
from dataclasses import dataclass

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


@dataclass
class Message(ABC):
    """Base class for all messages sent between provers and verifiers.
    
    Parameters
    ----------
    from_verifier : bool
        Whether the message is sent from the verifier to the prover, or vice versa.
    message : Any
        The content of the message.
    verifier_guess : int
        The verifier's guess. Only used when the message is sent from the verifier to
        the prover. One of three values:
            - 0: the verifier doesn't make a guess yet
            - 1: the verifier guesses that the graphs are isomorphic
            - 2: the verifier guesses that the graphs are not isomorphic
    """

    from_verifier: bool
    message: Any
    verifier_guess: int


class MessageExchange(list[Message]):
    """A message exchange between provers and verifiers."""

    def __init__(self, iterable=None):
        if iterable is None:
            iterable = []
        for message in iterable:
            if not isinstance(message, Message):
                raise TypeError(f"Expected a Message object, got {type(message)}")
        super().__init__(iterable)

    def __setitem__(self, index, message):
        if isinstance(message, Message):
            super().__setitem__(index, message)
        else:
            raise TypeError(f"Expected a Message object, got {type(message)}")
        
    def append(self, message):
        if isinstance(message, Message):
            super().append(message)
        else:
            raise TypeError(f"Expected a Message object, got {type(message)}")
        
    def extend(self, iterable):
        for message in iterable:
            if isinstance(message, Message):
                super().append(message)
            else:
                raise TypeError(f"Expected a Message object, got {type(message)}")


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

    @abstractmethod
    def rollout(self, *args, **kwargs) -> MessageExchange:
        """Perform a rollout of the scenario."""
        pass