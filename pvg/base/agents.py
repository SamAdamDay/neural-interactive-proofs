from abc import ABC, abstractmethod
from typing import Optional, Any
from dataclasses import dataclass

import torch
import torch.nn as nn

from pvg.parameters import Parameters


class AgentPart(nn.Module, ABC):
    """Base class for all agent parts: bodies and heads.
    
    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    device : str or torch.device, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    def __init__(self, params: Parameters, device: Optional[str | torch.device] = None):
        super().__init__()
        self.params = params
        if device is None:
            device = "cpu"
        self.device = device

    @abstractmethod
    def to(device: str | torch.device):
        """Move the agent to the given device."""
        pass


class AgentBody(AgentPart, ABC):
    """Base class for all agent bodies, which compute representations for heads."""

    pass


class AgentHead(AgentPart, ABC):
    """Base class for all agent heads."""

    pass


class AgentPolicyHead(AgentHead, ABC):
    """Base class for all agent policy heads."""

    pass


class AgentCriticHead(AgentHead, ABC):
    """Base class for all agent critic heads."""

    pass
