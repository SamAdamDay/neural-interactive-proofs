"""Base classes for building agents.

An agent is composed of a body and one or more heads. The body computes a representation
of the environment state, and the heads use this representation to compute the agent's
policy, value function, etc.

All modules are TensorDictModules, which means they take and return TensorDicts. Input
and output keys are specified in the module's `input_keys` and `output_keys` attributes.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
from dataclasses import dataclass

import torch

from tensordict.nn import TensorDictModuleBase

from pvg.parameters import Parameters


class AgentPart(TensorDictModuleBase, ABC):
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


class AgentValueHead(AgentHead, ABC):
    """Base class for all agent value heads, to the value of a state."""

    pass


class AgentCriticHead(AgentHead, ABC):
    """Base class for all agent critic heads, to the value of a state-action pair."""

    pass


class SoloAgentHead(AgentHead, ABC):
    """Base class for all solo agent heads, which attempt the task on their own."""

    pass