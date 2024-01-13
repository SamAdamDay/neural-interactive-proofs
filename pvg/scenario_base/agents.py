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
from torch.nn.parameter import Parameter as TorchParameter

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase

from pvg.parameters import Parameters, TrainerType, ScenarioType
from pvg.utils.types import TorchDevice


class AgentPart(TensorDictModuleBase, ABC):
    """Base class for all agent parts: bodies and heads.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    device : TorchDevice, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    def __init__(self, params: Parameters, device: Optional[TorchDevice] = None):
        super().__init__()
        self.params = params
        if device is None:
            device = "cpu"
        self.device = device

    @abstractmethod
    def to(device: TorchDevice):
        """Move the agent to the given device."""
        pass


class DummyAgentPartMixin(AgentPart, ABC):
    """A mixin for agent parts which are dummy (e.g. random or constant).

    Adds a dummy parameter to the agent part, so that PyTorch can calculate gradients
    and so that tensordict can determine the device.
    """

    def __init__(self, params: Parameters, device: TorchDevice | None = None):
        super().__init__(params, device)
        self.dummy_parameter = TorchParameter(torch.tensor(0.0, device=self.device))

    def to(self, device: TorchDevice):
        """Move the agent to the given device."""
        self.device = device
        self.dummy_parameter = self.dummy_parameter.to(device)


class AgentBody(AgentPart, ABC):
    """Base class for all agent bodies, which compute representations for heads.
    
    Representations should have dimension `params.d_representation`.
    """


class DummyAgentBody(DummyAgentPartMixin, AgentBody, ABC):
    """A dummy agent body which does nothing."""


class AgentHead(AgentPart, ABC):
    """Base class for all agent heads."""


class AgentPolicyHead(AgentHead, ABC):
    """Base class for all agent policy heads."""


class RandomAgentPolicyHead(DummyAgentPartMixin, AgentPolicyHead, ABC):
    """A policy head which samples actions randomly."""


class AgentValueHead(AgentHead, ABC):
    """Base class for all agent value heads, to the value of a state."""


class ConstantAgentValueHead(DummyAgentPartMixin, AgentValueHead, ABC):
    """A value head which returns a constant value."""


class AgentCriticHead(AgentHead, ABC):
    """Base class for all agent critic heads, to the value of a state-action pair."""


class SoloAgentHead(AgentHead, ABC):
    """Base class for all solo agent heads, which attempt the task on their own."""


class CombinedBody(TensorDictModuleBase, ABC):
    """A module which combines all the agent bodies together.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    bodies : dict[str, AgentBody]
        The agent bodies to combine.
    """

    def __init__(self, params: Parameters, bodies: dict[str, AgentBody]):
        super().__init__()
        self.params = params
        self.bodies = bodies

        # Add the bodies as submodules, so that PyTorch knows about them
        for agent_name, body in bodies.items():
            self.add_module(agent_name, body)

    @abstractmethod
    def forward(self, data: TensorDictBase) -> TensorDict:
        """Forward pass through the combined body.

        Parameters
        ----------
        data : TensorDict
            The input to the combined body.

        Returns
        -------
        body_output : TensorDict
            The output of the combined body.
        """
        pass


class CombinedPolicyHead(TensorDictModuleBase, ABC):
    """A module which combines all the agent policy heads together.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    policy_heads : dict[str, AgentPolicyHead]
        The agent policy heads to combine.
    """

    def __init__(self, params: Parameters, policy_heads: dict[str, AgentPolicyHead]):
        super().__init__()
        self.params = params
        self.policy_heads = policy_heads

        # Add the policy heads as submodules, so that PyTorch knows about them
        for agent_name, policy_head in policy_heads.items():
            self.add_module(agent_name, policy_head)

    @abstractmethod
    def forward(self, data: TensorDictBase) -> TensorDict:
        """Forward pass through the combined policy head.

        Parameters
        ----------
        data : TensorDict
            The input to the combined policy head.

        Returns
        -------
        policy_output : TensorDict
            The output of the combined policy head.
        """
        pass


class CombinedValueHead(TensorDictModuleBase, ABC):
    """A module which combines all the agent value heads together.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    value_heads : dict[str, AgentValueHead]
        The agent value heads to combine.
    """

    def __init__(self, params: Parameters, value_heads: dict[str, AgentValueHead]):
        super().__init__()
        self.params = params
        self.value_heads = value_heads

        # Add the value heads as submodules, so that PyTorch knows about them
        for agent_name, value_head in value_heads.items():
            self.add_module(agent_name, value_head)

    @abstractmethod
    def forward(self, data: TensorDictBase) -> TensorDict:
        """Forward pass through the combined value head.

        Parameters
        ----------
        data : TensorDict
            The input to the combined value head.

        Returns
        -------
        value_output : TensorDict
            The output of the combined value head.
        """


@dataclass
class Agent:
    """A class which holds all the parts of an agent for an experiment."""

    body: AgentBody
    policy_head: Optional[AgentPolicyHead] = None
    value_head: Optional[AgentValueHead] = None
    critic_head: Optional[AgentCriticHead] = None
    solo_head: Optional[SoloAgentHead] = None

    def __post_init__(self):
        if self.policy_head is None and self.solo_head is None:
            raise ValueError(
                "An agent must have either a policy head or a solo head, or both."
            )

        if self.policy_head is not None and (
            self.value_head is None and self.critic_head is None
        ):
            raise ValueError(
                "An agent with a policy head must have either a value head or a critic "
                "head, or both."
            )
