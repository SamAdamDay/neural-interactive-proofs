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

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase

import dacite

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


class CombinedBody(TensorDictModuleBase, ABC):
    """A module which combines all the agent bodies together.
    
    Parameters
    ----------
    bodies : dict[str, AgentBody]
        The agent bodies to combine.
    """

    agent_names = ["prover", "verifier"]

    def __init__(self, bodies: dict[str, AgentBody]):
        super().__init__()
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
    policy_heads : dict[str, AgentPolicyHead]
        The agent policy heads to combine.
    """

    agent_names = ["prover", "verifier"]

    def __init__(self, policy_heads: dict[str, AgentPolicyHead]):
        super().__init__()
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
    value_heads : dict[str, AgentValueHead]
        The agent value heads to combine.
    """

    agent_names = ["prover", "verifier"]

    def __init__(self, value_heads: dict[str, AgentValueHead]):
        super().__init__()
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
        pass


@dataclass
class Agent:
    """A class which holds all the parts of an agent for an experiment."""

    body: AgentBody
    policy_head: Optional[AgentPolicyHead] = None
    value_head: Optional[AgentValueHead] = None
    critic_head: Optional[AgentCriticHead] = None
    solo_head: Optional[SoloAgentHead] = None

    @classmethod
    def from_dict(cls, data):
        return dacite.from_dict(data_class=cls, data=data)

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
