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


class AgentsBuilder(ABC):
    """Base class for building agents.
    
    All scenarios should subclass this class and add the class attributes below.

    Attributes
    ----------
    scenario : Scenario
        The scenario for which this builder builds agents.
    body_class : type[AgentBody]
        The class for the agent body.
    policy_head_class : type[AgentPolicyHead]
        The class for the agent policy head.
    value_head_class : type[AgentValueHead]
        The class for the agent value head.
    solo_head_class : type[SoloAgentHead]
        The class for the solo agent head.
    """

    scenario: ScenarioType

    body_class: type[AgentBody]
    policy_head_class: type[AgentPolicyHead]
    value_head_class: type[AgentValueHead]
    solo_head_class: type[SoloAgentHead]

    @classmethod
    def build(
        cls,
        params: Parameters,
        device: TorchDevice,
    ) -> dict[str, Agent]:
        """Build the agents for the given scenario.

        Parameters
        ----------
        params : Parameters
            The parameters of the experiment.
        device : TorchDevice
            The device to use for the agents.

        Returns
        -------
        agents : dict[str, Agent]
            A dictionary mapping agent names to `Agent` objects, which contain the agent
            parts.
        """

        if params.scenario != cls.scenario:
            raise ValueError(
                f"Cannot build agents for scenario {params.scenario} "
                f"with {cls.__name__} parameters."
            )

        agent_names = ["prover", "verifier"]

        agents = {}
        for agent_name in agent_names:
            agent_dict = {}

            agent_dict["body"] = cls.body_class(
                params=params,
                device=device,
                agent_name=agent_name,
            )

            if params.trainer == TrainerType.PPO:
                agent_dict["policy_head"] = cls.policy_head_class(
                    params=params,
                    device=device,
                    agent_name=agent_name,
                )
                agent_dict["value_head"] = cls.value_head_class(
                    params=params,
                    device=device,
                    agent_name=agent_name,
                )
            elif params.trainer == TrainerType.SOLO_AGENT:
                agent_dict["solo_head"] = cls.solo_head_class(
                    params=params,
                    device=device,
                    agent_name=agent_name,
                )

            agents[agent_name] = Agent.from_dict(agent_dict)

        return agents