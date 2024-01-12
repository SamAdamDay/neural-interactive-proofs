from abc import ABC, abstractmethod
from typing import ClassVar

import torch

from pvg.parameters import Parameters, ScenarioType, TrainerType
from pvg.experiment_settings import ExperimentSettings
from pvg.scenario_base.data import Dataset, DataLoader
from pvg.scenario_base.agents import (
    AgentBody,
    AgentPolicyHead,
    AgentValueHead,
    SoloAgentHead,
    CombinedBody,
    CombinedPolicyHead,
    CombinedValueHead,
    Agent,
)
from pvg.scenario_base.environment import Environment
from pvg.utils.types import TorchDevice


class ScenarioInstance(ABC):
    """A base class for holding the components of an experiment.

    All scenarios should subclass this class define the class attributes below.

    The principal aim of this class is to abstract away the details of the particular
    experiment being run.

    Parameters
    ----------
    params : Parameters
        The params of the experiment.
    device : TorchDevice
        The device to use for training.

    Attributes
    ----------
    dataset : Dataset
        The dataset for the experiment.
    agents : dict[str, Agent]
        The agents for the experiment.
    environment : Optional[Environment]
        The environment for the experiment, if the experiment is RL.
    combined_body : Optional[CombinedBody]
        The combined body of the agents, if the agents are combined.
    combined_policy_head : Optional[CombinedPolicyHead]
        The combined policy head of the agents, if the agents are combined.
    combined_value_head : Optional[CombinedValueHead]
        The combined value head of the agents, if the agents are combined.

    Class Attributes
    ----------------
    scenario : ScenarioType
        The scenario for which this holder holds components.
    dataset_class : type[Dataset]
        The dataset class for the experiment.
    dataloader_class : type[DataLoader]
        The data loader class to use for the experiment.
    body_class : type[AgentBody]
        The class for the agent body.
    policy_head_class : type[AgentPolicyHead]
        The class for the agent policy head.
    value_head_class : type[AgentValueHead]
        The class for the agent value head.
    solo_head_class : type[SoloAgentHead]
        The class for the solo agent head.
    combined_body_class : type[CombinedBody]
        The class for the combined bodies of the agents.
    combined_policy_head_class : type[CombinedPolicyHead]
        The class for the combined policy heads of the agents.
    combined_value_head_class : type[CombinedValueHead]
        The class for the combined value heads of the agents.
    """

    scenario: ClassVar[ScenarioType]

    dataset_class: ClassVar[type[Dataset]]
    dataloader_class: ClassVar[type[DataLoader]]

    environment_class: ClassVar[type[Environment]]

    body_class: ClassVar[type[AgentBody]]
    policy_head_class: ClassVar[type[AgentPolicyHead]]
    value_head_class: ClassVar[type[AgentValueHead]]
    solo_head_class: ClassVar[type[SoloAgentHead]]
    combined_body_class: ClassVar[type[CombinedBody]]
    combined_policy_head_class: ClassVar[type[CombinedPolicyHead]]
    combined_value_head_class: ClassVar[type[CombinedValueHead]]

    def __init__(self, params: Parameters, settings: ExperimentSettings):
        if params.scenario != self.scenario:
            raise ValueError(
                f"Cannot build agents for scenario {params.scenario} "
                f"with {self.__name__} parameters."
            )

        self.params = params
        self.settings = settings

        self.device = settings.device

        self.dataset = self.dataset_class(params, settings)

        # Create the agents
        self.agents: dict[str, Agent] = {}
        for agent_name in params.agents:
            agent_dict = {}

            agent_dict["body"] = self.body_class(
                params=params,
                device=self.device,
                agent_name=agent_name,
            )

            if params.trainer == TrainerType.PPO:
                agent_dict["policy_head"] = self.policy_head_class(
                    params=params,
                    device=self.device,
                    agent_name=agent_name,
                )
                agent_dict["value_head"] = self.value_head_class(
                    params=params,
                    device=self.device,
                    agent_name=agent_name,
                )
            elif params.trainer == TrainerType.SOLO_AGENT:
                agent_dict["solo_head"] = self.solo_head_class(
                    params=params,
                    device=self.device,
                    agent_name=agent_name,
                )

            self.agents[agent_name] = Agent.from_dict(agent_dict)

        # Build additional components if the trainer is an RL trainer
        if self.params.trainer == TrainerType.PPO:
            # Create the environment
            self.environment = self.environment_class(params, settings)

            # Create the combined agents
            self.combined_body = self.combined_body_class(
                params,
                {name: self.agents[name].body for name in params.agents},
            )
            self.combined_policy_head = self.combined_policy_head_class(
                params,
                {name: self.agents[name].policy_head for name in params.agents},
            )
            self.combined_value_head = self.combined_value_head_class(
                params,
                {name: self.agents[name].value_head for name in params.agents},
            )
