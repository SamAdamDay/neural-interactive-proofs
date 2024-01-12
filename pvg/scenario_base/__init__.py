"""Base classes for all PVG scenario components.

A scenario consists of a dataset and a definition of the agents.

Contains base classes for:

- Handling data
- The RL environment
- Building agents
- Holding the components of a scenario

Scenarios should subclass the `ScenarioInstance` class, and its `build` factory class
method is used to build the agents using the given parameters.
"""

from .data import Dataset, DataLoader
from .environment import Environment
from .agents import (
    AgentPart,
    AgentBody,
    AgentHead,
    AgentPolicyHead,
    AgentCriticHead,
    AgentValueHead,
    SoloAgentHead,
    Agent,
    CombinedBody,
    CombinedPolicyHead,
    CombinedValueHead,
)
from .scenario_instance import ScenarioInstance
