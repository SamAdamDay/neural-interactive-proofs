"""Base classes for all PVG scenario components.

A scenario consists of a dataset and a definition of the agents.

Contains base classes for:

- Handling data
- Building agents
- Holding the components of a scenario

Scenarios should subclass the `AgentsBuilder` class, and its `build` factory class
method is used to build the agents using the given parameters.
"""

from .data import Dataset, DataLoader
from .agents import (
    AgentPart,
    AgentBody,
    AgentHead,
    AgentPolicyHead,
    AgentCriticHead,
    AgentValueHead,
    SoloAgentHead,
    Agent,
    AgentsBuilder,
)
from .component_holder import ComponentHolder
