"""Base classes for all PVG scenario components.

A scenario consists of a dataset and a definition of the agents.

Contains base classes for:

- Handling data
- Building agents
"""

from .data import Dataset, DataLoader, load_dataset
from .agents import (
    AgentPart,
    AgentBody,
    AgentHead,
    AgentPolicyHead,
    AgentCriticHead,
    AgentValueHead,
)
