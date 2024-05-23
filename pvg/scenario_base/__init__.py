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
    AgentHooks,
    AgentPart,
    AgentBody,
    AgentHead,
    DummyAgentBody,
    AgentPolicyHead,
    RandomAgentPolicyHead,
    AgentValueHead,
    ConstantAgentValueHead,
    SoloAgentHead,
    CombinedAgentPart,
    CombinedBody,
    CombinedPolicyHead,
    CombinedValueHead,
    Agent,
)
from .rollout_samples import (
    RolloutSamples,
    register_rollout_samples_class,
    build_rollout_samples,
)
