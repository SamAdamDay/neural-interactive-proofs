"""All components for the graph isomorphism task.

Has classes for:

- Handling data
- Defining the RL environment
- Generating a dataset
- Building agents

Examples
--------
>>> from pvg.parameters import HyperParameters, Scenario, Trainer
>>> from pvg.graph_isomorphism import GraphIsomorphismAgentsBuilder
>>> hyper_params = HyperParameters(
...     Scenario.GRAPH_ISOMORPHISM, Trainer.SOLO_AGENT, "eru10000"
... )
>>> agents = GraphIsomorphismAgentsBuilder.build(hyper_params, "cpu")
"""

from .data import GraphIsomorphismDataset
from .environment import GraphIsomorphismEnvironment
from .agents import (
    GraphIsomorphismAgentPart,
    GraphIsomorphismAgentBody,
    GraphIsomorphismAgentPolicyHead,
    GraphIsomorphismAgentValueHead,
    GraphIsomorphismSoloAgentHead,
    GraphIsomorphismAgentHooks,
)
from .rollout_samples import GraphIsomorphismRolloutSamples
