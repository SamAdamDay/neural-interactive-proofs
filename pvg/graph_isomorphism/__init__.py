"""All components for the graph isomorphism task.

Has classes for:

- Handling data
- Generating a dataset
- Building agents

The `GraphIsomorphismAgentsBuilder.build` factory function is used to build the agents
using the given parameters.

Examples
--------
>>> from pvg.parameters import Parameters, Scenario, Trainer
>>> from pvg.graph_isomorphism import GraphIsomorphismAgentsBuilder
>>> params = Parameters(Scenario.GRAPH_ISOMORPHISM, Trainer.SOLO_AGENT, "eru10000")
>>> agents = GraphIsomorphismAgentsBuilder.build(params, "cpu")
"""

import torch

from .data import GraphIsomorphismDataset
from .dataset_generation import generate_gi_dataset, GraphIsomorphicDatasetConfig
from .agents import (
    GraphIsomorphismAgentPart,
    GraphIsomorphismAgentBody,
    GraphIsomorphismAgentPolicyHead,
    GraphIsomorphismAgentValueHead,
    GraphIsomorphismSoloAgentHead,
    GraphIsomorphismAgentsBuilder
)
from .component_holder import GraphIsomorphismComponentHolder
