"""All components for the graph isomorphism task.

Has classes for:

- Handling data
- Generating a dataset
- Building agents
"""

from .data import GraphIsomorphismDataset, GraphIsomorphismData
from .dataset_generation import generate_gi_dataset, GraphIsomorphicDatasetConfig
from .agents import (
    GraphIsomorphismAgentBody,
    GraphIsomorphismAgentPolicyHead,
    GraphIsomorphismAgentValueHead,
)
