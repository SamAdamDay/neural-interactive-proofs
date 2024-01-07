"""All components for the graph isomorphism task.

Has classes for:

- Handling data
- Generating a dataset
- Building agents

The `build_agents` factory function is used to build the agents using the given
parameters.

Examples
--------
>>> from pvg.parameters import Parameters, Scenario, Trainer
>>> from pvg.graph_isomorphism import build_agents
>>> params = Parameters(Scenario.GRAPH_ISOMORPHISM, Trainer.SOLO_AGENT, "eru10000")
>>> agents = build_agents(params, "cpu")
"""

import torch

from pvg.parameters import Parameters, Scenario, Trainer

from .data import GraphIsomorphismDataset
from .dataset_generation import generate_gi_dataset, GraphIsomorphicDatasetConfig
from .agents import (
    GraphIsomorphismAgentPart,
    GraphIsomorphismAgentBody,
    GraphIsomorphismAgentPolicyHead,
    GraphIsomorphismAgentValueHead,
    GraphIsomorphismSoloAgentHead,
)


def build_agents(
    params: Parameters, device: str | torch.device
) -> dict[str, dict[str, GraphIsomorphismAgentPart]]:
    """Build the agents for the graph isomorphism task using the given parameters.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    device : str or torch.device
        The device to use for the agents.

    Returns
    -------
    agents : dict[str, dict[str, GraphIsomorphismAgentPart]]
        A dictionary mapping agent names dicts of agent parts. The agent parts are:

        - "body": The agents's body.
        - "policy_head" (optional): The agents's policy head.
        - "value_head" (optional): The agents's value head.
        - "solo_head" (optional): The agents's solo agent head.
    """

    if params.scenario != Scenario.GRAPH_ISOMORPHISM:
        raise ValueError(
            f"Cannot build agents for scenario {params.scenario} "
            "with graph isomorphism parameters."
        )

    agent_names = ["prover", "verifier"]

    agents = {}
    for agent_name in agent_names:
        agents[agent_name] = {}

        agents[agent_name]["body"] = GraphIsomorphismAgentBody(
            params=params,
            device=device,
            agent_name=agent_name,
        )

        if params.trainer == Trainer.PPO:
            agents[agent_name]["policy_head"] = GraphIsomorphismAgentPolicyHead(
                params=params,
                device=device,
                agent_name=agent_name,
            )
            agents[agent_name]["value_head"] = GraphIsomorphismAgentValueHead(
                params=params,
                device=device,
                agent_name=agent_name,
            )
        elif params.trainer == Trainer.SOLO_AGENT:
            agents[agent_name]["solo_head"] = GraphIsomorphismSoloAgentHead(
                params=params,
                device=device,
                agent_name=agent_name,
            )

    return agents
