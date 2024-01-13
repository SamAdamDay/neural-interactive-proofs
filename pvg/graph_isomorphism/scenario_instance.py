"""A class which holds the components of a graph isomorphism experiment."""

from pvg.parameters import ScenarioType
from pvg.scenario_base import DataLoader, ScenarioInstance
from pvg.graph_isomorphism.data import GraphIsomorphismDataset
from pvg.graph_isomorphism.agents import (
    GraphIsomorphismAgentBody,
    GraphIsomorphismDummyAgentBody,
    GraphIsomorphismAgentPolicyHead,
    GraphIsomorphismRandomAgentPolicyHead,
    GraphIsomorphismAgentValueHead,
    GraphIsomorphismConstantAgentValueHead,
    GraphIsomorphismSoloAgentHead,
    GraphIsomorphismCombinedBody,
    GraphIsomorphismCombinedPolicyHead,
    GraphIsomorphismCombinedValueHead,
)
from pvg.graph_isomorphism.environment import GraphIsomorphismEnvironment


class GraphIsomorphismScenarioInstance(ScenarioInstance):
    """A class which holds the components of a graph isomorphism experiment.

    Attributes
    ----------
    dataset : Dataset
        The dataset for the experiment.
    dataloader_class : type[DataLoader]
        The data loader class to use for the experiment.
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

    Parameters
    ----------
    params : Parameters
        The params of the experiment.
    device : TorchDevice
        The device to use for training.
    """

    scenario = ScenarioType.GRAPH_ISOMORPHISM

    dataset_class = GraphIsomorphismDataset
    dataloader_class = DataLoader

    environment_class = GraphIsomorphismEnvironment

    body_class = GraphIsomorphismAgentBody
    dummy_body_class = GraphIsomorphismDummyAgentBody
    policy_head_class = GraphIsomorphismAgentPolicyHead
    random_policy_head_class = GraphIsomorphismRandomAgentPolicyHead
    value_head_class = GraphIsomorphismAgentValueHead
    constant_value_head_class = GraphIsomorphismConstantAgentValueHead
    solo_head_class = GraphIsomorphismSoloAgentHead
    combined_body_class = GraphIsomorphismCombinedBody
    combined_policy_head_class = GraphIsomorphismCombinedPolicyHead
    combined_value_head_class = GraphIsomorphismCombinedValueHead
