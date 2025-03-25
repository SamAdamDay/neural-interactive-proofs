"""A data structure for holding the components of an experiment."""

from dataclasses import dataclass
from typing import Optional
from nip.scenario_base.data import Dataset
from nip.scenario_base.agents import (
    CombinedBody,
    CombinedPolicyHead,
    CombinedValueHead,
    CombinedWhole,
    Agent,
    PureTextSharedModelGroup,
)
from nip.scenario_base.environment import Environment
from nip.protocols import ProtocolHandler
from nip.message_regression import MessageRegressor


@dataclass
class ScenarioInstance:
    """A dataclass for holding the components of an experiment.

    The principal aim of this class is to abstract away the details of the particular
    experiment being run.

    Attributes
    ----------
    train_dataset : Dataset
        The train dataset for the experiment.
    test_dataset : Dataset
        The test dataset for the experiment.
    protocol_handler : ProtocolHandler
        The interaction protocol handler for the experiment.
    message_regressor : MessageRegressor
        The message regressor for the experiment, which is used to test if the label can
        be inferred purely from the messages.
    agents : dict[str, Agent]
        The agents for the experiment. Each 'agent' is a dictionary containing all of
        the agent parts.
    train_environment : Optional[Environment]
        The train environment for the experiment, if the experiment is RL.
    test_environment : Optional[Environment]
        The environment for testing the agents, which uses the test dataset.
    combined_whole : Optional[CombinedWholeAgent]
        If the agents are not split into parts, this holds the combination of the whole
        agents.
    combined_body : Optional[CombinedBody]
        The combined body of the agents, if the agents are combined the actor and critic
        share the same body.
    combined_policy_body : Optional[CombinedBody]
        The combined policy body of the agents, if the agents are combined and the actor
        and critic have separate bodies.
    combined_value_body : Optional[CombinedBody]
        The combined value body of the agents, if the agents are combined and the actor
        and critic have separate bodies.
    combined_policy_head : Optional[CombinedPolicyHead]
        The combined policy head of the agents, if the agents are combined.
    combined_value_head : Optional[CombinedValueHead]
        The combined value head of the agents, if the agents are combined.
    shared_model_groups : Optional[dict[str, PureTextSharedModelGroup]]
        The shared model groups for pure-text environments. Agents in the same group
        share the same model. A dictionary with the group name as the key and the shared
        model group as the value.
    """

    train_dataset: Dataset
    test_dataset: Dataset
    agents: dict[str, Agent]
    protocol_handler: ProtocolHandler
    message_regressor: MessageRegressor
    train_environment: Optional[Environment] = None
    test_environment: Optional[Environment] = None
    combined_whole: Optional[CombinedWhole] = None
    combined_body: Optional[CombinedBody] = None
    combined_policy_body: Optional[CombinedBody] = None
    combined_value_body: Optional[CombinedBody] = None
    combined_policy_head: Optional[CombinedPolicyHead] = None
    combined_value_head: Optional[CombinedValueHead] = None
    shared_model_groups: Optional[dict[str, PureTextSharedModelGroup]] = None
