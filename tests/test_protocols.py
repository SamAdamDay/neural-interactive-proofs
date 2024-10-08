from itertools import product

import pytest

import torch

from pvg.experiment_settings import ExperimentSettings
from pvg.parameters import (
    Parameters,
    InteractionProtocolType,
    ScenarioType,
    TrainerType,
    CommonProtocolParameters,
)
from pvg.protocols import build_protocol_handler, ZeroKnowledgeProtocol


@pytest.mark.parametrize(
    "base_protocol",
    [
        InteractionProtocolType.PVG,
        InteractionProtocolType.DEBATE,
        InteractionProtocolType.ABSTRACT_DECISION_PROBLEM,
        InteractionProtocolType.MNIP,
        InteractionProtocolType.MULTI_CHANNEL_TEST,
    ],
)
def test_zero_knowledge_active_agent_mask(base_protocol: InteractionProtocolType):
    """Test that the active agent mask is correctly set for the zero-knowledge protocol.

    Parameters
    ----------
    base_protocol : InteractionProtocolType
        The base protocol to use. This is extended to a zero-knowledge protocol.
    """

    params = Parameters(
        scenario=ScenarioType.GRAPH_ISOMORPHISM,
        trainer=TrainerType.VANILLA_PPO,
        dataset="test",
        interaction_protocol=base_protocol,
        protocol_common=CommonProtocolParameters(zero_knowledge=True),
    )
    settings = ExperimentSettings(device="cpu", test_run=True)
    protocol_handler: ZeroKnowledgeProtocol = build_protocol_handler(params, settings)

    agent_names = protocol_handler.agent_names
    channel_names = protocol_handler.message_channel_names
    max_message_rounds = protocol_handler.max_message_rounds
    base_protocol_handler = protocol_handler.base_protocol
    base_agent_names = base_protocol_handler.agent_names
    base_channel_names = base_protocol_handler.message_channel_names

    round = torch.arange(0, max_message_rounds)

    # Get the masks of which agents are active in which rounds and channels
    active_agents_mask = protocol_handler.get_active_agents_mask_from_rounds(round)
    base_active_agent_mask = base_protocol_handler.get_active_agents_mask_from_rounds(
        round
    )

    # Build a list which for each round contains a set of the agent-channel pairs that
    # are expected to be active in the zero-knowledge protocol.
    expected_active_agents: list[set[tuple[str, str]]] = []
    for round_id in range(max_message_rounds):
        expected_active_agents_round = set()
        for (agent_id, agent_name), (channel_id, channel_name) in product(
            enumerate(base_agent_names), enumerate(base_channel_names)
        ):
            if not base_active_agent_mask[round_id, agent_id, channel_id]:
                continue
            if agent_name == base_protocol_handler.verifier_name:
                expected_active_agents_round.add(
                    (agent_name, f"{channel_name}_standard")
                )
                expected_active_agents_round.add(
                    ("adversarial_verifier", f"{channel_name}_adversarial")
                )
            else:
                expected_active_agents_round.add(
                    (agent_name, f"{channel_name}_standard")
                )
                expected_active_agents_round.add(
                    (agent_name, f"{channel_name}_adversarial")
                )
        for channel_name in base_channel_names:
            expected_active_agents_round.add(("simulator", f"{channel_name}_simulator"))
        expected_active_agents.append(expected_active_agents_round)

    # Build a corresponding list of the actual active agents.
    active_agents: list[set[tuple[str, str]]] = []
    for round_id in range(max_message_rounds):
        active_agents_round = set()
        for (agent_id, agent_name), (channel_id, channel_name) in product(
            enumerate(agent_names), enumerate(channel_names)
        ):
            if active_agents_mask[round_id, agent_id, channel_id]:
                active_agents_round.add((agent_name, channel_name))
        active_agents.append(active_agents_round)

    # Check that the expected and actual active agents match.
    for round_id in range(max_message_rounds):
        message = f"Active agents in round {round_id} do not match expectation."
        assert active_agents[round_id] == expected_active_agents[round_id], message
