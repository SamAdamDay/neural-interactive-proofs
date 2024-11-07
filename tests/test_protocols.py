from itertools import product

import pytest

import torch
from torch.testing import assert_close
from torch import Tensor

from tensordict import TensorDict

from einops import rearrange, repeat

from jaxtyping import Int, Bool, Float

from pvg.experiment_settings import ExperimentSettings
from pvg.parameters import (
    HyperParameters,
    InteractionProtocolType,
    ScenarioType,
    TrainerType,
    CommonProtocolParameters,
    PvgProtocolParameters,
    ZkProtocolParameters,
)
from pvg.protocols import build_protocol_handler, ZeroKnowledgeProtocol
from pvg.utils.maths import set_seed


@pytest.mark.parametrize(
    "base_protocol",
    [
        InteractionProtocolType.PVG,
        InteractionProtocolType.DEBATE,
        InteractionProtocolType.ABSTRACT_DECISION_PROBLEM,
        InteractionProtocolType.MNIP,
        InteractionProtocolType.MERLIN_ARTHUR,
        InteractionProtocolType.MULTI_CHANNEL_TEST,
    ],
)
def test_zero_knowledge_channel_names(base_protocol: InteractionProtocolType):
    """Make sure that the zero-knowledge protocol has the correct channel names.

    Parts of the protocol rely on having the channel names in a particular order, so if
    this every changes, the protocol will need to be updated.

    Parameters
    ----------
    base_protocol : InteractionProtocolType
        The base protocol to use. This is extended to a zero-knowledge protocol.
    """

    hyper_params = HyperParameters(
        scenario=ScenarioType.GRAPH_ISOMORPHISM,
        trainer=TrainerType.VANILLA_PPO,
        dataset="test",
        interaction_protocol=base_protocol,
        protocol_common=CommonProtocolParameters(zero_knowledge=True),
    )
    settings = ExperimentSettings(device="cpu", test_run=True)
    protocol_handler: ZeroKnowledgeProtocol = build_protocol_handler(
        hyper_params, settings
    )

    base_protocol_handler = protocol_handler.base_protocol

    expected_channel_names = [
        f"{name}_{kind}"
        for kind, name in product(
            ["standard", "adversarial", "simulator"],
            base_protocol_handler.message_channel_names,
        )
    ]

    assert protocol_handler.message_channel_names == expected_channel_names


@pytest.mark.parametrize(
    "base_protocol",
    [
        InteractionProtocolType.PVG,
        InteractionProtocolType.DEBATE,
        InteractionProtocolType.ABSTRACT_DECISION_PROBLEM,
        InteractionProtocolType.MNIP,
        InteractionProtocolType.MERLIN_ARTHUR,
        InteractionProtocolType.MULTI_CHANNEL_TEST,
    ],
)
def test_zero_knowledge_agent_names(base_protocol: InteractionProtocolType):
    """Make sure that the zero-knowledge protocol has the correct agent names.

    The agent names should be the base agent names plus the adversarial verifier and
    simulator.

    Parameters
    ----------
    base_protocol : InteractionProtocolType
        The base protocol to use. This is extended to a zero-knowledge protocol.
    """

    hyper_params = HyperParameters(
        scenario=ScenarioType.GRAPH_ISOMORPHISM,
        trainer=TrainerType.VANILLA_PPO,
        dataset="test",
        interaction_protocol=base_protocol,
        protocol_common=CommonProtocolParameters(zero_knowledge=True),
    )
    settings = ExperimentSettings(device="cpu", test_run=True)
    protocol_handler: ZeroKnowledgeProtocol = build_protocol_handler(
        hyper_params, settings
    )

    base_protocol_handler = protocol_handler.base_protocol

    expected_agent_names = base_protocol_handler.agent_names + [
        "adversarial_verifier",
        "simulator",
    ]

    assert protocol_handler.agent_names == expected_agent_names


@pytest.mark.parametrize(
    "base_protocol",
    [
        InteractionProtocolType.PVG,
        InteractionProtocolType.DEBATE,
        InteractionProtocolType.ABSTRACT_DECISION_PROBLEM,
        InteractionProtocolType.MNIP,
        InteractionProtocolType.MERLIN_ARTHUR,
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

    hyper_params = HyperParameters(
        scenario=ScenarioType.GRAPH_ISOMORPHISM,
        trainer=TrainerType.VANILLA_PPO,
        dataset="test",
        interaction_protocol=base_protocol,
        protocol_common=CommonProtocolParameters(zero_knowledge=True),
    )
    settings = ExperimentSettings(device="cpu", test_run=True)
    protocol_handler: ZeroKnowledgeProtocol = build_protocol_handler(
        hyper_params, settings
    )

    agent_names = protocol_handler.agent_names
    channel_names = protocol_handler.message_channel_names
    max_message_rounds = protocol_handler.max_message_rounds
    base_protocol_handler = protocol_handler.base_protocol
    base_agent_names = base_protocol_handler.agent_names
    base_channel_names = base_protocol_handler.message_channel_names

    round_id = torch.arange(0, max_message_rounds)
    seed = torch.randint(0, 1000, (max_message_rounds,))

    # Get the masks of which agents are active in which rounds and channels
    active_agents_mask = protocol_handler.get_active_agents_mask_from_rounds_and_seed(
        round_id, seed
    )
    base_active_agent_mask = (
        base_protocol_handler.get_active_agents_mask_from_rounds_and_seed(
            round_id, seed
        )
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
                expected_active_agents_round.add(
                    ("simulator", f"{channel_name}_simulator")
                )
            else:
                expected_active_agents_round.add(
                    (agent_name, f"{channel_name}_standard")
                )
                expected_active_agents_round.add(
                    (agent_name, f"{channel_name}_adversarial")
                )
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


def test_zero_knowledge_pvg_step_method():
    """Test the implementation of taking a step in the zero-knowledge PVG protocol."""

    message_size = 1
    num_message_logits = 2
    simulator_reward_coefficient = 0.342
    aux_prover_reward_coefficient = 0.123

    hyper_params = HyperParameters(
        scenario=ScenarioType.GRAPH_ISOMORPHISM,
        trainer=TrainerType.VANILLA_PPO,
        dataset="test",
        interaction_protocol=InteractionProtocolType.PVG,
        protocol_common=CommonProtocolParameters(
            zero_knowledge=True, verifier_first=True
        ),
        pvg_protocol=PvgProtocolParameters(),
        message_size=message_size,
        zk_protocol=ZkProtocolParameters(
            simulator_reward_coefficient=simulator_reward_coefficient,
            aux_prover_reward_coefficient=aux_prover_reward_coefficient,
            use_multiple_simulators=False,
        ),
    )
    settings = ExperimentSettings(device="cpu", test_run=True)
    protocol_handler: ZeroKnowledgeProtocol = build_protocol_handler(
        hyper_params, settings
    )

    agent_names = protocol_handler.agent_names
    channel_names = protocol_handler.message_channel_names
    num_channels = len(channel_names)
    max_message_rounds = protocol_handler.max_message_rounds
    base_protocol_handler = protocol_handler.base_protocol
    base_channel_names = base_protocol_handler.message_channel_names
    num_base_channels = len(base_channel_names)

    verifier_index = protocol_handler.verifier_index
    adversarial_verifier_index = agent_names.index("adversarial_verifier")
    simulator_indices = agent_names.index("simulator") if not protocol_handler.use_multiple_simulators else [protocol_handler.agent_names.index(a) for a in protocol_handler.agent_names if "simulator" in a]

    batch_size = (32, 8)

    set_seed(0)

    env_td = TensorDict(
        dict(
            y=torch.randint(0, 2, (*batch_size, 1)),
            round=torch.randint(0, max_message_rounds, batch_size),
            seed=torch.randint(0, 1000, batch_size),
            done=torch.randint(0, 2, batch_size, dtype=torch.bool),
            terminated=torch.randint(0, 2, batch_size, dtype=torch.bool),
            agents=TensorDict(
                dict(
                    decision=torch.randint(0, 3, (*batch_size, len(agent_names))),
                    done=torch.randint(
                        0, 2, (*batch_size, len(agent_names)), dtype=torch.bool
                    ),
                ),
                batch_size=(*batch_size, len(agent_names)),
            ),
            message_cosine_similarity=2 * torch.rand(*batch_size) - 1,
            decision_cosine_similarity=2 * torch.rand(*batch_size) - 1,
        ),
        batch_size=batch_size,
    )

    # Get the index of the first active verifier channel, if the verifier is active at
    # all.
    verifier_first_active_channel: Int[Tensor, "..."] = torch.argmax(
        base_protocol_handler.get_active_agents_mask_from_rounds_and_seed(
            env_td["round"], env_td["seed"]
        )[..., verifier_index, :].int(),
        dim=-1,
    )

    # Get the mask of the first active channel for the adversarial verifier
    adversarial_verifier_first_active_channel: Bool[Tensor, "... channel"] = (
        torch.zeros((*batch_size, num_channels), dtype=torch.bool)
    )
    adversarial_verifier_first_active_channel.scatter_(
        -1, verifier_first_active_channel[..., None] + num_base_channels, True
    )

    # Get the mask of the first active channel for the simulator
    simulator_first_active_channel: Bool[Tensor, "... channel"] = torch.zeros(
        (*batch_size, num_channels), dtype=torch.bool
    )
    simulator_first_active_channel.scatter_(
        -1, verifier_first_active_channel[..., None] + 2 * num_base_channels, True
    )

    # Create the first set of message logits. In the first channel in which the verifier
    # is active, the logits are 1.0 followed by zeros. In the other channels, the logits
    # are all zeros.
    message_logits_adversarial = torch.where(
        adversarial_verifier_first_active_channel[..., None],
        torch.tensor([1.0] + [0.0] * (num_message_logits - 1)).expand(
            *batch_size, num_channels, num_message_logits
        ),
        0,
    )

    # Create the second set of message logits. In the first channel in which the
    # verifier is active, the logits are: cosine_similarity, sqrt(1 - cosine_similarity),
    # followed by zeros. In the other channels, the logits are all zeros.
    # TODO update to allow for multiple simulators
    source: Float[Tensor, "... logit"] = torch.stack(
        [
            env_td["message_cosine_similarity"],
            torch.sqrt(1 - torch.pow(env_td["message_cosine_similarity"], 2)),
        ]
        + [torch.zeros(batch_size)] * (num_message_logits - 2),
        dim=-1,
    )
    message_logits_simulator = torch.where(
        simulator_first_active_channel[..., None],
        repeat(source, "... logit -> ... channel logit", channel=num_channels),
        0,
    ) 

    # Create two sets of decision logits, whose cosine similarities are those specified
    # TODO update to multiple simulators
    decision_logits_a = torch.tensor([1.0, 0.0, 0.0]).expand(*batch_size, 3).clone()
    decision_logits_b = torch.zeros_like(decision_logits_a) 
    decision_logits_b[..., 0] = env_td["decision_cosine_similarity"]
    decision_logits_b[..., 1] = torch.sqrt(
        1 - torch.pow(env_td["decision_cosine_similarity"], 2)
    )

    # Build the combined message logits tensor. We explicitly set the logits for the
    # adversarial verifier and simulator, and leave the rest as random to make sure they
    # are not used.
    message_logits = torch.randn(
        *batch_size,
        len(agent_names),
        len(channel_names),
        message_size,
        num_message_logits,
    )
    message_logits[..., adversarial_verifier_index, :, 0, :] = (
        message_logits_adversarial
    )
    message_logits[..., adversarial_verifier_index, :, 1:, :] = 0
    message_logits[..., simulator_indices, :, 0, :] = message_logits_simulator
    message_logits[..., simulator_indices, :, 1:, :] = 0

    # Build the combined decision logits tensor. Similar to the message logits, we
    # leave the logits for the non-verifier agents as random.
    decision_logits = torch.randn(*batch_size, len(agent_names), 3)
    decision_logits[..., adversarial_verifier_index, :] = decision_logits_a
    decision_logits[..., simulator_indices, :] = decision_logits_b

    # Sanity check that the cosine similarities are as expected
    grouped_message_logits = rearrange(
        message_logits,
        "... agent (group base_channel) position logit "
        "-> ... agent group base_channel position logit",
        group=3,
    )

    # TODO update to multiple simulators
    assert_close(
        torch.cosine_similarity(
            rearrange(
                grouped_message_logits[..., adversarial_verifier_index, 1, :, :, :],
                "... base_channel position logit -> ... (position base_channel logit)",
            ),
            rearrange(
                grouped_message_logits[..., simulator_indices, 2, :, :, :],
                "... base_channel position logit -> ... (position base_channel logit)",
            ),
            dim=-1,
        ),
        env_td["message_cosine_similarity"],
    )
    # TODO update to multiple simulators
    assert_close(
        torch.cosine_similarity(
            decision_logits[..., adversarial_verifier_index, :],
            decision_logits[..., simulator_indices, :],
            dim=-1,
        ),
        env_td["decision_cosine_similarity"],
    )

    # Add the logits to the environment tensor dictionary
    env_td["agents", "main_message_logits"] = message_logits
    env_td["agents", "decision_logits"] = decision_logits

    # Run the step method for both the zero-knowledge and base protocols
    shared_done, agent_done, terminated, reward = (
        protocol_handler.step_interaction_protocol(env_td)
    )
    _, base_agent_done, base_terminated, base_reward = (
        protocol_handler.base_protocol.step_interaction_protocol(env_td)
    )

    original_shared_done = env_td["done"]
    original_adversarial_verifier_done = env_td["agents", "done"][
        ..., adversarial_verifier_index
    ]
    verifier_done = agent_done[..., verifier_index]
    adversarial_verifier_done = agent_done[..., adversarial_verifier_index]

    # Get masks for when the verifier is active and when it can guess
    verifier_active_mask = (
        base_protocol_handler.get_active_agents_mask_from_rounds_and_seed(
            env_td["round"], env_td["seed"]
        )[..., verifier_index, :]
    )
    verifier_active_mask: Bool[Tensor, "..."] = verifier_active_mask.any(dim=-1)
    verifier_can_guess_mask: Bool[Tensor, "..."] = (
        base_protocol_handler.get_verifier_guess_mask_from_rounds_and_seed(
            env_td["round"], env_td["seed"]
        )
    )

    # Make sure the verifier done signal is the same in both cases
    assert_close(verifier_done, base_agent_done[..., verifier_index])

    # Make sure the adversarial verifier done signal is correct. The adversarial
    # verifier is done if it makes a guess (i.e. output is 0 or 1) when this is
    # permitted, or if it was already done.
    expected_adversarial_verifier_done = (
        env_td["agents", "decision"][..., adversarial_verifier_index] != 2
    )
    expected_adversarial_verifier_done = (
        expected_adversarial_verifier_done & verifier_active_mask
    )
    expected_adversarial_verifier_done = (
        expected_adversarial_verifier_done | original_adversarial_verifier_done
    )
    assert_close(adversarial_verifier_done, expected_adversarial_verifier_done)

    # Make sure the shared done signal is correct
    expected_shared_done = (
        verifier_done & adversarial_verifier_done
    ) | original_shared_done
    assert_close(shared_done, expected_shared_done)

    # Make sure the dones for the non-verifier agents are the same as the shared done
    for agent_index, agent_name in enumerate(agent_names):
        if agent_name in ["verifier", "adversarial_verifier"]:
            continue
        assert_close(
            agent_done[..., agent_index],
            shared_done,
            msg=f"Done signal for {agent_name!r} not the same as shared done.",
        )

    # Make sure the terminated signal is the same in both cases
    assert_close(terminated, base_terminated)

    # Make sure the verifier reward is the same in both cases
    assert_close(reward[..., verifier_index], base_reward[..., verifier_index])

    # Check that the simulator rewards are as expected. The rewards are proportional to
    # the cosine similarity between the adversarial verifier and simulator logits.
    expected_simulator_reward = (
        env_td["message_cosine_similarity"] + env_td["decision_cosine_similarity"]
    )
    expected_simulator_reward = (
        env_td["message_cosine_similarity"] * verifier_active_mask
    )
    expected_simulator_reward += (
        env_td["decision_cosine_similarity"] * verifier_can_guess_mask
    )
    expected_simulator_reward *= simulator_reward_coefficient
    assert_close(reward[..., simulator_indices], expected_simulator_reward)

    # Make sure the adversarial verifier reward the negative of the simulator reward
    assert_close(reward[..., adversarial_verifier_index], -reward[..., simulator_indices].mean(dim=-1))

    # Make sure the prover rewards are the base prover rewards plus the simulator reward
    # multiplied by the coefficient.
    for prover_name in base_protocol_handler.prover_names:
        prover_id = agent_names.index(prover_name)
        base_prover_reward = base_reward[..., prover_id]
        expected_prover_reward = base_prover_reward + (
            expected_simulator_reward * aux_prover_reward_coefficient
        )
        assert_close(
            reward[..., prover_id],
            expected_prover_reward,
            msg=f"Prover {prover_name!r} reward not as expected.",
        )
