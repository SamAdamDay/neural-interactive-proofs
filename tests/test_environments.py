import torch
import torch.nn.functional as F
from torch.testing import assert_close

from tensordict import TensorDict

from torchrl.envs.utils import check_env_specs

from pvg.parameters import (
    Parameters,
    ScenarioType,
    TrainerType,
    AgentsParameters,
    GraphIsomorphismAgentParameters,
    RlTrainerParameters,
    CommonProtocolParameters,
    PvgProtocolParameters,
)
from pvg.experiment_settings import ExperimentSettings
from pvg.graph_isomorphism import GraphIsomorphismEnvironment, GraphIsomorphismDataset
from pvg.image_classification import (
    ImageClassificationEnvironment,
    ImageClassificationDataset,
)
from pvg.protocols import build_protocol_handler


def test_environment_specs():
    """Test that the environment has the correct specs."""

    scenario_types = [ScenarioType.GRAPH_ISOMORPHISM, ScenarioType.IMAGE_CLASSIFICATION]
    dataset_classes = [GraphIsomorphismDataset, ImageClassificationDataset]
    environment_classes = [GraphIsomorphismEnvironment, ImageClassificationEnvironment]
    for scenario_type, dataset_class, environment_class in zip(
        scenario_types, dataset_classes, environment_classes
    ):
        params = Parameters(scenario_type, TrainerType.VANILLA_PPO, "test")
        protocol_handler = build_protocol_handler(params)
        settings = ExperimentSettings(device="cpu", test_run=True, pin_memory=False)
        dataset = dataset_class(params, settings, protocol_handler)
        env = environment_class(params, settings, dataset, protocol_handler)
        check_env_specs(env)


def test_graph_isomorphism_environment_step():
    """Make sure the GI environment step method works as expected."""

    batch_size = 12
    max_message_rounds = 6

    # Set up the environment.
    params = Parameters(
        ScenarioType.GRAPH_ISOMORPHISM,
        TrainerType.VANILLA_PPO,
        "test",
        rl=RlTrainerParameters(
            frames_per_batch=batch_size * max_message_rounds,
            steps_per_env_per_iteration=max_message_rounds,
        ),
        agents=AgentsParameters(
            prover=GraphIsomorphismAgentParameters(),
            verifier=GraphIsomorphismAgentParameters(),
        ),
        protocol_common=CommonProtocolParameters(
            prover_reward=1,
            verifier_reward=2,
            verifier_terminated_penalty=-4,
            verifier_no_guess_reward=8,
            verifier_incorrect_penalty=-16,
        ),
        pvg_protocol=PvgProtocolParameters(
            max_message_rounds=max_message_rounds,
            min_message_rounds=1,
            verifier_first=False,
        ),
        # agents=AgentsParameters(
        #     prover0=GraphIsomorphismAgentParameters(),
        #     prover1=GraphIsomorphismAgentParameters(),
        #     verifier=GraphIsomorphismAgentParameters(),
        # ),
        # debate_protocol=ProtocolParameters(
        #     protocol=InteractionProtocolType.DEBATE,
        #     prover_reward=1,
        #     verifier_reward=2,
        #     verifier_terminated_penalty=-4,
        #     verifier_no_guess_reward=8,
        #     verifier_incorrect_penalty=-16,
        # ),
    )
    settings = ExperimentSettings(device="cpu", test_run=True)
    protocol_handler = build_protocol_handler(params)
    dataset = GraphIsomorphismDataset(params, settings, protocol_handler)
    env = GraphIsomorphismEnvironment(params, settings, dataset, protocol_handler)

    max_num_nodes = env.max_num_nodes

    # This test setup only works when the max number of nodes in the "test" dataset is
    # 8. If this changes, this test will need to be updated.
    assert max_num_nodes == 8

    # Set up the TensorDict to feed into the environment.
    env_td = TensorDict(
        dict(
            adjacency=torch.zeros(
                batch_size, max_num_nodes, max_num_nodes, dtype=torch.int32
            ),
            node_mask=torch.ones(batch_size, max_num_nodes, dtype=torch.bool),
            round=torch.remainder(
                torch.arange(batch_size, dtype=torch.long), max_message_rounds
            ),
            message_history=torch.zeros(
                batch_size, 2, max_num_nodes, max_message_rounds, dtype=torch.float32
            ),
            x=torch.zeros(
                batch_size, 2, max_num_nodes, max_message_rounds, dtype=torch.float32
            ),
            y=torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1]),
            agents=dict(
                node_selected=torch.tensor(
                    [
                        [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12],
                        [7, 6, 5, 4, 3, 2, 1, 0, 8, 15, 14, 13],
                    ]
                ).transpose(0, 1),
                decision=torch.tensor(
                    [[0] * batch_size, [0, 0, 1, 2, 2, 2, 0, 0, 1, 1, 1, 1]]
                ).transpose(0, 1),
            ),
            done=torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0], dtype=torch.bool),
        ),
        batch_size=batch_size,
    )

    # Define the expected output.
    expected_message_history = torch.zeros(
        batch_size, 2, max_num_nodes, max_message_rounds, dtype=torch.float32
    )
    expected_message = torch.zeros(batch_size, 2, 8, dtype=torch.float32)
    for i in range(batch_size):
        round = env_td["round"][i]
        agent_index = round % 2
        message = env_td["agents", "node_selected"][i, agent_index]
        expected_message[i] = F.one_hot(message, 2 * max_num_nodes).view(2, 8)
        graph_id = message // max_num_nodes
        expected_message_history[i, graph_id, message % max_num_nodes, round] = 1
    expected_next = TensorDict(
        dict(
            adjacency=env_td["adjacency"],
            node_mask=env_td["node_mask"],
            message_history=expected_message_history,
            x=expected_message_history,
            round=env_td["round"] + 1,
            done=torch.tensor([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1], dtype=torch.bool),
            message=expected_message,
            agents=dict(
                reward=torch.tensor(
                    [
                        [0, 0],
                        [0, 2],
                        [0, 0],
                        [0, 8],
                        [0, 0],
                        [0, -4],
                        [0, 0],
                        [0, -16],
                        [0, 0],
                        [1, -16],
                        [0, 0],
                        [1, 2],
                    ],
                    dtype=torch.float32,
                )
            ),
        ),
        batch_size=batch_size,
    )

    # Run the step method.
    next = env._step(env_td)

    # Check that the output is as expected.
    assert_close(next["adjacency"], expected_next["adjacency"])
    assert_close(next["node_mask"], expected_next["node_mask"])
    assert_close(next["message_history"], expected_next["message_history"])
    assert_close(next["round"], expected_next["round"])
    assert_close(next["done"], expected_next["done"])
    assert_close(next["message"], expected_next["message"])
    assert_close(next["agents", "reward"], expected_next["agents", "reward"])
