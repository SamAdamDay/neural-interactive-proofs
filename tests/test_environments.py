"""Tests for the environment modules.

These tests check that the environments are correctly specified, and that the step
functions work as expected.
"""

import pytest

import torch
import torch.nn.functional as F
from torch.testing import assert_close

from tensordict import TensorDict

from torchrl.envs.utils import check_env_specs

from einops import rearrange, repeat

from pvg.scenario_base import Environment, TensorDictDataset
from pvg.parameters import (
    HyperParameters,
    ScenarioType,
    TrainerType,
    AgentsParameters,
    GraphIsomorphismAgentParameters,
    ImageClassificationAgentParameters,
    RlTrainerParameters,
    CommonProtocolParameters,
    PvgProtocolParameters,
    ImageClassificationParameters,
)
from pvg.experiment_settings import ExperimentSettings
from pvg.graph_isomorphism import GraphIsomorphismEnvironment, GraphIsomorphismDataset
from pvg.image_classification import (
    ImageClassificationEnvironment,
    ImageClassificationDataset,
)
from pvg.protocols import build_protocol_handler
from pvg.utils.maths import set_seed


@pytest.mark.parametrize(
    "scenario_type, dataset_class, environment_class",
    [
        (
            ScenarioType.GRAPH_ISOMORPHISM,
            GraphIsomorphismDataset,
            GraphIsomorphismEnvironment,
        ),
        (
            ScenarioType.IMAGE_CLASSIFICATION,
            ImageClassificationDataset,
            ImageClassificationEnvironment,
        ),
    ],
    ids=["graph_isomorphism", "image_classification"],
)
def test_environment_specs(
    scenario_type: ScenarioType,
    dataset_class: type[TensorDictDataset],
    environment_class: type[Environment],
):
    """Test that the environment has the correct specs.

    Parameters
    ----------
    scenario_type : ScenarioType
        The scenario to test.
    dataset_class : type[TensorDictDataset]
        The dataset class to use for the scenario.
    environment_class : type[Environment]
        The environment class to use for the scenario.
    """

    hyper_params = HyperParameters(
        scenario_type, TrainerType.VANILLA_PPO, "test", message_size=3
    )
    settings = ExperimentSettings(
        device="cpu", test_run=True, pin_memory=False, ignore_cache=True
    )
    protocol_handler = build_protocol_handler(hyper_params, settings)
    dataset = dataset_class(hyper_params, settings, protocol_handler)
    env = environment_class(hyper_params, settings, dataset, protocol_handler)
    check_env_specs(env)


def test_graph_isomorphism_environment_step():
    """Make sure the GI environment step method works as expected."""

    batch_size = 12
    max_message_rounds = 6
    message_size = 1

    # Set up the environment.
    hyper_params = HyperParameters(
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
            verifier_first=False,
        ),
        pvg_protocol=PvgProtocolParameters(
            max_message_rounds=max_message_rounds,
            min_message_rounds=1,
        ),
        message_size=message_size,
    )
    settings = ExperimentSettings(device="cpu", test_run=True, ignore_cache=True)
    protocol_handler = build_protocol_handler(hyper_params, settings)
    dataset = GraphIsomorphismDataset(hyper_params, settings, protocol_handler)
    env = GraphIsomorphismEnvironment(hyper_params, settings, dataset, protocol_handler)

    max_num_nodes = env.max_num_nodes
    num_message_channels = protocol_handler.num_message_channels

    # This test setup only works when the max number of nodes in the "test" dataset is
    # 8. If this changes, this test will need to be updated.
    assert max_num_nodes == 8

    # This test setup only works when the number of message channels is 1. If this
    # changes, this test will need to be updated.
    assert num_message_channels == 1

    # Set up the TensorDict to feed into the environment.
    done = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0], dtype=torch.bool)
    env_td = TensorDict(
        dict(
            adjacency=torch.zeros(
                batch_size, max_num_nodes, max_num_nodes, dtype=torch.int32
            ),
            node_mask=torch.ones(batch_size, max_num_nodes, dtype=torch.bool),
            round=torch.remainder(
                torch.arange(batch_size, dtype=torch.long), max_message_rounds
            ),
            seed=torch.zeros(batch_size, dtype=torch.int64),
            message_history=torch.zeros(
                batch_size,
                max_message_rounds,
                num_message_channels,
                message_size,
                2,
                max_num_nodes,
                dtype=torch.float32,
            ),
            x=torch.zeros(
                batch_size,
                max_message_rounds,
                num_message_channels,
                message_size,
                2,
                max_num_nodes,
                dtype=torch.float32,
            ),
            y=torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1]),
            agents=TensorDict(
                dict(
                    node_selected=rearrange(
                        torch.tensor(
                            [
                                [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12],
                                [7, 6, 5, 4, 3, 2, 1, 0, 8, 15, 14, 13],
                            ]
                        ),
                        "agent batch -> batch agent 1 1",
                    ),
                    decision=torch.tensor(
                        [[0] * batch_size, [0, 0, 1, 2, 2, 2, 0, 0, 1, 1, 1, 1]]
                    ).transpose(0, 1),
                    done=repeat(done, "batch -> batch 2"),
                ),
                batch_size=(batch_size, 2),
            ),
            done=done,
            terminated=torch.zeros(batch_size, dtype=torch.bool),
        ),
        batch_size=batch_size,
    )

    # Define the expected output.
    expected_message_history = torch.zeros(
        batch_size,
        max_message_rounds,
        num_message_channels,
        message_size,
        2,
        max_num_nodes,
        dtype=torch.float32,
    )
    expected_message = torch.zeros(
        batch_size,
        num_message_channels,
        message_size,
        2,
        max_num_nodes,
        dtype=torch.float32,
    )
    for i in range(batch_size):
        round_id = env_td["round"][i]
        agent_index = round_id % 2
        message = env_td["agents", "node_selected"][i, agent_index]
        expected_message[i] = F.one_hot(message, 2 * max_num_nodes).view(
            num_message_channels, message_size, 2, max_num_nodes
        )
        graph_id = message // max_num_nodes
        expected_message_history[
            i, round_id, 0, 0, graph_id, message % max_num_nodes
        ] = 1
    expected_done = torch.tensor([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1], dtype=torch.bool)
    expected_next = TensorDict(
        dict(
            adjacency=env_td["adjacency"],
            node_mask=env_td["node_mask"],
            message_history=expected_message_history,
            x=expected_message_history,
            round=env_td["round"] + 1,
            done=expected_done,
            message=expected_message,
            agents=TensorDict(
                dict(
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
                    ),
                    done=repeat(expected_done, "batch -> batch 2"),
                ),
                batch_size=(batch_size, 2),
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


def test_image_classification_environment_step():
    """Make sure the IC environment step method works as expected."""

    batch_size = 12
    max_message_rounds = 6
    message_size = 1

    # Set up the environment.
    hyper_params = HyperParameters(
        ScenarioType.IMAGE_CLASSIFICATION,
        TrainerType.VANILLA_PPO,
        "test",
        rl=RlTrainerParameters(
            frames_per_batch=batch_size * max_message_rounds,
            steps_per_env_per_iteration=max_message_rounds,
        ),
        agents=AgentsParameters(
            prover=ImageClassificationAgentParameters(),
            verifier=ImageClassificationAgentParameters(),
        ),
        protocol_common=CommonProtocolParameters(
            prover_reward=1,
            verifier_reward=2,
            verifier_terminated_penalty=-4,
            verifier_no_guess_reward=8,
            verifier_incorrect_penalty=-16,
            verifier_first=False,
        ),
        pvg_protocol=PvgProtocolParameters(
            max_message_rounds=max_message_rounds,
            min_message_rounds=1,
        ),
        image_classification=ImageClassificationParameters(num_block_groups=2),
    )
    settings = ExperimentSettings(device="cpu", test_run=True)
    protocol_handler = build_protocol_handler(hyper_params, settings)
    dataset = ImageClassificationDataset(hyper_params, settings, protocol_handler)
    env = ImageClassificationEnvironment(
        hyper_params, settings, dataset, protocol_handler
    )

    image_width = env.image_width
    image_height = env.image_height
    latent_width = env.latent_width
    latent_height = env.latent_height
    num_image_channels = env.dataset_num_channels

    num_message_channels = protocol_handler.num_message_channels

    # Build the tensor of latent pixels selected by the agents.
    set_seed(0)
    latent_pixel_selected = torch.randint(
        0,
        latent_height * latent_width,
        (batch_size, 2, num_image_channels, message_size),
    )

    # This test setup only works when the number of message channels is 1. If this
    # changes, this test will need to be updated.
    assert num_message_channels == 1

    # Set up the TensorDict to feed into the environment.
    done = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0], dtype=torch.bool)
    env_td = TensorDict(
        dict(
            image=torch.zeros(
                batch_size,
                num_image_channels,
                image_height,
                image_width,
                dtype=torch.int32,
            ),
            round=torch.remainder(
                torch.arange(batch_size, dtype=torch.long), max_message_rounds
            ),
            seed=torch.zeros(batch_size, dtype=torch.int64),
            message_history=torch.zeros(
                batch_size,
                max_message_rounds,
                num_message_channels,
                message_size,
                latent_height,
                latent_width,
                dtype=torch.float32,
            ),
            y=torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1]),
            agents=TensorDict(
                dict(
                    latent_pixel_selected=latent_pixel_selected,
                    decision=torch.tensor(
                        [[0] * batch_size, [0, 0, 1, 2, 2, 2, 0, 0, 1, 1, 1, 1]]
                    ).transpose(0, 1),
                    done=repeat(done, "batch -> batch 2"),
                ),
                batch_size=(batch_size, 2),
            ),
            done=done,
            terminated=torch.zeros(batch_size, dtype=torch.bool),
        ),
        batch_size=batch_size,
    )

    # Define the expected output.
    expected_message_history = torch.zeros(
        batch_size,
        max_message_rounds,
        num_message_channels,
        message_size,
        latent_height,
        latent_width,
        dtype=torch.float32,
    )
    expected_message = torch.zeros(
        batch_size,
        num_message_channels,
        message_size,
        latent_height,
        latent_width,
        dtype=torch.float32,
    )
    for i in range(batch_size):
        round_id = env_td["round"][i]
        agent_index = round_id % 2
        message = env_td["agents", "latent_pixel_selected"][i, agent_index]
        expected_message[i] = F.one_hot(message, latent_height * latent_width).view(
            num_message_channels, message_size, latent_height, latent_width
        )
        y, x = divmod(message.item(), latent_width)
        expected_message_history[i, round_id, 0, 0, y, x] = 1
    expected_done = torch.tensor([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1], dtype=torch.bool)
    expected_next = TensorDict(
        dict(
            image=env_td["image"],
            message_history=expected_message_history,
            x=expected_message_history,
            round=env_td["round"] + 1,
            done=expected_done,
            message=expected_message,
            agents=TensorDict(
                dict(
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
                    ),
                    done=torch.stack(
                        [expected_done, torch.zeros_like(expected_done)], dim=-1
                    ),
                ),
                batch_size=(batch_size, 2),
            ),
        ),
        batch_size=batch_size,
    )

    # Run the step method.
    next = env._step(env_td)

    # Check that the output is as expected.
    assert_close(next["image"], expected_next["image"])
    assert_close(next["message_history"], expected_next["message_history"])
    assert_close(next["round"], expected_next["round"])
    assert_close(next["done"], expected_next["done"])
    assert_close(next["message"], expected_next["message"])
    assert_close(next["agents", "reward"], expected_next["agents", "reward"])
