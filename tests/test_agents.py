import pytest

import torch

from tensordict import TensorDict
from tensordict.nn import TensorDictSequential

from einops import repeat

from pvg import (
    Parameters,
    ScenarioType,
    TrainerType,
    InteractionProtocolType,
    AgentsParameters,
    GraphIsomorphismAgentParameters,
    ImageClassificationAgentParameters,
    RlTrainerParameters,
)
from pvg.parameters import AgentParameters, AGENT_NAMES
from pvg.factory import build_scenario_instance
from pvg.experiment_settings import ExperimentSettings
from pvg.scenario_base import (
    DataLoader,
    TensorDictEnvironment,
    TensorDictAgentPartMixin,
)


def test_graph_isomorphism_combined_agents():
    """Test the combined agents for the graph isomorphism scenario with three agents

    The idea is to catch dimension bugs caused by the fact that we normally have two
    agents and two graphs. If these dimension are mixed up this should be caught here.
    """

    # Very basic parameters with three agents
    agent_params = GraphIsomorphismAgentParameters(
        num_gnn_layers=1,
        d_gnn=1,
        d_gin_mlp=1,
        num_heads=2,
        num_transformer_layers=1,
        d_transformer=2,
        d_transformer_mlp=1,
        d_node_selector=1,
        num_node_selector_layers=1,
        d_decider=1,
        num_decider_layers=1,
        d_value=1,
        num_value_layers=1,
        normalize_message_history=False,
    )
    params = Parameters(
        scenario=ScenarioType.GRAPH_ISOMORPHISM,
        trainer=TrainerType.VANILLA_PPO,
        dataset="eru10000",
        interaction_protocol=InteractionProtocolType.MERLIN_ARTHUR,
        agents=AgentsParameters(
            prover0=agent_params,
            prover1=agent_params,
            verifier=agent_params,
        ),
        rl=RlTrainerParameters(
            use_shared_body=True,
        ),
    )

    # The experiment settings
    settings = ExperimentSettings(device="cpu", test_run=True, ignore_cache=True)

    # Build the scenario instance
    scenario_instance = build_scenario_instance(
        params=params,
        settings=settings,
    )

    # Combine the agents into a single actor
    combined_actor = TensorDictSequential(
        scenario_instance.combined_body, scenario_instance.combined_policy_head
    )

    # Get the dataloader
    dataloader = DataLoader(dataset=scenario_instance.train_dataset, batch_size=8)

    # Make sure the combined actor can process a batch
    batch = next(iter(dataloader))
    batch["round"] = torch.zeros(batch.batch_size, dtype=torch.int64)
    batch["decision_restriction"] = torch.zeros(batch.batch_size, dtype=torch.int64)
    message_shape = batch["x"].shape[:-5] + batch["x"].shape[-4:]
    batch["message"] = torch.zeros(message_shape, dtype=torch.float32)
    combined_actor(batch)


@pytest.mark.parametrize(
    "scenario_type, agent_params_class",
    [
        (ScenarioType.GRAPH_ISOMORPHISM, GraphIsomorphismAgentParameters),
        (ScenarioType.IMAGE_CLASSIFICATION, ImageClassificationAgentParameters),
    ],
    ids=["graph_isomorphism", "image_classification"],
)
def test_channel_visibility(
    scenario_type: ScenarioType, agent_params_class: type[AgentParameters]
):
    """Test that the agents only see the channels they are supposed to see.

    Uses the multi-channel test protocol to check that the agents only see the channels
    they are supposed to see.

    Replaces the bodies of the agents with a test body that checks the data received by
    the agent. The test body checks that the data has the expected number of channels
    and that all the expected channels are present in the data.
    """

    torch.manual_seed(0)

    num_message_channels = 3
    max_message_rounds = 8
    batch_size = 5

    # Build the experiment components
    basic_agent_params = agent_params_class.construct_test_params()
    params = Parameters(
        scenario_type,
        TrainerType.VANILLA_PPO,
        "test",
        agents=AgentsParameters(
            **{
                agent_name: basic_agent_params
                for agent_name in AGENT_NAMES[
                    InteractionProtocolType.MULTI_CHANNEL_TEST
                ]
            }
        ),
        rl=RlTrainerParameters(frames_per_batch=max_message_rounds * batch_size),
        interaction_protocol=InteractionProtocolType.MULTI_CHANNEL_TEST,
    )
    settings = ExperimentSettings(
        device="cpu", test_run=True, pin_memory=False, ignore_cache=True
    )
    scenario_instance = build_scenario_instance(params, settings)

    protocol_handler = scenario_instance.protocol_handler
    channel_names = protocol_handler.message_channel_names
    environment: TensorDictEnvironment = scenario_instance.train_environment
    combined_body = scenario_instance.combined_body
    old_bodies = combined_body.bodies.copy()

    class TestBody(TensorDictAgentPartMixin):
        """A body that checks the data received by the agent"""

        def forward(self, data: TensorDict, *args, **kwargs) -> TensorDict:

            # Check that the data has the expected number of channels (i.e. the number
            # visible to the agent)
            expected_num_channels = len(self.visible_message_channel_names)
            assert_message = (
                f"Agent {self.agent_name!r} did not receive the expected number of "
                f"channels. Expected {expected_num_channels}, got "
                f"{data['x'].shape[2]}"
            )
            assert data["x"].shape[2] == expected_num_channels, assert_message

            # Check that all the expected channels are present in the data
            for i, channel_id in enumerate(self.visible_message_channel_indices):
                assert_message = (
                    f"Channel {channel_names[channel_id]!r} not found in data "
                    f"for agent {self.agent_name!r}"
                )
                assert (data["x"][:, :, i, ...] == channel_id).all(), assert_message

            # Run the forward method of the original body
            return old_bodies[self.agent_name].forward(data, *args, **kwargs)

        @property
        def agent_level_in_keys(self):
            return old_bodies[self.agent_name].agent_level_in_keys

        @property
        def env_level_in_keys(self):
            return old_bodies[self.agent_name].env_level_in_keys

        @property
        def agent_level_out_keys(self):
            return old_bodies[self.agent_name].agent_level_out_keys

        @property
        def env_level_out_keys(self):
            return old_bodies[self.agent_name].env_level_out_keys

        def to(self, *args, **kwargs):
            return self

    # Replace the bodies with the test bodies
    for agent_name in combined_body.bodies.keys():
        combined_body.bodies[agent_name] = TestBody(
            params, settings, agent_name, protocol_handler
        )

    # The protocol should have 3 channels and 8 rounds for the test to work. If this
    # changes, the test will need to be updated.
    assert num_message_channels == protocol_handler.num_message_channels
    assert max_message_rounds == protocol_handler.max_message_rounds

    # Construct the input tensordict for the combined body
    input_td = environment.observation_spec.rand()
    input_td["round"] = torch.arange(batch_size) % max_message_rounds
    extra_dims = {
        f"dim_{i}": dim for i, dim in enumerate(environment.main_message_space_shape)
    }
    input_td["x"] = repeat(
        torch.arange(num_message_channels, dtype=input_td["x"].dtype),
        f"channel -> batch round channel position {' '.join(extra_dims.keys())}",
        batch=batch_size,
        round=max_message_rounds,
        position=1,
        **extra_dims,
    )

    # Run the combined body on the input tensordict
    combined_body(input_td)
