import torch

from tensordict.nn import TensorDictSequential

from pvg import (
    Parameters,
    ScenarioType,
    TrainerType,
    AgentsParameters,
    GraphIsomorphismAgentParameters,
)
from pvg.scenario_instance import build_scenario_instance
from pvg.experiment_settings import ExperimentSettings
from pvg.scenario_base import DataLoader


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
        normalize_message_history=True,
    )
    params = Parameters(
        scenario=ScenarioType.GRAPH_ISOMORPHISM,
        trainer=TrainerType.VANILLA_PPO,
        dataset="eru10000",
        agents=AgentsParameters(
            agent1=agent_params,
            agent2=agent_params,
            agent3=agent_params,
        ),
    )

    # The experiment settings
    settings = ExperimentSettings(device="cpu", test_run=True)

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
    batch["message"] = torch.zeros_like(batch["y"])
    batch["ignore_message"] = torch.ones_like(batch["y"], dtype=torch.bool)
    batch["round"] = torch.zeros(batch.batch_size, dtype=torch.int64)
    batch["decision_restriction"] = torch.zeros(batch.batch_size, dtype=torch.int64)
    combined_actor(batch)
