from pvg import (
    Parameters,
    GraphIsomorphismParameters,
    GraphIsomorphismAgentParameters,
    SoloAgentParameters,
    PpoParameters,
    ScenarioType,
    TrainerType,
    run_experiment,
)
from pvg.utils.output import DummyTqdm

def test_run_experiment():
    """Test running experiments with very basic parameters."""

    # Very basic parameters for each scenario
    scenario_params = {
        ScenarioType.GRAPH_ISOMORPHISM: GraphIsomorphismParameters(
            prover=GraphIsomorphismAgentParameters(
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
                d_critic=1,
                num_critic_transformer_layers=1,
                num_critic_layers=1,
            ),
            verifier=GraphIsomorphismAgentParameters(
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
                d_critic=1,
                num_critic_transformer_layers=1,
                num_critic_layers=1,
            ),
        )
    }

    # Very basic parameters for each trainer
    trainer_params = {
        TrainerType.SOLO_AGENT: SoloAgentParameters(
            num_epochs=1,
            batch_size=1,
        ),
        TrainerType.PPO: PpoParameters(
            num_iterations=8,
            num_epochs=4,
            minibatch_size=64,
        ),
    }

    for scenario_type, scenario_param in scenario_params.items():
        for trainer_type, trainer_param in trainer_params.items():
            # Construct the parameters
            params_dict = dict(
                scenario=scenario_type, trainer=trainer_type, dataset="test"
            )
            params_dict[str(scenario_type)] = scenario_param
            params_dict[str(trainer_type)] = trainer_param
            params = Parameters.from_dict(params_dict)

            # Run the experiment in test mode
            run_experiment(params, tqdm_func=DummyTqdm, test_run=True)
