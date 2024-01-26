from pvg import (
    Parameters,
    AgentsParameters,
    GraphIsomorphismAgentParameters,
    SoloAgentParameters,
    PpoParameters,
    SpgParameters,
    SpgVariant,  # TODO Ideally combine this with SpgParameters
    ScenarioType,
    TrainerType,
    run_experiment,
)
from pvg.utils.output import DummyTqdm


def test_run_experiment():
    """Test running experiments with very basic parameters."""

    # Very agent parameters for each scenario
    agents_params_dict = {
        ScenarioType.GRAPH_ISOMORPHISM: AgentsParameters(
            [
                (
                    "prover",
                    GraphIsomorphismAgentParameters(
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
                ),
                (
                    "verifier",
                    GraphIsomorphismAgentParameters(
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
                ),
            ]
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
        TrainerType.SPG: SpgParameters(
            variant=SpgVariant.LOLA,
            names=tuple(
                list(agents_params_dict.values())[0].keys()
            ),  # Assuming only one scenario
            stackelberg_sequence=(("verifier",), ("prover",)),
            num_iterations=8,
            num_epochs=4,
            minibatch_size=64,
        ),
    }

    for scenario_type, agents_param in agents_params_dict.items():
        for trainer_type, trainer_param in trainer_params.items():
            # Construct the parameters
            params = Parameters.from_dict(
                {
                    "scenario": scenario_type,
                    "trainer": trainer_type,
                    "dataset": "test",
                    "agents": agents_param,
                    str(trainer_type): trainer_param,
                }
            )

            # Run the experiment in test mode
            run_experiment(params, tqdm_func=DummyTqdm, test_run=True)
