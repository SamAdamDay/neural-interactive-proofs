from sklearn.model_selection import ParameterGrid

from pvg import (
    Parameters,
    AgentsParameters,
    GraphIsomorphismAgentParameters,
    ImageClassificationAgentParameters,
    SoloAgentParameters,
    CommonPpoParameters,
    SpgParameters,
    SpgVariant,  # TODO Ideally combine this with SpgParameters
    ScenarioType,
    TrainerType,
    run_experiment,
    prepare_experiment,
)
from pvg.utils.output import DummyTqdm


def test_prepare_run_experiment():
    """Test preparing and running experiments with very basic parameters.

    Tests all combinations of:
    - Scenario type
    - Trainer type
    - Whether the agents are random
    - Whether to pretrain the agents
    """

    # Very basic agent parameters for each scenario
    agents_params_dict = {
        ScenarioType.GRAPH_ISOMORPHISM: AgentsParameters(
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
            ),
        ),
        ScenarioType.IMAGE_CLASSIFICATION: AgentsParameters(
            prover=ImageClassificationAgentParameters(
                num_convs_per_group=1,
                d_latent_pixel_selector=1,
                num_latent_pixel_selector_layers=1,
                d_decider=1,
                num_decider_layers=1,
                d_value=1,
                num_value_layers=1,
            ),
            verifier=ImageClassificationAgentParameters(
                num_convs_per_group=1,
                d_latent_pixel_selector=1,
                num_latent_pixel_selector_layers=1,
                d_decider=1,
                num_decider_layers=1,
                d_value=1,
                num_value_layers=1,
            ),
        ),
    }

    # Very basic parameters for each trainer
    trainer_params = {
        # TrainerType.SOLO_AGENT: SoloAgentParameters(
        #     num_epochs=1,
        #     batch_size=1,
        # ),
        TrainerType.VANILLA_PPO: CommonPpoParameters(
            num_iterations=8,
            num_epochs=4,
            minibatch_size=64,
        ),
        # TrainerType.SPG: SpgParameters(
        #     variant=SpgVariant.PSOS,
        # ),
    }

    basic_params = {
        "scenario": list(agents_params_dict.items()),
        "trainer": list(trainer_params.items()),
        "is_random": [True, False],
        "pretrain_agents": [True, False],
    }

    for param_spec in ParameterGrid(basic_params):
        scenario_type, agents_param = param_spec["scenario"]
        trainer_type, trainer_param = param_spec["trainer"]
        is_random = param_spec["is_random"]
        pretrain_agents = param_spec["pretrain_agents"]

        if trainer_type == TrainerType.SOLO_AGENT and (pretrain_agents or is_random):
            continue

        # Construct the parameters
        if is_random:
            agents_param = AgentsParameters(
                prover={"is_random": True},
                verifier=agents_param["verifier"],
            )
        params = Parameters(
            **{
                "scenario": scenario_type,
                "trainer": trainer_type,
                "dataset": "test",
                "agents": agents_param,
                str(trainer_type): trainer_param,
            }
        )

        # Prepare the experiment
        prepare_experiment(params=params, test_run=True)

        # Run the experiment in test mode
        run_experiment(params, tqdm_func=DummyTqdm, test_run=True)


test_prepare_run_experiment()
