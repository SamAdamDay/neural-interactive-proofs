import pytest

from pvg import (
    Parameters,
    ScenarioType,
    TrainerType,
    AgentsParameters,
    GraphIsomorphismAgentParameters,
    PpoParameters,
    ExperimentSettings,
)
from pvg.graph_isomorphism import GraphIsomorphismScenarioInstance
from pvg.trainers.ppo import PpoTrainer


def test_gi_ppo_train_optimizer_groups():
    """Test that the graph isomorphism PPO optimizer groups are correct."""

    # Parameters for the agents which make them very simple
    basic_agent_params = dict(
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
    )

    # Define the the different parameter options to test
    params_list = [
        Parameters(
            ScenarioType.GRAPH_ISOMORPHISM,
            TrainerType.PPO,
            "test",
            ppo=PpoParameters(lr=3.0),
            agents=AgentsParameters(
                [
                    (
                        "prover",
                        GraphIsomorphismAgentParameters(
                            body_lr_factor=1.0, **basic_agent_params
                        ),
                    ),
                    (
                        "verifier",
                        GraphIsomorphismAgentParameters(
                            body_lr_factor=1.0, **basic_agent_params
                        ),
                    ),
                ]
            ),
        ),
        Parameters(
            ScenarioType.GRAPH_ISOMORPHISM,
            TrainerType.PPO,
            "test",
            ppo=PpoParameters(lr=3.0),
            agents=AgentsParameters(
                [
                    (
                        "prover",
                        GraphIsomorphismAgentParameters(
                            body_lr_factor=0.1, **basic_agent_params
                        ),
                    ),
                    (
                        "verifier",
                        GraphIsomorphismAgentParameters(
                            body_lr_factor=1.0, **basic_agent_params
                        ),
                    ),
                ]
            ),
        ),
        Parameters(
            ScenarioType.GRAPH_ISOMORPHISM,
            TrainerType.PPO,
            "test",
            ppo=PpoParameters(lr=3.0),
            agents=AgentsParameters(
                [
                    (
                        "prover",
                        GraphIsomorphismAgentParameters(
                            body_lr_factor=1.0, **basic_agent_params
                        ),
                    ),
                    (
                        "verifier",
                        GraphIsomorphismAgentParameters(
                            body_lr_factor=0.1, **basic_agent_params
                        ),
                    ),
                ]
            ),
        ),
        Parameters(
            ScenarioType.GRAPH_ISOMORPHISM,
            TrainerType.PPO,
            "test",
            ppo=PpoParameters(lr=3.0, body_lr_factor=0.01),
            agents=AgentsParameters(
                [
                    (
                        "prover",
                        GraphIsomorphismAgentParameters(
                            body_lr_factor=0.1, **basic_agent_params
                        ),
                    ),
                    (
                        "verifier",
                        GraphIsomorphismAgentParameters(
                            body_lr_factor=1.0, **basic_agent_params
                        ),
                    ),
                ]
            ),
        ),
    ]

    # Define the expected learning rates for the prover body, the verifier body, and the
    # rest of the parameters
    expected_lrs = [
        dict(prover=3.0, verifier=3.0, rest=3.0),
        dict(prover=0.3, verifier=3.0, rest=3.0),
        dict(prover=3.0, verifier=0.3, rest=3.0),
        dict(prover=0.03, verifier=0.03, rest=3.0),
    ]

    for i, params in enumerate(params_list):
        # Create the experiment settings and scenario instance to pass to the trainer
        settings = ExperimentSettings(device="cpu", test_run=True)
        scenario_instance = GraphIsomorphismScenarioInstance(params, settings)

        # Create the trainer and get the loss module and optimizer
        trainer = PpoTrainer(params, scenario_instance, settings)
        trainer._train_setup()
        loss_module, _ = trainer._get_loss_module_and_gae()
        optimizer = trainer._get_optimizer(loss_module)

        # Run through all the loss module parameters and make sure they are in the
        # optimizer with the correct learning rate
        for param_name, param in loss_module.named_parameters():
            # Look for the parameter in the optimizer
            optimizer_has_param = False
            for param_group in optimizer.param_groups:
                for optmizer_param in param_group["params"]:
                    if param is optmizer_param:
                        optimizer_lr = param_group["lr"]
                        optimizer_has_param = True
                        break

            # Make sure the optimizer has the parameter
            assert optimizer_has_param

            # Check that the learning rate is correct
            if param_name.startswith("actor_params.module_0_prover"):
                assert optimizer_lr == pytest.approx(expected_lrs[i]["prover"])
            elif param_name.startswith("actor_params.module_0_verifier"):
                assert optimizer_lr == pytest.approx(expected_lrs[i]["verifier"])
            else:
                assert optimizer_lr == pytest.approx(expected_lrs[i]["rest"])
