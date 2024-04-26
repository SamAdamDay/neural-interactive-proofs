from itertools import product
import re

import pytest

from pvg import (
    Parameters,
    ScenarioType,
    TrainerType,
    AgentsParameters,
    GraphIsomorphismAgentParameters,
    RlTrainerParameters,
    ExperimentSettings,
)
from pvg.trainers.vanilla_ppo import VanillaPpoTrainer
from pvg.scenario_instance import build_scenario_instance


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
            TrainerType.VANILLA_PPO,
            "test",
            rl=RlTrainerParameters(lr=3.0, body_lr_factor=None),
            agents=AgentsParameters(
                prover=GraphIsomorphismAgentParameters(
                    body_lr_factor={"actor": 1.0, "critic": 1.0},
                    gnn_lr_factor={"actor": 1.0, "critic": 1.0},
                    **basic_agent_params,
                ),
                verifier=GraphIsomorphismAgentParameters(
                    body_lr_factor={"actor": 1.0, "critic": 1.0},
                    gnn_lr_factor={"actor": 1.0, "critic": 1.0},
                    **basic_agent_params,
                ),
            ),
            functionalize_modules=False,
        ),
        Parameters(
            ScenarioType.GRAPH_ISOMORPHISM,
            TrainerType.VANILLA_PPO,
            "test",
            rl=RlTrainerParameters(lr=3.0, body_lr_factor=None),
            agents=AgentsParameters(
                prover=GraphIsomorphismAgentParameters(
                    body_lr_factor={"actor": 0.1, "critic": 0.1},
                    gnn_lr_factor={"actor": 1.0, "critic": 1.0},
                    **basic_agent_params,
                ),
                verifier=GraphIsomorphismAgentParameters(
                    body_lr_factor={"actor": 1.0, "critic": 1.0},
                    gnn_lr_factor={"actor": 1.0, "critic": 1.0},
                    **basic_agent_params,
                ),
            ),
            functionalize_modules=False,
        ),
        Parameters(
            ScenarioType.GRAPH_ISOMORPHISM,
            TrainerType.VANILLA_PPO,
            "test",
            rl=RlTrainerParameters(lr=3.0, body_lr_factor=None),
            agents=AgentsParameters(
                prover=GraphIsomorphismAgentParameters(
                    body_lr_factor={"actor": 1.0, "critic": 1.0},
                    gnn_lr_factor={"actor": 1.0, "critic": 1.0},
                    **basic_agent_params,
                ),
                verifier=GraphIsomorphismAgentParameters(
                    body_lr_factor={"actor": 0.1, "critic": 0.1},
                    gnn_lr_factor={"actor": 1.0, "critic": 1.0},
                    **basic_agent_params,
                ),
            ),
            functionalize_modules=False,
        ),
        Parameters(
            ScenarioType.GRAPH_ISOMORPHISM,
            TrainerType.VANILLA_PPO,
            "test",
            rl=RlTrainerParameters(
                lr=3.0, body_lr_factor={"actor": 0.01, "critic": 0.01}
            ),
            agents=AgentsParameters(
                prover=GraphIsomorphismAgentParameters(
                    body_lr_factor={"actor": 0.1, "critic": 0.1},
                    gnn_lr_factor={"actor": 1.0, "critic": 1.0},
                    **basic_agent_params,
                ),
                verifier=GraphIsomorphismAgentParameters(
                    body_lr_factor={"actor": 1.0, "critic": 1.0},
                    gnn_lr_factor={"actor": 1.0, "critic": 1.0},
                    **basic_agent_params,
                ),
            ),
            functionalize_modules=False,
        ),
        Parameters(
            ScenarioType.GRAPH_ISOMORPHISM,
            TrainerType.VANILLA_PPO,
            "test",
            rl=RlTrainerParameters(lr=3.0, body_lr_factor=None),
            agents=AgentsParameters(
                prover=GraphIsomorphismAgentParameters(
                    body_lr_factor={"actor": 1.0, "critic": 1.0},
                    gnn_lr_factor={"actor": 0.1, "critic": 0.1},
                    **basic_agent_params,
                ),
                verifier=GraphIsomorphismAgentParameters(
                    body_lr_factor={"actor": 1.0, "critic": 1.0},
                    gnn_lr_factor={"actor": 1.0, "critic": 1.0},
                    **basic_agent_params,
                ),
            ),
            functionalize_modules=False,
        ),
        Parameters(
            ScenarioType.GRAPH_ISOMORPHISM,
            TrainerType.VANILLA_PPO,
            "test",
            rl=RlTrainerParameters(lr=3.0, body_lr_factor=None),
            agents=AgentsParameters(
                prover=GraphIsomorphismAgentParameters(
                    body_lr_factor={"actor": 0.1, "critic": 0.1},
                    gnn_lr_factor={"actor": 0.1, "critic": 0.1},
                    **basic_agent_params,
                ),
                verifier=GraphIsomorphismAgentParameters(
                    body_lr_factor={"actor": 1.0, "critic": 1.0},
                    gnn_lr_factor={"actor": 1.0, "critic": 1.0},
                    **basic_agent_params,
                ),
            ),
            functionalize_modules=False,
        ),
        Parameters(
            ScenarioType.GRAPH_ISOMORPHISM,
            TrainerType.VANILLA_PPO,
            "test",
            rl=RlTrainerParameters(
                lr=3.0, body_lr_factor={"actor": 0.1, "critic": 0.1}
            ),
            agents=AgentsParameters(
                prover=GraphIsomorphismAgentParameters(
                    body_lr_factor={"actor": 1.0, "critic": 1.0},
                    gnn_lr_factor={"actor": 0.1, "critic": 0.1},
                    **basic_agent_params,
                ),
                verifier=GraphIsomorphismAgentParameters(
                    body_lr_factor={"actor": 1.0, "critic": 1.0},
                    gnn_lr_factor={"actor": 1.0, "critic": 1.0},
                    **basic_agent_params,
                ),
            ),
            functionalize_modules=False,
        ),
        Parameters(
            ScenarioType.GRAPH_ISOMORPHISM,
            TrainerType.VANILLA_PPO,
            "test",
            rl=RlTrainerParameters(lr=3.0, body_lr_factor=None),
            agents=AgentsParameters(
                prover=GraphIsomorphismAgentParameters(
                    body_lr_factor={"actor": 1.0, "critic": 1.0},
                    gnn_lr_factor={"actor": 1.0, "critic": 1.0},
                    **basic_agent_params,
                ),
                verifier=GraphIsomorphismAgentParameters(
                    body_lr_factor={"actor": 1.0, "critic": 1.0},
                    gnn_lr_factor={"actor": 0.1, "critic": 0.1},
                    **basic_agent_params,
                ),
            ),
            functionalize_modules=False,
        ),
    ]

    # Define the expected learning rates for the prover body, the verifier body, and the
    # rest of the parameters
    expected_lrs = [
        dict(
            prover_gnn=3.0,
            prover_body=3.0,
            verifier_gnn=3.0,
            verifier_body=3.0,
            rest=3.0,
        ),
        dict(
            prover_gnn=0.3,
            prover_body=0.3,
            verifier_gnn=3.0,
            verifier_body=3.0,
            rest=3.0,
        ),
        dict(
            prover_gnn=3.0,
            prover_body=3.0,
            verifier_gnn=0.3,
            verifier_body=0.3,
            rest=3.0,
        ),
        dict(
            prover_gnn=0.03,
            prover_body=0.03,
            verifier_gnn=0.03,
            verifier_body=0.03,
            rest=3.0,
        ),
        dict(
            prover_gnn=0.3,
            prover_body=3.0,
            verifier_gnn=3.0,
            verifier_body=3.0,
            rest=3.0,
        ),
        dict(
            prover_gnn=0.03,
            prover_body=0.3,
            verifier_gnn=3.0,
            verifier_body=3.0,
            rest=3.0,
        ),
        dict(
            prover_gnn=0.03,
            prover_body=0.3,
            verifier_gnn=0.3,
            verifier_body=0.3,
            rest=3.0,
        ),
        dict(
            prover_gnn=3.0,
            prover_body=3.0,
            verifier_gnn=0.3,
            verifier_body=3.0,
            rest=3.0,
        ),
    ]

    for use_shared_body, (i, params) in product([True, False], enumerate(params_list)):
        # Create the experiment settings and scenario instance to pass to the trainer
        params.rl.use_shared_body = use_shared_body
        settings = ExperimentSettings(device="cpu", test_run=True)
        scenario_instance = build_scenario_instance(params, settings)

        # Create the trainer and get the loss module and optimizer
        trainer = VanillaPpoTrainer(params, scenario_instance, settings)
        trainer._train_setup()
        loss_module, _ = trainer._get_loss_module_and_gae()
        optimizer, _ = trainer._get_optimizer_and_param_freezer(loss_module)

        def get_network_part(param_name: str) -> str:
            """Determine which part of the network the parameter is in."""
            if use_shared_body:
                if param_name.startswith("actor_network.module.0.prover.gnn"):
                    return "prover_gnn"
                if param_name.startswith("actor_network.module.0.verifier.gnn"):
                    return "verifier_gnn"
                if param_name.startswith("actor_network.module.0.prover"):
                    return "prover_body"
                if param_name.startswith("actor_network.module.0.verifier"):
                    return "verifier_body"
                return "rest"
            else:
                if re.match(
                    "actor_network.module.0.module.[0-9]+.prover.gnn", param_name
                ) or param_name.startswith("critic_network.module.0.prover.gnn"):
                    return "prover_gnn"
                if re.match(
                    "actor_network.module.0.module.[0-9]+.verifier.gnn",
                    param_name,
                ) or param_name.startswith("critic_network.module.0.verifier.gnn"):
                    return "verifier_gnn"
                if re.match(
                    "actor_network.module.0.module.[0-9]+.prover", param_name
                ) or param_name.startswith("critic_network.module.0.prover"):
                    return "prover_body"
                if re.match(
                    "actor_network.module.0.module.[0-9]+.verifier",
                    param_name,
                ) or param_name.startswith("critic_network.module.0.verifier"):
                    return "verifier_body"
                return "rest"

        # Run through all the loss module parameters and make sure they are in the
        # optimizer with the correct learning rate
        for param_name, param in loss_module.named_parameters():
            # Look for the parameter in the optimizer
            optimizer_has_param = False
            for param_group in optimizer.param_groups:
                for optimizer_param in param_group["params"]:
                    if param is optimizer_param:
                        optimizer_lr = param_group["lr"]
                        optimizer_has_param = True
                        break

            # Make sure the optimizer has the parameter
            assert (
                optimizer_has_param
            ), f"Optimizer does not have parameter {param_name}"

            # Check that the learning rate is correct
            network_part = get_network_part(param_name)
            assert optimizer_lr == pytest.approx(expected_lrs[i][network_part]), (
                f"Parameter {param_name} has learning rate {optimizer_lr} "
                f"instead of {expected_lrs[i][network_part]}. Matched network part: "
                f"{network_part}"
            )
