from itertools import product
import re
import dataclasses

import pytest

import torch

from sklearn.model_selection import ParameterGrid

from pvg import (
    Parameters,
    ScenarioType,
    TrainerType,
    AgentsParameters,
    GraphIsomorphismAgentParameters,
    ImageClassificationAgentParameters,
    RlTrainerParameters,
    ExperimentSettings,
)
from pvg.trainers import build_trainer
from pvg.trainers.rl_trainer_base import ReinforcementLearningTrainer
from pvg.trainers.vanilla_ppo import VanillaPpoTrainer
from pvg.scenario_instance import build_scenario_instance


def _optimizer_has_parameter(
    optimizer: torch.optim.Optimizer, parameter: torch.nn.Parameter, return_group=False
) -> bool | tuple[bool, dict | None]:
    """Check if the optimizer has a parameter in any of its groups.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer to check.
    parameter : torch.nn.Parameter
        The parameter to check for.
    return_group : bool, default=False
        Whether to return the group that the parameter is in.

    Returns
    -------
    has_parameter : bool
        Whether the optimizer has the parameter.
    group : dict, optional
        The group that the parameter is in. Only returned if `return_group` is True.
    """
    for param_group in optimizer.param_groups:
        for optimizer_param in param_group["params"]:
            if parameter is optimizer_param:
                if return_group:
                    return True, param_group
                return True
    if return_group:
        return False, None
    return False


def test_gi_ppo_train_optimizer_groups():
    """Test that the graph isomorphism PPO optimizer groups are correct."""

    basic_agent_params = GraphIsomorphismAgentParameters.construct_test_params()

    # Define the the different parameter options to test
    params_list = [
        Parameters(
            ScenarioType.GRAPH_ISOMORPHISM,
            TrainerType.VANILLA_PPO,
            "test",
            rl=RlTrainerParameters(lr=3.0, body_lr_factor=None),
            agents=AgentsParameters(
                prover=dataclasses.replace(
                    basic_agent_params,
                    body_lr_factor={"actor": 1.0, "critic": 1.0},
                    gnn_lr_factor={"actor": 1.0, "critic": 1.0},
                ),
                verifier=dataclasses.replace(
                    basic_agent_params,
                    body_lr_factor={"actor": 1.0, "critic": 1.0},
                    gnn_lr_factor={"actor": 1.0, "critic": 1.0},
                ),
            ),
            functionalize_modules=False,
        ),
        Parameters(
            ScenarioType.GRAPH_ISOMORPHISM,
            TrainerType.VANILLA_PPO,
            "test",
            rl=RlTrainerParameters(lr=3.0),
            agents=AgentsParameters(
                prover=dataclasses.replace(
                    basic_agent_params,
                    body_lr_factor={"actor": 0.1, "critic": 0.1},
                    gnn_lr_factor={"actor": 1.0, "critic": 1.0},
                ),
                verifier=dataclasses.replace(
                    basic_agent_params,
                    body_lr_factor={"actor": 1.0, "critic": 1.0},
                    gnn_lr_factor={"actor": 1.0, "critic": 1.0},
                ),
            ),
            functionalize_modules=False,
        ),
        Parameters(
            ScenarioType.GRAPH_ISOMORPHISM,
            TrainerType.VANILLA_PPO,
            "test",
            rl=RlTrainerParameters(lr=3.0),
            agents=AgentsParameters(
                prover=dataclasses.replace(
                    basic_agent_params,
                    body_lr_factor={"actor": 1.0, "critic": 1.0},
                    gnn_lr_factor={"actor": 1.0, "critic": 1.0},
                ),
                verifier=dataclasses.replace(
                    basic_agent_params,
                    body_lr_factor={"actor": 0.1, "critic": 0.1},
                    gnn_lr_factor={"actor": 1.0, "critic": 1.0},
                ),
            ),
            functionalize_modules=False,
        ),
        Parameters(
            ScenarioType.GRAPH_ISOMORPHISM,
            TrainerType.VANILLA_PPO,
            "test",
            rl=RlTrainerParameters(lr=3.0),
            agents=AgentsParameters(
                prover=dataclasses.replace(
                    basic_agent_params,
                    body_lr_factor={"actor": 1.0, "critic": 1.0},
                    gnn_lr_factor={"actor": 0.1, "critic": 0.1},
                ),
                verifier=dataclasses.replace(
                    basic_agent_params,
                    body_lr_factor={"actor": 1.0, "critic": 1.0},
                    gnn_lr_factor={"actor": 1.0, "critic": 1.0},
                ),
            ),
            functionalize_modules=False,
        ),
        Parameters(
            ScenarioType.GRAPH_ISOMORPHISM,
            TrainerType.VANILLA_PPO,
            "test",
            rl=RlTrainerParameters(lr=3.0),
            agents=AgentsParameters(
                prover=dataclasses.replace(
                    basic_agent_params,
                    body_lr_factor={"actor": 0.1, "critic": 0.1},
                    gnn_lr_factor={"actor": 0.1, "critic": 0.1},
                ),
                verifier=dataclasses.replace(
                    basic_agent_params,
                    body_lr_factor={"actor": 1.0, "critic": 1.0},
                    gnn_lr_factor={"actor": 1.0, "critic": 1.0},
                ),
            ),
            functionalize_modules=False,
        ),
        Parameters(
            ScenarioType.GRAPH_ISOMORPHISM,
            TrainerType.VANILLA_PPO,
            "test",
            rl=RlTrainerParameters(lr=3.0),
            agents=AgentsParameters(
                prover=dataclasses.replace(
                    basic_agent_params,
                    body_lr_factor={"actor": 1.0, "critic": 1.0},
                    gnn_lr_factor={"actor": 1.0, "critic": 1.0},
                ),
                verifier=dataclasses.replace(
                    basic_agent_params,
                    body_lr_factor={"actor": 1.0, "critic": 1.0},
                    gnn_lr_factor={"actor": 0.1, "critic": 0.1},
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
                    "actor_network.module.0.module.0.prover.gnn", param_name
                ) or param_name.startswith("critic_network.module.0.prover.gnn"):
                    return "prover_gnn"
                if re.match(
                    "actor_network.module.0.module.0.verifier.gnn",
                    param_name,
                ) or param_name.startswith("critic_network.module.0.verifier.gnn"):
                    return "verifier_gnn"
                if re.match(
                    "actor_network.module.0.module.0.prover", param_name
                ) or param_name.startswith("critic_network.module.0.prover"):
                    return "prover_body"
                if re.match(
                    "actor_network.module.0.module.0.verifier",
                    param_name,
                ) or param_name.startswith("critic_network.module.0.verifier"):
                    return "verifier_body"
                return "rest"

        # Run through all the loss module parameters and make sure they are in the
        # optimizer with the correct learning rate
        for param_name, param in loss_module.named_parameters():
            # Make sure the optimizer has the parameter, and get its group
            has_parameter, param_group = _optimizer_has_parameter(
                optimizer, param, return_group=True
            )
            assert has_parameter, f"Optimizer does not have parameter {param_name}"

            # Check that the learning rate is correct
            optimizer_lr = param_group["lr"]
            network_part = get_network_part(param_name)
            assert optimizer_lr == pytest.approx(expected_lrs[i][network_part]), (
                f"Parameter {param_name} has learning rate {optimizer_lr} "
                f"instead of {expected_lrs[i][network_part]}. Matched network part: "
                f"{network_part}"
            )


def test_loss_parameters_in_optimizer():
    """Make sure that all the loss parameters are in the optimizer.

    This test does less than `test_gi_ppo_train_optimizer_groups`, which also makes sure
    that learning rates are correct, but it is applied to more param combinations.
    """

    # Construct basic agent parameters for each scenario
    basic_agent_params = {}
    basic_agent_params[
        ScenarioType.GRAPH_ISOMORPHISM
    ] = GraphIsomorphismAgentParameters.construct_test_params()
    basic_agent_params[
        ScenarioType.IMAGE_CLASSIFICATION
    ] = ImageClassificationAgentParameters.construct_test_params()

    param_specs = [
        {
            "scenario": [
                ScenarioType.GRAPH_ISOMORPHISM,
                ScenarioType.IMAGE_CLASSIFICATION,
            ],
            "trainer": [TrainerType.VANILLA_PPO, TrainerType.REINFORCE],
            "use_shared_body": [True, False],
            "functionalize_modules": [True, False],
        }
    ]

    # Create the common experiment settings
    settings = ExperimentSettings(device="cpu", test_run=True)

    for param_spec in ParameterGrid(param_specs):
        # Construct the parameters
        params = Parameters(
            scenario=param_spec["scenario"],
            trainer=param_spec["trainer"],
            dataset="test",
            agents=AgentsParameters(
                prover=basic_agent_params[param_spec["scenario"]],
                verifier=basic_agent_params[param_spec["scenario"]],
            ),
            pretrain_agents=False,
            rl=RlTrainerParameters(use_shared_body=param_spec["use_shared_body"]),
            functionalize_modules=param_spec["functionalize_modules"],
            d_representation=1,
        )

        # Create the trainer and get the loss module and optimizer
        scenario_instance = build_scenario_instance(params, settings)
        trainer: ReinforcementLearningTrainer = build_trainer(
            params, scenario_instance, settings
        )
        trainer._train_setup()
        loss_module, _ = trainer._get_loss_module_and_gae()
        optimizer, _ = trainer._get_optimizer_and_param_freezer(loss_module)

        # Run through all the loss module parameters and make sure they are in the
        # optimizer
        for param_name, param in loss_module.named_parameters():
            assert (
                _optimizer_has_parameter(optimizer, param),
                f"Optimizer does not have parameter {param_name}",
            )
