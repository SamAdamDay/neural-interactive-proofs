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
    LrFactors,
)
from pvg.trainers import build_trainer
from pvg.trainers.rl_tensordict_base import ReinforcementLearningTrainer
from pvg.trainers.vanilla_ppo import VanillaPpoTrainer
from pvg.factory import build_scenario_instance


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


# A list of LR specifications and corresponding expected computed LRs
specs_and_expected_lrs = [
    dict(
        spec=dict(),
        expected_lrs=dict(
            prover_gnn=3.0,
            prover_body=3.0,
            verifier_gnn=3.0,
            verifier_body=3.0,
            rest=3.0,
        ),
    ),
    dict(
        spec=dict(prover_body_lr_factor=0.1),
        expected_lrs=dict(
            prover_gnn=0.3,
            prover_body=0.3,
            verifier_gnn=3.0,
            verifier_body=3.0,
            rest=3.0,
        ),
    ),
    dict(
        spec=dict(verifier_body_lr_factor=0.1),
        expected_lrs=dict(
            prover_gnn=3.0,
            prover_body=3.0,
            verifier_gnn=0.3,
            verifier_body=0.3,
            rest=3.0,
        ),
    ),
    dict(
        spec=dict(prover_gnn_lr_factor=0.1),
        expected_lrs=dict(
            prover_gnn=0.3,
            prover_body=3.0,
            verifier_gnn=3.0,
            verifier_body=3.0,
            rest=3.0,
        ),
    ),
    dict(
        spec=dict(prover_body_lr_factor=0.1, prover_gnn_lr_factor=0.1),
        expected_lrs=dict(
            prover_gnn=0.03,
            prover_body=0.3,
            verifier_gnn=3.0,
            verifier_body=3.0,
            rest=3.0,
        ),
    ),
    dict(
        spec=dict(verifier_gnn_lr_factor=0.1),
        expected_lrs=dict(
            prover_gnn=3.0,
            prover_body=3.0,
            verifier_gnn=0.3,
            verifier_body=3.0,
            rest=3.0,
        ),
    ),
    dict(
        spec=dict(use_shared_body=False),
        expected_lrs=dict(
            prover_actor_gnn=3.0,
            prover_critic_gnn=3.0,
            prover_actor_body=3.0,
            prover_critic_body=3.0,
            verifier_actor_gnn=3.0,
            verifier_critic_gnn=3.0,
            verifier_actor_body=3.0,
            verifier_critic_body=3.0,
            rest=3.0,
        ),
    ),
    dict(
        spec=dict(use_shared_body=False, prover_actor_body_lr_factor=0.1),
        expected_lrs=dict(
            prover_actor_gnn=0.3,
            prover_critic_gnn=3.0,
            prover_actor_body=0.3,
            prover_critic_body=3.0,
            verifier_actor_gnn=3.0,
            verifier_critic_gnn=3.0,
            verifier_actor_body=3.0,
            verifier_critic_body=3.0,
            rest=3.0,
        ),
    ),
    dict(
        spec=dict(use_shared_body=False, prover_critic_body_lr_factor=0.1),
        expected_lrs=dict(
            prover_actor_gnn=3.0,
            prover_critic_gnn=0.3,
            prover_actor_body=3.0,
            prover_critic_body=0.3,
            verifier_actor_gnn=3.0,
            verifier_critic_gnn=3.0,
            verifier_actor_body=3.0,
            verifier_critic_body=3.0,
            rest=3.0,
        ),
    ),
    dict(
        spec=dict(
            use_shared_body=False,
            prover_actor_body_lr_factor=0.1,
            prover_actor_gnn_lr_factor=0.1,
            prover_critic_body_lr_factor=10.0,
            prover_critic_gnn_lr_factor=10.0,
        ),
        expected_lrs=dict(
            prover_actor_gnn=0.03,
            prover_critic_gnn=300.0,
            prover_actor_body=0.3,
            prover_critic_body=30.0,
            verifier_actor_gnn=3.0,
            verifier_critic_gnn=3.0,
            verifier_actor_body=3.0,
            verifier_critic_body=3.0,
            rest=3.0,
        ),
    ),
    dict(
        spec=dict(
            use_shared_body=False,
            prover_actor_body_lr_factor=0.1,
            prover_actor_gnn_lr_factor=0.1,
            prover_critic_body_lr_factor=10.0,
            prover_critic_gnn_lr_factor=10.0,
            verifier_actor_body_lr_factor=0.01,
            verifier_actor_gnn_lr_factor=0.01,
            verifier_critic_body_lr_factor=100.0,
            verifier_critic_gnn_lr_factor=100.0,
        ),
        expected_lrs=dict(
            prover_actor_gnn=0.03,
            prover_critic_gnn=300.0,
            prover_actor_body=0.3,
            prover_critic_body=30.0,
            verifier_actor_gnn=0.0003,
            verifier_critic_gnn=30000.0,
            verifier_actor_body=0.03,
            verifier_critic_body=300.0,
            rest=3.0,
        ),
    ),
]


@pytest.mark.parametrize(
    ("lr_spec", "expected_lrs"),
    [(item["spec"], item["expected_lrs"]) for item in specs_and_expected_lrs],
)
def test_gi_ppo_train_optimizer_groups(lr_spec: dict, expected_lrs: dict):
    """Test that the graph isomorphism PPO optimizer groups are correct."""

    # Get the specification, using default values if not provided
    lr: float = lr_spec.get("lr", 3.0)
    use_shared_body: bool = lr_spec.get("use_shared_body", True)
    prover_body_lr_factor: float = lr_spec.get("prover_body_lr_factor", 1.0)
    prover_gnn_lr_factor: float = lr_spec.get("prover_gnn_lr_factor", 1.0)
    verifier_body_lr_factor: float = lr_spec.get("verifier_body_lr_factor", 1.0)
    verifier_gnn_lr_factor: float = lr_spec.get("verifier_gnn_lr_factor", 1.0)
    prover_actor_body_lr_factor: float = lr_spec.get("prover_actor_body_lr_factor", 1.0)
    prover_critic_body_lr_factor: float = lr_spec.get(
        "prover_critic_body_lr_factor", 1.0
    )
    prover_actor_gnn_lr_factor: float = lr_spec.get("prover_actor_gnn_lr_factor", 1.0)
    prover_critic_gnn_lr_factor: float = lr_spec.get("prover_critic_gnn_lr_factor", 1.0)
    verifier_actor_body_lr_factor: float = lr_spec.get(
        "verifier_actor_body_lr_factor", 1.0
    )
    verifier_critic_body_lr_factor: float = lr_spec.get(
        "verifier_critic_body_lr_factor", 1.0
    )
    verifier_actor_gnn_lr_factor: float = lr_spec.get(
        "verifier_actor_gnn_lr_factor", 1.0
    )
    verifier_critic_gnn_lr_factor: float = lr_spec.get(
        "verifier_critic_gnn_lr_factor", 1.0
    )

    # If we're using the shared body, we copy the body learning rates to the GNN
    # learning rates to the actor and critic
    if use_shared_body:
        prover_actor_body_lr_factor = prover_body_lr_factor
        prover_critic_body_lr_factor = prover_body_lr_factor
        prover_actor_gnn_lr_factor = prover_gnn_lr_factor
        prover_critic_gnn_lr_factor = prover_gnn_lr_factor
        verifier_actor_body_lr_factor = verifier_body_lr_factor
        verifier_critic_body_lr_factor = verifier_body_lr_factor
        verifier_actor_gnn_lr_factor = verifier_gnn_lr_factor
        verifier_critic_gnn_lr_factor = verifier_gnn_lr_factor

    # Construct the parameters
    basic_agent_params = GraphIsomorphismAgentParameters.construct_test_params()
    params = Parameters(
        ScenarioType.GRAPH_ISOMORPHISM,
        TrainerType.VANILLA_PPO,
        "test",
        rl=RlTrainerParameters(lr=lr, use_shared_body=use_shared_body),
        agents=AgentsParameters(
            prover=dataclasses.replace(
                basic_agent_params,
                body_lr_factor=LrFactors(
                    actor=prover_actor_body_lr_factor,
                    critic=prover_critic_body_lr_factor,
                ),
                gnn_lr_factor=LrFactors(
                    actor=prover_actor_gnn_lr_factor,
                    critic=prover_critic_gnn_lr_factor,
                ),
            ),
            verifier=dataclasses.replace(
                basic_agent_params,
                body_lr_factor=LrFactors(
                    actor=verifier_actor_body_lr_factor,
                    critic=verifier_critic_body_lr_factor,
                ),
                gnn_lr_factor=LrFactors(
                    actor=verifier_actor_gnn_lr_factor,
                    critic=verifier_critic_gnn_lr_factor,
                ),
            ),
        ),
        functionalize_modules=False,
    )

    # Create the experiment settings and scenario instance to pass to the trainer
    settings = ExperimentSettings(device="cpu", test_run=True, ignore_cache=True)
    scenario_instance = build_scenario_instance(params, settings)

    # Create the trainer and get the loss module and optimizer
    trainer = VanillaPpoTrainer(params, scenario_instance, settings)
    trainer._build_operators()
    loss_module, _ = trainer._get_loss_module_and_gae()
    optimizer, _ = trainer._get_optimizer_and_param_freezer(loss_module)

    def get_network_part(param_name: str) -> str:
        """Determine which part of the network the parameter is in."""
        if params.rl.use_shared_body:
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
            if param_name.startswith("actor_network.module.0.module.0.prover.gnn"):
                return "prover_actor_gnn"
            if param_name.startswith("critic_network.module.0.prover.gnn"):
                return "prover_critic_gnn"
            if param_name.startswith("actor_network.module.0.module.0.verifier.gnn"):
                return "verifier_actor_gnn"
            if param_name.startswith("critic_network.module.0.verifier.gnn"):
                return "verifier_critic_gnn"
            if param_name.startswith("actor_network.module.0.module.0.prover"):
                return "prover_actor_body"
            if param_name.startswith("critic_network.module.0.prover"):
                return "prover_critic_body"
            if param_name.startswith("actor_network.module.0.module.0.verifier"):
                return "verifier_actor_body"
            if param_name.startswith("critic_network.module.0.verifier"):
                return "verifier_critic_body"
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
        assert optimizer_lr == pytest.approx(expected_lrs[network_part]), (
            f"Parameter {param_name} has learning rate {optimizer_lr} "
            f"instead of {expected_lrs[network_part]}. Matched network part: "
            f"{network_part}"
        )


@pytest.mark.parametrize(
    "param_spec",
    ParameterGrid(
        {
            "scenario": [
                ScenarioType.GRAPH_ISOMORPHISM,
                ScenarioType.IMAGE_CLASSIFICATION,
            ],
            "trainer": [TrainerType.VANILLA_PPO, TrainerType.REINFORCE],
            "use_shared_body": [True, False],
            "functionalize_modules": [True, False],
        }
    ),
)
def test_loss_parameters_in_optimizer(param_spec):
    """Make sure that all the loss parameters are in the optimizer.

    This test does less than `test_gi_ppo_train_optimizer_groups`, which also makes sure
    that learning rates are correct, but it is applied to more param combinations.
    """

    # Construct basic agent parameters for each scenario
    basic_agent_params = {}
    basic_agent_params[ScenarioType.GRAPH_ISOMORPHISM] = (
        GraphIsomorphismAgentParameters.construct_test_params()
    )
    basic_agent_params[ScenarioType.IMAGE_CLASSIFICATION] = (
        ImageClassificationAgentParameters.construct_test_params()
    )

    # Create the common experiment settings
    settings = ExperimentSettings(device="cpu", test_run=True)

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
    trainer._build_operators()
    loss_module, _ = trainer._get_loss_module_and_gae()
    optimizer, _ = trainer._get_optimizer_and_param_freezer(loss_module)

    # Run through all the loss module parameters and make sure they are in the
    # optimizer
    for param_name, param in loss_module.named_parameters():
        test = _optimizer_has_parameter(optimizer, param)
        assert test, f"Optimizer does not have parameter {param_name}"
