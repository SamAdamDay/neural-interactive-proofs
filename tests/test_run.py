"""Tests for running experiments with different parameters.

These are basic tests which just run various experiments to make sure there are no
errors. They do not check that the experiments are correct.
"""

import pytest

from sklearn.model_selection import ParameterGrid

from nip import (
    HyperParameters,
    GraphIsomorphismAgentParameters,
    ImageClassificationAgentParameters,
    CodeValidationAgentParameters,
    CommonProtocolParameters,
    SoloAgentParameters,
    RlTrainerParameters,
    CommonPpoParameters,
    SpgParameters,
    run_experiment,
    prepare_experiment,
)
from nip.parameters import AGENT_NAMES
from nip.utils.output import DummyTqdm

# Specification for creating a grid of parameters using ParameterGrid
param_specs = [
    # Test the two tensor scenarios with the vanilla PPO trainer
    {
        "scenario": [
            "graph_isomorphism",
            "image_classification",
        ],
        "message_size": [3],
    },
    # Test the code validation scenario with the expert iteration trainer with various
    # protocols
    {
        "scenario": ["code_validation"],
        "trainer": ["pure_text_ei"],
        "protocol": [
            "nip",
            "debate",
            "adp",
            "merlin_arthur",
            "mnip",
            "solo_verifier",
        ],
    },
    # Test pretraining the agents
    {
        "pretrain_agents": [True],
    },
    # Test random agents
    {
        "is_random": [True],
    },
    # Test the KL penalty loss
    {
        "ppo_loss": ["kl_penalty"],
    },
    # Test the using non-shared bodies
    {
        "scenario": [
            "graph_isomorphism",
            "image_classification",
        ],
        "use_shared_body": [False],
    },
    # Test the SPG trainer with different variants
    {
        "trainer": ["spg"],
        "spg_variant": [
            "spg",
            "pspg",
            "lola",
            "pola",
            "psos",
            "sos",
        ],
    },
    # Test the other trainers
    {
        "trainer": [
            "solo_agent",
            "reinforce",
        ],
    },
    # Test the other protocols
    {
        "protocol": [
            "debate",
            "adp",
            "merlin_arthur",
            "mnip",
        ],
    },
    # Test the zero-knowledge protocols
    {
        "zero_knowledge": [True],
        "protocol": [
            "nip",
            "debate",
            "adp",
            "merlin_arthur",
            "mnip",
        ],
    },
    # Test manual architectures
    {
        "manual_architecture": [
            ("prover",),
            ("verifier",),
            ("prover", "verifier"),
        ],
    },
    # Test the including a linear message space
    {
        "scenario": [
            "graph_isomorphism",
            "image_classification",
        ],
        "include_linear_message": [True],
    },
]


@pytest.mark.parametrize("param_spec", ParameterGrid(param_specs))
def test_prepare_run_experiment(param_spec: dict):
    """Test preparing and running experiments with very basic parameters.

    Tests all combinations of:
    - Scenario type
    - Trainer type
    - Whether the agents are random
    - Whether to pretrain the agents
    """

    # Very basic agent parameters for each scenario
    basic_agent_params = {}
    basic_agent_params["graph_isomorphism"] = (
        GraphIsomorphismAgentParameters.construct_test_params()
    )
    basic_agent_params["image_classification"] = (
        ImageClassificationAgentParameters.construct_test_params()
    )
    basic_agent_params["code_validation"] = (
        CodeValidationAgentParameters.construct_test_params()
    )

    # Very basic parameters for each trainer
    rl_params = RlTrainerParameters(
        num_iterations=2,
        num_epochs=1,
        minibatch_size=1,
        frames_per_batch=8,
    )
    trainer_params = {
        "solo_agent": SoloAgentParameters(
            num_epochs=1,
            batch_size=1,
        ),
        "vanilla_ppo": None,
        "spg": SpgParameters(),
        "reinforce": None,
        "pure_text_ei": None,
    }
    common_ppo_params = CommonPpoParameters()

    # Extract the parameters, using defaults if not specified
    scenario_type = param_spec.get("scenario", "graph_isomorphism")
    trainer_type = param_spec.get("trainer", "vanilla_ppo")
    ppo_loss_type = param_spec.get("ppo_loss", "clip")
    protocol_type = param_spec.get("protocol", "nip")
    zero_knowledge = param_spec.get("zero_knowledge", False)
    is_random = param_spec.get("is_random", False)
    pretrain_agents = param_spec.get("pretrain_agents", False)
    manual_architecture = param_spec.get("manual_architecture", None)
    use_shared_body = param_spec.get("use_shared_body", True)
    include_linear_message = param_spec.get("include_linear_message", False)
    message_size = param_spec.get("message_size", 1)

    # Construct the PPO parameters
    common_ppo_params.loss_type = ppo_loss_type

    # Construct the RL parameters
    rl_params.use_shared_body = use_shared_body

    # Construct the agent parameters
    agents_param = {}
    agent_names = list(AGENT_NAMES[protocol_type])
    if zero_knowledge:
        agent_names.extend(["simulator", "adversarial_verifier"])
    for agent_name in agent_names:
        if is_random and agent_name != "verifier":
            agents_param[agent_name] = {"is_random": True}
        else:
            agents_param[agent_name] = basic_agent_params[scenario_type]
    if manual_architecture is not None:
        for agent_name in manual_architecture:
            agents_param[agent_name].use_manual_architecture = True

    # Construct the trainer parameters
    trainer_param = trainer_params[trainer_type]
    if trainer_type == "spg":
        trainer_param.variant = param_spec["spg_variant"]

    # Construct the parameters
    hyper_params = HyperParameters(
        **{
            "scenario": scenario_type,
            "trainer": trainer_type,
            "dataset": "test",
            "interaction_protocol": protocol_type,
            "protocol_common": CommonProtocolParameters(
                zero_knowledge=zero_knowledge,
            ),
            "agents": agents_param,
            "pretrain_agents": pretrain_agents,
            "rl": rl_params,
            "ppo": common_ppo_params,
            trainer_type: trainer_param,
            "d_representation": 1,
            "include_linear_message_space": include_linear_message,
            "message_size": message_size,
            "seed": 109,
        }
    )

    # Prepare the experiment
    prepare_experiment(hyper_params=hyper_params, test_run=True, ignore_cache=True)

    # Run the experiment in test mode
    run_experiment(
        hyper_params,
        tqdm_func=DummyTqdm,
        test_run=True,
        ignore_cache=True,
        pin_memory=False,
        num_rollout_workers=0,
    )
