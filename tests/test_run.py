from sklearn.model_selection import ParameterGrid

from pvg import (
    Parameters,
    GraphIsomorphismAgentParameters,
    ImageClassificationAgentParameters,
    SoloAgentParameters,
    RlTrainerParameters,
    CommonPpoParameters,
    SpgParameters,
    SpgVariant,
    ScenarioType,
    PpoLossType,
    TrainerType,
    InteractionProtocolType,
    run_experiment,
    prepare_experiment,
)
from pvg.parameters import AGENT_NAMES
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
    basic_agent_params = {}
    basic_agent_params[ScenarioType.GRAPH_ISOMORPHISM] = (
        GraphIsomorphismAgentParameters.construct_test_params()
    )
    basic_agent_params[ScenarioType.IMAGE_CLASSIFICATION] = (
        ImageClassificationAgentParameters.construct_test_params()
    )

    # Very basic parameters for each trainer
    rl_params = RlTrainerParameters(
        num_iterations=2,
        num_epochs=1,
        minibatch_size=1,
        frames_per_batch=8,
    )
    trainer_params = {
        TrainerType.SOLO_AGENT: SoloAgentParameters(
            num_epochs=1,
            batch_size=1,
        ),
        TrainerType.VANILLA_PPO: None,
        TrainerType.SPG: SpgParameters(),
        TrainerType.REINFORCE: None,
    }
    common_ppo_params = CommonPpoParameters()

    param_specs = [
        # Test combinations of scenario, random provers, and pretraining
        {
            "scenario": [
                ScenarioType.GRAPH_ISOMORPHISM,
                ScenarioType.IMAGE_CLASSIFICATION,
            ],
            "trainer": [TrainerType.VANILLA_PPO],
            "ppo_loss": [PpoLossType.CLIP],
            "protocol": [InteractionProtocolType.PVG],
            "spg_variant": [SpgVariant.SPG],
            "is_random": [True, False],
            "pretrain_agents": [True, False],
            "manual_architecture": [None],
            "use_shared_body": [True],
        },
        # Test the KL penalty loss
        {
            "scenario": [ScenarioType.GRAPH_ISOMORPHISM],
            "trainer": [TrainerType.VANILLA_PPO],
            "ppo_loss": [PpoLossType.KL_PENALTY],
            "protocol": [InteractionProtocolType.PVG],
            "spg_variant": [SpgVariant.SPG],
            "is_random": [False],
            "pretrain_agents": [False],
            "manual_architecture": [None],
            "use_shared_body": [True],
        },
        # Test the using non-shared bodies
        {
            "scenario": [
                ScenarioType.GRAPH_ISOMORPHISM,
                ScenarioType.IMAGE_CLASSIFICATION,
            ],
            "trainer": [TrainerType.VANILLA_PPO],
            "ppo_loss": [PpoLossType.CLIP],
            "protocol": [InteractionProtocolType.PVG],
            "spg_variant": [SpgVariant.SPG],
            "is_random": [False],
            "pretrain_agents": [True],
            "manual_architecture": [None],
            "use_shared_body": [False],
        },
        # Test the other trainers
        {
            "scenario": [ScenarioType.GRAPH_ISOMORPHISM],
            "trainer": [TrainerType.SPG, TrainerType.SOLO_AGENT, TrainerType.REINFORCE],
            "ppo_loss": [PpoLossType.CLIP],
            "protocol": [InteractionProtocolType.PVG],
            "spg_variant": [SpgVariant.SPG],
            "is_random": [False],
            "pretrain_agents": [False],
            "manual_architecture": [None],
            "use_shared_body": [True],
        },
        # Test the SPG trainer with different variants
        {
            "scenario": [ScenarioType.GRAPH_ISOMORPHISM],
            "trainer": [TrainerType.SPG],
            "ppo_loss": [PpoLossType.CLIP],
            "protocol": [InteractionProtocolType.PVG],
            "spg_variant": [
                SpgVariant.SPG,
                SpgVariant.PSPG,
                SpgVariant.LOLA,
                SpgVariant.POLA,
                SpgVariant.PSOS,
                SpgVariant.SOS,
            ],
            "is_random": [False],
            "pretrain_agents": [False],
            "manual_architecture": [None],
            "use_shared_body": [True],
        },
        # Test the other protocols
        {
            "scenario": [ScenarioType.GRAPH_ISOMORPHISM],
            "trainer": [TrainerType.VANILLA_PPO],
            "ppo_loss": [PpoLossType.CLIP],
            "protocol": [
                InteractionProtocolType.DEBATE,
                InteractionProtocolType.ABSTRACT_DECISION_PROBLEM,
                InteractionProtocolType.MERLIN_ARTHUR,
            ],
            "spg_variant": [SpgVariant.SPG],
            "is_random": [True],
            "pretrain_agents": [False],
            "manual_architecture": [None],
            "use_shared_body": [True],
        },
        # Test manual architectures
        {
            "scenario": [ScenarioType.GRAPH_ISOMORPHISM],
            "trainer": [TrainerType.VANILLA_PPO],
            "ppo_loss": [PpoLossType.CLIP],
            "protocol": [InteractionProtocolType.PVG],
            "spg_variant": [SpgVariant.SPG],
            "is_random": [False],
            "pretrain_agents": [False],
            "manual_architecture": [("prover",), ("verifier",), ("prover", "verifier")],
            "use_shared_body": [True],
        },
    ]

    for param_spec in ParameterGrid(param_specs):
        scenario_type = param_spec["scenario"]
        trainer_type = param_spec["trainer"]
        ppo_loss_type = param_spec["ppo_loss"]
        protocol_type = param_spec["protocol"]
        is_random = param_spec["is_random"]
        pretrain_agents = param_spec["pretrain_agents"]
        manual_architecture = param_spec["manual_architecture"]
        use_shared_body = param_spec["use_shared_body"]

        # Construct the PPO parameters
        common_ppo_params.loss_type = ppo_loss_type

        # Construct the RL parameters
        rl_params.use_shared_body = use_shared_body

        # Construct the agent parameters
        agents_param = {}
        for agent_name in AGENT_NAMES[protocol_type]:
            if is_random and agent_name != "verifier":
                agents_param[agent_name] = {"is_random": True}
            else:
                agents_param[agent_name] = basic_agent_params[scenario_type]
        if manual_architecture is not None:
            for agent_name in manual_architecture:
                agents_param[agent_name].use_manual_architecture = True

        # Construct the trainer parameters
        trainer_param = trainer_params[trainer_type]
        if trainer_type == TrainerType.SPG:
            trainer_param.variant = param_spec["spg_variant"]

        # Construct the parameters
        params = Parameters(
            **{
                "scenario": scenario_type,
                "trainer": trainer_type,
                "dataset": "test",
                "interaction_protocol": protocol_type,
                "agents": agents_param,
                "pretrain_agents": pretrain_agents,
                "rl": rl_params,
                "ppo": common_ppo_params,
                str(trainer_type): trainer_param,
                "d_representation": 1,
            }
        )

        # Prepare the experiment
        prepare_experiment(params=params, test_run=True, ignore_cache=True)

        # Run the experiment in test mode
        run_experiment(
            params,
            tqdm_func=DummyTqdm,
            test_run=True,
            ignore_cache=True,
            pin_memory=False,
        )
