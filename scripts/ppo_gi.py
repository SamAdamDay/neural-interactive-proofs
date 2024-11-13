"""PPO graph isomorphism experiment using a grid of hyperparameters."""

from argparse import Namespace
import os
import logging
import dataclasses
from copy import deepcopy

import torch

from pvg import (
    HyperParameters,
    AgentsParameters,
    RandomAgentParameters,
    GraphIsomorphismAgentParameters,
    RlTrainerParameters,
    CommonPpoParameters,
    SpgParameters,
    ScenarioType,
    TrainerType,
    PpoLossType,
    ActivationType,
    SpgVariant,
    IhvpVariant,
    InteractionProtocolType,
    CommonProtocolParameters,
    PvgProtocolParameters,
    DebateProtocolParameters,
    ZkProtocolParameters,
    DatasetParameters,
    ConstantUpdateSchedule,
    AlternatingPeriodicUpdateSchedule,
    ContiguousPeriodicUpdateSchedule,
    run_experiment,
    prepare_experiment,
    PreparedExperimentInfo,
)
from pvg.utils.experiments import (
    MultiprocessHyperparameterExperiment,
    SequentialHyperparameterExperiment,
    ExperimentFunctionArguments,
)

MULTIPROCESS = True

param_grid = dict(
    trainer=[TrainerType.VANILLA_PPO],
    interaction_protocol=[InteractionProtocolType.PVG],
    dataset_name=["eru10000"],
    num_iterations=[5000],
    num_epochs=[10],
    minibatch_size=[256],
    frames_per_batch=[2048],
    gamma=[0.95],
    lmbda=[0.95],
    ppo_loss_type=[PpoLossType.CLIP],
    clip_epsilon=[0.2],
    kl_target=[0.01],
    kl_beta=[1.0],
    kl_decrement=[0.5],
    kl_increment=[2.0],
    critic_coef=[1.0],
    loss_critic_type=["smooth_l1"],
    clip_value=[False],
    entropy_eps=[0.001],
    lr=[0.0003],
    anneal_lr=[False],
    max_grad_norm=[0.5],
    body_lr_factor=[{"actor": 1.0, "critic": 1.0}],
    gnn_lr_factor=[{"actor": 1.0, "critic": 1.0}],
    use_dual_gnn=[False],
    use_orthogonal_initialisation=[True],
    prover_num_layers=[5],
    prover_num_value_layers=[2],
    prover_dim_value_layers=[16],
    prover_lr_factor=[{"actor": 1.0, "critic": 1.0}],
    prover_manual_architecture=[False],
    verifier_num_layers=[2],
    verifier_num_value_layers=[2],
    verifier_dim_value_layers=[16],
    verifier_lr_factor=[{"actor": 1.0, "critic": 1.0}],
    verifier_manual_architecture=[False],
    num_transformer_layers=[1],
    num_transformer_heads=[4],
    include_round_in_decider=[True],
    d_representation=[16],
    use_batch_norm=[False],
    random_prover=[False],
    use_shared_body=[False],
    pretrain_agents=[False],
    activation_function=[ActivationType.TANH],
    spg_variant=[SpgVariant.SPG],
    stackelberg_sequence=[None],
    ihvp_variant=[IhvpVariant.NYSTROM],
    ihvp_num_iterations=[5],
    ihvp_rank=[5],
    ihvp_rho=[0.1],
    shared_reward=[False],
    verifier_terminated_penalty=[-1.0],
    verifier_no_guess_reward=[0.05],
    normalize_advantage=[True],
    normalize_observations=[True],
    include_linear_message=[False],
    message_size=[1],
    min_message_rounds=[2],
    max_message_rounds=[8],
    verifier_first=[True],
    debate_sequential=[False],
    debate_prover0_first=[True],
    zero_knowledge=[True],
    use_multiple_simulators=[True],
    simulator_reward_coefficient=[None],
    aux_prover_reward_coefficient=[0.0,0.1,0.5,1.0],
    # update_spec can be `None`, `(num_verifier_iterations, num_prover_iterations)` or
    # `(num_verifier_iterations, num_prover0_iterations, num_prover1_iterations)`.
    update_spec=[None],
    max_train_size=[None],
    seed=[8144, 820, 4173, 3992],
)


def _construct_params(combo: dict, cmd_args: Namespace) -> HyperParameters:

    # Set the pretrain_agents flag. This can be forced to False with the --no-pretrain
    # flag.
    if cmd_args.no_pretrain:
        pretrain_agents = False
    else:
        pretrain_agents = combo["pretrain_agents"]

    # Create the parameters object
    verifier_update_schedule = ConstantUpdateSchedule()
    prover_update_schedule = ConstantUpdateSchedule()
    prover0_update_schedule = ConstantUpdateSchedule()
    prover1_update_schedule = ConstantUpdateSchedule()
    if isinstance(combo["update_spec"], tuple) and len(combo["update_spec"]) == 2:
        period = combo["update_spec"][0] + combo["update_spec"][1]
        verifier_update_schedule = AlternatingPeriodicUpdateSchedule(
            period, combo["update_spec"][0], first_agent=True
        )
        prover_update_schedule = AlternatingPeriodicUpdateSchedule(
            period, combo["update_spec"][0], first_agent=False
        )
    elif isinstance(combo["update_spec"], tuple) and len(combo["update_spec"]) == 3:
        period = sum(combo["update_spec"])
        verifier_update_schedule = ContiguousPeriodicUpdateSchedule(
            period, 0, combo["update_spec"][0]
        )
        prover0_update_schedule = ContiguousPeriodicUpdateSchedule(
            period,
            combo["update_spec"][0],
            combo["update_spec"][0] + combo["update_spec"][1],
        )
        prover1_update_schedule = ContiguousPeriodicUpdateSchedule(
            period,
            combo["update_spec"][0] + combo["update_spec"][1],
            period,
        )
    if combo["prover_manual_architecture"]:
        prover_lr_factor = {"actor": 0.0, "critic": 0.0}
    else:
        prover_lr_factor = combo["prover_lr_factor"]
    if combo["verifier_manual_architecture"]:
        verifier_lr_factor = {"actor": 0.0, "critic": 0.0}
    else:
        verifier_lr_factor = combo["verifier_lr_factor"]
    if combo["random_prover"]:
        prover_params = RandomAgentParameters()
    else:
        prover_params = GraphIsomorphismAgentParameters(
            num_gnn_layers=combo["prover_num_layers"],
            activation_function=combo["activation_function"],
            agent_lr_factor=prover_lr_factor,
            num_transformer_layers=combo["num_transformer_layers"],
            num_heads=combo["num_transformer_heads"],
            num_value_layers=combo["prover_num_value_layers"],
            d_value=combo["prover_dim_value_layers"],
            use_manual_architecture=combo["prover_manual_architecture"],
            use_dual_gnn=combo["use_dual_gnn"],
            body_lr_factor=combo["body_lr_factor"],
            gnn_lr_factor=combo["gnn_lr_factor"],
            include_round_in_decider=combo["include_round_in_decider"],
            use_batch_norm=combo["use_batch_norm"],
            use_orthogonal_initialisation=combo["use_orthogonal_initialisation"],
            update_schedule=prover_update_schedule,
        )
    verifier_params = GraphIsomorphismAgentParameters(
        num_gnn_layers=combo["verifier_num_layers"],
        activation_function=combo["activation_function"],
        agent_lr_factor=verifier_lr_factor,
        num_transformer_layers=combo["num_transformer_layers"],
        num_heads=combo["num_transformer_heads"],
        num_value_layers=combo["verifier_num_value_layers"],
        d_value=combo["verifier_dim_value_layers"],
        use_manual_architecture=combo["verifier_manual_architecture"],
        use_dual_gnn=combo["use_dual_gnn"],
        body_lr_factor=combo["body_lr_factor"],
        gnn_lr_factor=combo["gnn_lr_factor"],
        include_round_in_decider=combo["include_round_in_decider"],
        use_batch_norm=combo["use_batch_norm"],
        use_orthogonal_initialisation=combo["use_orthogonal_initialisation"],
        update_schedule=verifier_update_schedule,
    )
    agents_params_dict = {}
    if combo["interaction_protocol"] in (
        InteractionProtocolType.PVG,
        InteractionProtocolType.ABSTRACT_DECISION_PROBLEM,
    ):
        # agents_params = AgentsParameters(verifier=verifier_params, prover=prover_params)
        agents_params_dict["verifier"] = verifier_params
        agents_params_dict["prover"] = prover_params
    elif combo["interaction_protocol"] in (
        InteractionProtocolType.DEBATE,
        InteractionProtocolType.MERLIN_ARTHUR,
        InteractionProtocolType.MNIP,
    ):
        prover0_params = dataclasses.replace(
            prover_params, update_schedule=prover0_update_schedule
        )
        prover1_params = dataclasses.replace(
            prover_params, update_schedule=prover1_update_schedule
        )
        agents_params_dict["verifier"] = verifier_params
        agents_params_dict["prover0"] = prover0_params
        agents_params_dict["prover1"] = prover1_params
        # agents_params = AgentsParameters(
        #     verifier=verifier_params, prover0=prover0_params, prover1=prover1_params
        # )
    else:
        raise NotImplementedError(
            f"Unknown interaction protocol: {combo['interaction_protocol']}"
        )
    if combo["zero_knowledge"]:
        # We need to create fresh copies of the parameters for the adversarial verifier and simulator agents
        original_agent_names = list(agents_params_dict.keys())
        agents_params_dict["adversarial_verifier"] = dataclasses.replace(verifier_params)
        if combo["use_multiple_simulators"]:
            for agent_name in original_agent_names:
                agents_params_dict[f"simulator_{agent_name}"] = dataclasses.replace(agents_params_dict[agent_name])
        else:
            agents_params_dict["simulator"] = dataclasses.replace(verifier_params)

    hyper_params = HyperParameters(
        scenario=ScenarioType.GRAPH_ISOMORPHISM,
        trainer=combo["trainer"],
        dataset=combo["dataset_name"],
        # agents=agents_params,
        agents=AgentsParameters(**agents_params_dict),
        rl=RlTrainerParameters(
            frames_per_batch=combo["frames_per_batch"],
            num_iterations=combo["num_iterations"],
            num_epochs=combo["num_epochs"],
            minibatch_size=combo["minibatch_size"],
            lr=combo["lr"],
            anneal_lr=combo["anneal_lr"],
            max_grad_norm=combo["max_grad_norm"],
            normalize_observations=combo["normalize_observations"],
            use_shared_body=combo["use_shared_body"],
            gamma=combo["gamma"],
            lmbda=combo["lmbda"],
            loss_critic_type=combo["loss_critic_type"],
            clip_value=combo["clip_value"],
        ),
        ppo=CommonPpoParameters(
            loss_type=combo["ppo_loss_type"],
            clip_epsilon=combo["clip_epsilon"],
            kl_target=combo["kl_target"],
            kl_beta=combo["kl_beta"],
            kl_decrement=combo["kl_decrement"],
            kl_increment=combo["kl_increment"],
            critic_coef=combo["critic_coef"],
            entropy_eps=combo["entropy_eps"],
            normalize_advantage=combo["normalize_advantage"],
        ),
        spg=SpgParameters(
            variant=combo["spg_variant"],
            stackelberg_sequence=combo["stackelberg_sequence"],
            ihvp_variant=combo["ihvp_variant"],
            ihvp_num_iterations=combo["ihvp_num_iterations"],
            ihvp_rank=combo["ihvp_rank"],
            ihvp_rho=combo["ihvp_rho"],
        ),
        interaction_protocol=combo["interaction_protocol"],
        protocol_common=CommonProtocolParameters(
            shared_reward=combo["shared_reward"],
            verifier_terminated_penalty=combo["verifier_terminated_penalty"],
            verifier_no_guess_reward=combo["verifier_no_guess_reward"],
            verifier_first=combo["verifier_first"],
            zero_knowledge=combo["zero_knowledge"],
        ),
        pvg_protocol=PvgProtocolParameters(
            min_message_rounds=combo["min_message_rounds"],
            max_message_rounds=combo["max_message_rounds"],
        ),
        debate_protocol=DebateProtocolParameters(
            min_message_rounds=combo["min_message_rounds"],
            max_message_rounds=combo["max_message_rounds"],
            sequential=combo["debate_sequential"],
            prover0_first=combo["debate_prover0_first"],
        ),
        zk_protocol=ZkProtocolParameters(
            simulator_reward_coefficient = combo["simulator_reward_coefficient"],
            aux_prover_reward_coefficient = combo["aux_prover_reward_coefficient"],
            use_multiple_simulators = combo["use_multiple_simulators"],
        ),
        pretrain_agents=pretrain_agents,
        d_representation=combo["d_representation"],
        include_linear_message_space=combo["include_linear_message"],
        message_size=combo["message_size"],
        dataset_options=DatasetParameters(
            max_train_size=combo["max_train_size"],
        ),
        seed=combo["seed"],
    )

    return hyper_params


def experiment_fn(arguments: ExperimentFunctionArguments):
    combo = arguments.combo
    cmd_args = arguments.cmd_args
    logger = arguments.child_logger_adapter

    logger.info(f"Starting run {arguments.run_id}")
    logger.debug(f"Combo: {combo}")

    device = torch.device(f"cuda:{cmd_args.gpu_num}")

    hyper_params = _construct_params(combo, cmd_args)

    # Make sure W&B doesn't print anything when the logger level is higher than DEBUG
    if logger.level > logging.DEBUG:
        os.environ["WANDB_SILENT"] = "true"

    if cmd_args.use_wandb:
        wandb_tags = [cmd_args.tag] if cmd_args.tag != "" else []
    else:
        wandb_tags = []

    # Train and test the agents
    run_experiment(
        hyper_params,
        device=device,
        dataset_on_device=cmd_args.dataset_on_device,
        enable_efficient_attention=cmd_args.enable_efficient_attention,
        logger=logger,
        tqdm_func=arguments.tqdm_func,
        ignore_cache=cmd_args.ignore_cache,
        use_wandb=cmd_args.use_wandb,
        wandb_project=cmd_args.wandb_project,
        wandb_entity=cmd_args.wandb_entity,
        run_id=arguments.run_id,
        wandb_tags=wandb_tags,
        global_tqdm_step_fn=arguments.global_tqdm_step_fn,
        wandb_group=arguments.common_run_name,
    )


def run_id_fn(combo_index: int | None, cmd_args: Namespace) -> str:
    if combo_index is None:
        return f"ppo_gi_{cmd_args.run_infix}"
    return f"ppo_gi_{cmd_args.run_infix}_{combo_index}"


def run_preparer_fn(combo: dict, cmd_args: Namespace) -> PreparedExperimentInfo:
    hyper_params = _construct_params(combo, cmd_args)
    return prepare_experiment(
        hyper_params=hyper_params, ignore_cache=cmd_args.ignore_cache
    )


if __name__ == "__main__":
    if MULTIPROCESS:
        experiment_class = MultiprocessHyperparameterExperiment
        extra_args = dict(default_num_workers=4)
    else:
        experiment_class = SequentialHyperparameterExperiment
        extra_args = dict()

    experiment = experiment_class(
        param_grid=param_grid,
        experiment_fn=experiment_fn,
        run_id_fn=run_id_fn,
        run_preparer_fn=run_preparer_fn,
        experiment_name="PPO_GI",
        **extra_args,
    )

    experiment.parser.add_argument(
        "--no-dataset-on-device",
        action="store_false",
        dest="dataset_on_device",
        default=True,
        help="Don't store the whole dataset on the device.",
    )

    experiment.parser.add_argument(
        "--enable-efficient-attention",
        action="store_true",
        default=False,
        help="Enable efficient attention scaled dot product backend (may be buggy).",
    )

    experiment.run()
