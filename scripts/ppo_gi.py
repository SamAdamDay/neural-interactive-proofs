"""PPO graph isomorphism experiment using a grid of hyperparameters."""

from argparse import Namespace
import os
import logging

import torch

from pvg import (
    Parameters,
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
    CommonProtocolParameters,
    PvgProtocolParameters,
    ConstantUpdateSchedule,
    AlternatingPeriodicUpdateSchedule,
    run_experiment,
    prepare_experiment,
)
from pvg.utils.experiments import (
    MultiprocessHyperparameterExperiment,
    SequentialHyperparameterExperiment,
)

MULTIPROCESS = True

param_grid = dict(
    trainer=[TrainerType.VANILLA_PPO],
    dataset_name=["eru10000"],
    num_iterations=[5000],
    num_epochs=[10],
    minibatch_size=[256],
    frames_per_batch=[2024],
    gamma=[0.9],
    lmbda=[0.95],
    ppo_loss_type=[PpoLossType.CLIP],
    clip_epsilon=[0.2],
    kl_target=[0.01],
    kl_beta=[1.0],
    kl_decrement=[0.5],
    kl_increment=[2.0],
    critic_coef=[1.0],
    entropy_eps=[0.001],
    lr=[0.003],
    max_grad_norm=[0.5],
    body_lr_factor=[0.1],
    gnn_lr_factor=[1.0],
    use_dual_gnn=[False],
    prover_num_layers=[5],
    prover_num_value_layers=[2],
    prover_dim_value_layers=[16],
    prover_lr_factor=[1.0],
    prover_manual_architecture=[False],
    verifier_num_layers=[2],
    verifier_num_value_layers=[2],
    verifier_dim_value_layers=[16],
    verifier_lr_factor=[1.0],
    verifier_manual_architecture=[False],
    num_transformer_layers=[1],
    num_transformer_heads=[4],
    include_round_in_decider=[True],
    d_representation=[16],
    use_batch_norm=[True],
    random_prover=[False],
    use_shared_body=[True],
    pretrain_agents=[True],
    activation_function=[ActivationType.TANH],
    spg_variant=[SpgVariant.SPG],
    stackelberg_sequence=[(("verifier",), ("prover",))],
    ihvp_variant=[IhvpVariant.NYSTROM],
    ihvp_num_iterations=[5],
    ihvp_rank=[5],
    ihvp_rho=[0.1],
    shared_reward=[False],
    verifier_terminated_penalty=[-1.0],
    verifier_no_guess_reward=[0.05],
    normalize_advantage=[True],
    normalize_message_history=[True],
    min_message_rounds=[2],
    max_message_rounds=[8],
    verifier_first=[True],
    # update_spec can be `None` or `(num_verifier_iterations, num_prover_iterations)`
    update_spec=[(25, 25)],
    seed=[8144, 820, 4173, 3992],
)


def experiment_fn(
    combo: dict,
    run_id: str,
    cmd_args: Namespace,
    tqdm_func: callable,
    logger: logging.Logger,
):
    logger.info(f"Starting run {run_id}")
    logger.debug(f"Combo: {combo}")

    device = torch.device(f"cuda:{cmd_args.gpu_num}")

    # Make sure W&B doesn't print anything when the logger level is higher than DEBUG
    if logger.level > logging.DEBUG:
        os.environ["WANDB_SILENT"] = "true"

    # Set the pretrain_agents flag. This can be forced to False with the --no-pretrain
    # flag.
    if cmd_args.no_pretrain:
        pretrain_agents = False
    else:
        pretrain_agents = combo["pretrain_agents"]

    # Create the parameters object
    if combo["update_spec"] is None:
        verifier_update_schedule = ConstantUpdateSchedule()
        prover_update_schedule = ConstantUpdateSchedule()
    else:
        period = combo["update_spec"][0] + combo["update_spec"][1]
        verifier_update_schedule = AlternatingPeriodicUpdateSchedule(
            period, combo["update_spec"][0], first_agent=True
        )
        prover_update_schedule = AlternatingPeriodicUpdateSchedule(
            period, combo["update_spec"][0], first_agent=False
        )
    if combo["prover_manual_architecture"]:
        prover_lr_factor = 0.0
    else:
        prover_lr_factor = combo["prover_lr_factor"]
    if combo["verifier_manual_architecture"]:
        verifier_lr_factor = 0.0
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
            normalize_message_history=combo["normalize_message_history"],
            use_manual_architecture=combo["prover_manual_architecture"],
            use_dual_gnn=combo["use_dual_gnn"],
            gnn_lr_factor=combo["gnn_lr_factor"],
            include_round_in_decider=combo["include_round_in_decider"],
            use_batch_norm=combo["use_batch_norm"],
            update_schedule=prover_update_schedule,
        )
    params = Parameters(
        scenario=ScenarioType.GRAPH_ISOMORPHISM,
        trainer=combo["trainer"],
        dataset=combo["dataset_name"],
        agents=AgentsParameters(
            verifier=GraphIsomorphismAgentParameters(
                num_gnn_layers=combo["verifier_num_layers"],
                activation_function=combo["activation_function"],
                agent_lr_factor=verifier_lr_factor,
                num_transformer_layers=combo["num_transformer_layers"],
                num_heads=combo["num_transformer_heads"],
                num_value_layers=combo["verifier_num_value_layers"],
                d_value=combo["verifier_dim_value_layers"],
                normalize_message_history=combo["normalize_message_history"],
                use_manual_architecture=combo["verifier_manual_architecture"],
                use_dual_gnn=combo["use_dual_gnn"],
                gnn_lr_factor=combo["gnn_lr_factor"],
                include_round_in_decider=combo["include_round_in_decider"],
                use_batch_norm=combo["use_batch_norm"],
                update_schedule=verifier_update_schedule,
            ),
            prover=prover_params,
        ),
        rl=RlTrainerParameters(
            frames_per_batch=combo["frames_per_batch"],
            num_iterations=combo["num_iterations"],
            num_epochs=combo["num_epochs"],
            minibatch_size=combo["minibatch_size"],
            body_lr_factor=combo["body_lr_factor"],
            lr=combo["lr"],
            max_grad_norm=combo["max_grad_norm"],
            use_shared_body=combo["use_shared_body"],
            gamma=combo["gamma"],
            lmbda=combo["lmbda"],
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
        protocol_common=CommonProtocolParameters(
            shared_reward=combo["shared_reward"],
            verifier_terminated_penalty=combo["verifier_terminated_penalty"],
            verifier_no_guess_reward=combo["verifier_no_guess_reward"],
        ),
        pvg_protocol=PvgProtocolParameters(
            min_message_rounds=combo["min_message_rounds"],
            max_message_rounds=combo["max_message_rounds"],
            verifier_first=combo["verifier_first"],
        ),
        pretrain_agents=pretrain_agents,
        d_representation=combo["d_representation"],
        seed=combo["seed"],
    )

    if cmd_args.use_wandb:
        wandb_tags = [cmd_args.tag] if cmd_args.tag != "" else []
    else:
        wandb_tags = []

    # Train and test the agents
    run_experiment(
        params,
        device=device,
        dataset_on_device=cmd_args.dataset_on_device,
        logger=logger,
        tqdm_func=tqdm_func,
        ignore_cache=cmd_args.ignore_cache,
        use_wandb=cmd_args.use_wandb,
        wandb_project=cmd_args.wandb_project,
        wandb_entity=cmd_args.wandb_entity,
        run_id=run_id,
        wandb_tags=wandb_tags,
    )


def run_id_fn(combo_index: int, cmd_args: Namespace):
    return f"ppo_gi_{cmd_args.run_infix}_{combo_index}"


def run_preparer_fn(combo: dict, cmd_args: Namespace):
    params = Parameters(
        scenario=ScenarioType.GRAPH_ISOMORPHISM,
        trainer=TrainerType.VANILLA_PPO,
        dataset=combo["dataset_name"],
    )
    prepare_experiment(params=params, ignore_cache=cmd_args.ignore_cache)


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

    experiment.run()
