"""PPO image classification experiment using a grid of hyperparameters."""

from argparse import Namespace
import os
import logging

import torch

from pvg import (
    Parameters,
    AgentsParameters,
    RandomAgentParameters,
    ImageClassificationAgentParameters,
    ImageClassificationParameters,
    RlTrainerParameters,
    CommonPpoParameters,
    SoloAgentParameters,
    SpgParameters,
    ScenarioType,
    TrainerType,
    PpoLossType,
    ActivationType,
    BinarificationMethodType,
    ImageBuildingBlockType,
    CommonProtocolParameters,
    PvgProtocolParameters,
    ConstantUpdateSchedule,
    AlternatingPeriodicUpdateSchedule,
    SpgVariant,
    IhvpVariant,
    Guess,
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
    dataset_name=["cifar10"],
    num_iterations=[5000],
    num_epochs=[10],
    minibatch_size=[256],
    frames_per_batch=[256],
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
    body_lr_factor=[{"actor": 1.0, "critic": 1.0}],
    prover_blocks_per_group=[4],
    prover_num_decider_layers=[3],
    prover_lr_factor=[{"actor": 1.0, "critic": 1.0}],
    prover_block_type=[ImageBuildingBlockType.CONV2D],
    prover_pretrained_embeddings_model=["resnet18"],
    verifier_blocks_per_group=[1],
    verifier_num_decider_layers=[2],
    verifier_lr_factor=[{"actor": 1.0, "critic": 1.0}],
    verifier_block_type=[ImageBuildingBlockType.CONV2D],
    verifier_pretrained_embeddings_model=[None],
    num_block_groups=[1],
    initial_num_channels=[16],
    random_prover=[False],
    pretrain_agents=[False],
    binarification_method=[BinarificationMethodType.SELECT_TWO],
    binarification_seed=[None],
    selected_classes=[None],
    activation_function=[ActivationType.TANH],
    pretrain_num_epochs=[50],
    pretrain_batch_size=[256],
    pretrain_learning_rate=[0.001],
    pretrain_no_body_lr_factor=[True],
    spg_variant=[SpgVariant.SPG],
    stackelberg_sequence=[(("verifier",), ("prover",))],
    ihvp_variant=[IhvpVariant.NYSTROM],
    ihvp_num_iterations=[5],
    ihvp_rank=[5],
    ihvp_rho=[0.1],
    shared_reward=[False],
    normalize_advantage=[True],
    normalize_observations=[True],
    min_message_rounds=[2],
    max_message_rounds=[8],
    # update_spec can be `None` or `(num_verifier_iterations, num_prover_iterations)`
    update_spec=[None],
    seed=[8144],
)


def _construct_params(combo: dict, cmd_args: Namespace) -> Parameters:

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
    if combo["random_prover"]:
        prover_params = RandomAgentParameters()
    else:
        prover_params = ImageClassificationAgentParameters(
            num_blocks_per_group=combo["prover_blocks_per_group"],
            num_decider_layers=combo["prover_num_decider_layers"],
            activation_function=combo["activation_function"],
            agent_lr_factor=combo["prover_lr_factor"],
            body_lr_factor=combo["body_lr_factor"],
            update_schedule=prover_update_schedule,
            building_block_type=combo["prover_block_type"],
            pretrained_embeddings_model=combo["prover_pretrained_embeddings_model"],
        )
    params = Parameters(
        scenario=ScenarioType.IMAGE_CLASSIFICATION,
        trainer=combo["trainer"],
        dataset=combo["dataset_name"],
        agents=AgentsParameters(
            verifier=ImageClassificationAgentParameters(
                num_blocks_per_group=combo["verifier_blocks_per_group"],
                num_decider_layers=combo["verifier_num_decider_layers"],
                activation_function=combo["activation_function"],
                body_lr_factor=combo["body_lr_factor"],
                agent_lr_factor=combo["verifier_lr_factor"],
                update_schedule=verifier_update_schedule,
                building_block_type=combo["verifier_block_type"],
                pretrained_embeddings_model=combo[
                    "verifier_pretrained_embeddings_model"
                ],
            ),
            prover=prover_params,
        ),
        rl=RlTrainerParameters(
            frames_per_batch=combo["frames_per_batch"],
            num_iterations=combo["num_iterations"],
            num_epochs=combo["num_epochs"],
            minibatch_size=combo["minibatch_size"],
            normalize_observations=combo["normalize_observations"],
            lr=combo["lr"],
            gamma=combo["gamma"],
            lmbda=combo["lmbda"],
            anneal_lr=combo["anneal_lr"],
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
        image_classification=ImageClassificationParameters(
            num_block_groups=combo["num_block_groups"],
            initial_num_channels=combo["initial_num_channels"],
        ),
        solo_agent=SoloAgentParameters(
            num_epochs=combo["pretrain_num_epochs"],
            batch_size=combo["pretrain_batch_size"],
            learning_rate=combo["pretrain_learning_rate"],
            body_lr_factor_override=combo["pretrain_no_body_lr_factor"],
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
        ),
        pvg_protocol=PvgProtocolParameters(
            min_message_rounds=combo["min_message_rounds"],
            max_message_rounds=combo["max_message_rounds"],
        ),
        pretrain_agents=pretrain_agents,
        seed=combo["seed"],
    )

    return params


def experiment_fn(arguments: ExperimentFunctionArguments):
    combo = arguments.combo
    cmd_args = arguments.cmd_args
    logger = arguments.child_logger_adapter

    logger.info(f"Starting run {arguments.run_id}")
    logger.debug(f"Combo: {combo}")

    device = torch.device(f"cuda:{cmd_args.gpu_num}")

    # Make sure W&B doesn't print anything when the logger level is higher than DEBUG
    if logger.level > logging.DEBUG:
        os.environ["WANDB_SILENT"] = "true"

    if cmd_args.use_wandb:
        wandb_tags = [cmd_args.tag] if cmd_args.tag != "" else []
    else:
        wandb_tags = []

    params = _construct_params(combo, cmd_args)

    # Train and test the agents
    run_experiment(
        params,
        device=device,
        logger=logger,
        dataset_on_device=cmd_args.dataset_on_device,
        enable_efficient_attention=cmd_args.enable_efficient_attention,
        tqdm_func=arguments.tqdm_func,
        ignore_cache=cmd_args.ignore_cache,
        use_wandb=cmd_args.use_wandb,
        wandb_project=cmd_args.wandb_project,
        wandb_entity=cmd_args.wandb_entity,
        run_id=arguments.run_id,
        wandb_tags=wandb_tags,
        global_tqdm_step_fn=arguments.global_tqdm_step_fn,
    )


def run_id_fn(combo_index: int, cmd_args: Namespace) -> str:
    return f"ppo_ic_{cmd_args.run_infix}_{combo_index}"


def run_preparer_fn(combo: dict, cmd_args: Namespace) -> PreparedExperimentInfo:
    params = _construct_params(combo, cmd_args)
    return prepare_experiment(params=params, ignore_cache=cmd_args.ignore_cache)


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
        experiment_name="PPO_IC",
        **extra_args,
    )

    experiment.parser.add_argument(
        "--dataset-on-device",
        action="store_true",
        dest="dataset_on_device",
        help="Store the whole dataset on the device (needs more GPU memory).",
    )

    experiment.parser.add_argument(
        "--enable-efficient-attention",
        action="store_true",
        default=False,
        help="Enable efficient attention scaled dot product backend (may be buggy).",
    )

    experiment.run()
