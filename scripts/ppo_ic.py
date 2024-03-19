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
    CommonPpoParameters,
    SoloAgentParameters,
    SpgParameters,
    ScenarioType,
    TrainerType,
    ActivationType,
    BinarificationMethodType,
    CommonProtocolParameters,
    PvgProtocolParameters,
    SpgVariant,
    IhvpVariant,
    run_experiment,
    prepare_experiment,
)
from pvg.utils.experiments import (
    MultiprocessHyperparameterExperiment,
    SequentialHyperparameterExperiment,
)

MULTIPROCESS = False

param_grid = dict(
    trainer=[TrainerType.VANILLA_PPO],
    dataset_name=["svhn"],
    num_iterations=[1000],
    num_epochs=[1],
    minibatch_size=[2],
    gamma=[0.9],
    lmbda=[0.95],
    clip_epsilon=[0.2],
    entropy_eps=[0.001],
    lr=[0.003],
    body_lr_factor=[1.0],
    prover_convs_per_group=[1],
    prover_num_decider_layers=[1],
    prover_lr_factor=[1.0],
    verifier_convs_per_group=[1],
    verifier_num_decider_layers=[1],
    verifier_lr_factor=[0.1],
    num_conv_groups=[1],
    initial_num_channels=[1],
    random_prover=[False],
    pretrain_agents=[False],
    binarification_method=[BinarificationMethodType.MERGE],
    binarification_seed=[None],
    selected_classes=[None],
    activation_function=[ActivationType.TANH],
    pretrain_num_epochs=[50],
    pretrain_batch_size=[256],
    pretrain_learning_rate=[0.001],
    pretrain_body_lr_factor=[1.0],
    spg_variant=[SpgVariant.SPG],
    stackelberg_sequence=[(("verifier",), ("prover",))],
    ihvp_variant=[IhvpVariant.NYSTROM],
    ihvp_num_iterations=[5],
    ihvp_rank=[5],
    ihvp_rho=[0.1],
    shared_reward=[False],
    normalize_advantage=[True],
    normalize_message_history=[True],
    min_message_rounds=[0],
    max_message_rounds=[8],
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
    if combo["random_prover"]:
        prover_params = RandomAgentParameters()
    else:
        prover_params = ImageClassificationAgentParameters(
            num_convs_per_group=combo["prover_convs_per_group"],
            num_decider_layers=combo["prover_num_decider_layers"],
            activation_function=combo["activation_function"],
            agent_lr_factor=combo["prover_lr_factor"],
            normalize_message_history=combo["normalize_message_history"],
        )
    params = Parameters(
        scenario=ScenarioType.IMAGE_CLASSIFICATION,
        trainer=combo["trainer"],
        dataset=combo["dataset_name"],
        agents=AgentsParameters(
            verifier=ImageClassificationAgentParameters(
                num_convs_per_group=combo["verifier_convs_per_group"],
                num_decider_layers=combo["verifier_num_decider_layers"],
                activation_function=combo["activation_function"],
                agent_lr_factor=combo["verifier_lr_factor"],
                normalize_message_history=combo["normalize_message_history"],
            ),
            prover=prover_params,
        ),
        ppo=CommonPpoParameters(
            num_iterations=combo["num_iterations"],
            num_epochs=combo["num_epochs"],
            minibatch_size=combo["minibatch_size"],
            gamma=combo["gamma"],
            lmbda=combo["lmbda"],
            clip_epsilon=combo["clip_epsilon"],
            entropy_eps=combo["entropy_eps"],
            body_lr_factor=combo["body_lr_factor"],
            lr=combo["lr"],
            normalize_advantage=combo["normalize_advantage"],
        ),
        image_classification=ImageClassificationParameters(
            num_conv_groups=combo["num_conv_groups"],
            initial_num_channels=combo["initial_num_channels"],
        ),
        solo_agent=SoloAgentParameters(
            num_epochs=combo["pretrain_num_epochs"],
            batch_size=combo["pretrain_batch_size"],
            learning_rate=combo["pretrain_learning_rate"],
            body_lr_factor=combo["pretrain_body_lr_factor"],
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

    if cmd_args.use_wandb:
        wandb_tags = [cmd_args.tag] if cmd_args.tag != "" else []
    else:
        wandb_tags = []

    # Train and test the agents
    run_experiment(
        params,
        device=device,
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
    return f"ppo_ic_{cmd_args.run_infix}_{combo_index}"


def run_preparer_fn(combo: dict, cmd_args: Namespace):
    params = Parameters(
        scenario=ScenarioType.IMAGE_CLASSIFICATION,
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
        experiment_name="PPO_IC",
        **extra_args,
    )
    experiment.run()
