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
    CommonPpoParameters,
    SpgParameters,
    ScenarioType,
    TrainerType,
    ActivationType,
    SpgVariant,
    IhvpVariant,
    PvgProtocolParameters,
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
    num_iterations=[10000],
    num_epochs=[4],
    minibatch_size=[256],
    gamma=[0.9],
    lmbda=[0.95],
    clip_epsilon=[0.2],
    entropy_eps=[0.001],
    lr=[0.003],
    body_lr_factor=[0.1],
    prover_num_layers=[5],
    prover_lr_factor=[1.0],
    verifier_num_layers=[2],
    verifier_lr_factor=[1.0],
    num_transformer_layers=[1],
    random_prover=[False],
    pretrain_agents=[True],
    activation_function=[ActivationType.TANH],
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

    # Create the parameters object
    if combo["random_prover"]:
        prover_params = RandomAgentParameters()
    else:
        prover_params = GraphIsomorphismAgentParameters(
            num_gnn_layers=combo["prover_num_layers"],
            activation_function=combo["activation_function"],
            agent_lr_factor=combo["prover_lr_factor"],
            num_transformer_layers=combo["num_transformer_layers"],
            normalize_message_history=combo["normalize_message_history"],
        )
    params = Parameters(
        scenario=ScenarioType.GRAPH_ISOMORPHISM,
        trainer=TrainerType.VANILLA_PPO,
        dataset=combo["dataset_name"],
        agents=AgentsParameters(
            verifier=GraphIsomorphismAgentParameters(
                num_gnn_layers=combo["verifier_num_layers"],
                activation_function=combo["activation_function"],
                agent_lr_factor=combo["verifier_lr_factor"],
                num_transformer_layers=combo["num_transformer_layers"],
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
        spg=SpgParameters(
            variant=combo["spg_variant"],
            stackelberg_sequence=combo["stackelberg_sequence"],
            ihvp_variant=combo["ihvp_variant"],
            ihvp_num_iterations=combo["ihvp_num_iterations"],
            ihvp_rank=combo["ihvp_rank"],
            ihvp_rho=combo["ihvp_rho"],
        ),
        pvg_protocol=PvgProtocolParameters(
            shared_reward=combo["shared_reward"],
        ),
        pretrain_agents=combo["pretrain_agents"],
        min_message_rounds=combo["min_message_rounds"],
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
    experiment.run()
