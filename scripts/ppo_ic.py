"""PPO image classification experiment using a grid of hyperparameters.

This script runs through a grid of hyperparameters, specified in the ``param_grid``
dict. If the ``MULTIPROCESS`` variable is set to True, the experiments are run using a
pool of workers (specified by the ``--num-workers`` command line argument). Otherwise,
the experiments are run sequentially.

Additional settings, like whether to log to W&B can be set via command line arguments.
Run the script with the ``--help`` flag to see all available arguments.
"""

from argparse import Namespace
import os
import logging

import torch

from nip import (
    HyperParameters,
    AgentsParameters,
    RandomAgentParameters,
    ImageClassificationAgentParameters,
    ImageClassificationParameters,
    RlTrainerParameters,
    CommonPpoParameters,
    SoloAgentParameters,
    SpgParameters,
    CommonProtocolParameters,
    NipProtocolParameters,
    ConstantUpdateSchedule,
    AlternatingPeriodicUpdateSchedule,
    run_experiment,
    prepare_experiment,
    PreparedExperimentInfo,
)
from nip.utils.experiments import (
    MultiprocessHyperparameterExperiment,
    SequentialHyperparameterExperiment,
    ExperimentFunctionArguments,
)

MULTIPROCESS = True

param_grid = dict(
    trainer=["vanilla_ppo"],
    dataset_name=["cifar10"],
    num_iterations=[5000],
    num_epochs=[10],
    minibatch_size=[256],
    frames_per_batch=[256],
    gamma=[0.95],
    lmbda=[0.95],
    ppo_loss_type=["clip"],
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
    prover_block_type=["conv2d"],
    prover_pretrained_embeddings_model=["resnet18"],
    verifier_blocks_per_group=[1],
    verifier_num_decider_layers=[2],
    verifier_lr_factor=[{"actor": 1.0, "critic": 1.0}],
    verifier_block_type=["conv2d"],
    verifier_pretrained_embeddings_model=[None],
    num_block_groups=[1],
    initial_num_channels=[16],
    random_prover=[False],
    pretrain_agents=[False],
    binarification_method=["select_two"],
    binarification_seed=[None],
    selected_classes=[None],
    activation_function=["tanh"],
    pretrain_num_epochs=[50],
    pretrain_batch_size=[256],
    pretrain_learning_rate=[0.001],
    pretrain_no_body_lr_factor=[True],
    spg_variant=["spg"],
    stackelberg_sequence=[(("verifier",), ("prover",))],
    ihvp_variant=["nystrom"],
    ihvp_num_iterations=[5],
    ihvp_rank=[5],
    ihvp_rho=[0.1],
    shared_reward=[False],
    normalize_advantage=[True],
    normalize_observations=[True],
    include_linear_message=[False],
    message_size=[1],
    min_message_rounds=[3],
    max_message_rounds=[8],
    # update_spec can be `None` or `(num_verifier_iterations, num_prover_iterations)`
    update_spec=[None],
    seed=[8144, 820, 4173, 3992],
)


def _construct_params(combo: dict, cmd_args: Namespace) -> HyperParameters:
    """Construct the hyperparameters object for the experiment.

    Parameters
    ----------
    combo : dict
        The hyperparameter combination to use (from the `param_grid` grid).
    cmd_args : Namespace
        The command line arguments.

    Returns
    -------
    hyper_params : HyperParameters
        The hyperparameters object.
    """

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
    hyper_params = HyperParameters(
        scenario="image_classification",
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
        nip_protocol=NipProtocolParameters(
            min_message_rounds=combo["min_message_rounds"],
            max_message_rounds=combo["max_message_rounds"],
        ),
        pretrain_agents=pretrain_agents,
        include_linear_message_space=combo["include_linear_message"],
        message_size=combo["message_size"],
        seed=combo["seed"],
    )

    return hyper_params


def experiment_fn(arguments: ExperimentFunctionArguments):
    """Run a single experiment.

    Parameters
    ----------
    arguments : ExperimentFunctionArguments
        The arguments for the experiment.
    """

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

    hyper_params = _construct_params(combo, cmd_args)

    # Train and test the agents
    run_experiment(
        hyper_params,
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
        wandb_group=arguments.common_run_name,
    )


def run_id_fn(combo_index: int | None, cmd_args: Namespace) -> str:
    """Generate the run ID for a given hyperparameter combination.

    Parameters
    ----------
    combo_index : int | None
        The index of the hyperparameter combination. If None, the run ID is for the
        entire experiment.
    cmd_args : Namespace
        The command line arguments.

    Returns
    -------
    run_id : str
        The run ID.
    """
    if combo_index is None:
        return f"ppo_ic_{cmd_args.run_infix}"
    return f"ppo_ic_{cmd_args.run_infix}_{combo_index}"


def run_preparer_fn(combo: dict, cmd_args: Namespace) -> PreparedExperimentInfo:
    """Prepare the experiment for a single run.

    Parameters
    ----------
    combo : dict
        The hyperparameter combination to use (from the `param_grid` grid).
    cmd_args : Namespace
        The command line arguments.

    Returns
    -------
    prepared_experiment_info : PreparedExperimentInfo
        The prepared experiment data.
    """
    hyper_params = _construct_params(combo, cmd_args)
    return prepare_experiment(
        hyper_params=hyper_params, ignore_cache=cmd_args.ignore_cache
    )


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

# Set the `parser` module attribute to enable the script auto-documented by Sphinx
parser = experiment.parser

if __name__ == "__main__":

    experiment.run()
