"""Script for running Expert Iteration (EI) with the code validation task.

This script runs through a grid of hyperparameters, specified in the ``param_grid``
dict, and runs EI experiments for the code validation task for each.

Additional settings, like whether to log to W&B, the number of rollout workers to use,
and whether to use the dummy API, can be set via command line arguments. Run the script
with the ``--help`` flag to see all available arguments.
"""

from argparse import Namespace
import os
import logging
from datetime import datetime

from nip import (
    HyperParameters,
    AgentsParameters,
    CodeValidationAgentParameters,
    RlTrainerParameters,
    TextRlParameters,
    CommonProtocolParameters,
    NipProtocolParameters,
    DebateProtocolParameters,
    PureTextEiParameters,
    CodeValidationParameters,
    BaseRunParameters,
    run_experiment,
    prepare_experiment,
    PreparedExperimentInfo,
    DatasetParameters,
)
from nip.utils.experiments import (
    SequentialHyperparameterExperiment,
    ExperimentFunctionArguments,
)
from nip.utils.env import get_env_var

param_grid = dict(
    interaction_protocol=["nip"],
    dataset_name=["lrhammond/buggy-apps"],
    apps_difficulty=["interview"],
    num_iterations=[8],
    rollouts_per_iteration=[200],
    verifier_model=["OpenAI/gpt-4o-mini-2024-07-18"],
    verifier_temperature=[None],
    verifier_top_p=[None],
    verifier_guess_replacement_proportion=[0.0],
    verifier_guess_replacement_annealing=["linear"],
    verifier_guess_replacement_annealing_rate=[0.1],
    prover_model=["OpenAI/gpt-4o-2024-08-06"],
    prover_temperature=[None],
    prover_top_p=[None],
    freeze_prover=[False],
    provers_share_model=[True],
    fine_tune_from_scratch=[True],
    fine_tune_on_all_previous_rollouts=[True],
    rollout_selection_method=["threshold"],
    weighting_use_replacement=[True],
    shared_reward=[False],
    randomize_prover_stance=[False],
    min_message_rounds=[1],
    max_message_rounds=[9],
    verifier_first=[True],
    debate_sequential=[False],
    debate_prover0_first=[True],
    max_train_size=[None],
    max_test_size=[None],
    test_scheme=["none"],
    num_test_iterations=[1],
    rerun_tests=[None],
)


def _construct_params(combo: dict, cmd_args: Namespace) -> HyperParameters:
    """Construct the hyperparameters object for the experiment.

    Parameters
    ----------
    combo : dict
        The hyperparameter combination to use (from the ``param_grid`` grid).
    cmd_args : Namespace
        The command line arguments.

    Returns
    -------
    hyper_params : HyperParameters
        The hyperparameters object.
    """

    verifier_model_provider, _, verifier_model_name = combo["verifier_model"].partition(
        "/"
    )
    prover_model_provider, _, prover_model_name = combo["prover_model"].partition("/")

    agents_params_dict = dict(
        verifier=CodeValidationAgentParameters(
            model_name=verifier_model_name,
            model_provider=verifier_model_provider,
            temperature=combo["verifier_temperature"],
            top_p=combo["verifier_top_p"],
            use_dummy_api=cmd_args.use_dummy_api,
            fine_tune_from_scratch=combo["fine_tune_from_scratch"],
        ),
    )

    prover_params_dict = dict(
        model_name=prover_model_name,
        model_provider=prover_model_provider,
        temperature=combo["prover_temperature"],
        top_p=combo["prover_top_p"],
        use_dummy_api=cmd_args.use_dummy_api,
        freeze_agent=combo["freeze_prover"],
        fine_tune_from_scratch=combo["fine_tune_from_scratch"],
    )

    if combo["provers_share_model"]:
        prover_params_dict["shared_model_group"] = "provers_group"
    else:
        prover_params_dict["shared_model_group"] = None

    if combo["interaction_protocol"] in ["nip", "adp"]:
        agents_params_dict["prover"] = CodeValidationAgentParameters(
            **prover_params_dict
        )
    elif combo["interaction_protocol"] in [
        "debate",
        "mnip",
        "merlin_arthur",
    ]:
        agents_params_dict["prover0"] = CodeValidationAgentParameters(
            **prover_params_dict
        )
        agents_params_dict["prover1"] = CodeValidationAgentParameters(
            **prover_params_dict
        )
    elif combo["interaction_protocol"] == "solo_verifier":
        pass
    else:
        raise NotImplementedError(
            f"This script does not currently support the "
            f"{combo['interaction_protocol']} protocol."
        )

    if combo["rerun_tests"] is not None:
        base_run_params = BaseRunParameters(
            base_run_type="rerun_tests",
            run_id=combo["rerun_tests"],
            wandb_project=get_env_var("WANDB_CV_PROJECT"),
        )
    else:
        base_run_params = BaseRunParameters(base_run_type="none")

    return HyperParameters(
        scenario="code_validation",
        trainer="pure_text_ei",
        dataset=combo["dataset_name"],
        rl=RlTrainerParameters(
            rollouts_per_iteration=combo["rollouts_per_iteration"],
            frames_per_batch=None,
            num_iterations=combo["num_iterations"],
            num_test_iterations=combo["num_test_iterations"],
        ),
        text_rl=TextRlParameters(
            test_scheme=combo["test_scheme"],
            fine_tune_on_all_previous_rollouts=combo[
                "fine_tune_on_all_previous_rollouts"
            ],
            verifier_guess_replacement_proportion=combo[
                "verifier_guess_replacement_proportion"
            ],
            verifier_guess_replacement_annealing=combo[
                "verifier_guess_replacement_annealing"
            ],
            verifier_guess_replacement_annealing_rate=combo[
                "verifier_guess_replacement_annealing_rate"
            ],
        ),
        pure_text_ei=PureTextEiParameters(
            rollout_selection_method=combo["rollout_selection_method"],
            weighting_use_replacement=combo["weighting_use_replacement"],
        ),
        agents=AgentsParameters(**agents_params_dict),
        interaction_protocol=combo["interaction_protocol"],
        protocol_common=CommonProtocolParameters(
            shared_reward=combo["shared_reward"],
            verifier_first=combo["verifier_first"],
            randomize_prover_stance=combo["randomize_prover_stance"],
        ),
        nip_protocol=NipProtocolParameters(
            min_message_rounds=combo["min_message_rounds"],
            max_message_rounds=combo["max_message_rounds"],
        ),
        debate_protocol=DebateProtocolParameters(
            min_message_rounds=combo["min_message_rounds"],
            max_message_rounds=combo["max_message_rounds"],
            sequential=combo["debate_sequential"],
            prover0_first=combo["debate_prover0_first"],
        ),
        code_validation=CodeValidationParameters(
            apps_difficulty=combo["apps_difficulty"],
        ),
        dataset_options=DatasetParameters(
            max_train_size=combo["max_train_size"],
            max_test_size=combo["max_test_size"],
        ),
        base_run=base_run_params,
    )


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
    logger.setLevel(logging.INFO)

    logger.info(f"Starting run {arguments.run_id}")
    logger.debug(f"Combo: {combo}")

    hyper_params = _construct_params(combo, cmd_args)

    if cmd_args.num_rollout_workers is None:
        if cmd_args.use_dummy_api:
            num_rollout_workers = 0
        else:
            num_rollout_workers = 8
    else:
        num_rollout_workers = cmd_args.num_rollout_workers

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
        logger=logger,
        tqdm_func=arguments.tqdm_func,
        ignore_cache=cmd_args.ignore_cache,
        use_wandb=cmd_args.use_wandb,
        wandb_project=cmd_args.wandb_project,
        wandb_entity=cmd_args.wandb_entity,
        run_id=arguments.run_id,
        allow_resuming_wandb_run=True,
        allow_overriding_wandb_config=True,
        wandb_tags=wandb_tags,
        wandb_group=arguments.common_run_name,
        num_rollout_workers=num_rollout_workers,
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
    if cmd_args.run_infix == "" and cmd_args.use_dummy_api:
        run_infix = f"test_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    elif cmd_args.run_infix == "":
        raise ValueError(
            "When not using the dummy API, the run_infix argument must be provided."
        )
    else:
        run_infix = cmd_args.run_infix
    if combo_index is None:
        return f"ei_cv_{run_infix}"
    return f"ei_cv_{run_infix}_{combo_index}"


def run_preparer_fn(combo: dict, cmd_args: Namespace) -> PreparedExperimentInfo:
    """Prepare the experiment for a single run.

    Parameters
    ----------
    combo : dict
        The hyperparameter combination to use (from the ``param_grid`` grid).
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


experiment = SequentialHyperparameterExperiment(
    param_grid=param_grid,
    experiment_fn=experiment_fn,
    run_id_fn=run_id_fn,
    run_preparer_fn=run_preparer_fn,
    experiment_name="EI_VC",
    arg_parser_description="Run Code Validation experiments with Expert Iteration, "
    "running from a hyperparameter grid in sequence.",
    default_wandb_project=get_env_var("WANDB_CV_PROJECT", ""),
    allow_resuming_wandb_run=True,
    add_run_infix_argument=False,
)

experiment.parser.add_argument(
    "run_infix",
    type=str,
    help="Infix to add to the run ID to distinguish between different runs. Defaults to 'test_{time_now}' when using dummy API; otherwise raises an error.",
    nargs="?",
    default="",
)

experiment.parser.add_argument(
    "--num-rollout-workers",
    type=int,
    default=None,
    help="Number of workers to use for sampling rollouts. Defaults 0 when using dummy API, 8 otherwise.",
)

experiment.parser.add_argument(
    "--dummy",
    action="store_true",
    dest="use_dummy_api",
    help="Whether to use the dummy API for the agents. Useful for testing.",
)

# Set the ``parser`` module attribute to enable the script auto-documented by Sphinx
parser = experiment.parser

if __name__ == "__main__":
    experiment.run()
