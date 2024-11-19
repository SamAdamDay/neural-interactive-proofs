"""Script for running Expert Iteration with the code validation task."""

from argparse import Namespace
import os
import logging
from datetime import datetime


from pvg import (
    HyperParameters,
    AgentsParameters,
    CodeValidationAgentParameters,
    RlTrainerParameters,
    TextRlParameters,
    ScenarioType,
    TrainerType,
    InteractionProtocolType,
    CommonProtocolParameters,
    PvgProtocolParameters,
    DebateProtocolParameters,
    PureTextEiParameters,
    CodeValidationParameters,
    BaseRunParameters,
    run_experiment,
    prepare_experiment,
    PreparedExperimentInfo,
)
from pvg.utils.experiments import (
    SequentialHyperparameterExperiment,
    ExperimentFunctionArguments,
)
from pvg.constants import WANDB_CV_PROJECT

param_grid = dict(
    interaction_protocol=[InteractionProtocolType.PVG],
    dataset_name=["lrhammond/buggy-apps"],
    apps_difficulty=["interview"],
    num_iterations=[8],
    rollouts_per_iteration=[200],
    verifier_model=["gpt-4o-mini-2024-07-18"],
    verifier_temperature=[None],
    verifier_top_p=[None],
    verifier_guess_replacement_proportion=[0.0],
    verifier_guess_replacement_annealing=["linear"],
    verifier_guess_replacement_annealing_rate=[0.1],
    prover_model=["gpt-4o-2024-08-06"],
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
    min_message_rounds=[0],
    max_message_rounds=[9],
    verifier_first=[True],
    debate_sequential=[False],
    debate_prover0_first=[True],
    test_scheme=["none"],
    num_test_iterations=[1],
    rerun_tests=[None],
)


def _construct_params(combo: dict, cmd_args: Namespace) -> HyperParameters:

    agents_params_dict = dict(
        verifier=CodeValidationAgentParameters(
            model_name=combo["verifier_model"],
            temperature=combo["verifier_temperature"],
            top_p=combo["verifier_top_p"],
            use_dummy_api=cmd_args.use_dummy_api,
            fine_tune_from_scratch=combo["fine_tune_from_scratch"],
        ),
    )

    prover_params_dict = dict(
        model_name=combo["prover_model"],
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

    if combo["interaction_protocol"] in [
        InteractionProtocolType.PVG,
        InteractionProtocolType.ABSTRACT_DECISION_PROBLEM,
    ]:
        agents_params_dict["prover"] = CodeValidationAgentParameters(
            **prover_params_dict
        )
    elif combo["interaction_protocol"] in [
        InteractionProtocolType.DEBATE,
        InteractionProtocolType.MNIP,
        InteractionProtocolType.MERLIN_ARTHUR,
    ]:
        agents_params_dict["prover0"] = CodeValidationAgentParameters(
            **prover_params_dict
        )
        agents_params_dict["prover1"] = CodeValidationAgentParameters(
            **prover_params_dict
        )
    elif combo["interaction_protocol"] == InteractionProtocolType.SOLO_VERIFIER:
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
            wandb_project=WANDB_CV_PROJECT,
        )
    else:
        base_run_params = BaseRunParameters(base_run_type="none")

    return HyperParameters(
        scenario=ScenarioType.CODE_VALIDATION,
        trainer=TrainerType.PURE_TEXT_EI,
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
        code_validation=CodeValidationParameters(
            apps_difficulty=combo["apps_difficulty"],
        ),
        base_run=base_run_params,
    )


def experiment_fn(arguments: ExperimentFunctionArguments):
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
    hyper_params = _construct_params(combo, cmd_args)
    return prepare_experiment(
        hyper_params=hyper_params, ignore_cache=cmd_args.ignore_cache
    )


if __name__ == "__main__":

    experiment = SequentialHyperparameterExperiment(
        param_grid=param_grid,
        experiment_fn=experiment_fn,
        run_id_fn=run_id_fn,
        run_preparer_fn=run_preparer_fn,
        experiment_name="EI_VC",
        arg_parser_description="Run Code Validation experiments with Expert Iteration, "
        "running from a hyperparameter grid in sequence.",
        default_wandb_project=WANDB_CV_PROJECT,
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

    experiment.run()
