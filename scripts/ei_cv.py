"""Script for running Expert Iteration with the code validation task."""

from argparse import Namespace
import os
import logging


from pvg import (
    Parameters,
    AgentsParameters,
    CodeValidationAgentParameters,
    RlTrainerParameters,
    ScenarioType,
    TrainerType,
    InteractionProtocolType,
    CommonProtocolParameters,
    PvgProtocolParameters,
    DebateProtocolParameters,
    PureTextEiParameters,
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
    num_iterations=[10],
    rollouts_per_iteration=[200],
    verifier_model=["gpt-4o-mini-2024-07-18"],
    verifier_temperature=[None],
    verifier_top_p=[None],
    prover_model=["gpt-4o-2024-08-06"],
    prover_temperature=[None],
    prover_top_p=[None],
    freeze_prover=[False],
    fine_tune_from_scratch=[False],
    shared_reward=[False],
    min_message_rounds=[0],
    max_message_rounds=[9],
    verifier_first=[True],
    debate_sequential=[False],
    debate_prover0_first=[True],
    run_test_loop=[False],
    use_dummy_api=[False],
)


def _construct_params(combo: dict, cmd_args: Namespace) -> Parameters:

    agents_params_dict = dict(
        verifier=CodeValidationAgentParameters(
            model_name=combo["verifier_model"],
            temperature=combo["verifier_temperature"],
            top_p=combo["verifier_top_p"],
            use_dummy_api=combo["use_dummy_api"],
            fine_tune_from_scratch=combo["fine_tune_from_scratch"],
        ),
    )

    prover_params_dict = dict(
        model_name=combo["prover_model"],
        temperature=combo["prover_temperature"],
        top_p=combo["prover_top_p"],
        use_dummy_api=combo["use_dummy_api"],
        freeze_agent=combo["freeze_prover"],
        fine_tune_from_scratch=combo["fine_tune_from_scratch"],
    )

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
    else:
        raise NotImplementedError(
            f"This script does not currently support the "
            f"{combo['interaction_protocol']} protocol."
        )

    return Parameters(
        scenario=ScenarioType.CODE_VALIDATION,
        trainer=TrainerType.PURE_TEXT_EI,
        dataset=combo["dataset_name"],
        rl=RlTrainerParameters(
            rollouts_per_iteration=combo["rollouts_per_iteration"],
            frames_per_batch=None,
            num_iterations=combo["num_iterations"],
        ),
        pure_text_ei=PureTextEiParameters(
            run_test_loop=combo["run_test_loop"],
        ),
        agents=AgentsParameters(**agents_params_dict),
        interaction_protocol=combo["interaction_protocol"],
        protocol_common=CommonProtocolParameters(
            shared_reward=combo["shared_reward"],
            verifier_first=combo["verifier_first"],
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
    )


def experiment_fn(arguments: ExperimentFunctionArguments):
    combo = arguments.combo
    cmd_args = arguments.cmd_args
    logger = arguments.child_logger_adapter
    logger.setLevel(logging.INFO)

    logger.info(f"Starting run {arguments.run_id}")
    logger.debug(f"Combo: {combo}")

    params = _construct_params(combo, cmd_args)

    # Make sure W&B doesn't print anything when the logger level is higher than DEBUG
    if logger.level > logging.DEBUG:
        os.environ["WANDB_SILENT"] = "true"

    if cmd_args.use_wandb:
        wandb_tags = [cmd_args.tag] if cmd_args.tag != "" else []
    else:
        wandb_tags = []

    # Train and test the agents
    run_experiment(
        params,
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
        num_rollout_workers=cmd_args.num_rollout_workers,
    )


def run_id_fn(combo_index: int | None, cmd_args: Namespace) -> str:
    if combo_index is None:
        return f"ei_vc_{cmd_args.run_infix}"
    return f"ei_vc_{cmd_args.run_infix}_{combo_index}"


def run_preparer_fn(combo: dict, cmd_args: Namespace) -> PreparedExperimentInfo:
    params = _construct_params(combo, cmd_args)
    return prepare_experiment(params=params, ignore_cache=cmd_args.ignore_cache)


if __name__ == "__main__":

    experiment = SequentialHyperparameterExperiment(
        param_grid=param_grid,
        experiment_fn=experiment_fn,
        run_id_fn=run_id_fn,
        run_preparer_fn=run_preparer_fn,
        experiment_name="EI_VC",
        default_wandb_project=WANDB_CV_PROJECT,
        allow_resuming_wandb_run=True,
    )

    experiment.parser.add_argument(
        "--num-rollout-workers",
        type=int,
        default=4,
        help="Number of workers to use for sampling rollouts.",
    )

    experiment.run()
