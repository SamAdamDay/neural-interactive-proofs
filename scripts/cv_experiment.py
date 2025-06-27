"""Script for running code validation experiments.

This script runs through a grid of hyperparameters, specified in the ``param_grid``
dict, and runs experiments for the code validation task for each. The grid specifies the
trainer to use, the interaction protocol and many other parameters.

Additional settings, like whether to log to W&B, the number of rollout workers to use,
and whether to use the dummy API, can be set via command line arguments. Run the script
with the ``--help`` flag to see all available arguments.
"""

from abc import ABC
from argparse import Namespace
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

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
    PureTextMaltParameters,
    CodeValidationParameters,
    BaseRunParameters,
    run_experiment,
    prepare_experiment,
    PreparedExperimentInfo,
    DatasetParameters,
    TrainerType,
    InteractionProtocolType,
    AppsDifficultyType,
    TestSchemeType,
    VerifierDecisionSpectrumType,
)
from nip.utils.experiments import (
    SequentialHyperparameterExperiment,
    ExperimentFunctionArguments,
    RunIDFunctionArguments,
)
from nip.utils.data import unflatten_dict_keys
from nip.utils.env import get_env_var

script_name = os.path.basename(__file__)
logger = logging.getLogger(f"nip.scripts.{script_name}")

scripts_dir = Path(__file__).parent
config_dir = scripts_dir / "config" / "cv_experiment"


class AgentConfig(BaseModel, ABC):
    """Base class for agent configuration.

    This class is used to define the common parameters for agents in the code validation
    experiment. It is inherited by specific agent configurations.

    These parameters map roughly onto the
    :class:`nip.parameters.agents.CodeValidationAgentParameters
    <CodeValidationAgentParameters>` class, with some differences for convenience.
    """

    model: str = "OpenAI/gpt-4o-mini-2024-07-18"
    system_prompt_template: str | None = None
    lm_server_scheme_host: str = "http://localhost"
    lm_server_port: int = 5000
    vllm_server_port: int = 8000
    temperature: float | None = None
    num_epochs: int = 3
    dpo_beta: float = 0.1
    lora_rank: int = 32
    lora_alpha_scale: float = 1.0
    lora_dropout: float = 0.05
    top_p: float | None = None
    repetition_penalty: float | None = None
    supervisor_message: str = "all"
    freeze: bool = False
    quantization: str = "none"


class VerifierConfig(AgentConfig):
    """Configuration for the verifier agent in the code validation experiment.

    This class inherits from :class:`AgentConfig` and adds specific parameters for the
    verifier agent.

    The extra config values map onto various hyper-parameters in the
    :class:`nip.parameters.HyperParameters <HyperParameters>`
    """

    guess_replacement_proportion: float = 0.0
    guess_replacement_annealing: str = "linear"
    guess_replacement_annealing_rate: float = 0.1
    decision_spectrum: VerifierDecisionSpectrumType = "accept_reject"


class ProverConfig(AgentConfig):
    """Configuration for the prover agent(s) in the code validation experiment.

    This class inherits from :class:`AgentConfig` and adds specific parameters for the
    prover agent(s).

    The extra config values map onto various hyper-parameters in the
    :class:`nip.parameters.HyperParameters <HyperParameters>`
    """

    max_words: int = 150
    invalid_response_penalty: float | None = None


class CodeValidationExperimentConfig(BaseModel):
    """Configuration for the code validation experiment.

    These parameters map roughly onto the :class:`nip.parameters.HyperParameters
    <HyperParameters>` class, with some differences for convenience.
    """

    trainer: TrainerType = "pure_text_ei"
    interaction_protocol: InteractionProtocolType = "nip"
    dataset_name: str = "lrhammond/buggy-apps"
    apps_difficulty: AppsDifficultyType = "interview"
    num_iterations: int = 1
    rollouts_per_iteration: int = 20
    learning_rate: float = 1e-6
    malt_num_initial_ei_iterations: int = 0
    malt_pair_selection_method: Literal["positive_negative", "interval"] = "interval"
    provers_share_model: bool = True
    fine_tune_from_scratch: bool = False
    fine_tune_on_all_previous_rollouts: bool = False
    rollout_selection_method: Literal["threshold", "weighted_sampling"] = "threshold"
    weighting_use_replacement: bool = True
    shared_reward: bool = False
    randomize_prover_stance: bool = False
    min_message_rounds: int = 1
    max_message_rounds: int = 9
    verifier_first: bool = True
    debate_sequential: bool = False
    debate_prover0_first: bool = True
    max_train_size: int | None = None
    max_test_size: int | None = None
    test_scheme: TestSchemeType = "none"
    num_test_iterations: int = 1
    test_dataset_split: str = "validation"
    rerun_tests: str | None = None
    """Which run ID to rerun tests from.

    If set, this should be the ID of a previous experiment. This experiment will be
    stepped through, running the tests as specified by "test_scheme" and other test
    parameters.
    """
    force_more_iterations: bool = False
    """Whether to force more iterations of an already finished run.

    If set to true and the script is run with a run ID which already exists, the script
    will continue running the experiment from the last completed iteration, even if the
    previous had fewer iterations than specified in this configuration file. This is
    useful for continuing promising experiments which have already completed.
    """
    seed: int = 6198

    verifier: VerifierConfig = VerifierConfig()
    """Configuration for the verifier agent."""
    prover: ProverConfig = ProverConfig()
    """Configuration for the prover agent(s)."""


def _construct_params(
    config: CodeValidationExperimentConfig, cmd_args: Namespace
) -> HyperParameters:
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

    verifier_model_provider, _, verifier_model_name = config.verifier.model.partition(
        "/"
    )
    prover_model_provider, _, prover_model_name = config.prover.model.partition("/")

    agents_params_dict = dict(
        verifier=CodeValidationAgentParameters(
            model_name=verifier_model_name,
            model_provider=verifier_model_provider,
            system_prompt_template_path=config.verifier.system_prompt_template,
            language_model_server_scheme_host=config.verifier.lm_server_scheme_host,
            language_model_server_port=config.verifier.lm_server_port,
            vllm_server_port=config.verifier.vllm_server_port,
            temperature=config.verifier.temperature,
            top_p=config.verifier.top_p,
            repetition_penalty=config.verifier.repetition_penalty,
            use_dummy_api=cmd_args.use_dummy_api,
            freeze_agent=config.verifier.freeze,
            fine_tune_from_scratch=config.fine_tune_from_scratch,
            use_supervisor_message=config.verifier.supervisor_message,
            dpo_beta=config.verifier.dpo_beta,
            lora_rank=config.verifier.lora_rank,
            lora_alpha_scale=config.verifier.lora_alpha_scale,
            lora_dropout=config.verifier.lora_dropout,
            quantization=config.verifier.quantization,
            num_epochs=config.verifier.num_epochs,
        ),
    )

    prover_params_dict = dict(
        model_name=prover_model_name,
        model_provider=prover_model_provider,
        system_prompt_template_path=config.prover.system_prompt_template,
        language_model_server_scheme_host=config.prover.lm_server_scheme_host,
        language_model_server_port=config.prover.lm_server_port,
        vllm_server_port=config.prover.vllm_server_port,
        temperature=config.prover.temperature,
        top_p=config.prover.top_p,
        repetition_penalty=config.prover.repetition_penalty,
        use_dummy_api=cmd_args.use_dummy_api,
        freeze_agent=config.prover.freeze,
        fine_tune_from_scratch=config.fine_tune_from_scratch,
        use_supervisor_message=config.prover.supervisor_message,
        dpo_beta=config.prover.dpo_beta,
        max_response_words=config.prover.max_words,
        lora_rank=config.prover.lora_rank,
        lora_alpha_scale=config.prover.lora_alpha_scale,
        lora_dropout=config.prover.lora_dropout,
        quantization=config.prover.quantization,
        num_epochs=config.prover.num_epochs,
    )

    if config.provers_share_model:
        prover_params_dict["shared_model_group"] = "provers_group"
    else:
        prover_params_dict["shared_model_group"] = None

    if config.interaction_protocol in ["nip", "adp"]:
        agents_params_dict["prover"] = CodeValidationAgentParameters(
            **prover_params_dict
        )
    elif config.interaction_protocol in [
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
    elif config.interaction_protocol == "solo_verifier":
        pass
    else:
        raise NotImplementedError(
            f"This script does not currently support the "
            f"{config.interaction_protocol} protocol."
        )

    if config.rerun_tests is not None:
        base_run_params = BaseRunParameters(
            base_run_type="rerun_tests",
            run_id=config.rerun_tests,
            wandb_project=get_env_var("WANDB_CV_PROJECT"),
        )
    else:
        base_run_params = BaseRunParameters(base_run_type="none")

    return HyperParameters(
        scenario="code_validation",
        trainer=config.trainer,
        dataset=config.dataset_name,
        test_dataset_split=config.test_dataset_split,
        rl=RlTrainerParameters(
            lr=config.learning_rate,
            rollouts_per_iteration=config.rollouts_per_iteration,
            frames_per_batch=None,
            num_iterations=config.num_iterations,
            num_test_iterations=config.num_test_iterations,
        ),
        text_rl=TextRlParameters(
            test_scheme=config.test_scheme,
            fine_tune_on_all_previous_rollouts=config.fine_tune_on_all_previous_rollouts,
            verifier_guess_replacement_proportion=config.verifier.guess_replacement_proportion,
            verifier_guess_replacement_annealing=config.verifier.guess_replacement_annealing,
            verifier_guess_replacement_annealing_rate=config.verifier.guess_replacement_annealing_rate,
        ),
        pure_text_ei=PureTextEiParameters(
            rollout_selection_method=config.rollout_selection_method,
            weighting_use_replacement=config.weighting_use_replacement,
        ),
        pure_text_malt=PureTextMaltParameters(
            num_initial_ei_iterations=config.malt_num_initial_ei_iterations,
            pair_selection_method=config.malt_pair_selection_method,
        ),
        agents=AgentsParameters(**agents_params_dict),
        interaction_protocol=config.interaction_protocol,
        protocol_common=CommonProtocolParameters(
            shared_reward=config.shared_reward,
            verifier_first=config.verifier_first,
            randomize_prover_stance=config.randomize_prover_stance,
            verifier_decision_spectrum=config.verifier.decision_spectrum,
            prover_invalid_response_penalty=config.prover.invalid_response_penalty,
        ),
        nip_protocol=NipProtocolParameters(
            min_message_rounds=config.min_message_rounds,
            max_message_rounds=config.max_message_rounds,
        ),
        debate_protocol=DebateProtocolParameters(
            min_message_rounds=config.min_message_rounds,
            max_message_rounds=config.max_message_rounds,
            sequential=config.debate_sequential,
            prover0_first=config.debate_prover0_first,
        ),
        code_validation=CodeValidationParameters(
            apps_difficulty=config.apps_difficulty,
        ),
        dataset_options=DatasetParameters(
            max_test_size=config.max_test_size,
        ),
        base_run=base_run_params,
        seed=config.seed,
    )


def experiment_fn(arguments: ExperimentFunctionArguments):
    """Run a single experiment.

    Parameters
    ----------
    arguments : ExperimentFunctionArguments
        The arguments for the experiment.
    """

    combo = unflatten_dict_keys(arguments.combo, separator=".")
    config = CodeValidationExperimentConfig(**combo)
    cmd_args = arguments.cmd_args

    logger.setLevel(arguments.log_level)

    logger.info(f"Starting run {arguments.run_id}")
    logger.debug(f"Combo: {config}")

    hyper_params = _construct_params(config, cmd_args)

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
        force_more_iterations=config.force_more_iterations,
        resume_if_safe=cmd_args.resume_if_safe,
    )


def run_id_fn(arguments: RunIDFunctionArguments) -> str:
    """Generate the run ID for a given hyperparameter combination.

    Parameters
    ----------
    arguments : RunIDFunctionArguments
        The arguments for generating the run ID, including:

        - combo_index: The index of the hyperparameter combination.
        - cmd_args: The command line arguments.
        - config_file_stem: The stem of the configuration file, if provided.

    Returns
    -------
    run_id : str
        The run ID.
    """

    cmd_args = arguments.cmd_args

    if cmd_args.run_infix == "" and cmd_args.use_dummy_api:
        run_infix = f"test_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    elif cmd_args.run_infix == "":
        raise ValueError(
            "When not using the dummy API, the run_infix argument must be provided."
        )
    else:
        run_infix = cmd_args.run_infix

    if arguments.combo_index is not None:
        run_suffix = f"{run_infix}_{arguments.combo_index}"
    else:
        run_suffix = f"{run_infix}"

    if arguments.config_file_stem is not None:
        return f"cv_{arguments.config_file_stem}_{run_suffix}"
    else:
        return f"cv_{run_suffix}"


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
    combo = unflatten_dict_keys(combo, separator=".")
    config = CodeValidationExperimentConfig(**combo)
    hyper_params = _construct_params(config, cmd_args)
    return prepare_experiment(
        hyper_params=hyper_params, ignore_cache=cmd_args.ignore_cache
    )


experiment = SequentialHyperparameterExperiment(
    experiment_fn=experiment_fn,
    run_id_fn=run_id_fn,
    run_preparer_fn=run_preparer_fn,
    experiment_name="CV",
    arg_parser_description="Run Code Validation experiments, "
    "running from a hyperparameter grid in sequence.",
    config_file_base_path=config_dir,
    default_wandb_project=get_env_var("WANDB_CV_PROJECT", ""),
    allow_resuming_wandb_run=True,
    add_run_infix_argument=False,
)

experiment.parser.add_argument(
    "run_infix",
    type=str,
    help="Infix to add to the run ID to distinguish between different runs. "
    "Defaults to 'test_{time_now}' when using dummy API; otherwise raises an error.",
    nargs="?",
    default="",
)

experiment.parser.add_argument(
    "--dummy",
    action="store_true",
    dest="use_dummy_api",
    help="Whether to use the dummy API for the agents. Useful for testing.",
)

experiment.parser.add_argument(
    "--resume-if-safe",
    action="store_true",
    dest="resume_if_safe",
    help="If the run already exists, whether to resume it if major version numbers "
    "match.",
)

# Set the ``parser`` module attribute to enable the script auto-documented by Sphinx
parser = experiment.parser

if __name__ == "__main__":
    experiment.run()
