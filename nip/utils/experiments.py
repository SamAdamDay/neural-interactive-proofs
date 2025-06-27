"""Experiment runners.

Contains utilities for running hyperparameter experiments. These can be run either
sequentially or in parallel using a pool of workers.

The workflow is as follows:

1. Call the constructor with the hyperparameter grid and the experiment function.
2. (Optional) Add any additional arguments to the arg parser using
   ``experiment.parser.add_argument``.
3. Call ``experiment.run()`` to run the experiment.

See the docstrings of the classes for more details.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from typing import Callable, Optional, Iterable
import textwrap
import os
from abc import ABC, abstractmethod
import logging
from functools import partial
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
import json

import json5

import yaml

from sklearn.model_selection import ParameterGrid

import torch

import wandb
from wandb import AlertLevel as WandbAlertLevel

from tqdm import tqdm

from pydantic import TypeAdapter

from tqdm_multiprocess.logger import setup_logger_tqdm
from tqdm_multiprocess import TqdmMultiProcessPool
from tqdm_multiprocess.std import init_worker

from nip.run import PreparedExperimentInfo
from nip.utils.env import get_env_var
from nip.utils.data import flatten_dict_keys
from nip.utils.types import ExperimentConfig
from nip.utils.logging import set_logger_handler


logger = logging.getLogger(__name__)


def _identity(string: str) -> str:
    """Return the input string.

    Hack to be able to pickle the command arguments.
    See: https://stackoverflow.com/a/71010038

    Parameters
    ----------
    string : str
        The input string.

    Returns
    -------
    string : str
        The input string.
    """
    return string


class PrefixLoggerAdapter(logging.LoggerAdapter):
    """A logger adapter that adds a prefix to the log message."""

    def __init__(self, logger: logging.Logger, prefix: str):
        super().__init__(logger, {})
        self.prefix = prefix

    def process(self, msg, kwargs):
        """Process the log message, adding the prefix."""
        return f"{self.prefix}{msg}", kwargs

    @property
    def level(self):
        """Get the log level of the logger."""
        return self.logger.level


class MultiLineFormatter(logging.Formatter):
    """Multi-line formatter.

    https://stackoverflow.com/a/66855071
    """

    def get_header_length(self, record):
        """Get the header length of a given record."""
        return len(
            super().format(
                logging.LogRecord(
                    name=record.name,
                    level=record.levelno,
                    pathname=record.pathname,
                    lineno=record.lineno,
                    msg="",
                    args=(),
                    exc_info=None,
                )
            )
        )

    def format(self, record):
        """Format a record with added indentation."""
        indent = " " * self.get_header_length(record)
        head, *trailing = super().format(record).splitlines(True)
        return head + "".join(indent + line for line in trailing)


class TqdmMultiProcessPoolMaxTasks(TqdmMultiProcessPool):
    """A TqdmMultiProcessPool that allows setting maxtasksperchild."""

    def __init__(self, process_count, max_tasks_per_child=None):
        self.mp_manager = multiprocessing.Manager()
        self.logging_queue = self.mp_manager.Queue()
        self.tqdm_queue = self.mp_manager.Queue()
        self.global_tqdm_queue = self.mp_manager.Queue()
        self.process_count = process_count
        worker_init_function = partial(init_worker, self.logging_queue)
        self.mp_pool = multiprocessing.Pool(
            self.process_count,
            worker_init_function,
            maxtasksperchild=max_tasks_per_child,
        )


@dataclass
class RunIDFunctionArguments:
    """Arguments to the function which determines the run name.

    Parameters
    ----------
    combo_index: int | None
        The index of the hyperparameter combination. If None, the run name will be for
        the whole experiment.
    cmd_args : Namespace
        The command line arguments.
    config_file_stem : str | None
        The stem of the config file, if given.
    """

    combo_index: int | None
    cmd_args: Namespace
    config_file_stem: str | None = None


@dataclass
class ExperimentFunctionArguments:
    """Arguments to the function which runs a single experiment.

    Parameters
    ----------
    combo : dict
        A single combination of hyperparameters.
    run_id : str
        A unique identifier for the run.
    cmd_args : Namespace
        The command line arguments.
    common_run_name : str
        A name for the experiment that is common to all runs.
    tqdm_func : callable
        A function used to create a tqdm progress bar.
    global_tqdm_step_fn : callable
        A function used to update the global progress bar.
    log_level : int
        The log level to use for the experiment. Can be used with ``logging.setLevel``.
    """

    combo: dict
    run_id: str
    cmd_args: Namespace
    common_run_name: str
    tqdm_func: callable
    global_tqdm_step_fn: callable = lambda: ...
    log_level: int = logging.INFO


class HyperparameterExperiment(ABC):
    """A base class to run an experiment over a grid of hyperparameters.

    Runs each combination of hyperparameters in the grid as a separate experiment.
    """

    def __init__(
        self,
        experiment_fn: Callable[[ExperimentFunctionArguments], None],
        param_grid: Optional[dict] = None,
        run_id_fn: Optional[Callable[[RunIDFunctionArguments], str]] = None,
        experiment_name: str = "EXPERIMENT",
        run_preparer_fn: Optional[
            Callable[[dict, Namespace], PreparedExperimentInfo]
        ] = None,
        arg_parser_description: str = "Run hyperparameter experiments",
        default_config_filename: Optional[str] = None,
        config_file_base_path: Optional[str | Path] = None,
        default_wandb_project: Optional[str] = None,
        allow_resuming_wandb_run: bool = False,
        add_run_infix_argument: bool = True,
    ):
        if run_id_fn is None:

            def run_id_fn(arguments: RunIDFunctionArguments) -> str:
                if arguments.combo_index is None:
                    return f"{experiment_name.lower()}"
                return f"{experiment_name.lower()}_{arguments.combo_index}"

        if param_grid is not None:
            self.param_grid = flatten_dict_keys(param_grid, separator=".")
        else:
            self.param_grid = None
        self.experiment_fn = experiment_fn
        self.run_id_fn = run_id_fn
        self.experiment_name = experiment_name
        self.run_preparer_fn = run_preparer_fn
        self.allow_resuming_wandb_run = allow_resuming_wandb_run

        if config_file_base_path is None:
            config_file_base_path = Path.cwd()
        self.config_file_base_path = Path(config_file_base_path)

        if default_wandb_project is None:
            default_wandb_project = get_env_var("WANDB_PROJECT", "")

        self.config_file_stem: Optional[str] = None

        # Set up the arg parser
        self.parser = ArgumentParser(
            description=arg_parser_description,
            formatter_class=ArgumentDefaultsHelpFormatter,
        )

        if self.param_grid is None:

            if default_config_filename is not None:
                default_kwarg = {"default": default_config_filename}
            else:
                default_kwarg = {}

            if config_file_base_path is not None:
                relative_text = f" (relative to {str(config_file_base_path)!r})"
            else:
                relative_text = ""

            self.parser.add_argument(
                "--config-file",
                "-c",
                type=str,
                help=f"The path to the file containing the hyperparameter grid"
                f"{relative_text}. The file extension can be omitted.",
                **default_kwarg,
            )

            self.parser.epilog = (
                "The config file should be a JSON, JSON5, or YAML file containing a "
                "dictionary with keys 'kind' and 'parameters'. If 'kind' is "
                "'single_experiment', then 'parameters' should be a dictionary "
                "with the hyperparameters to use. If 'kind' is 'grid', then "
                "'parameters' should be a dictionary with keys as hyperparameter names "
                "and values as lists of values to try."
            )

        # Add parser arguments for controlling logging output
        self.parser.add_argument(
            "-d", "--debug", help="Print debug messages", action="store_true"
        )
        self.parser.add_argument(
            "-q",
            "--quiet",
            help="Print less output (set log level to WARNING)",
            action="store_true",
        )

        # Add parser arguments for W&B
        if add_run_infix_argument:
            self.parser.add_argument(
                "--run-infix",
                type=str,
                help="The string to add in the middle of the run ID",
                default="a",
            )
        self.parser.add_argument(
            "--use-wandb",
            action="store_true",
            help="Whether to use W&B to log the experiment",
        )
        self.parser.add_argument(
            "--wandb-project",
            type=str,
            help="The name of the W&B project to use",
            default=default_wandb_project,
        )
        self.parser.add_argument(
            "--wandb-entity",
            type=str,
            help="The name of the W&B entity to use",
            default=get_env_var("WANDB_ENTITY", ""),
        )
        self.parser.add_argument(
            "--tag",
            type=str,
            default="",
            help="An optional tag for the W&B run",
        )

        # Other experiment settings
        self.parser.add_argument(
            "--gpu-num",
            type=int,
            help="The (0-based) number of the GPU to use",
            default=0,
        )
        self.parser.add_argument(
            "--ignore-cache",
            action="store_true",
            help="Ignore the dataset and model cache and rebuild from scratch.",
        )
        self.parser.add_argument(
            "--no-pretrain",
            dest="no_pretrain",
            action="store_true",
            help="Don't pretrain the agents, regardless of the hyperparameters",
        )

        # Create a logging formatter
        self.logging_formatter = MultiLineFormatter(
            fmt="[%(asctime)s %(levelname)s] %(message)s", datefmt="%x %X"
        )

        self.cmd_args: Optional[Namespace] = None

    @abstractmethod
    def _run(self):
        """Run the experiment.

        This is the function that actually runs the experiment, and should be
        implemented by subclasses.
        """
        pass

    @property
    def common_run_name(self) -> str:
        """A name for the experiment that is common to all runs."""
        if self.cmd_args is None:
            raise ValueError("The command line arguments have not been parsed yet.")
        return self.run_id_fn(
            RunIDFunctionArguments(
                combo_index=None,
                cmd_args=self.cmd_args,
                config_file_stem=self.config_file_stem,
            )
        )

    @property
    def combinations(self) -> ParameterGrid:
        """An iterator over the combinations of hyperparameters."""
        return ParameterGrid(self.param_grid)

    @property
    def enumerated_combinations(self) -> Iterable[tuple[int, dict]]:
        """An iterator over the combinations of hyperparameters plus an enumeration."""
        return enumerate(self.combinations)

    def check_no_extant_runs(self):
        """Make sure there are no runs with the same ID as any run in this experiment.

        Raises
        ------
        ValueError
            If there is a run with the same ID as any run in this experiment.
        """

        if self.cmd_args.use_wandb and not self.allow_resuming_wandb_run:

            api = wandb.Api()

            # Get the names of the runs we'll be running
            num_combinations = len(list(self.combinations))
            run_names = [
                self.run_id_fn(
                    RunIDFunctionArguments(
                        combo_index=i,
                        cmd_args=self.cmd_args,
                        config_file_stem=self.config_file_stem,
                    )
                )
                for i in range(num_combinations)
            ]

            # Check if any already exist
            runs = api.runs(
                path=f"{self.cmd_args.wandb_entity}/{self.cmd_args.wandb_project}",
                filters={"$or": [{"name": run_name} for run_name in run_names]},
            )
            try:
                first_run = runs[0]
            except IndexError:
                pass
            else:
                raise ValueError(
                    f"A run with the ID {first_run.id} already exists in the project"
                )

    def run(self):
        """Run the experiment."""

        self.cmd_args = self.parser.parse_args()

        if "config_file" in self.cmd_args:
            if self.cmd_args.config_file is not None:
                self._load_param_grid(self.cmd_args.config_file)
            else:
                self.param_grid = {}

        self.check_no_extant_runs()

        setup_logger_tqdm(formatter=self.logging_formatter)

        if self.cmd_args.debug:
            self.experiment_log_level = logging.DEBUG
        elif not self.cmd_args.quiet:
            self.experiment_log_level = logging.INFO
        else:
            self.experiment_log_level = logging.WARNING

        self._run()

        # Send a W&B alert to say the experiment is finished
        if self.cmd_args.use_wandb:
            os.environ["WANDB_SILENT"] = "true"
            dummy_run = wandb.init(
                id=get_env_var("WANDB_DUMMY_RUN_NAME"),
                project=get_env_var("WANDB_DUMMY_RUN_PROJECT"),
                entity=get_env_var("WANDB_DUMMY_RUN_ENTITY"),
            )
            wandb.alert(
                title=f"{self.common_run_name} finished",
                text=(
                    f"The hyperparameter experiment for {self.experiment_name!r}"
                    f" has finished."
                ),
                level=WandbAlertLevel.INFO,
            )
            dummy_run.finish()

    def _load_param_grid(self, config_filename: str):
        """Load the hyperparameter grid from a config file.

        Parameters
        ----------
        config_filename : str
            The path to the config file given as a command line argument. If this is an
            absolute path, it will be used as is. If it is a relative path, it will be
            resolved relative to the config_file_base_path given in the constructor.

        Raises
        ------
        ValueError
            If the config file format is not supported, or the content of the file
            does not match the expected format.
        """

        file_path = self.config_file_base_path / config_filename
        file_extension = file_path.suffix.lower()

        extension_loaders = {
            ".json": json.load,
            ".json5": json5.load,
            ".yaml": yaml.safe_load,
            ".yml": yaml.safe_load,
        }

        for extension, loader in extension_loaders.items():
            if file_extension == extension:
                with open(file_path, "r") as f:
                    config: ExperimentConfig = loader(f)
                self.config_file_stem = file_path.stem
                break
        else:
            for extension, loader in extension_loaders.items():
                test_file_path = file_path.with_name(f"{config_filename}{extension}")
                if test_file_path.exists():
                    with open(test_file_path, "r") as f:
                        config: ExperimentConfig = loader(f)
                    self.config_file_stem = config_filename
                    break
            else:
                raise ValueError(
                    f"Config file {config_filename!r} is not valid, nor could a valid "
                    f"file be found after appending any of the supported extensions: "
                    f"{', '.join(extension_loaders.keys())}."
                )

        TypeAdapter(ExperimentConfig).validate_python(config)

        if "kind" not in config or "parameters" not in config:
            raise ValueError(
                "The config file must contain a dictionary with keys 'kind' and "
                "'parameters'."
            )
        if not isinstance(config["parameters"], dict):
            raise ValueError(
                "The 'parameters' key in the config file must be a dictionary."
            )

        flattened_param_grid = flatten_dict_keys(config["parameters"], separator=".")

        if config["kind"] == "single_experiment":
            self.param_grid = {
                key: [value] for key, value in flattened_param_grid.items()
            }
        elif config["kind"] == "grid":
            self.param_grid = flattened_param_grid
        else:
            raise ValueError(
                f"Unsupported config file kind: {config['kind']!r}. "
                "Supported kinds are: 'single_experiment', 'grid'."
            )

    def _setup_logger(self, combo_index: int, num_combos: int):
        """Set up the logger for a single run.

        Parameters
        ----------
        combo_index : int
            The index of the current combination in the list of combinations.
        num_combos : int
            The total number of combinations.
        """

        # The root logger is set to WARNING, so that we don't get too much output from
        # other packages. The package-level logger is configured separately.
        logging.getLogger().setLevel(logging.WARNING)

        package_logger = logging.getLogger(__name__.partition(".")[0])
        package_logger.setLevel(self.experiment_log_level)
        package_logger.propagate = False

        handler = logging.StreamHandler()

        info_prefix = f"[{combo_index+1}/{num_combos}]"
        formatter = MultiLineFormatter(
            fmt=f"[%(asctime)s %(levelname)s] {info_prefix} %(message)s",
            datefmt="%x %X",
        )
        handler.setFormatter(formatter)

        set_logger_handler(package_logger, handler)


class SequentialHyperparameterExperiment(HyperparameterExperiment):
    """A class to run an experiment over a grid of hyperparameters in sequence.

    Runs each combination of hyperparameters in the grid as a separate experiment. If
    there is an error in one of the experiments, all subsequent experiments are skipped.

    A summary of the results is printed at the end.

    The workflow is as follows:

    1. Call the constructor with the hyperparameter grid and the experiment function.
    2. (Optional) Add any additional arguments to the arg parser using
       ``self.parser.add_argument``.
    3. Call ``self.run()`` to run the experiment.

    Parameters
    ----------
    experiment_fn : Callable[[ExperimentFunctionArguments], None]
        A function that takes a single hyperparameter combination and runs the
        experiment. The arguments are specified in the
        :class:`ExperimentFunctionArguments` dataclass.
    param_grid : dict, optional
        A dictionary mapping hyperparameter names to lists of values to try. If not
        given, a positional argument "config_file" will be added to the parser, and this
        will be used to load the hyperparameter grid from a config file.
    run_id_fn : Callable[[RunIDFunctionArguments], str], optional
        A function that takes a single hyperparameter combination and returns a unique
        identifier for the run. If None, the default is to use the experiment name and
        the combination index. The arguments are specified in the
        :class:`RunIDFunctionArguments` dataclass.
    run_preparer_fn : Callable[[dict, Namespace], PreparedExperimentInfo], optional
        A function that takes a single hyperparameter combination and prepares the run
        for it. It should return a ``PreparedExperimentInfo`` instance. This is
        optional. It should take the form:

        .. code-block:: python

            run_preparer_fn(combo, cmd_args)

        where ``combo`` is a single combination of hyperparameters and ``cmd_args`` is
        the command line arguments.
    experiment_name : str, default="EXPERIMENT"
        The name of the experiment.
    arg_parser_description : str, default="Run hyperparameter experiments sequentially"
        The description of the argument parser.
    default_config_filename : Optional[str], default=None
        The default config filename to use if the param_grid is not given. If None, no
        default is set, and if the user does not provide a config file, an empty grid is
        used.
    config_file_base_path : Optional[str | Path], default=None
        The base path to use for the config file. If None, the current working directory
        is used. If the config file is specified as a relative path, it will be resolved
        relative to this base path. If this is an absolute path, it will be used as is.
    default_wandb_project : Optional[str], default=None
        The default W&B project to use. If None, the default is to use the WANDB_PROJECT
        environment variable.
    allow_resuming_wandb_run : bool, default=False
        Whether to allow resuming a W&B run with the same ID as a run in this
        experiment.
    add_run_infix_argument : bool, default=True
        Whether to add an argument to the parser for adding an infix to the run ID.
    output_width : int, default=70
        The width of the output to print (after the logging prefix).
    """

    def __init__(
        self,
        experiment_fn: Callable[
            [dict, str, Namespace, Callable, logging.LoggerAdapter, str], None
        ],
        param_grid: Optional[dict] = None,
        run_id_fn: Optional[Callable[[RunIDFunctionArguments], str]] = None,
        run_preparer_fn: Optional[
            Callable[[dict, Namespace], PreparedExperimentInfo]
        ] = None,
        experiment_name: str = "EXPERIMENT",
        arg_parser_description: str = "Run hyperparameter experiments sequentially",
        default_config_filename: Optional[str] = None,
        config_file_base_path: Optional[str | Path] = None,
        default_wandb_project: Optional[str] = None,
        allow_resuming_wandb_run: bool = False,
        add_run_infix_argument: bool = True,
        output_width: int = 70,
    ):
        super().__init__(
            param_grid=param_grid,
            experiment_fn=experiment_fn,
            run_id_fn=run_id_fn,
            experiment_name=experiment_name,
            run_preparer_fn=run_preparer_fn,
            arg_parser_description=arg_parser_description,
            config_file_base_path=config_file_base_path,
            default_config_filename=default_config_filename,
            default_wandb_project=default_wandb_project,
            allow_resuming_wandb_run=allow_resuming_wandb_run,
            add_run_infix_argument=add_run_infix_argument,
        )

        self.output_width = output_width

        # Add various arguments
        self.parser.add_argument(
            "--combo-groups",
            type=int,
            default=1,
            help="Into how many groups to split the experiment combinations",
        )
        self.parser.add_argument(
            "--combo-num",
            type=int,
            default=0,
            help="Which combo group to run this time",
        )
        self.parser.add_argument(
            "--num-skip",
            type=int,
            default=0,
            help="The number of initial combos to skip. Useful to resume a group",
        )

    def _run_single_experiment(
        self,
        combinations: list[tuple[int, dict]],
        combo: dict,
        combo_index: int,
        cmd_args: Namespace,
    ) -> bool:
        """Run an experiment for a single combination of hyperparameters."""

        # Create a unique run_id for this run
        run_id = self.run_id_fn(
            RunIDFunctionArguments(
                combo_index=combo_index,
                cmd_args=cmd_args,
                config_file_stem=self.config_file_stem,
            )
        )

        self._setup_logger(combo_index, len(combinations))

        # Print the run_id and the hyper-parameters
        logger.info("")
        logger.info("=" * self.output_width)
        title = f"| {self.experiment_name} | Run ID: {run_id}"
        title += (" " * (self.output_width - 1 - len(title))) + "|"
        title = textwrap.fill(title, self.output_width)
        logger.info(title)
        logger.info("=" * self.output_width)

        # Run the experiment
        self.experiment_fn(
            ExperimentFunctionArguments(
                combo=combo,
                run_id=run_id,
                cmd_args=cmd_args,
                tqdm_func=tqdm,
                common_run_name=self.common_run_name,
                log_level=self.experiment_log_level,
            )
        )

        return True

    def _run(self):
        cmd_args = self.cmd_args

        # Filter to combos
        combinations = self.enumerated_combinations
        combinations = filter(
            lambda x: x[0] % cmd_args.combo_groups == cmd_args.combo_num, combinations
        )
        combinations = list(combinations)[cmd_args.num_skip :]

        # Prepare the runs
        if self.run_preparer_fn is not None:
            for i, combo in combinations:
                self.run_preparer_fn(combo, cmd_args)

        # Keep track of the results of the runs
        run_results = []
        for combo_num in range(len(combinations)):
            run_results.append("SKIPPED")

        try:
            # Run the experiment for each sampled combination of parameters
            for i, (combo_index, combo) in enumerate(combinations):
                # Set the status of the current run to failed until proven otherwise
                run_results[i] = "FAILED"

                self._run_single_experiment(combinations, combo, combo_index, cmd_args)

                run_results[i] = "SUCCEEDED"

        finally:
            # Print a summary of the experiment results
            logger.info("")
            logger.info("")
            logger.info("=" * self.output_width)
            title = f"| SUMMARY | GROUP {cmd_args.combo_num}/{cmd_args.combo_groups}"
            title += (" " * (self.output_width - 1 - len(title))) + "|"
            title = textwrap.fill(title, self.output_width)
            logger.info(title)
            logger.info("=" * self.output_width)
            for result, (combo_num, combo) in zip(run_results, combinations):
                logger.info("")
                logger.info(f"COMBO {combo_num}")
                logger.info(textwrap.fill(str(combo)))
                logger.info(result)


class MultiprocessHyperparameterExperiment(HyperparameterExperiment):
    """A class to run an experiment over a grid of hyperparameters in parallel.

    Runs each combination of hyperparameters in the grid as a separate experiment using
    a pool of workers.

    The workflow is as follows:

    1. Call the constructor with the hyperparameter grid and the experiment function.
    2. (Optional) Add any additional arguments to the arg parser using
       ``self.parser.add_argument``.
    3. Call ``self.run()`` to run the experiment.

    Parameters
    ----------
    experiment_fn : Callable[[ExperimentFunctionArguments], None]
        A function that takes a single hyperparameter combination and runs the
        experiment. The arguments are specified in the ``ExperimentFunctionArguments``
        dataclass.
    param_grid : dict, optional
        A dictionary mapping hyperparameter names to lists of values to try. If not
        given, a positional argument "config_file" will be added to the parser,
        and this will be used to load the hyperparameter grid from a config file.
    run_id_fn : Callable[[RunIDFunctionArguments], str], optional
        A function that takes a single hyperparameter combination and returns a unique
        identifier for the run. If None, the default is to use the experiment name and
        the combination index. The arguments are specified in the
        :class:`RunIDFunctionArguments` dataclass.
    run_preparer_fn : Callable[[dict, Namespace], PreparedExperimentInfo], optional
        A function that takes a single hyperparameter combination and prepares the run
        for it. It should return a ``PreparedExperimentInfo`` instance. This is
        optional. It should take the form:

        .. code-block:: python

            run_preparer_fn(combo, cmd_args)

        where ``combo`` is a single combination of hyperparameters and ``cmd_args`` is
        the command line arguments.
    experiment_name : str, default="EXPERIMENT"
        The name of the experiment.
    arg_parser_description : str, default="Run hyperparameter experiments in parallel"
        The description of the argument parser.
    default_config_filename : Optional[str], default=None
        The default config filename to use if the param_grid is not given. If None, no
        default is set, and if the user does not provide a config file, an empty grid
        is used.
    config_file_base_path : Optional[str | Path], default=None
        The base path to use for the config file. If None, the current working
        directory is used. If the config file is specified as a relative path, it
        will be resolved relative to this base path. If this is an absolute path, it
        will be used as is.
    default_wandb_project : Optional[str], default=None
        The default W&B project to use. If None, the default is to use the WANDB_PROJECT
        environment variable.
    allow_resuming_wandb_run : bool, default=False
        Whether to allow resuming a W&B run with the same ID as a run in this
        experiment.
    add_run_infix_argument : bool, default=True
        Whether to add an argument to the parser for adding an infix to the run ID.
    default_num_workers : int, default=1
        The default number of workers to use for multiprocessing.
    """

    def __init__(
        self,
        experiment_fn: Callable[[ExperimentFunctionArguments], None],
        param_grid: Optional[dict] = None,
        run_id_fn: Optional[Callable[[RunIDFunctionArguments], str]] = None,
        run_preparer_fn: Optional[
            Callable[[dict, Namespace], PreparedExperimentInfo]
        ] = None,
        experiment_name: str = "EXPERIMENT",
        arg_parser_description: str = "Run hyperparameter experiments in parallel",
        default_config_filename: Optional[str] = None,
        config_file_base_path: Optional[str | Path] = None,
        default_wandb_project: Optional[str] = None,
        allow_resuming_wandb_run: bool = False,
        add_run_infix_argument: bool = True,
        default_num_workers: int = 1,
    ):
        super().__init__(
            param_grid=param_grid,
            experiment_fn=experiment_fn,
            run_id_fn=run_id_fn,
            experiment_name=experiment_name,
            run_preparer_fn=run_preparer_fn,
            arg_parser_description=arg_parser_description,
            config_file_base_path=config_file_base_path,
            default_config_filename=default_config_filename,
            default_wandb_project=default_wandb_project,
            allow_resuming_wandb_run=allow_resuming_wandb_run,
            add_run_infix_argument=add_run_infix_argument,
        )

        # Needed so that we can pickle the command arguments
        # https://stackoverflow.com/a/71010038
        self.parser.register("type", None, _identity)

        # Add various arguments
        self.parser.add_argument(
            "--num-workers",
            type=int,
            default=default_num_workers,
            help="The number of workers to use for multiprocessing",
        )
        self.parser.add_argument(
            "--max-tasks-per-child",
            type=int,
            default=1,
            help=(
                "The maximum number of tasks each worker can run before being replaced"
            ),
        )
        self.parser.add_argument(
            "--num-skip",
            type=int,
            default=0,
            help="The number of initial tasks to skip. Useful to resume an experiment",
        )

    def _task_fn(
        self,
        combinations: list[dict],
        combo_index: int,
        cmd_args: Namespace,
        fine_grained_global_tqdm: bool,
        tqdm_func: Callable,
        global_tqdm: tqdm,
    ) -> bool:
        """Run a task on a single worker.

        Parameters
        ----------
        combinations : list[dict]
            The list of combinations of hyperparameters.
        combo_index : int
            The index of the current combination.
        cmd_args : Namespace
            The command line arguments.
        fine_grained_global_tqdm : bool
            Whether to update the global progress bar after each iteration. If False,
            the global progress bar is only updated after each experiment is finished.
        tqdm_func : Callable
            The tqdm function to use in the experiment to create new progress bars. This
            argument is provided by ``tqdm_multiprocess``.
        global_tqdm : tqdm
            The global progress bar. This argument is provided by ``tqdm_multiprocess``.
        """

        # Create a unique run_id for this run
        run_id = self.run_id_fn(
            RunIDFunctionArguments(
                combo_index=combo_index,
                cmd_args=cmd_args,
                config_file_stem=self.config_file_stem,
            )
        )

        self._setup_logger(combo_index, len(combinations))

        # The tqdm function to use. Set the leave argument to False because otherwise
        # tqdm doesn't display multiple progress bars properly due to a bug
        # https://github.com/tqdm/tqdm/issues/1496
        info_prefix = f"[{combo_index+1}/{len(combinations)}] "
        tqdm_func = partial(
            tqdm_func,
            leave=False,
            bar_format=info_prefix + "{desc}: {percentage:3.0f}%|{bar}{r_bar}",
        )

        if fine_grained_global_tqdm:

            def global_tqdm_step_fn():
                global_tqdm.update(1)

        else:

            def global_tqdm_step_fn():
                pass

        # Run the experiment
        self.experiment_fn(
            ExperimentFunctionArguments(
                combo=combinations[combo_index],
                run_id=run_id,
                cmd_args=cmd_args,
                tqdm_func=tqdm_func,
                global_tqdm_step_fn=global_tqdm_step_fn,
                common_run_name=self.common_run_name,
                log_level=self.experiment_log_level,
            )
        )

        # Update the global progress bar if we're not doing it after each iteration
        if not fine_grained_global_tqdm:
            global_tqdm.update(1)

        # Log that this run is finished
        logger.info(f"{info_prefix}{run_id} finished")

        return True

    def _run(self):
        cmd_args = self.cmd_args

        # Set the torch multiprocessing start method to spawn, to avoid issues with CUDA
        torch.multiprocessing.set_start_method("spawn", force=True)

        # Set up Weights & Biases on this process. Later, each worker will init its own
        # W&B run.
        if cmd_args.use_wandb:
            wandb.setup()

        # Get all configurations of hyperparameters, and turn this into a list of tasks
        combinations = list(self.combinations)

        # Prepare the runs and compute the total number of iterations
        if self.run_preparer_fn is not None:
            total_iterations = 0
            for combo in combinations:
                info = self.run_preparer_fn(combo, cmd_args)
                total_iterations += info.total_num_iterations
            fine_grained_global_tqdm = True
        else:
            total_iterations = len(combinations)
            fine_grained_global_tqdm = False

        # Create a list of tasks
        tasks = [
            (
                self._task_fn,
                (
                    combinations,
                    combo_index,
                    cmd_args,
                    fine_grained_global_tqdm,
                ),
            )
            for combo_index in range(len(combinations))
        ]
        tasks = tasks[cmd_args.num_skip :]

        # Create a pool of workers
        pool = TqdmMultiProcessPoolMaxTasks(
            cmd_args.num_workers, max_tasks_per_child=cmd_args.max_tasks_per_child
        )

        with tqdm(
            total=total_iterations, dynamic_ncols=True, miniters=1, smoothing=0.1
        ) as global_progress:
            global_progress.set_description("Total progress")
            pool.map(global_progress, tasks, lambda x: None, lambda x: None)
