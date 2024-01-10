"""Experiment runners.

Contains utilities for running hyperparameter experiments. These can be run either
sequentially or in parallel using a pool of workers. 

The workflow is as follows:

1. Call the constructor with the hyperparameter grid and the experiment function.
2. (Optional) Add any additional arguments to the arg parser using
   `experiment.parser.add_argument`.
3. Call `experiment.run()` to run the experiment.

See the docstrings of the classes for more details.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from typing import Callable, Optional
import textwrap
import os
from abc import ABC, abstractmethod
import logging
from functools import partial

from sklearn.model_selection import ParameterGrid

import torch

from tqdm import tqdm

from tqdm_multiprocess.logger import setup_logger_tqdm
from tqdm_multiprocess import TqdmMultiProcessPool


# Hack to be able to pickle the command arguments
# https://stackoverflow.com/a/71010038
def identity(string):
    return string


class PrefixLoggerAdapter(logging.LoggerAdapter):
    """A logger adapter that adds a prefix to the log message."""

    def __init__(self, logger: logging.Logger, prefix: str):
        super().__init__(logger, {})
        self.prefix = prefix

    def process(self, msg, kwargs):
        return f"{self.prefix}{msg}", kwargs

    @property
    def level(self):
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


class HyperparameterExperiment(ABC):
    """A base class to run an experiment over a grid of hyperparameters.

    Runs each combination of hyperparameters in the grid as a separate experiment.
    """

    def __init__(
        self,
        param_grid: dict,
        experiment_fn: Callable[
            [dict, str, Namespace, Callable, logging.LoggerAdapter], None
        ],
        run_id_fn: Optional[Callable[[int, Namespace], str]] = None,
        experiment_name: str = "EXPERIMENT",
        arg_parser_description: str = "Run hyperparameter experiments",
    ):
        if run_id_fn is None:
            run_id_fn = (
                lambda combo_index, _: f"{experiment_name.lower()}_{combo_index}"
            )

        self.param_grid = param_grid
        self.experiment_fn = experiment_fn
        self.run_id_fn = run_id_fn
        self.experiment_name = experiment_name

        # Set up the arg parser
        self.parser = ArgumentParser(
            description=arg_parser_description,
            formatter_class=ArgumentDefaultsHelpFormatter,
        )

        # Add parser arguments for controlling logging output
        self.parser.add_argument(
            "-d", "--debug", help="Print debug messages", action="store_true"
        )
        self.parser.add_argument(
            "-v",
            "--verbose",
            help="Print additional info messages",
            action="store_true",
        )

        # Create a logging formatter
        self.logging_formatter = MultiLineFormatter(
            fmt="[%(asctime)s %(levelname)s] %(message)s", datefmt="%x %X"
        )

    @abstractmethod
    def _run(self, cmd_args: Namespace, base_logger: logging.Logger):
        """The function that actually runs the experiment, to be implemented."""
        pass

    def run(self):
        """Run the experiment."""

        # Get the arguments
        cmd_args = self.parser.parse_args()

        # Set up the logger
        base_logger = logging.getLogger(__name__)
        setup_logger_tqdm(formatter=self.logging_formatter)

        # Set the log level inside the experiment function
        if cmd_args.debug:
            self.experiment_log_level = logging.DEBUG
        elif cmd_args.verbose:
            self.experiment_log_level = logging.INFO
        else:
            self.experiment_log_level = logging.WARNING

        # Run the experiment
        self._run(cmd_args, base_logger)


class SequentialHyperparameterExperiment(HyperparameterExperiment):
    """A class to run an experiment over a grid of hyperparameters in sequence.

    Runs each combination of hyperparameters in the grid as a separate experiment. If
    there is an error in one of the experiments, all subsequent experiments are skipped.

    A summary of the results is printed at the end.

    The workflow is as follows:

    1. Call the constructor with the hyperparameter grid and the experiment function.
    2. (Optional) Add any additional arguments to the arg parser using
       `self.parser.add_argument`.
    3. Call `self.run()` to run the experiment.

    Parameters
    ----------
    param_grid : dict
        A dictionary mapping hyperparameter names to lists of values to try.
    experiment_fn : Callable[[dict, str, Namespace, Callable, logging.LoggerAdapter],
    None]
        A function that takes a single hyperparameter combination and runs the
        experiment. It should take the form:
            experiment_fn(combo, run_id, cmd_args, tqdm_func, child_logger_adapter)
        where `combo` is a single combination of hyperparameters, `run_id` is a unique
        identifier for the run, `cmd_args` is the command line arguments, `tqdm_func` is
        a function used to create a tqdm progress bar, and `child_logger_adapter` is a
        logger adapter to use for logging.
    run_id_fn : Callable[[int, Namespace], str], optional
        A function that takes a single hyperparameter combination and returns a unique
        identifier for the run. If None, the default is to use the experiment name and
        the combination index. It should take the form:
            run_id_fn(combo_index, cmd_args)
        where `combo_index` is the index of the combination in the ParameterGrid and
        `cmd_args` is the command line arguments.
    experiment_name : str, default="EXPERIMENT"
        The name of the experiment.
    output_width : int, default=70
        The width of the output to print (after the logging prefix).
    """

    def __init__(
        self,
        param_grid: dict,
        experiment_fn: Callable[
            [dict, str, Namespace, Callable, logging.LoggerAdapter], None
        ],
        run_id_fn: Optional[Callable[[int, Namespace], str]] = None,
        experiment_name: str = "EXPERIMENT",
        output_width: int = 70,
    ):
        super().__init__(
            param_grid=param_grid,
            experiment_fn=experiment_fn,
            run_id_fn=run_id_fn,
            experiment_name=experiment_name,
            arg_parser_description="Run hyperparameter experiments sequentially",
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
        base_logger: logging.Logger,
    ) -> bool:
        """Run an experiment for a single combination of hyperparameters."""

        info_prefix = f"[{combo_index}/{len(combinations)}] "

        # Create a unique run_id for this run
        run_id = self.run_id_fn(combo_index, cmd_args)

        # Set up the logger
        child_logger = logging.getLogger(f"{base_logger.name}.{run_id}")
        child_logger.setLevel(self.experiment_log_level)
        child_logger_adapter = PrefixLoggerAdapter(child_logger, info_prefix)

        # The tqdm function to use
        tqdm_func = partial(
            tqdm,
            bar_format=info_prefix + "{desc}: {percentage:3.0f}%|{bar}{r_bar}",
        )

        # Print the run_id and the Parameters
        base_logger.info("")
        base_logger.info("=" * self.output_width)
        title = f"| {self.experiment_name} | Run ID: {run_id}"
        title += (" " * (self.output_width - 1 - len(title))) + "|"
        title = textwrap.fill(title, self.output_width)
        base_logger.info(title)
        base_logger.info("=" * self.output_width)

        # Run the experiment
        self.experiment_fn(
            combo,
            run_id,
            cmd_args,
            tqdm_func,
            child_logger_adapter,
        )

        return True

    def _run(self, cmd_args: Namespace, base_logger: logging.Logger):
        # An iterator over the configurations of hyperparameters
        param_iter = ParameterGrid(self.param_grid)

        # Enumerate these to keep track of them
        combinations = enumerate(param_iter)

        # Filter to combos
        combinations = filter(
            lambda x: x[0] % cmd_args.combo_groups == cmd_args.combo_num, combinations
        )
        combinations = list(combinations)[cmd_args.num_skip :]

        # Keep track of the results of the runs
        run_results = []
        for combo_num in range(len(combinations)):
            run_results.append("SKIPPED")

        try:
            # Run the experiment for each sampled combination of parameters
            for i, (combo_index, combo) in enumerate(combinations):
                # Set the status of the current run to failed until proven otherwise
                run_results[i] = "FAILED"

                self._run_single_experiment(
                    combinations, combo, combo_index, cmd_args, base_logger
                )

                run_results[i] = "SUCCEEDED"

        finally:
            # Print a summary of the experiment results
            base_logger.info("")
            base_logger.info("")
            base_logger.info("=" * self.output_width)
            title = f"| SUMMARY | GROUP {cmd_args.combo_num}/{cmd_args.combo_groups}"
            title += (" " * (self.output_width - 1 - len(title))) + "|"
            title = textwrap.fill(title, self.output_width)
            base_logger.info(title)
            base_logger.info("=" * self.output_width)
            for result, (combo_num, combo) in zip(run_results, combinations):
                base_logger.info("")
                base_logger.info(f"COMBO {combo_num}")
                base_logger.info(textwrap.fill(str(combo)))
                base_logger.info(result)


class MultiprocessHyperparameterExperiment(HyperparameterExperiment):
    """A class to run an experiment over a grid of hyperparameters in parallel.

    Runs each combination of hyperparameters in the grid as a separate experiment using
    a pool of workers.

    The workflow is as follows:

    1. Call the constructor with the hyperparameter grid and the experiment function.
    2. (Optional) Add any additional arguments to the arg parser using
       `self.parser.add_argument`.
    3. Call `self.run()` to run the experiment.

    Parameters
    ----------
    param_grid : dict
        A dictionary mapping hyperparameter names to lists of values to try.
    experiment_fn : Callable[[dict, str, Namespace, Callable, logging.LoggerAdapter],
    None]
        A function that takes a single hyperparameter combination and runs the
        experiment. It should take the form:
            experiment_fn(combo, run_id, cmd_args, tqdm_func, child_logger_adapter)
        where `combo` is a single combination of hyperparameters, `run_id` is a unique
        identifier for the run, `cmd_args` is the command line arguments, `tqdm_func` is
        a function used to create a tqdm progress bar, and `child_logger_adapter` is a
        logger adapter to use for logging.
    run_id_fn : Callable[[int, Namespace], str], optional
        A function that takes a single hyperparameter combination and returns a unique
        identifier for the run. If None, the default is to use the experiment name and
        the combination index. It should take the form:
            run_id_fn(combo_index, cmd_args)
        where `combo_index` is the index of the combination in the ParameterGrid and
        `cmd_args` is the command line arguments.
    experiment_name : str, default="EXPERIMENT"
        The name of the experiment.
    """

    def __init__(
        self,
        param_grid: dict,
        experiment_fn: Callable[
            [dict, str, Namespace, Callable, logging.LoggerAdapter], None
        ],
        run_id_fn: Optional[Callable[[int, Namespace], str]] = None,
        experiment_name: str = "EXPERIMENT",
    ):
        super().__init__(
            param_grid=param_grid,
            experiment_fn=experiment_fn,
            run_id_fn=run_id_fn,
            experiment_name=experiment_name,
            arg_parser_description="Run hyperparameter experiments in parallel",
        )

        # Needed so that we can pickle the command arguments
        # https://stackoverflow.com/a/71010038
        self.parser.register("type", None, identity)

        # Add various arguments
        self.parser.add_argument(
            "--num-workers",
            type=int,
            default=1,
            help="The number of workers to use for multiprocessing",
        )

    def _task_fn(
        self,
        combinations: list[dict],
        combo_index: int,
        cmd_args: Namespace,
        base_logger: logging.Logger,
        tqdm_func: Callable,
        global_tqdm: tqdm,
    ) -> bool:
        info_prefix = f"[{combo_index}/{len(combinations)}] "

        # Create a unique run_id for this run
        run_id = self.run_id_fn(combo_index, cmd_args)

        # Set up the logger
        child_logger = logging.getLogger(f"{base_logger.name}.{run_id}")
        child_logger.setLevel(self.experiment_log_level)
        child_logger_adapter = PrefixLoggerAdapter(child_logger, info_prefix)

        # The tqdm function to use. Set the leave argument to False because tqdm because
        # otherwise tqdm doesn't display multiple progress bars properly due to a bug
        # https://github.com/tqdm/tqdm/issues/1496
        tqdm_func = partial(
            tqdm_func,
            leave=False,
            bar_format=info_prefix + "{desc}: {percentage:3.0f}%|{bar}{r_bar}",
        )

        # Run the experiment
        self.experiment_fn(
            combinations[combo_index],
            run_id,
            cmd_args,
            tqdm_func,
            child_logger_adapter,
        )

        # Update the global progress bar and log that this run is finished
        global_tqdm.update(1)
        base_logger.info(f"{info_prefix}{run_id} finished")

        return True

    def _run(self, cmd_args: Namespace, base_logger: logging.Logger):
        # Set the torch multiprocessing start method to spawn, to avoid issues with CUDA
        torch.multiprocessing.set_start_method("spawn", force=True)

        # Get all configurations of hyperparameters, and turn this into a list of tasks
        combinations = list(ParameterGrid(self.param_grid))

        # Create a list of tasks
        tasks = [
            (self._task_fn, (combinations, combo_index, cmd_args, base_logger))
            for combo_index in range(len(combinations))
        ]

        # Create a pool of workers
        pool = TqdmMultiProcessPool(cmd_args.num_workers)

        with tqdm(total=len(combinations), dynamic_ncols=True) as global_progress:
            global_progress.set_description("Total progress")
            pool.map(global_progress, tasks, lambda x: None, lambda x: None)
