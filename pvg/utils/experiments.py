from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from typing import Callable, Optional
import textwrap
import os
from abc import ABC, abstractmethod
import logging
from functools import partial

from sklearn.model_selection import ParameterGrid

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


class HyperparameterExperiment(ABC):
    """A base class to run an experiment over a grid of hyperparameters.

    Runs each combination of hyperparameters in the grid as a separate experiment.
    """

    def __init__(
        self,
        param_grid: dict,
        experiment_fn: Callable[[dict, str, Namespace], None],
        run_id_fn: Optional[Callable[[int, Namespace], str]] = None,
        experiment_name: str = "EXPERIMENT",
    ):
        if run_id_fn is None:
            run_id_fn = (
                lambda combo_index, _: f"{experiment_name.lower()}_{combo_index}"
            )

        self.param_grid = param_grid
        self.experiment_fn = experiment_fn
        self.run_id_fn = run_id_fn
        self.experiment_name = experiment_name

    @abstractmethod
    def run(self):
        pass


class SequentialHyperparameterExperiment(HyperparameterExperiment):
    """A class to run an experiment over a grid of hyperparameters.

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
    experiment_fn : Callable[[dict, str, Namespace], None]
        A function that takes a single hyperparameter combination and runs the
        experiment. It should take the form:
            experiment_fn(combo, run_id, cmd_args)
        where combo is a single combination of hyperparameters, run_id is a unique
        identifier for the run, and cmd_args is the command line arguments.
    run_id_fn : Callable[[int, Namespace], str], optional
        A function that takes a single hyperparameter combination and returns a unique
        identifier for the run. If None, the default is to use the experiment name and
        the combination index. It should take the form:
            run_id_fn(combo_index, cmd_args)
        where combo_index is the index of the combination in the ParameterGrid and
        cmd_args is the command line arguments.
    experiment_name : str, default="EXPERIMENT"
        The name of the experiment.
    output_width : int, default=79
        The width of the output to print.
    """

    def __init__(
        self,
        param_grid: dict,
        experiment_fn: Callable[[dict, str, Namespace], None],
        run_id_fn: Optional[Callable[[int, Namespace], str]] = None,
        experiment_name: str = "EXPERIMENT",
        output_width: int = 79,
    ):
        super().__init__(
            param_grid=param_grid,
            experiment_fn=experiment_fn,
            run_id_fn=run_id_fn,
            experiment_name=experiment_name,
        )

        self.output_width = output_width

        # Set up the arg parser
        self.parser = ArgumentParser(
            description="Run hyperparameter experiments sequentially",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )

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

    def run(self):
        """Run the experiment."""
        # Get the arguments
        cmd_args = self.parser.parse_args()

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

                # Create a unique run_id for this trial
                run_id = self.run_id_fn(combo_index, cmd_args)

                # Print the run_id and the Parameters
                print()
                print()
                print("=" * self.output_width)
                title = f"| {self.experiment_name} | Run ID: {run_id}"
                title += (" " * (self.output_width - 1 - len(title))) + "|"
                title = textwrap.fill(title, self.output_width)
                print(title)
                print("=" * self.output_width)
                print()
                print()

                self.experiment_fn(combo, run_id, cmd_args)

                run_results[i] = "SUCCEEDED"

        finally:
            # Print a summary of the experiment results
            print()
            print()
            print("=" * self.output_width)
            title = f"| SUMMARY | GROUP {cmd_args.combo_num}/{cmd_args.combo_groups}"
            title += (" " * (self.output_width - 1 - len(title))) + "|"
            title = textwrap.fill(title, self.output_width)
            print(title)
            print("=" * self.output_width)
            for result, (combo_num, combo) in zip(run_results, combinations):
                print()
                print(f"COMBO {combo_num}")
                print(textwrap.fill(str(combo)))
                print(result)


class MultiprocessHyperparameterExperiment(HyperparameterExperiment):
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
        )

        # Set up the arg parser
        self.parser = ArgumentParser(
            description="Run hyperparameter experiments in parallel",
            formatter_class=ArgumentDefaultsHelpFormatter,
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
        self.parser.add_argument(
            '-d', '--debug',
            help="Print debug messages",
            action="store_const", dest="log_level", const=logging.DEBUG,
            default=logging.WARNING,
        )
        self.parser.add_argument(
            '-v', '--verbose',
            help="Print additional info messages",
            action="store_const", dest="log_level", const=logging.INFO,
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
        child_logger.setLevel(cmd_args.log_level)
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

    def run(self):
        # Get the arguments
        cmd_args = self.parser.parse_args()

        # Get all configurations of hyperparameters, and turn this into a list of tasks
        combinations = list(ParameterGrid(self.param_grid))

        # Set up the logger
        base_logger = logging.getLogger(__name__)
        setup_logger_tqdm()

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
