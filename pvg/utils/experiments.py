from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from typing import Callable, Optional
import textwrap

from sklearn.model_selection import ParameterGrid


class HyperparameterExperiment:
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
        if run_id_fn is None:
            run_id_fn = (
                lambda combo_index, _: f"{experiment_name.lower()}_{combo_index}"
            )

        self.param_grid = param_grid
        self.experiment_fn = experiment_fn
        self.run_id_fn = run_id_fn
        self.experiment_name = experiment_name
        self.output_width = output_width

        # Set up the arg parser
        self.parser = ArgumentParser(
            description="Run the supervised MMH experiments",
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
