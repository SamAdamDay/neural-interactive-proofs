"""Script for running a Gaussian Process Bayesian Optimiser on a manual function.

A manual function is one where the user manually inputs the values of the function upon
prompting. This can be used for experiments run separately from this script.

The script is interruptable, and will save the current state of the optimiser to a file
if interrupted. Passing the same experiment name will load the state of the optimiser
from the file, and continue from where it left off.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from skopt import gp_minimize

parser = ArgumentParser(
    description=__doc__.partition("\n")[0],
    epilog=__doc__.partition("\n")[2],
    formatter_class=ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "experiment_name",
    type=str,
    help="The name of the experiment which is being run.",
)
