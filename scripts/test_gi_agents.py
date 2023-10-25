from abc import ABC
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform

from sklearn.model_selection import ParameterGrid

from jaxtyping import Float

from tqdm import tqdm

from pvg.scenarios import GraphIsomorphismAgent
from pvg.data import GraphIsomorphismDataset, GraphIsomorphismData
from pvg.parameters import Parameters, GraphIsomorphismParameters

TEST_SIZE = 0.2

# Set up the arg parser
parser = ArgumentParser(
    description="Run the supervised MMH experiments",
    formatter_class=ArgumentDefaultsHelpFormatter,
)

# Add various arguments
parser.add_argument("run_letter", type=str, help="The letter to use for the W&B run")
parser.add_argument(
    "--combo-groups",
    type=int,
    default=1,
    help="Into how many groups to split the experiment combinations",
)
parser.add_argument(
    "--combo-num",
    type=int,
    default=0,
    help="Which combo group to run this time",
)
parser.add_argument(
    "--num-skip",
    type=int,
    default=0,
    help="The number of initial combos to skip. Useful to resume a group",
)
parser.add_argument(
    "--gpu-num", type=int, default=0, help="The (0-indexed) GPU number to use"
)

# Get the arguments
cmd_args = parser.parse_args()

# The different hyperparameters to test
param_grid = {
    "dataset_name": ["er10000"],
    "d_decider": [16],
    "num_epochs": [1000],
    "learning_rate": [0.003],
    "SCHEDULER_PATIENCE": [2000],
    "SCHEDULER_FACTOR": [0.5],
    "freeze": [True, False],
    "seed": [8144, 820, 4173, 3992],
}

# An interator over the configurations of hyperparameters
param_iter = ParameterGrid(param_grid)

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
        run_id = f"mh_supervised_{cmd_args.run_letter}_{combo_index}"

        # Print the run_id and the Parameters
        print()
        print()
        print("=" * 79)
        title = f"| SUPERVISED MMH EXPERIMENT | Run ID: {run_id}"
        title += (" " * (78 - len(title))) + "|"
        print(title)
        print("=" * 79)
        print()
        print()

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        torch_generator = torch.Generator().manual_seed(SEED)

        params = Parameters(
            scenario="graph_isomorphism",
            trainer="test",
            dataset=DATASET_NAME,
            max_message_rounds=1,
            graph_isomorphism=GraphIsomorphismParameters(
                prover_d_gnn=8,
                verifier_d_gnn=8,
            )
        )

        # Set up the parameters for the experiment
        parameters = Parameters(**combo)

        # Make the experiment and run it
        tags = [cmd_args.tag] if cmd_args.tag != "" else []
        args = Experiment.make_experiment(
            parameters=parameters,
            run_id=run_id,
            project_name=cmd_args.project_name,
            tags=tags,
        )
        experiment = Experiment(**args)
        experiment.run()

        run_results[i] = "SUCCEEDED"

finally:
    # Print a summary of the experiment results
    print()
    print()
    print("=" * 79)
    title = f"| SUMMARY | GROUP {cmd_args.combo_num}/{cmd_args.combo_groups}"
    title += (" " * (78 - len(title))) + "|"
    print(title)
    print("=" * 79)
    for result, (combo_num, combo) in zip(run_results, combinations):
        print()
        print(f"COMBO {combo_num}")
        print(combo)
        print(result)
