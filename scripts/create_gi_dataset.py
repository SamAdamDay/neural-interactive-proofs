"""Generate a graph isomorphism dataset"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch

from pvg.graph_isomorphism.dataset_generation import (
    generate_gi_dataset,
    GraphIsomorphicDatasetConfig,
)

# Set up the arg parser
parser = ArgumentParser(
    description="Generate a graph isomorphism dataset",
    formatter_class=ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "name",
    type=str,
    help="The name of the dataset",
)
parser.add_argument(
    "--num-graphs",
    type=int,
    default=1000,
    help="The number of graphs to generate",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=800000,
    help="The batch size to use for generating the graphs",
)

# Get the arguments
cmd_args = parser.parse_args()

# Create the config
config = GraphIsomorphicDatasetConfig(num_samples=cmd_args.num_graphs)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Generate the dataset
generate_gi_dataset(
    config, cmd_args.name, batch_size=cmd_args.batch_size, device=device
)
