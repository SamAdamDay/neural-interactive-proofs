"""Generate the validation dataset."""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pvg.constants import OPENROUTER_API_KEY

from pvg.code_validation.dataset_generation import (
    generate_and_save_cv_dataset,
    CodeValidationDatasetConfig,
)

# Set up the arg parser
parser = ArgumentParser(
    description="Generate a code validation dataset",
    formatter_class=ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--split",
    type=str,
    default=None,
    help="Whether to draw problems from the train or test split of the APPS dataset",
)
parser.add_argument(
    "--num_data",
    type=int,
    default=10000,
    help="How many problems the dataset should contain (per split per difficulty level)",
)
parser.add_argument(
    "--save_after",
    type=int,
    default=10,
    help="The number of problems added after which to save (and possibly push) the dataset",
)
parser.add_argument(
    "--openrouter_api_key",
    type=str,
    default=OPENROUTER_API_KEY,
    help="The OpenRouter API key to use for generating responses",
)

# Get the arguments
cmd_args = parser.parse_args()

# Create the config
config = CodeValidationDatasetConfig(
    split=cmd_args.split,
    num_data=cmd_args.num_data,
    save_after=cmd_args.save_after,
    openrouter_api_key=cmd_args.openrouter_api_key,
)

# Generate the dataset
generate_and_save_cv_dataset(config)
