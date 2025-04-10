"""Generate a Buggy APPS dataset.

Our Buggy APPS dataset is based on the APPS dataset :cite:p:`Hendrycks2021`, which
consists of problem statements and code solutions. The Buggy APPS dataset is
augmented with buggy code solutions, generated by asking a large language model to
introduce a subtle bug into each correct solution.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from nip.code_validation.dataset_generation import (
    generate_and_save_cv_dataset,
    CodeValidationDatasetConfig,
)
from nip.utils.env import get_env_var

# Set up the arg parser
parser = ArgumentParser(
    description="Generate a Buggy APPS dataset",
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

if __name__ == "__main__":

    # Get the arguments
    cmd_args = parser.parse_args()

    # Create the config
    config = CodeValidationDatasetConfig(
        split=cmd_args.split,
        num_data=cmd_args.num_data,
        save_after=cmd_args.save_after,
        openrouter_api_key=get_env_var("OPENROUTER_API_KEY"),
    )

    # Generate the dataset
    generate_and_save_cv_dataset(config)
