"""Visualise the rollouts of a MALT experiment as a forest of trees.

The MALT :cite:p:`Motwani2024` trainer samples a set of trees of responses. These are
stored in a flat array. This script provides a command line interface to visualise the
trees in a forest.

By default, the visualisation is saved to the 'analysis' subdirectory of the checkpoint.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from nip.trainers import PureTextEiTrainer
from nip.utils.malt_forest import MaltForestVisualiser
from nip.utils.env import get_env_var

parser = ArgumentParser(
    description=__doc__.strip(),
    formatter_class=ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "run_id",
    type=str,
    help="The id of the wandb run to download the rollouts from.",
)

parser.add_argument(
    "iteration",
    type=str,
    help="The experiment iteration of the rollouts. Either an integer or 'test'.",
)

parser.add_argument(
    "--stdout",
    "-s",
    action="store_true",
    help="Whether to print the visualisation to stdout. Otherwise, it is saved to the "
    "'analysis' subdirectory of the checkpoint.",
)

parser.add_argument(
    "--format",
    "-f",
    choices=["yaml", "jsonl"],
    default="yaml",
    help="The format to visualise the MALT forest in.",
)

parser.add_argument(
    "--wandb_project",
    type=str,
    default=get_env_var("WANDB_CV_PROJECT"),
    help="The wandb project to use.",
)

parser.add_argument(
    "--wandb_entity",
    type=str,
    default=get_env_var("WANDB_ENTITY"),
    help="The wandb entity to use.",
)

if __name__ == "__main__":

    cmd_args = parser.parse_args()

    malt_forest_visualiser = MaltForestVisualiser.from_wandb_run(
        run_id=cmd_args.run_id,
        iteration=cmd_args.iteration,
        wandb_project=cmd_args.wandb_project,
        wandb_entity=cmd_args.wandb_entity,
    )
    malt_forest_visualisation = malt_forest_visualiser.visualise(
        format=cmd_args.format,
    )

    if cmd_args.stdout:
        print(malt_forest_visualisation)  # noqa: T201
    else:
        checkpoint_dir = PureTextEiTrainer.get_checkpoint_base_dir_from_run_id(
            cmd_args.run_id
        )
        file_path = (
            checkpoint_dir
            / "analysis"
            / "malt_forest"
            / f"{cmd_args.iteration}.{cmd_args.format}"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as file:
            file.write(malt_forest_visualisation)
        print(f"Saved visualisation to {file_path!s}.")  # noqa: T201
