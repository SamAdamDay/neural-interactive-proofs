"""Download code validation checkpoints from W&B to the local directory."""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from nip.utils.checkpoints import download_checkpoint
from nip.utils.env import get_env_var

parser = ArgumentParser(
    description=__doc__,
    formatter_class=ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "checkpoint_names",
    type=str,
    nargs="+",
    help="The names of the checkpoints to download.",
)

parser.add_argument(
    "--wandb_entity",
    type=str,
    default=get_env_var("WANDB_ENTITY"),
    help="The wandb entity to use.",
)

parser.add_argument(
    "--wandb_project",
    type=str,
    default=get_env_var("WANDB_CV_PROJECT"),
    help="The wandb project to use.",
)

parser.add_argument(
    "--overwrite",
    "-o",
    action="store_true",
    help="Whether to overwrite existing checkpoints.",
)

if __name__ == "__main__":

    # Get the arguments
    cmd_args = parser.parse_args()

    for checkpoint_name in cmd_args.checkpoint_names:
        print(f"Downloading checkpoint {checkpoint_name!r} from W&B...")  # noqa: T201
        download_checkpoint(
            cmd_args.wandb_project,
            checkpoint_name,
            wandb_entity=cmd_args.wandb_entity,
            include_everything=True,
            handle_existing="overwrite" if cmd_args.overwrite else "skip",
        )
