"""Script for creating or joining a tmux session for running the nip project."""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch

from libtmux import Server
from libtmux.constants import PaneDirection

DEFAULT_BASE_DIRECTORY = "/root/neural-interactive-proofs"

parser = ArgumentParser(
    description="Create or join a tmux session for running the nip project.",
    formatter_class=ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--session", type=str, default="nip", help="Name of the tmux session."
)
parser.add_argument(
    "--single-pane",
    "-p",
    action="store_true",
    help="Create a single pane for running experiments, rather than one per GPU.",
)
parser.add_argument(
    "--speed-test",
    "-s",
    action="store",
    nargs="?",
    default="",
    const="ppo_gi",
    type=str,
    help="Run a speed test for the given script (can exclude the '.py').",
)
parser.add_argument(
    "--force-new-session",
    "-f",
    action="store_true",
    help="Force the creation of a new session, even if one already exists.",
)
parser.add_argument(
    "--base-directory",
    action="store",
    nargs=1,
    default=DEFAULT_BASE_DIRECTORY,
    type=str,
    help="Base directory for the tmux session.",
)


def main():
    """Main function for the script."""

    cmd_args = parser.parse_args()

    server = Server()

    # If the session already exists and we're not forcing a new session, attach to it
    if server.has_session(cmd_args.session) and not cmd_args.force_new_session:
        session = server.sessions.get(session_name=cmd_args.session)
        session.attach()
        return

    # Create a new session
    session = server.new_session(
        cmd_args.session,
        kill_session=cmd_args.force_new_session,
        start_directory=cmd_args.base_directory,
        window_name="nip",
    )
    window = session.windows.get(window_name="nip")

    # Create a pane on the right with nvitop running there
    monitor_pane = window.split(direction=PaneDirection.Right, attach=False)
    monitor_pane.send_keys("nvitop")

    # Create panes for running experiments
    experiment_panes = [window.active_pane]
    if not cmd_args.single_pane:
        num_gpus = torch.cuda.device_count()
        for i in range(0, num_gpus - 1):
            pane_size = f"{((num_gpus - i - 1) * 100) // (num_gpus - i)}%"
            previous_pane = experiment_panes[-1]
            new_pane = previous_pane.split(
                direction=PaneDirection.Below,
                attach=False,
                size=pane_size,
                start_directory=cmd_args.base_directory,
            )
            experiment_panes.append(new_pane)

    # Run speed tests in each pane, if requested
    if cmd_args.speed_test != "":
        script = f"python scripts/{cmd_args.speed_test}"
        if not cmd_args.speed_test.endswith(".py"):
            script += ".py"
        script += " --no-pretrain"
        for i, pane in enumerate(experiment_panes):
            pane_script = script
            if i > 0:
                pane_script += f" --gpu-num {i}"
            pane.send_keys(pane_script)

    # Attach to the session
    session.attach()


if __name__ == "__main__":
    main()
