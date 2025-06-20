"""Run the self-hosting language model server.

This server controls a vLLM server for language model inference and provides an Open-AI
compatible API for training.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import subprocess
from shutil import which
import os

from nip.constants import REPOSITORY_ROOT
from nip.utils.env import get_env_var


parser = ArgumentParser(
    description=__doc__.partition("\n\n")[0],
    epilog=__doc__.partition("\n\n")[2],
    formatter_class=ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--lm-server-port",
    type=int,
    default=get_env_var("DEFAULT_LM_SERVER_PORT"),
    help="The port on which the main language model server will run.",
)

parser.add_argument(
    "--vllm-port",
    type=int,
    default=get_env_var("DEFAULT_VLLM_SERVER_PORT"),
    help="The port on which the vLLM server will run.",
)

parser.add_argument(
    "--max-training-jobs",
    type=int,
    default=1,
    help="The maximum number of concurrent training jobs allowed.",
)

parser.add_argument(
    "--vllm-num-gpus",
    type=str,
    default="auto",
    help="The number of GPUs to use for the vLLM server. "
    "If set to 'auto', it will use all available GPUs.",
)

parser.add_argument(
    "--vllm-clear-cache",
    action="store_true",
    help="Whether to clear the Hugging Face model cache before starting the server. "
    "This only removes cached models other than the one being loaded.",
)

parser.add_argument(
    "--accelerate-config",
    type=str,
    default="accelerate_config.yaml.jinja2",
    help="Path to the configuration file for the accelerate library. Can be either a "
    "regular file or a Jinja2 template. If empty, no configuration file will be passed "
    "to the `accelerate` command.",
)

parser.add_argument(
    "--log-to-file",
    action="store_true",
    help="Whether to log vLLM and trainer to files instead of stdout and stderr.",
    default=False,
)

parser.add_argument(
    "--external",
    action="store_true",
    help="Whether to run the server in external mode, allowing it to be accessed from "
    "outside. Otherwise, it will only be accessible from localhost.",
    default=False,
)

parser.add_argument(
    "--dev",
    action="store_true",
    help="Whether to run the FastAPI server in development mode, which enables "
    "auto-reload.",
    default=False,
)


def main():
    """Run the language model server."""

    args = parser.parse_args()

    if args.external:
        host = "0.0.0.0"
    else:
        host = "127.0.0.1"

    new_env_variables = {
        "VLLM_PORT": str(args.vllm_port),
        "SUBPROCESS_OUTPUT_DESTINATION": (
            "log_file" if args.log_to_file else "stdout_std_err"
        ),
        "MAX_TRAINING_JOBS": str(args.max_training_jobs),
        "VLLM_NUM_GPUS": args.vllm_num_gpus,
        "ACCELERATE_CONFIG_PATH": args.accelerate_config,
    }

    subprocess.run(
        [
            which("uv"),
            "run",
            "fastapi",
            "dev" if args.dev else "run",
            "--host",
            host,
            "--port",
            str(args.lm_server_port),
            "nip/language_model_server/server.py",
        ],
        env=dict(os.environ, **new_env_variables),
        cwd=REPOSITORY_ROOT,
    )


if __name__ == "__main__":
    main()
