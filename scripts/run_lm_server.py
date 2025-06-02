"""Run the self-hosting language model server.

This server controls a vLLM server for language model inference and provides an Open-AI
compatible API for training.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging

from flask import Flask

from nip.language_model_server.server import LanguageModelServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(name)s:%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

parser = ArgumentParser(
    description=__doc__.partition("\n\n")[0],
    epilog=__doc__.partition("\n\n")[2],
    formatter_class=ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--lm-server-port",
    type=int,
    default=5000,
    help="The port on which the main language model server will run.",
)

parser.add_argument(
    "--vllm-port",
    type=int,
    default=8000,
    help="The port on which the vLLM server will run.",
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
    "--debug",
    action="store_true",
    help="Whether to run the server in debug mode, which enables Flask's debug "
    "features such as automatic reloading and detailed error messages.",
    default=False,
)

if __name__ == "__main__":
    args = parser.parse_args()

    app = Flask(__name__)

    with LanguageModelServer(
        app,
        vllm_port=args.vllm_port,
        subprocess_output_destination=(
            "log_file" if args.log_to_file else "stdout_stderr"
        ),
    ) as lm_server:

        if args.external:
            hostname = "0.0.0.0"
        else:
            hostname = "localhost"

        app.run(host=hostname, port=args.lm_server_port)
