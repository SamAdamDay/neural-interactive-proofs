"""Run the self-hosting language model server.

This server controls a vLLM server for language model inference and provides an Open-AI
compatible API for training.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import uvicorn

from pydantic_settings import BaseSettings, CliApp, CliSettingsSource

from nip.language_model_server.config import Settings
from nip.constants import REPOSITORY_ROOT, PACKAGE_ROOT
from nip.utils.env import get_env_var, set_env_variables
from nip.utils.os import change_directory


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

cli_settings = CliSettingsSource(Settings, root_parser=parser)


def main():
    """Run the language model server."""

    args = parser.parse_args()

    settings = CliApp.run(Settings, cli_args=args, cli_settings=cli_settings)

    settings.subprocess_output_destination = (
        "log_file" if args.log_to_file else "stdout_std_err"
    )

    if args.external:
        host = "0.0.0.0"
    else:
        host = "127.0.0.1"

    new_env_variables_lower_case = settings.model_dump()
    new_env_variables = {
        key.upper(): str(value) for key, value in new_env_variables_lower_case.items()
    }

    with change_directory(REPOSITORY_ROOT):
        with set_env_variables(new_env_variables):
            uvicorn.run(
                "nip.language_model_server.server:app",
                host=host,
                port=args.lm_server_port,
                log_level="debug" if args.debug else "info",
                log_config={
                    "version": 1,
                    "disable_existing_loggers": False,
                    "formatters": {
                        "default": {
                            "format": "\033[37;100m %(levelname)s \033[0m "
                            "%(asctime)s %(message)s",
                            "datefmt": "%d/%m/%y %H:%M:%S",
                        },
                        "access": {
                            "format": "    \033[47m %(levelname)s \033[0m %(message)s",
                            "datefmt": "%d/%m/%y %H:%M:%S",
                        },
                    },
                    "handlers": {
                        "default": {
                            "formatter": "default",
                            "class": "logging.StreamHandler",
                            "stream": "ext://sys.stderr",
                        },
                        "access": {
                            "formatter": "access",
                            "class": "logging.StreamHandler",
                            "stream": "ext://sys.stdout",
                        },
                    },
                    "loggers": {
                        "uvicorn.error": {
                            "level": "INFO",
                            "handlers": ["default"],
                            "propagate": False,
                        },
                        "uvicorn.access": {
                            "level": "INFO",
                            "handlers": ["access"],
                            "propagate": False,
                        },
                    },
                    "root": {
                        "level": "DEBUG" if args.debug else "INFO",
                        "handlers": ["default"],
                        "propagate": False,
                    },
                },
                reload=args.dev,
                reload_dirs=[str(PACKAGE_ROOT)],
            )


if __name__ == "__main__":
    main()
