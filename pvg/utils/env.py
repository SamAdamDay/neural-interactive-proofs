"""Utilities for working with environment variables."""

from dotenv import load_dotenv

from pvg.constants import ENV_FILE

env_loaded = False


def load_env_once():
    """Load the environment variables once.

    If the environment variables have already been loaded, this function does nothing.
    """

    global env_loaded

    if not env_loaded:
        load_dotenv(ENV_FILE)
        env_loaded = True


def reload_env():
    """Reload the environment variables.

    This function reloads the environment variables from the .env file even if they have
    already been loaded.
    """
    load_dotenv(ENV_FILE)
