"""Utilities for working with environment variables."""

import os
from typing import Callable

from dotenv import load_dotenv

from pvg.constants import ENV_FILE
from pvg.utils.types import NOT_GIVEN

env_loaded = False


class EnvironmentVariableDoesNotExistError(Exception):
    """An exception raised when an environment variable is not set."""

    def __init__(self, var_name: str):
        self.var_name = var_name
        super().__init__(f"The environment variable {var_name!r} is not set.")


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


def get_env_var(var_name: str, default=NOT_GIVEN) -> str:
    """Get the value of an environment variable, raising an error if not set.

    Parameters
    ----------
    var_name : str
        The name of the environment variable to get.
    default : str, optional
        The default value to return if the environment variable is not set. If not
        provided, an error is raised if the environment variable is not set.

    Returns
    -------
    env_value : str
        The value of the environment variable.

    Raises
    ------
    EnvironmentVariableDoesNotExistError
        If the environment variable is not set.
    """

    load_env_once()

    env_value = os.getenv(var_name)

    if env_value is None:
        if default is not NOT_GIVEN:
            return default
        raise EnvironmentVariableDoesNotExistError(var_name)

    return env_value


def env_var_default_factory(var_name: str, default=NOT_GIVEN) -> Callable[[], str]:
    """Create a factory function for getting an environment variable with a default.

    This is useful for setting the value of an environment variable as the default value
    of a parameter in a dataclass.

    Example
    -------
    ```python
    import dataclasses

    @dataclasses.dataclass
    class MyParameters:
        my_var: str = dataclasses.field(
            default_factory=env_var_default_factory("MY_VAR")
        )

    Parameters
    ----------
    var_name : str
        The name of the environment variable to get.
    default : str, optional
        The default value to return if the environment variable is not set. If not
        provided, an error is raised if the environment variable is not set.

    Returns
    -------
    factory : Callable[[], str]
        A factory function that returns the value of the environment variable if set,
        and the default value otherwise.
    """

    def factory() -> str:
        return get_env_var(var_name, default=default)

    return factory
