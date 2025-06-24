"""Utilities for operating system interactions."""

import os
from contextlib import contextmanager


@contextmanager
def change_directory(new_path: str):
    """Context manager to temporarily change the current working directory.

    Parameters
    ----------
    new_path : str
        The path to change to.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    """

    original_path = os.getcwd()
    os.chdir(new_path)
    try:
        yield
    finally:
        os.chdir(original_path)
