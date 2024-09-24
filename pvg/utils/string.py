"""Utility functions for string manipulation."""

import random
from string import ascii_letters


def random_string(length: int, capitalise=True) -> str:
    """Generate a random string of the given length.

    Parameters
    ----------
    length : int
        The length of the random string to generate.
    capitalise : bool
        Whether to capitalise the first letter of the random string.

    Returns
    -------
    random_string : str
        The generated random string.
    """

    random_string = "".join(random.choices(ascii_letters, k=length))

    if capitalise:
        random_string = random_string.capitalize()

    return random_string
