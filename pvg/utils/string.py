"""Utility functions for string manipulation."""

import random
from string import ascii_letters
import hashlib


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


def get_hash_parity(input_string: str) -> int:
    """
    Hashes the input string using SHA-256 and checks the parity of the resulting hash.

    Args:
        input_string (str): The string to be hashed.

    Returns:
        int: 0 if the hash is even, 1 if the hash is odd.
    """

    hash_object = hashlib.sha256(input_string.encode())
    hex_dig = hash_object.hexdigest()
    hash_int = int(hex_dig, 16)
    return hash_int % 2
