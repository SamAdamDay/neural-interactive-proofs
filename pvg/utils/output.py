"""Utilities related to outputting things to the user."""
from typing import Iterable


class DummyTqdm:
    """A dummy tqdm class, to avoid printing progress bars.

    Parameters
    ----------
    *args
        Positional arguments to pass to tqdm.
    **kwargs
        Keyword arguments to pass to tqdm.
    """

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            self.iterable = args[0]

    def __iter__(self):
        return iter(self.iterable)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, *args):
        pass

    def close(self):
        pass
