"""Utilities for working with data."""

from typing import Optional

import torch

from tensordict.tensordict import TensorDict, TensorDictBase

from pvg.scenario_base import DataLoader
from pvg.utils.types import TorchDevice


def forgetful_cycle(iterable):
    """A version of cycle that doesn't save copies of the values"""
    while True:
        for i in iterable:
            yield i


class VariableDataCycler:
    """A loader that cycles through data, but allows the batch size to vary.

    Parameters
    ----------
    dataloader : DataLoader
        The base dataloader to use. This dataloader will be cycled through.
        device : TorchDevice, optional
            The device to move the data to. If None, the data will not be moved.
        non_blocking : bool, default=False
            Whether to move the data to the device with `non_blocking=True`.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        device: Optional[TorchDevice] = None,
        non_blocking: bool = False,
    ):
        self.dataloader = dataloader
        self.device = device
        self.non_blocking = non_blocking
        self.dataloader_iter = iter(forgetful_cycle(self.dataloader))
        self.remainder: Optional[list[TensorDict]] = None

    def get_batch(
        self,
        batch_size: int,
    ) -> TensorDict:
        """Get a batch of data from the dataloader with the given batch size.

        If the dataloader is exhausted, it will be reset.

        Parameters
        ----------
        batch_size : int
            The size of the batch to return.

        Returns
        -------
        batch : TensorDict
            A batch of data with the given batch size.
        """

        left_to_sample = batch_size
        batch_components: list[TensorDict] = []

        # Start by sampling from the remainder from the previous sampling
        if self.remainder is not None:
            batch_components.append(self.remainder[:left_to_sample])
            if len(self.remainder) <= left_to_sample:
                left_to_sample -= len(self.remainder)
                self.remainder = None
            else:
                self.remainder = self.remainder[left_to_sample:]
                left_to_sample = 0

        # Keep sampling batches until we have enough
        while left_to_sample > 0:
            batch: TensorDict = next(self.dataloader_iter)
            if self.device is not None:
                batch = batch.to(self.device, non_blocking=self.non_blocking)
            batch_components.append(batch[:left_to_sample])
            if len(batch) <= left_to_sample:
                left_to_sample -= len(batch)
            else:
                self.remainder = batch[left_to_sample:]
                left_to_sample = 0

        # Concatenate the batch components into a single batch
        batch = torch.cat(batch_components, dim=0)

        return batch

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.dataloader!r})"


def tensordict_to_numpy_dict(data: TensorDictBase) -> dict:
    """Convert a TensorDict to a dict of numpy arrays.

    Parameters
    ----------
    data : TensorDictBase
        The TensorDict to convert.

    Returns
    -------
    data : dict
        The dict of numpy arrays.
    """

    data = data.to_dict()

    def to_numpy_dict(data: dict) -> dict:
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.detach().cpu().numpy()
            else:
                data[key] = to_numpy_dict(value)
        return data

    return to_numpy_dict(data)


def max_length_iterator(iterable, maxlen):
    """An iterator that stops after a maximum length.

    Parameters
    ----------
    iterable : Iterable
        The iterable to iterate over.
    maxlen : int
        The maximum length to iterate over.
    """
    for i, item in enumerate(iterable):
        if i >= maxlen:
            break
        yield item


def nested_dict_keys(data: dict) -> list[tuple[str, ...]]:
    """Get the keys of a nested dict as a list of tuples of strings.

    A nested dict is a dict that may contain other dicts as values. This function
    recursively traverses the dict to get all the keys, where each key is a tuple of
    strings.

    Parameters
    ----------
    data : dict
        The nested dict to get the keys of.

    Returns
    -------
    keys : list
        The keys of the nested dict.

    Examples
    --------
    >>> data = {"a": {"b": 1, "c": 2}, "d": 3}
    >>> get_nested_dict_keys(data)
    [("a", "b"), ("a", "c"), ("d",)]
    """

    keys = []

    def get_keys(data, prefix=()):
        for key, value in data.items():
            if isinstance(value, dict):
                get_keys(value, prefix + (key,))
            else:
                keys.append(prefix + (key,))

    get_keys(data)
    return keys


def nested_dict_keys_stringified(data: dict, separator=".") -> list[str]:
    """Get the keys of a nested dict as a list of joined strings.

    A nested dict is a dict that may contain other dicts as values. This function
    recursively traverses the dict to get all the keys and joins them with a separator.

    Parameters
    ----------
    data : dict
        The nested dict to get the keys of.
    separator : str
        The separator to use between keys.

    Returns
    -------
    keys : list
        The keys of the nested dict.

    Examples
    --------
    >>> data = {"a": {"b": 1, "c": 2}, "d": 3}
    >>> get_nested_dict_keys(data)
    ["a.b", "a.c", "d"]
    """

    keys_tuple = nested_dict_keys(data)
    return [separator.join(key) for key in keys_tuple]


def pad_missing_conversations(tensor: torch.Tensor, indices: list[int], dim: int):

    if tensor.shape[0] == 1:
        return tensor.expand(dim, *tensor.shape[1:])
    else:
        z = torch.zeros(dim, *tensor.shape[1:])
        z[indices] = tensor
        return z
