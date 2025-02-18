"""Utilities for working with data."""

from typing import Optional, Any, Iterable, Iterator, TypeVar

import torch

from tensordict.tensordict import TensorDict, TensorDictBase

from nip.utils.types import TorchDevice
from nip.utils.nested_array_dict import NestedArrayDict, concatenate_nested_array_dicts

T = TypeVar("T")


def forgetful_cycle(iterable: Iterable[T]) -> Iterator[T]:
    """Return an iterator that cycles through an iterable without saving copies.

    This iterator will keep yielding items from the iterable in order, and will start
    over once the iterable is exhausted. However, unlike the `itertools.cycle` function,
    it does not save copies of the items in the iterable.

    Parameters
    ----------
    iterable : Iterable
        The iterable to cycle through.

    Yields
    ------
    item : Any
        The next item in the iterable, cycling through it indefinitely.
    """
    while True:
        for i in iterable:
            yield i


def truncated_iterator(iterable: Iterable[T], maxlen: int) -> Iterator[T]:
    """Return an iterator that stops after a maximum length.

    Parameters
    ----------
    iterable : Iterable
        The iterable to iterate over.
    maxlen : int
        The maximum length to iterate over.

    Yields
    ------
    item : Any
        The next item in the iterable.
    """

    for i, item in enumerate(iterable):
        if i >= maxlen:
            return
        yield item


class VariableDataCycler:
    """A loader that cycles through data, but allows the batch size to vary.

    If a default batch size is provided, it is possible to iterate infinitely over the
    data as follows:

    >>> data_cycler = VariableDataCycler(dataloader, default_batch_size=32)
    >>> for batch in data_cycler:
    ...     # Do something with the batch

    Parameters
    ----------
    dataloader : Iterable
        The base dataloader to use. This dataloader will be cycled through.
    device : TorchDevice, optional
        The device to move the data to. If None, the data will not be moved.
    non_blocking : bool, default=False
        Whether to move the data to the device with `non_blocking=True`.
    default_batch_size : int, optional
        The default batch size to use when getting a batch and iterating over the
        instance. If None, the batch size must be manually specified when getting a
        batch, and it is not possible to iterate over the instance.
    """

    def __init__(
        self,
        dataloader: Iterable,
        device: Optional[TorchDevice] = None,
        non_blocking: bool = False,
        default_batch_size: Optional[int] = None,
    ):
        self.dataloader = dataloader
        self.device = device
        self.non_blocking = non_blocking
        self.default_batch_size = default_batch_size
        self.dataloader_iter = iter(forgetful_cycle(self.dataloader))
        self.remainder: Optional[list[TensorDict | NestedArrayDict]] = None

    def __iter__(self):
        if self.default_batch_size is None:
            raise ValueError(
                "Cannot iterate over the VariableDataCycler without a default batch "
                "size."
            )
        while True:
            yield self.get_batch()

    def get_batch(
        self,
        batch_size: Optional[int] = None,
    ) -> TensorDict:
        """Get a batch of data from the dataloader with the given batch size.

        If the dataloader is exhausted, it will be reset.

        Parameters
        ----------
        batch_size : int, optional
            The size of the batch to return. If None, the default batch size will be
            used, if it was provided.

        Returns
        -------
        batch : TensorDict
            A batch of data with the given batch size.
        """

        if batch_size is None:
            if self.default_batch_size is None:
                raise ValueError(
                    "Must provide a batch size when the default batch size is not "
                    "specified."
                )
            batch_size = self.default_batch_size

        left_to_sample = batch_size
        batch_components: list[TensorDict | NestedArrayDict] = []

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
            batch: TensorDict | NestedArrayDict = next(self.dataloader_iter)
            if self.device is not None:
                batch = batch.to(self.device, non_blocking=self.non_blocking)
            batch_components.append(batch[:left_to_sample])
            if len(batch) <= left_to_sample:
                left_to_sample -= len(batch)
            else:
                self.remainder = batch[left_to_sample:]
                left_to_sample = 0

        # Concatenate the batch components into a single batch
        if isinstance(batch_components[0], TensorDict):
            batch = torch.cat(batch_components, dim=0)
        else:
            batch = concatenate_nested_array_dicts(batch_components, dim=0)

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


def is_nested_key(index: Any) -> bool:
    """Check whether an index is a nested key used in TensorDicts.

    TensorDicts use nested keys to access values. A nested key is either a string or a
    tuple of strings.

    Parameters
    ----------
    index : Any
        The index to check.

    Returns
    -------
    is_nested_key : bool
        Whether the index is a nested key.
    """

    if isinstance(index, str):
        return True

    if isinstance(index, tuple):
        return all(isinstance(key, str) for key in index)

    return False


def rename_dict_key(
    dictionary: dict, old_key: str, new_key: str, allow_non_existant: bool = False
):
    """Rename a key in a dictionary, modifying the dictionary in place.

    Parameters
    ----------
    dictionary : dict
        The dictionary to rename the key in.
    old_key : str
        The old key to rename.
    new_key : str
        The new key to rename to.
    allow_non_existant : bool, default=False
        Whether to allow the old key to not exist in the dictionary. In this case, the
        dictionary is returned unchanged.

    Raises
    ------
    KeyError
        If the old key is not found in the dictionary and `allow_non_existant` is False.
    """

    if old_key not in dictionary:
        if allow_non_existant:
            return dictionary
        raise KeyError(f"Key '{old_key}' not found in dictionary.")

    dictionary[new_key] = dictionary.pop(old_key)
