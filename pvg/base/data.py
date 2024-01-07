"""Base classes for handling data.

Dataloaders yield tensordicts.
"""

from abc import ABC, abstractmethod
from typing import Union, Any
import shutil
import os

import torch
from torch import Tensor

from tensordict import TensorDict

from pvg.parameters import Parameters


class Dataset(ABC):
    """Base class for all datasets.

    The dataset is stored a as a memory-mapped tensordict. See
    https://pytorch.org/tensordict/saving.html

    To subclass, implement the following methods:

    - `raw_dir` (property): The path to the directory containing the raw data.
    - `processed_dir` (property): The path to the directory containing the processed
      data.
    - `_build_tensor_dict`: Build the tensordict from the raw data.

    Parameters
    ----------
    params : Parameters
        The parameters for the experiment.
    ignore_cache : bool, default=False
        If True, the cache is ignored and the dataset is rebuilt from the raw data.
    num_threads : int, default=8
        The number of threads to use for saving the memory-mapped tensordict.
    """

    _tensor_dict: TensorDict

    def __init__(
        self, params: Parameters, ignore_cache: bool = False, num_threads: int = 8
    ):
        self.params = params

        if not os.path.isdir(self.processed_dir) or ignore_cache:
            # Delete the processed directory if it exists
            if os.path.isdir(self.processed_dir) and ignore_cache:
                shutil.rmtree(self.processed_dir)

            # Create the tensordict and save it to disk as a memory-mapped file
            self._tensor_dict = self._build_tensor_dict()
            self._tensor_dict.memmap_(self.processed_dir, num_threads=num_threads)

        else:
            # Load the memory-mapped tensordict
            self._tensor_dict = TensorDict.load_memmap(self.processed_dir)

    @property
    @abstractmethod
    def raw_dir(self) -> str:
        """The path to the directory containing the raw data."""
        pass

    @property
    @abstractmethod
    def processed_dir(self) -> str:
        """The path to the directory containing the processed data."""
        pass

    @abstractmethod
    def _build_tensor_dict(self) -> TensorDict:
        """Build the tensordict from the raw data."""
        pass

    def __getitem__(
        self, index: Union[None, int, slice, str, Tensor, list[Any], tuple[Any, ...]]
    ) -> TensorDict:
        return self._tensor_dict.__getitem__(index)

    def __getitems__(
        self, index: Union[None, int, slice, str, Tensor, list[Any], tuple[Any, ...]]
    ) -> TensorDict:
        return self._tensor_dict.__getitems__(index)

    def __len__(self) -> int:
        return len(self._tensor_dict)


class DataLoader(torch.utils.data.DataLoader):
    """The dataloader class, which may be subclassed to add functionality.

    Works with a `Dataset` subclass. The dataloader will yield tensordicts.

    Parameters
    ----------
    dataset : Dataset
        The dataset to load.
    """

    def __init__(self, dataset: Dataset, **kwargs):
        collate_fn = kwargs.pop("collate_fn", lambda x: x)
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)


def load_dataset(params: Parameters) -> Dataset:
    for value in globals().values():
        if issubclass(value, Dataset) and value.name == params.dataset:
            cls = value
            break
    return cls(params)
