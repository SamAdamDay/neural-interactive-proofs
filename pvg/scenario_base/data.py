"""Base classes for handling data.

Dataloaders yield tensordicts.
"""

from abc import ABC, abstractmethod
from typing import Union, Any
import shutil
import os
from textwrap import indent

import torch
from torch import Tensor

from tensordict import TensorDict
from tensordict.utils import _td_fields

from pvg.parameters import Parameters
from pvg.experiment_settings import ExperimentSettings


class Dataset(ABC):
    """Base class for all datasets.

    The dataset is stored a as a memory-mapped tensordict. See
    https://pytorch.org/tensordict/saving.html

    To subclass, implement the following methods:

    - `raw_dir` (property): The path to the directory containing the raw data.
    - `processed_dir` (property): The path to the directory containing the processed
      data.
    - `_build_tensor_dict`: Build the tensordict from the raw data.
    - `_download` (optional): Download the raw data.

    Parameters
    ----------
    params : Parameters
        The parameters for the experiment.
    settings : ExperimentSettings
        The settings for the experiment.
    """

    _tensor_dict: TensorDict

    def __init__(self, params: Parameters, settings: ExperimentSettings):
        self.params = params
        self.settings = settings

        # Download the raw data if this is implemented
        try:
            self._download()
        except NotImplementedError:
            pass

        if not os.path.isdir(self.processed_dir) or settings.ignore_cache:
            # Delete the processed directory if it exists
            if os.path.isdir(self.processed_dir) and settings.ignore_cache:
                shutil.rmtree(self.processed_dir)

            # Create the tensordict and save it to disk as a memory-mapped file
            self._tensor_dict = self._build_tensor_dict()
            self._tensor_dict.memmap_(
                self.processed_dir, num_threads=settings.num_dataset_threads
            )

        else:
            # Load the memory-mapped tensordict
            self._tensor_dict = TensorDict.load_memmap(self.processed_dir)

    def _download(self):
        """Download the raw data."""
        raise NotImplementedError

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

    def __repr__(self) -> str:
        # Adapted from TensorDictBase.__repr__
        fields = _td_fields(self._tensor_dict)
        field_str = indent(f"fields={{{fields}}}", 4 * " ")
        batch_size_str = indent(f"batch_size={self._tensor_dict.batch_size}", 4 * " ")
        device_str = indent(f"device={self._tensor_dict.device}", 4 * " ")
        is_shared_str = indent(f"is_shared={self._tensor_dict.is_shared()}", 4 * " ")
        string = ",\n".join([field_str, batch_size_str, device_str, is_shared_str])
        return f"{type(self).__name__}(\n{string})"


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
