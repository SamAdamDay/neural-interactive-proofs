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
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.dataloader import (
    _BaseDataLoaderIter,
    _SingleProcessDataLoaderIter,
    _MultiProcessingDataLoaderIter,
)

from tensordict import TensorDict, MemoryMappedTensor
from tensordict.utils import _td_fields, IndexType

from pvg.protocols import ProtocolHandler
from pvg.parameters import Parameters
from pvg.experiment_settings import ExperimentSettings
from pvg.utils.types import TorchDevice
from pvg.utils.data import is_nested_key


class CachedPretrainedEmbeddingsNotFound(Exception):
    """Raised when the cached embeddings for a pretrained model are not found."""

    def __init__(self, model_name: str):
        super().__init__(f"Cached embeddings for model {model_name} not found")
        self.model_name = model_name


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
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    train : bool
        Whether to load the training or test set.
    """

    _main_data: TensorDict

    def __init__(
        self,
        params: Parameters,
        settings: ExperimentSettings,
        protocol_handler: ProtocolHandler,
        train: bool = True,
    ):
        self.params = params
        self.settings = settings
        self.protocol_handler = protocol_handler
        self.train = train

        # Download the raw data if this is implemented
        try:
            self._download()
        except NotImplementedError:
            pass

        if not os.path.isdir(self.processed_dir) or settings.ignore_cache:
            # Delete the processed directory if it exists
            if os.path.isdir(self.processed_dir) and settings.ignore_cache:
                shutil.rmtree(self.processed_dir)

            # Create the tensordict of the dataset and save it to disk as a
            # memory-mapped file
            self._main_data = self._build_tensor_dict()
            self._main_data.memmap_(
                self.processed_dir, num_threads=settings.num_dataset_threads
            )

        else:
            # Load the memory-mapped tensordict
            self._main_data = TensorDict.load_memmap(self.processed_dir)

        if self.settings.dataset_on_device:
            self._main_data.to(self.settings.device)

        # Create a tensordict to store pretrained model embeddings
        self._pretrained_embeddings = TensorDict(
            {}, batch_size=self._main_data.batch_size, device=self._main_data.device
        )

    @property
    def device(self) -> TorchDevice:
        return self._main_data.device

    def load_pretrained_embeddings(self, model_name: str):
        """Load cached embeddings for a pretrained model.

        Parameters
        ----------
        model_name : str
            The name of the pretrained model.

        Raises
        ------
        CachedPretrainedEmbeddingsNotFound
            If the cached embeddings are not found.
        """

        if model_name in self._pretrained_embeddings.keys():
            raise ValueError(
                f"Pretrained embeddings for model {model_name} have already been loaded"
            )

        cache_path = os.path.join(self.pretrained_embeddings_dir, f"{model_name}.pt")

        if not os.path.isfile(cache_path):
            raise CachedPretrainedEmbeddingsNotFound(model_name)

        self._pretrained_embeddings[model_name] = MemoryMappedTensor.from_filename(
            cache_path
        )

    def add_pretrained_embeddings(
        self, model_name: str, embeddings: Tensor, overwrite_cache: bool = False
    ):
        """Add pretrained embeddings to the dataset and cache them.

        Parameters
        ----------
        model_name : str
            The name of the pretrained model.
        embeddings : Tensor
            The embeddings to add to the dataset.
        overwrite_cache : bool, default=False
            Whether to overwrite the cached embeddings if they already exist.
        """

        if model_name in self._pretrained_embeddings.keys():
            raise ValueError(
                f"Pretrained embeddings for model {model_name} have already been loaded"
            )

        cache_path = os.path.join(self.pretrained_embeddings_dir, f"{model_name}.pt")

        if os.path.isfile(cache_path) and not overwrite_cache:
            raise ValueError(
                f"Pretrained embeddings for model {model_name} already exist in the "
                "cache and `overwrite_cache` is False."
            )

        # Save the embeddings to disk as a memory-mapped tensor
        embeddings = MemoryMappedTensor.from_tensor(
            embeddings, filename=cache_path, existsok=overwrite_cache
        )

        self._pretrained_embeddings[model_name] = embeddings

    def _download(self):
        """Download the raw data."""
        raise NotImplementedError

    @property
    @abstractmethod
    def raw_dir(self) -> str:
        """The path to the directory containing the raw data."""

    @property
    @abstractmethod
    def processed_dir(self) -> str:
        """The path to the directory containing the processed data."""

    @property
    def pretrained_embeddings_dir(self) -> str:
        """The path to the directory containing cached pretrained model embeddings."""
        raise NotImplementedError

    @abstractmethod
    def _build_tensor_dict(self) -> TensorDict:
        """Build the tensordict from the raw data."""

    def __getitem__(self, index: IndexType) -> TensorDict | Tensor:

        # If the index is a nested key, we're trying to get an item of the tensordict.
        # First try to get the item from the main data, and if it doesn't exist, try to
        # get it from the pretrained embeddings.
        if is_nested_key(index):
            try:
                return self._main_data.__getitem__(index)
            except KeyError as e:
                try:
                    return self._pretrained_embeddings.__getitem__(index)
                except KeyError:
                    raise e

        # Get the main data and clone the structure (but not the tensors). This is
        # needed because the tensordict is memory-mapped and so locked.
        data = self._main_data.__getitem__(index).clone(recurse=False)

        # Add the pretrained embeddings if they exist, as a nested tensordict
        if len(self._pretrained_embeddings.keys()) > 0:
            pretrained_embeddings = self._pretrained_embeddings.__getitem__(index)
            data.update(dict(pretrained_embeddings=pretrained_embeddings))

        return data

    # All of the logic for getting multiple items is performed in __getitem__
    __getitems__ = __getitem__

    def __len__(self) -> int:
        return len(self._main_data)

    def __repr__(self) -> str:
        # Adapted from TensorDictBase.__repr__
        fields = _td_fields(self._main_data)
        field_str = indent(f"fields={{{fields}}}", 4 * " ")
        batch_size_str = indent(f"batch_size={self._main_data.batch_size}", 4 * " ")
        device_str = indent(f"device={self._main_data.device}", 4 * " ")
        is_shared_str = indent(f"is_shared={self._main_data.is_shared()}", 4 * " ")
        string = ",\n".join([field_str, batch_size_str, device_str, is_shared_str])
        return f"{type(self).__name__}(\n{string})"


class TensorDictSingleProcessDataLoaderIter(_SingleProcessDataLoaderIter):
    """Single process DataLoaderIter for tensordicts.

    This is a hack to allow the DataLoader to work with TensorDict memory pinning.
    """

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data: TensorDict = self._dataset_fetcher.fetch(index)  # may raise StopIteration

        if self._pin_memory:

            def pin_memory(tensor: Tensor):
                return tensor.pin_memory(self._pin_memory_device)

            data = data._fast_apply(pin_memory)

        return data


class DataLoader(TorchDataLoader):
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

    def _get_iterator(self) -> _BaseDataLoaderIter:
        if self.num_workers == 0:
            return TensorDictSingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)
