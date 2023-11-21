from abc import ABC, abstractmethod

import torch

from pvg.parameters import Parameters


class Dataset(torch.utils.data.Dataset, ABC):
    """Base class for all datasets."""

    pass


class DataLoader(torch.utils.data.DataLoader):
    """Base class for all dataloaders."""

    pass


def load_dataset(params: Parameters) -> Dataset:
    for value in globals().values():
        if issubclass(value, Dataset) and value.name == params.dataset:
            cls = value
            break
    return cls(params)
