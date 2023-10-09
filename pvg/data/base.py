from abc import ABC, abstractmethod

import torch

from pvg.parameters import Parameters


class Dataset(torch.utils.data.Dataset, ABC):
    """Base class for all datasets."""

    pass


class DataLoader(torch.utils.data.DataLoader):
    """Base class for all dataloaders."""

    pass
