"""Utilities for working with data."""


from typing import Optional

import torch

from tensordict.tensordict import TensorDict

from pvg.scenario_base import DataLoader


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
    """

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.dataloader_iter = iter(forgetful_cycle(self.dataloader))
        self.remainder: Optional[list[TensorDict]] = None

    def get_batch(self, batch_size: int) -> TensorDict:
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
