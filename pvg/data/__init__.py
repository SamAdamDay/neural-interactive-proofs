import torch

from .base import Dataset
from .graph_isomorphism import GraphIsomorphismDataset, GraphIsomorphismData
from ..parameters import Parameters


def load_dataset(params: Parameters) -> Dataset:
    for value in globals().values():
        if issubclass(value, Dataset) and value.name == params.dataset:
            cls = value
            break
    return cls(params)
