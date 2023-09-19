import torch

from .base_dataset import Dataset
from ..parameters import Parameters


def build_dataset(parameters: Parameters, device: str | torch.device) -> Dataset:
    pass
