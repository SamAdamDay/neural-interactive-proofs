import torch

from .base import Trainer
from ..parameters import Parameters


def build_trainer(params: Parameters, device: str | torch.device) -> Trainer:
    pass
