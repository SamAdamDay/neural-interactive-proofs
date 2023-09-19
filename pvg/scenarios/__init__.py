import torch

from .base_scenario import Scenario, Prover, Verifier
from ..parameters import Parameters


def build_scenario(parameters: Parameters, device: str | torch.device) -> Scenario:
    pass
