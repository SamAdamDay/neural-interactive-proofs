import torch

from .base import Scenario, Prover, Verifier
from .graph_isomorphism import (
    GraphIsomorphismScenario,
    GraphIsomorphismAgent,
    GraphIsomorphismProver,
    GraphIsomorphismVerifier,
)
from ..parameters import Parameters


def build_scenario(parameters: Parameters, device: str | torch.device) -> Scenario:
    for value in globals().values():
        if issubclass(value, Scenario) and value.name == parameters.scenario:
            cls = value
            break
    return cls(parameters, device)
