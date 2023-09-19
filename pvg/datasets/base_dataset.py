from abc import ABC, abstractmethod

import torch

from pvg.parameters import Parameters


class Dataset(ABC):
    """Base class for all datasets.

    Parameters
    ----------
    parameters : Parameters
        The parameters of the experiment.
    """

    def __init__(self, parameters: Parameters):
        self.parameters = parameters
