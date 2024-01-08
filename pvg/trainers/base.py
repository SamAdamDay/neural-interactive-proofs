"""Base classes for all trainers."""

from abc import ABC, abstractmethod

from pvg.parameters import Parameters
from pvg.scenario_base.component_holder import ComponentHolder
from pvg.experiment_settings import ExperimentSettings


class Trainer(ABC):
    """Base class for all trainers.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    device : TorchDevice
        The device to use for training.
    """

    def __init__(
        self,
        params: Parameters,
        component_holder: ComponentHolder,
        settings: ExperimentSettings,
    ):
        self.params = params
        self.component_holder = component_holder
        self.settings = settings

    @abstractmethod
    def train(self):
        """Train the agents."""
        pass
