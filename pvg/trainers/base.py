"""Base classes for all trainers."""

from abc import ABC, abstractmethod

from pvg.parameters import Parameters
from pvg.scenario_base.scenario_instance import ScenarioInstance
from pvg.experiment_settings import ExperimentSettings


class Trainer(ABC):
    """Base class for all trainers.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    scenario_instance : ScenarioInstance
        The components of the experiment.
    device : TorchDevice
        The device to use for training.
    """

    def __init__(
        self,
        params: Parameters,
        scenario_instance: ScenarioInstance,
        settings: ExperimentSettings,
    ):
        self.params = params
        self.scenario_instance = scenario_instance
        self.settings = settings

        self.device = self.settings.device

    @abstractmethod
    def train(self):
        """Train the agents."""
        pass
