"""Base classes for all trainers."""

from abc import ABC, abstractmethod

from pvg.parameters import Parameters
from pvg.scenario_instance import ScenarioInstance
from pvg.experiment_settings import ExperimentSettings


class Trainer(ABC):
    """Base class for all trainers.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    scenario_instance : ScenarioInstance
        The components of the experiment.
    settings : ExperimentSettings
        The instance-specific settings of the experiment, like device, logging, etc.
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

        self._agent_names = self.scenario_instance.protocol_handler.agent_names

        self.device = self.settings.device

    @abstractmethod
    def train(self):
        """Train the agents."""
        pass
