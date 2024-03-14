"""Base classes for preparation before running experiments

The use of this module is optional, but can help avoid problems when running multiple
experiments in parallel.
"""

from abc import ABC, abstractmethod

from pvg.parameters import Parameters, ScenarioType
from pvg.experiment_settings import ExperimentSettings
from pvg.protocols import ProtocolHandler


class RunPreparer(ABC):
    """Base class for preparing a run.

    This is useful e.g. for downloading data before running an experiment. Without this,
    if running multiple experiments in parallel, the initial runs will all start
    downloading data at the same time, which can cause problems.

    To subclass, implement the following methods:

    - `prepare_run`: Prepare the run.

    Parameters
    ----------
    params : Parameters
        The parameters for the experiment.
    settings : ExperimentSettings
        The settings for the experiment.
    """

    def __init__(
        self,
        params: Parameters,
        settings: ExperimentSettings,
        protocol_handler: ProtocolHandler,
    ):
        self.params = params
        self.settings = settings
        self.protocol_handler = protocol_handler

    @abstractmethod
    def prepare_run(self):
        """Prepare the run."""
        pass


RUN_PREPARER_REGISTRY: dict[ScenarioType, type[RunPreparer]] = {}


def register_run_preparer(scenario: ScenarioType):
    """Register a subclass of RunPreparer with a scenario.

    Parameters
    ----------
    scenario : ScenarioType
        The scenario with which to register the subclass.
    """

    def decorator(cls):
        RUN_PREPARER_REGISTRY[scenario] = cls
        return cls

    return decorator


def build_run_preparer(
    params: Parameters, settings: ExperimentSettings, protocol_handler: ProtocolHandler
) -> RunPreparer:
    """Build a subclass of RunPreparer based on the parameters.

    Parameters
    ----------
    params : Parameters
        The parameters for the experiment.
    settings : ExperimentSettings
        The settings for the experiment.

    Returns
    -------
    RunPreparer
        The prepared run.
    """
    return RUN_PREPARER_REGISTRY[params.scenario](params, settings, protocol_handler)
