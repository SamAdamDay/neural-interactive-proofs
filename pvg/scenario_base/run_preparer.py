"""Base classes for preparation before running experiments

The use of this module is optional, but can help avoid problems when running multiple
experiments in parallel.
"""

from abc import ABC, abstractmethod

from pvg.parameters import Parameters
from pvg.experiment_settings import ExperimentSettings


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

    def __init__(self, params: Parameters, settings: ExperimentSettings):
        self.params = params
        self.settings = settings

    @abstractmethod
    def prepare_run(self):
        """Prepare the run."""
        pass
