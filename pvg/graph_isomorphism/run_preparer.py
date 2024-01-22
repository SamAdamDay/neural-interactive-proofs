"""Prepare for doing a graph isomorphism run.

Downloads and processes the data for the run.
"""

from pvg.scenario_base import RunPreparer
from pvg.graph_isomorphism.data import GraphIsomorphismDataset


class GraphIsomorphismRunPreparer(RunPreparer):
    """Prepare for doing an image classification run.

    Downloads and processes the data for the run.

    Parameters
    ----------
    params : Parameters
        The parameters for the experiment.
    settings : ExperimentSettings
        The settings for the experiment.
    """

    def prepare_run(self):
        """Prepare the run.

        Downloads and processes the data for the run, then deletes the dataset object.
        """
        dataset = GraphIsomorphismDataset(params=self.params, settings=self.settings)
        del dataset
