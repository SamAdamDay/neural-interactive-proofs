"""Prepare for doing an image classification run.

Downloads and processes the data for the run.
"""

from pvg.parameters import ScenarioType
from pvg.scenario_base import RunPreparer, register_run_preparer
from pvg.image_classification.data import ImageClassificationDataset


@register_run_preparer(ScenarioType.IMAGE_CLASSIFICATION)
class ImageClassificationRunPreparer(RunPreparer):
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
        train_dataset = ImageClassificationDataset(
            params=self.params, settings=self.settings, train=True
        )
        test_dataset = ImageClassificationDataset(
            params=self.params, settings=self.settings, train=False
        )
        del train_dataset
        del test_dataset
