"""Script to download all datasets used in the project.

This is useful to ensure that all datasets are already in the Docker image. This script
can be called from the Dockerfile.
"""

from pvg import HyperParameters, ScenarioType, TrainerType, prepare_experiment
from pvg.image_classification.data import DATASET_WRAPPER_CLASSES

if __name__ == "__main__":

    hyper_params = HyperParameters(
        scenario="image_classification",
        trainer="solo_agent",
        dataset="test",
    )
    for dataset_name in DATASET_WRAPPER_CLASSES.keys():
        print(f"Downloading dataset {dataset_name}...")  # noqa: T201
        hyper_params.dataset = dataset_name
        prepare_experiment(hyper_params=hyper_params, ignore_cache=True)
