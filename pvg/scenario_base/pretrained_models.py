from abc import ABC, abstractmethod
from typing import Optional, Iterable

import torch
from torch import nn

from pvg.parameters import Parameters
from pvg.experiment_settings import ExperimentSettings
from pvg.scenario_base.data import Dataset, DataLoader


class PretrainedModel(ABC):
    """Base class for pretrained models

    Parameters
    ----------
    params : Parameters
        The parameters for the experiment
    settings : ExperimentSettings
        The settings for the experiment

    Class attributes
    ----------------
    name : str
        The name of the model, which should uniquely identify it
    dataset : str
        The name of the dataset the model was trained for
    allow_other_datasets : bool, default=False
        Whether the model can be used for datasets other than the one it was trained on
    """

    name: str
    dataset: str
    allow_other_datasets: bool = False

    def __init__(self, params: Parameters, settings: ExperimentSettings):
        super().__init__()
        self.params = params
        self.settings = settings
        self._model: Optional[nn.Module] = None

    @abstractmethod
    def generate_dataset_embeddings(
        self, datasets: Iterable[Dataset], delete_model: bool = True
    ):
        """Load the model and generate embeddings for the datasets"""

    @abstractmethod
    def load_model(self):
        """Load the model.

        Most implementations will store the model in `self._model`
        """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the embeddings of an datapoint or batch of datapoints

        Parameters
        ----------
        x : torch.Tensor
            The input datapoint(s) to compute embeddings for

        Returns
        -------
        torch.Tensor
            The embeddings of the input datapoint(s)
        """

    def delete_model(self):
        """Delete the loaded model."""
        self._model = None


PRETRAINED_MODEL_CLASSES: dict[str, type[PretrainedModel]] = {}


def register_pretrained_model_class(
    pretrained_model_cls: type[PretrainedModel],
) -> type[PretrainedModel]:
    """Decorator to register a pretrained model class, so it can be built by name

    Parameters
    ----------
    pretrained_model_cls : type[PretrainedModel]
        The class to register

    Returns
    -------
    pretrained_model_cls : type[PretrainedModel]
        The class that was registered, unchanged
    """

    if pretrained_model_cls.name in PRETRAINED_MODEL_CLASSES:
        raise ValueError(
            f"Pretrained model class with name {pretrained_model_cls.name} already"
            " exists"
        )

    PRETRAINED_MODEL_CLASSES[pretrained_model_cls.name] = pretrained_model_cls

    return pretrained_model_cls


def build_pretrained_model(
    model_name: str, params: Parameters, settings: ExperimentSettings
) -> PretrainedModel:
    """Build a pretrained model by name

    Parameters
    ----------
    model_name : str
        The name of the pretrained model to build
    params : Parameters
        The parameters for the experiment
    settings : ExperimentSettings
        The settings for the experiment
    """

    if model_name not in PRETRAINED_MODEL_CLASSES:
        raise ValueError(f"Unknown pretrained model: {model_name}")

    model_class = PRETRAINED_MODEL_CLASSES[model_name]

    if model_class.dataset != params.dataset and not model_class.allow_other_datasets:
        raise ValueError(
            f"Model {model_name} was trained on {model_class.dataset}, but the "
            f"experiment is using dataset {params.dataset}"
        )

    return model_class(params, settings)
