"""Base classes for dealing with pretrained models for transfer learning.

Pretrained models can by used to generate embeddings for datasets, which can then be
used by agents instead of raw data. This means the agent starts with a good level of
knowledge about the data.

This module contains the base class for pretrained models, as well as a registry for
them.
"""

from abc import ABC, abstractmethod
from typing import Optional, Iterable, TypeVar

import torch
from torch import nn

from nip.parameters import HyperParameters
from nip.experiment_settings import ExperimentSettings
from nip.scenario_base.data import TensorDictDataset
from nip.constants import HF_PRETRAINED_MODELS_USER


class PretrainedModel(ABC):
    """Base class for pretrained models, used to generate embeddings for datasets.

    Pretrained models can by used to generate embeddings for datasets, which can then be
    used by agents instead of raw data. This means the agent starts with a good level of
    knowledge about the data.

    Parameters
    ----------
    hyper_params : HyperParameters
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

    def __init__(self, hyper_params: HyperParameters, settings: ExperimentSettings):
        super().__init__()
        self.hyper_params = hyper_params
        self.settings = settings
        self._model: Optional[nn.Module] = None

    @abstractmethod
    def generate_dataset_embeddings(
        self, datasets: Iterable[TensorDictDataset], delete_model: bool = True
    ) -> torch.Tensor:
        """Load the model and generate embeddings for the datasets.

        Parameters
        ----------
        datasets : Iterable[TensorDictDataset]
            The datasets to generate embeddings for
        delete_model : bool, default=True
            Whether to delete the model after generating embeddings

        Returns
        -------
        embeddings : torch.Tensor
            The embeddings for the datasets
        """


PRETRAINED_MODEL_CLASSES: dict[str, type[PretrainedModel]] = {}

P = TypeVar("P", bound=PretrainedModel)


def register_pretrained_model_class(pretrained_model_cls: type[P]) -> type[P]:
    """Register a pretrained model class, so it can be built by name.

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


def get_pretrained_model_class(
    model_name: str, hyper_params: HyperParameters
) -> type[PretrainedModel]:
    """Get the class for a pretrained model by name.

    If the full model name is not found, it tries to find a model with the name:
        f"{HF_PRETRAINED_MODELS_USER}/{model_name}_{hyper_params.dataset}"

    Parameters
    ----------
    model_name : str
        The name of the pretrained model to build
    hyper_params : HyperParameters
        The parameters for the experiment
    """

    if model_name in PRETRAINED_MODEL_CLASSES:
        model_class = PRETRAINED_MODEL_CLASSES[model_name]
    else:
        augmented_model_name = (
            f"{HF_PRETRAINED_MODELS_USER}/{model_name}_{hyper_params.dataset}"
        )
        if augmented_model_name in PRETRAINED_MODEL_CLASSES:
            model_class = PRETRAINED_MODEL_CLASSES[augmented_model_name]
        else:
            raise ValueError(
                f"Unknown pretrained model {model_name!r} for dataset "
                f"{hyper_params.dataset!r}"
            )

    if (
        model_class.dataset != hyper_params.dataset
        and not model_class.allow_other_datasets
    ):
        raise ValueError(
            f"Model {model_name!r} was trained on {model_class.dataset!r}, but the "
            f"experiment is using dataset {hyper_params.dataset!r}"
        )

    return model_class
