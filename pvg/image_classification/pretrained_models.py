"""Classes for loading pretrained image models, to use their embeddings

Models are loaded from a 'hub' somewhere (e.g. PyTorch Image Models (timm) or Hugging
Face Transformers).

The models are then used to extract embeddings from images, which can be used to aid
agents in image classification tasks.
"""

from abc import ABC
from typing import Optional
from math import ceil

import torch
from torch import Tensor

import timm
from timm.models import ResNet

from jaxtyping import Float

from pvg.scenario_base.data import DataLoader
from pvg.scenario_base.pretrained_models import (
    PretrainedModel,
    register_pretrained_model_class,
)
from pvg.image_classification.data import (
    ImageClassificationDataset,
    DATASET_WRAPPER_CLASSES,
)
from pvg.utils.oop import classproperty
from pvg.constants import HF_PRETRAINED_MODELS_USER


class PretrainedImageModel(PretrainedModel, ABC):
    """Base class for pretrained image models

    Derived classes should define the class attributes below.

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
    embedding_downscale_factor : int, optional
        The factor by which the model decreases the image size to produce embeddings.
        Instead of defining this, you can define `embedding_width` and
        `embedding_height` directly.
    embedding_channels : int
        The number of channels in the embeddings produced by the model.
    """

    name: str
    dataset: str
    allow_other_datasets: bool = False
    embedding_downscale_factor: Optional[int] = None
    embedding_channels: int

    @torch.no_grad()
    def generate_dataset_embeddings(
        self, datasets: dict[str, ImageClassificationDataset], delete_model: bool = True
    ) -> dict[str, Tensor]:
        """Load the model and generate embeddings for the datasets

        Parameters
        ----------
        datasets : dict[str, ImageClassificationDataset]
            The datasets to generate embeddings for
        delete_model : bool, default=True
            Whether to delete the model after generating the embeddings

        Returns
        -------

        """

        self.load_model()
        self._model.eval()
        self._model.to(self.settings.device)

        batch_size = self.settings.pretrained_embeddings_batch_size

        embeddings = {}

        for dataset_name, dataset in datasets.items():

            dataloader = DataLoader(dataset, batch_size=batch_size)
            embeddings[dataset_name] = torch.empty(
                len(dataset),
                self.embedding_channels,
                self.embedding_height,
                self.embedding_width,
            )

            # Compute the embeddings for each batch of images
            pbar = self.settings.tqdm_func(
                total=len(dataloader),
                desc=f"Generating embeddings for {self.name}, {dataset_name}",
            )
            for idx, batch in enumerate(dataloader):
                images = batch["image"].to(self.settings.device)
                batch_embeddings = self.forward(images).to("cpu")
                embeddings[dataset_name][
                    idx * batch_size : (idx + 1) * batch_size
                ] = batch_embeddings
                pbar.update(1)
            pbar.close()

        if delete_model:
            del self._model

        return embeddings

    @classproperty
    def embedding_width(cls) -> int:
        """The width of the embeddings produced by the model

        Must be a factor of the dataset's image width.
        """

        if cls.embedding_downscale_factor is None:
            raise NotImplementedError(
                "You must define either `embedding_downscale_factor` or "
                "`embedding_width` and `embedding_height`"
            )

        return ceil(
            DATASET_WRAPPER_CLASSES[cls.dataset].width / cls.embedding_downscale_factor
        )

    @classproperty
    def embedding_height(cls) -> int:
        """The height of the embeddings produced by the model

        Must be a factor of the dataset's image height.
        """

        if cls.embedding_downscale_factor is None:
            raise NotImplementedError(
                "You must define either `embedding_downscale_factor` or "
                "`embedding_width` and `embedding_height`"
            )

        return ceil(
            DATASET_WRAPPER_CLASSES[cls.dataset].height / cls.embedding_downscale_factor
        )


class Resnet18PretrainedModel(PretrainedImageModel, ABC):
    """Base class for Resnet18 models

    These models are hosted on Hugging Face.

    They are loaded using the PyTorch Image Models (timm) library.
    """

    embedding_downscale_factor = 32
    embedding_channels = 512

    _model: ResNet

    @classproperty
    def name(cls):
        return f"{HF_PRETRAINED_MODELS_USER}/resnet18_{cls.dataset}"

    @classproperty
    def timm_uri(cls):
        return f"hf_hub:{cls.name}"

    def load_model(self):
        self._model = timm.create_model(self.timm_uri, pretrained=True)

    def forward(
        self, x: Float[Tensor, "batch 3 width height"]
    ) -> Float[Tensor, "batch embedding_channels embedding_width embedding_height"]:
        return self._model.forward_features(x)


@register_pretrained_model_class
class Resnet18Cifar10PretrainedModel(Resnet18PretrainedModel):
    """Resnet18 model trained on CIFAR-10"""

    dataset = "cifar10"
