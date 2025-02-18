"""Classes for loading pretrained image models, to use their embeddings.

Models are loaded from a 'hub' somewhere (e.g. PyTorch Image Models (timm) or Hugging
Face Transformers).

The models are then used to extract embeddings from images, which can be used to aid
agents in image classification tasks.
"""

from abc import ABC
from typing import Optional, Any
from math import ceil

import torch
from torch import Tensor
from torch.utils.data import DataLoader, default_collate

import timm
from timm.models import ResNet
from timm.data import resolve_data_config, create_transform

from jaxtyping import Float

from nip.scenario_base.data import TensorDictDataLoader
from nip.scenario_base.pretrained_models import (
    PretrainedModel,
    register_pretrained_model_class,
)
from nip.image_classification.data import (
    ImageClassificationDataset,
    DATASET_WRAPPER_CLASSES,
)
from nip.utils.oop import classproperty
from nip.constants import HF_PRETRAINED_MODELS_USER


class PretrainedImageModel(PretrainedModel, ABC):
    """Base class for pretrained image models."""

    embedding_width: int
    embedding_height: int


class Resnet18PretrainedModel(PretrainedImageModel, ABC):
    """Base class for pretrained ResNet models using PyTorch Image Models (timm).

    These models are hosted on Hugging Face and are loaded using the PyTorch Image
    Models (timm) library.

    Derived classes should define the class attributes below.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters for the experiment
    settings : ExperimentSettings
        The settings for the experiment

    Class attributes
    ----------------
    dataset : str
        The name of the dataset the model was trained for
    allow_other_datasets : bool, default=False
        Whether the model can be used for datasets other than the one it was trained on
    """

    dataset: str
    allow_other_datasets: bool = False

    base_model_name = "resnet18"
    embedding_downscale_factor = 4
    embedding_channels = 512

    @classproperty
    def name(cls):
        """The name of the model."""  # noqa: D401
        return f"{HF_PRETRAINED_MODELS_USER}/{cls.base_model_name}_{cls.dataset}"

    @classproperty
    def timm_uri(cls):
        """The URI of the model in the timm library."""  # noqa: D401
        return f"hf_hub:{cls.name}"

    @torch.no_grad()
    def generate_dataset_embeddings(
        self, datasets: dict[str, ImageClassificationDataset], delete_model: bool = True
    ) -> dict[str, Tensor]:
        """Load the model and generate embeddings for the datasets.

        Parameters
        ----------
        datasets : dict[str, ImageClassificationDataset]
            The datasets to generate embeddings for
        delete_model : bool, default=True
            Whether to delete the model after generating the embeddings

        Returns
        -------
        embeddings : dict[str, Tensor]
            The embeddings for each dataset
        """

        # Load the model from the hub
        model: ResNet = timm.create_model(self.timm_uri, pretrained=True)
        model.eval()
        model.to(self.settings.device)

        batch_size = self.settings.pretrained_embeddings_batch_size

        embeddings = {}

        for dataset_name, dataset in datasets.items():

            # Load the base PyTorch dataset (expected by timm) with the transforms
            transform = self._get_transform(model)
            torch_dataset = dataset.build_torch_dataset(transform=transform)

            # Create the dataloader
            dataloader = TensorDictDataLoader(
                torch_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=default_collate,
            )

            # Initialize the embeddings tensor
            embeddings[dataset_name] = torch.empty(
                len(torch_dataset),
                self.embedding_channels,
                self.embedding_height,
                self.embedding_width,
            )

            # Compute the embeddings for each batch of images
            pbar = self.settings.tqdm_func(
                total=len(dataloader),
                desc=f"Generating embeddings for {self.name}, {dataset_name}",
            )
            for idx, (image, y) in enumerate(dataloader):
                image = image.to(self.settings.device)
                batch_embeddings = model.forward_features(image)
                batch_embeddings = batch_embeddings.to("cpu")
                embeddings[dataset_name][
                    idx * batch_size : (idx + 1) * batch_size
                ] = batch_embeddings
                pbar.update(1)
            pbar.close()

        if delete_model:
            del model

        return embeddings

    @classproperty
    def embedding_width(cls) -> int:
        """The width of the embeddings produced by the model.

        Must be a factor of the dataset's image width.
        """  # noqa: D401
        return ceil(
            DATASET_WRAPPER_CLASSES[cls.dataset].width / cls.embedding_downscale_factor
        )

    @classproperty
    def embedding_height(cls) -> int:
        """The height of the embeddings produced by the model.

        Must be a factor of the dataset's image height.
        """  # noqa: D401
        return ceil(
            DATASET_WRAPPER_CLASSES[cls.dataset].height / cls.embedding_downscale_factor
        )

    def _get_transform(self, model: ResNet) -> Any | None:
        """Get the transform to apply to images before passing them to the model.

        Returns
        -------
        transform : torchvision transform or None
            The transform to apply to images before passing them to the model
        """

        # Get the timm data configuration from the pretrained model
        data_cfg = resolve_data_config(model=model)

        return create_transform(
            input_size=data_cfg["input_size"],
            is_training=False,
            use_prefetcher=False,
            interpolation=data_cfg["interpolation"],
            mean=data_cfg["mean"],
            std=data_cfg["std"],
            crop_pct=data_cfg["crop_pct"],
            crop_mode=data_cfg["crop_mode"],
        )


@register_pretrained_model_class
class Resnet18Cifar10PretrainedModel(Resnet18PretrainedModel):
    """Resnet18 model trained on CIFAR-10."""

    dataset = "cifar10"
