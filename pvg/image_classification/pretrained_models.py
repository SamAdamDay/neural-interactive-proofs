"""Classes for loading pretrained image models, to use their embeddings

Models are loaded from a 'hub' somewhere (e.g. PyTorch Image Models (timm) or Hugging
Face Transformers).

The models are then used to extract embeddings from images, which can be used to aid
agents in image classification tasks.
"""

from abc import ABC

import torch
from torch import nn

import timm

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
    embedding_width : int
        The width of the embeddings produced by the model. Must be a factor of the
        dataset's image width.
    embedding_height : int
        The height of the embeddings produced by the model. Must be a factor of the
        dataset's image height.
    """

    name: str
    dataset: str
    allow_other_datasets: bool = False
    embedding_width: int
    embedding_height: int

    def generate_dataset_embeddings(
        self, datasets: dict[str, ImageClassificationDataset], delete_model: bool = True
    ) -> dict[str, torch.Tensor]:
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
                len(dataset), self.embedding_height, self.embedding_width
            )

            # Compute the embeddings for each batch of images
            pbar = self.settings.tqdm_func(
                total=4, desc=f"Generating embeddings for {self.name}, {dataset_name}"
            )
            for idx, batch in enumerate(dataloader):
                images = batch["image"].to(self.settings.device)
                embeddings[dataset_name][idx * batch_size : (idx + 1) * batch_size] = (
                    self.forward(images).to("cpu")
                )
                pbar.update(1)
            pbar.close()

        if delete_model:
            del self._model

        return embeddings


class Resnet18PretrainedModel(PretrainedImageModel, ABC):
    """Base class for Resnet18 models

    These models are hosted on Hugging Face: https://huggingface.co/SamAdamDay

    They are loaded using the PyTorch Image Models (timm) library.
    """

    @classproperty
    def name(cls):
        return f"SamAdamDay/resnet18_{cls.dataset}"

    @classproperty
    def embedding_width(cls):
        return DATASET_WRAPPER_CLASSES[cls.dataset].width // 2  # TODO

    @classproperty
    def embedding_height(cls):
        return DATASET_WRAPPER_CLASSES[cls.dataset].height // 2  # TODO

    @classproperty
    def timm_uri(cls):
        return f"hf_hub:{cls.name}"

    def load_model(self):
        self._model = timm.create_model(self.timm_uri, pretrained=True)


@register_pretrained_model_class()
class Resnet18Cifar10PretrainedModel(Resnet18PretrainedModel):
    """Resnet18 model trained on CIFAR-10"""

    dataset = "cifar10"
