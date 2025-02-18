"""Data classes for the image classification task.

Contains the `ImageClassificationDataset` class, which is a dataset for the image
classification task. The dataset is a binary classification problem, where the classes
are selected from a torchvision dataset. The dataset is binarified using one of the
following methods:

- `"merge"`: The classes are shuffled and merged into two
  classes.
- `"select_two"`: Two classes are selected from the original
    dataset.
- `"random"`: The classes are selected at random.
"""

import os
from typing import Optional, Any, TypeVar
from abc import ABC
from pathlib import Path

import torch
from torch.utils.data import DataLoader as TorchDataLoader

from torchvision.datasets import (
    VisionDataset,
    MNIST,
    CIFAR10,
    FashionMNIST,
    FakeData,
    CIFAR100,
    KMNIST,
    SVHN,
)
from torchvision import transforms

from tensordict import TensorDict

from nip.parameters import BinarificationMethodType
from nip.parameters import ScenarioType
from nip.factory import register_scenario_class
from nip.scenario_base import Dataset, TensorDictDataset
from nip.constants import IC_DATA_DIR


class TorchVisionDatasetWrapper(ABC):
    """A wrapper for TorchVision datasets, implementing a common interface.

    Derived classes should defined the class attributes below.

    Class attributes
    ----------------
    data_class : type[VisionDataset]
        The TorchVision dataset class.
    num_channels : int
        The number of channels of the images in the dataset.
    width : int
        The width of the images in the dataset.
    height : int
        The height of the images in the dataset.
    selected_classes : tuple[int, int]
        When selecting two classes from the original dataset, the default two to select.
    binarification_seed : int
        The seed used when doing a randomised binarification.
    """

    num_channels: int
    width: int
    height: int
    selected_classes: tuple[int, int] = (0, 1)
    binarification_seed: int = 0

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        download: bool = False,
    ):
        pass


DATASET_WRAPPER_CLASSES: dict[str, type[TorchVisionDatasetWrapper]] = {}

D = TypeVar("D", bound=TorchVisionDatasetWrapper)


def register_dataset_wrapper_class(dataset_name: str) -> callable:
    """Register a dataset wrapper class."""

    def decorator(wrapper_class: type[D]) -> type[D]:
        DATASET_WRAPPER_CLASSES[dataset_name] = wrapper_class
        return wrapper_class

    return decorator


@register_dataset_wrapper_class("test")
class TestDataset(FakeData, TorchVisionDatasetWrapper):
    """A fake dataset for testing."""

    num_channels = 1
    width = 28
    height = 28

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        download: bool = False,
    ):
        super().__init__(
            image_size=(self.num_channels, self.width, self.height),
            num_classes=2,
            transform=transform,
            target_transform=target_transform,
        )


@register_dataset_wrapper_class("mnist")
class MnistDatasetWrapper(MNIST, TorchVisionDatasetWrapper):
    """The MNIST dataset wrapper."""

    num_channels = 1
    width = 28
    height = 28


@register_dataset_wrapper_class("fashion_mnist")
class FashionMnistDatasetWrapper(FashionMNIST, MnistDatasetWrapper):
    """The Fashion-MNIST dataset wrapper."""

    pass


@register_dataset_wrapper_class("kmnist")
class KmnistDatasetWrapper(KMNIST, MnistDatasetWrapper):
    """The Kuzushiji-MNIST dataset wrapper."""

    pass


@register_dataset_wrapper_class("cifar10")
class Cifar10DatasetWrapper(CIFAR10, TorchVisionDatasetWrapper):
    """The CIFAR-10 dataset wrapper."""

    num_channels = 3
    width = 32
    height = 32
    selected_classes = (3, 5)


@register_dataset_wrapper_class("cifar100")
class Cifar100DatasetWrapper(CIFAR100, TorchVisionDatasetWrapper):
    """The CIFAR-100 dataset wrapper."""

    num_channels = 3
    width = 32
    height = 32


@register_dataset_wrapper_class("svhn")
class SvhnDatasetWrapper(SVHN, TorchVisionDatasetWrapper):
    """The Street View House Numbers dataset wrapper."""

    num_channels = 3
    width = 32
    height = 32
    binarification_seed = 2

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        download: bool = False,
    ) -> None:
        split = "train" if train else "test"
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )


@register_scenario_class("image_classification", Dataset)
class ImageClassificationDataset(TensorDictDataset):
    """A dataset for the image classification task.

    Uses a torchvision dataset, and removes all the classes apart from two (determined
    by `hyper_params.image_classification.selected_classes`).

    Shapes
    ------
    The dataset is a TensorDict with the following keys:
        - "image" (dataset_size num_channels height width): The images in the dataset.
        - "x" (dataset_size max_message_rounds height width): The pixel features, which
          are all zeros.
        - "y" (dataset_size): The labels of the images.
    """

    instance_keys = ("image", "x")

    x_dtype = torch.float32
    y_dtype = torch.int64

    def build_torch_dataset(
        self, *, transform: Optional[Any]
    ) -> TorchVisionDatasetWrapper:
        """Build the TorchVision dataset.

        Parameters
        ----------
        transform : Optional[Any]
            The transform to apply to the images.

        Returns
        -------
        dataset : TorchVisionDatasetWrapper
            The TorchVision dataset.
        """
        dataset_class = DATASET_WRAPPER_CLASSES[self.hyper_params.dataset]
        return dataset_class(
            root=self.raw_dir, train=self.train, transform=transform, download=True
        )

    def build_tensor_dict(self) -> TensorDict:
        """Build the dataset as a TensorDict from the raw data.

        Returns
        -------
        dataset : TensorDict
            The dataset as a TensorDict, with the keys "image", "x", and "y".
        """

        # Load the dataset
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        torch_dataset = self.build_torch_dataset(transform=transform)

        # Get the whole dataset as a single batch
        full_dataset_loader = TorchDataLoader(
            torch_dataset, batch_size=len(torch_dataset)
        )
        images, labels = next(iter(full_dataset_loader))

        # Keep track of the indices of the original dataset from which the final dataset
        # is produced. This is needed to reconstruct the dataset with `to_torch_dataset`
        rearrange_index = torch.arange(len(labels))

        # The generator used to turn the dataset into a binary classification problem
        binurification_generator = torch.Generator()
        binurification_generator.manual_seed(self.binarification_seed)

        if self.hyper_params.dataset_options.binarification_method == "merge":
            # Shuffle the classes and merge them into two classes
            num_classes = len(torch.unique(labels))
            shuffled_classes = torch.randperm(
                num_classes, generator=binurification_generator
            )
            shuffled_labels = shuffled_classes[labels]
            labels = torch.where(shuffled_labels < num_classes // 2, 0, 1)

        elif self.hyper_params.dataset_options.binarification_method == "select_two":
            # Select the classes we want for binary classification
            index = (labels == self.selected_classes[0]) | (
                labels == self.selected_classes[1]
            )
            images = images[index]
            labels = labels[index]
            rearrange_index = rearrange_index[index]
            labels = (labels == self.selected_classes[1]).to(self.y_dtype)

        elif self.hyper_params.dataset_options.binarification_method == "random":
            # Select labels at random
            labels = torch.randint(
                0,
                2,
                labels.shape,
                dtype=self.y_dtype,
                generator=binurification_generator,
            )

        else:
            raise ValueError(
                f"Unknown binarification method: "
                f"{self.hyper_params.dataset_options.binarification_method}"
            )

        # Make the dataset balanced if requested
        if self.hyper_params.dataset_options.make_balanced:
            permuted_indices = torch.randperm(len(labels))
            images = images[permuted_indices]
            labels = labels[permuted_indices]
            rearrange_index = rearrange_index[permuted_indices]
            index_0 = torch.where(labels == 0)[0]
            index_1 = torch.where(labels == 1)[0]
            num_classes_0 = (labels == 0).sum()
            num_classes_1 = (labels == 1).sum()
            if num_classes_0 > num_classes_1:
                index_0 = index_0[:num_classes_1]
            else:
                index_1 = index_1[:num_classes_0]
            index = torch.cat((index_0, index_1))
            images = images[index]
            labels = labels[index]
            rearrange_index = rearrange_index[index]

        # Create the pixel features, which are all zeros
        x = torch.zeros(
            images.shape[0],
            self.protocol_handler.max_message_rounds,
            self.protocol_handler.num_message_channels,
            self.hyper_params.message_size,
            *images.shape[-2:],
            dtype=self.x_dtype,
        )

        return TensorDict(
            dict(image=images, x=x, y=labels, _rearrange_index=rearrange_index),
            batch_size=images.shape[0],
        )

    @property
    def raw_dir(self) -> str:
        """The path to the directory containing the raw data."""
        return os.path.join(IC_DATA_DIR, self.hyper_params.dataset, "raw")

    @property
    def processed_dir(self) -> str:
        """The path to the directory containing the processed data."""

        processed_name = f"processed"
        processed_name += f"_{self.protocol_handler.max_message_rounds}"
        processed_name += f"_{self.protocol_handler.num_message_channels}"
        processed_name += f"_{self.hyper_params.message_size}"

        processed_name = str(self.binarification_method).lower()
        if self.binarification_method == "select_two":
            processed_name += f"_{self.selected_classes[0]}_{self.selected_classes[1]}"
        elif (
            self.binarification_method == "merge"
            or self.binarification_method == "random"
        ):
            processed_name += f"_{self.binarification_seed}"

        if self.train and self.hyper_params.dataset_options.max_train_size is not None:
            processed_name += f"_{self.hyper_params.dataset_options.max_train_size}"

        sub_dir = "train" if self.train else "test"

        return os.path.join(
            IC_DATA_DIR,
            self.hyper_params.dataset,
            processed_name,
            sub_dir,
        )

    @property
    def pretrained_embeddings_dir(self) -> str:
        """The path to the directory containing cached pretrained model embeddings."""
        sub_dir = "train" if self.train else "test"
        return os.path.join(
            IC_DATA_DIR, self.hyper_params.dataset, "pretrained_embeddings", sub_dir
        )

    @property
    def binarification_method(self) -> BinarificationMethodType:
        """The method used to binarify the dataset."""
        return self.hyper_params.dataset_options.binarification_method

    @property
    def binarification_seed(self) -> int:
        """The seed to use for shuffling the dataset before merging."""
        if self.hyper_params.dataset_options.binarification_seed is not None:
            return self.hyper_params.dataset_options.binarification_seed
        else:
            return DATASET_WRAPPER_CLASSES[
                self.hyper_params.dataset
            ].binarification_seed

    @property
    def selected_classes(self) -> tuple[int, int]:
        """The two classes selected for binary classification."""
        if self.hyper_params.dataset_options.selected_classes is not None:
            return self.hyper_params.dataset_options.selected_classes
        else:
            return DATASET_WRAPPER_CLASSES[self.hyper_params.dataset].selected_classes
