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

from pvg.parameters import BinarificationMethodType
from pvg.parameters import ScenarioType
from pvg.factory import register_scenario_class
from pvg.scenario_base import Dataset, TensorDictDataset
from pvg.constants import IC_DATA_DIR


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
    """Decorator to register a dataset wrapper class."""

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
    num_channels = 1
    width = 28
    height = 28


@register_dataset_wrapper_class("fashion_mnist")
class FashionMnistDatasetWrapper(FashionMNIST, MnistDatasetWrapper):
    pass


@register_dataset_wrapper_class("kmnist")
class KmnistDatasetWrapper(KMNIST, MnistDatasetWrapper):
    pass


@register_dataset_wrapper_class("cifar10")
class Cifar10DatasetWrapper(CIFAR10, TorchVisionDatasetWrapper):
    num_channels = 3
    width = 32
    height = 32
    selected_classes = (3, 5)


@register_dataset_wrapper_class("cifar100")
class Cifar100DatasetWrapper(CIFAR100, TorchVisionDatasetWrapper):
    num_channels = 3
    width = 32
    height = 32


@register_dataset_wrapper_class("svhn")
class SvhnDatasetWrapper(SVHN, TorchVisionDatasetWrapper):
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


@register_scenario_class(ScenarioType.IMAGE_CLASSIFICATION, Dataset)
class ImageClassificationDataset(TensorDictDataset):
    """A dataset for the image classification task.

    Uses a torchvision dataset, and removes all the classes apart from two (determined
    by `params.image_classification.selected_classes`).

    Shapes
    ------
    The dataset is a TensorDict with the following keys:
        - "image" (dataset_size num_channels height width): The images in the dataset.
        - "x" (dataset_size max_message_rounds height width): The pixel features, which
          are all zeros.
        - "y" (dataset_size): The labels of the images.
    """

    x_dtype = torch.float32
    y_dtype = torch.int64

    def build_torch_dataset(
        self, *, transform: Optional[Any]
    ) -> TorchVisionDatasetWrapper:
        dataset_class = DATASET_WRAPPER_CLASSES[self.params.dataset]
        return dataset_class(
            root=self.raw_dir, train=self.train, transform=transform, download=True
        )

    def build_tensor_dict(self) -> TensorDict:
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

        if (
            self.params.dataset_options.binarification_method
            == BinarificationMethodType.MERGE
        ):
            # Shuffle the classes and merge them into two classes
            num_classes = len(torch.unique(labels))
            shuffled_classes = torch.randperm(
                num_classes, generator=binurification_generator
            )
            shuffled_labels = shuffled_classes[labels]
            labels = torch.where(shuffled_labels < num_classes // 2, 0, 1)

        elif (
            self.params.dataset_options.binarification_method
            == BinarificationMethodType.SELECT_TWO
        ):
            # Select the classes we want for binary classification
            index = (labels == self.selected_classes[0]) | (
                labels == self.selected_classes[1]
            )
            images = images[index]
            labels = labels[index]
            rearrange_index = rearrange_index[index]
            labels = (labels == self.selected_classes[1]).to(self.y_dtype)

        elif (
            self.params.dataset_options.binarification_method
            == BinarificationMethodType.RANDOM
        ):
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
                f"{self.params.dataset_options.binarification_method}"
            )

        # Make the dataset balanced if requested
        if self.params.dataset_options.make_balanced:
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
            self.params.message_size,
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
        return os.path.join(IC_DATA_DIR, self.params.dataset, "raw")

    @property
    def processed_dir(self) -> str:
        """The path to the directory containing the processed data."""

        processed_name = f"processed"
        processed_name += f"_{self.protocol_handler.max_message_rounds}"
        processed_name += f"_{self.protocol_handler.num_message_channels}"
        processed_name += f"_{self.params.message_size}"

        processed_name = str(self.binarification_method).lower()
        if self.binarification_method == BinarificationMethodType.SELECT_TWO:
            processed_name += f"_{self.selected_classes[0]}_{self.selected_classes[1]}"
        elif (
            self.binarification_method == BinarificationMethodType.MERGE
            or self.binarification_method == BinarificationMethodType.RANDOM
        ):
            processed_name += f"_{self.binarification_seed}"

        sub_dir = "train" if self.train else "test"

        return os.path.join(
            IC_DATA_DIR,
            self.params.dataset,
            processed_name,
            sub_dir,
        )

    @property
    def pretrained_embeddings_dir(self) -> str:
        """The path to the directory containing cached pretrained model embeddings."""
        sub_dir = "train" if self.train else "test"
        return os.path.join(
            IC_DATA_DIR, self.params.dataset, "pretrained_embeddings", sub_dir
        )

    @property
    def binarification_method(self) -> BinarificationMethodType:
        """The method used to binarify the dataset."""
        return self.params.dataset_options.binarification_method

    @property
    def binarification_seed(self) -> int:
        """The seed to use for shuffling the dataset before merging."""
        if self.params.dataset_options.binarification_seed is not None:
            return self.params.dataset_options.binarification_seed
        else:
            return DATASET_WRAPPER_CLASSES[self.params.dataset].binarification_seed

    @property
    def selected_classes(self) -> tuple[int, int]:
        """The two classes selected for binary classification."""
        if self.params.dataset_options.selected_classes is not None:
            return self.params.dataset_options.selected_classes
        else:
            return DATASET_WRAPPER_CLASSES[self.params.dataset].selected_classes
