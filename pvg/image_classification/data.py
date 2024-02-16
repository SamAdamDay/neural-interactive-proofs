import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader as TorchDataLoader

import torchvision
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
from pvg.scenario_instance import register_scenario_class
from pvg.scenario_base import Dataset
from pvg.constants import IC_DATA_DIR


class TestDataset(FakeData):
    """A fake dataset for testing."""

    def __init__(
        self,
        root=None,
        train=None,
        transform=None,
        target_transform=None,
        download=None,
    ):
        super().__init__(
            image_size=(1, 28, 28),
            num_classes=2,
            transform=transform,
            target_transform=target_transform,
        )


class SVHNTrainParameter(SVHN):
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


@dataclass
class TorchVisionDatasetProperties:
    """Details of a TorchVision image dataset.

    Parameters
    ----------
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

    data_class: type[VisionDataset]
    num_channels: int
    width: int
    height: int
    selected_classes: tuple[int, int] = (0, 1)
    binarification_seed: int = 0


IMAGE_DATASETS: dict[str, TorchVisionDatasetProperties] = dict(
    test=TorchVisionDatasetProperties(
        data_class=TestDataset,
        num_channels=1,
        width=28,
        height=28,
    ),
    mnist=TorchVisionDatasetProperties(
        data_class=MNIST,
        num_channels=1,
        width=28,
        height=28,
    ),
    fashion_mnist=TorchVisionDatasetProperties(
        data_class=FashionMNIST,
        num_channels=1,
        width=28,
        height=28,
    ),
    cifar10=TorchVisionDatasetProperties(
        data_class=CIFAR10,
        num_channels=3,
        width=32,
        height=32,
        selected_classes=(3, 5),
    ),
    cifar100=TorchVisionDatasetProperties(
        data_class=CIFAR100,
        num_channels=3,
        width=32,
        height=32,
    ),
    kmnist=TorchVisionDatasetProperties(
        data_class=KMNIST,
        num_channels=1,
        width=28,
        height=28,
    ),
    svhn=TorchVisionDatasetProperties(
        data_class=SVHNTrainParameter,
        num_channels=3,
        width=32,
        height=32,
        binarification_seed=2,
    ),
)


@register_scenario_class(ScenarioType.IMAGE_CLASSIFICATION, Dataset)
class ImageClassificationDataset(Dataset):
    """A dataset for the image classification task.

    Uses a torchvision dataset, and removes all the classes apart from two (determined
    by `params.image_classification.selected_classes`).
    """

    x_dtype = torch.float32
    y_dtype = torch.int64

    def _build_tensor_dict(self) -> TensorDict:
        # Load the dataset
        dataset_class = IMAGE_DATASETS[self.params.dataset].data_class
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        torch_dataset = dataset_class(
            root=self.raw_dir, train=self.train, transform=transform, download=True
        )

        # Get the whole dataset as a single batch
        full_dataset_loader = TorchDataLoader(
            torch_dataset, batch_size=len(torch_dataset)
        )
        images, labels = next(iter(full_dataset_loader))

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

        # Create the pixel features, which are all zeros
        x = torch.zeros(
            images.shape[0],
            self.params.max_message_rounds,
            *images.shape[-2:],
            dtype=self.x_dtype,
        )

        return TensorDict(dict(image=images, x=x, y=labels), batch_size=images.shape[0])

    @property
    def raw_dir(self) -> str:
        """The path to the directory containing the raw data."""
        return os.path.join(IC_DATA_DIR, self.params.dataset, "raw")

    @property
    def processed_dir(self) -> str:
        """The path to the directory containing the processed data."""

        suffix = str(self.binarification_method).lower()
        if self.binarification_method == BinarificationMethodType.SELECT_TWO:
            suffix += f"_{self.selected_classes[0]}_{self.selected_classes[1]}"
        elif (
            self.binarification_method == BinarificationMethodType.MERGE
            or self.binarification_method == BinarificationMethodType.RANDOM
        ):
            suffix += f"_{self.binarification_seed}"

        sub_dir = "train" if self.train else "test"

        return os.path.join(
            IC_DATA_DIR,
            self.params.dataset,
            f"processed_{self.params.max_message_rounds}_{suffix}",
            sub_dir,
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
            return IMAGE_DATASETS[self.params.dataset].binarification_seed

    @property
    def selected_classes(self) -> tuple[int, int]:
        """The two classes selected for binary classification."""
        if self.params.dataset_options.selected_classes is not None:
            return self.params.dataset_options.selected_classes
        else:
            return IMAGE_DATASETS[self.params.dataset].selected_classes
