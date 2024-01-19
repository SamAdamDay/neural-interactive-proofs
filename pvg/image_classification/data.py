import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader as TorchDataLoader

import torchvision
from torchvision.datasets import VisionDataset, MNIST, CIFAR10, FashionMNIST
from torchvision import transforms

from tensordict import TensorDict

from pvg.scenario_base import Dataset
from pvg.constants import IC_DATA_DIR


@dataclass
class TorchVisionImageDataset:
    """Details of a TorchVision image dataset.

    Parameters
    ----------
    data_class : type[VisionDataset]
        The TorchVision dataset class.
    num_channels : int
        The number of channels of the images in the dataset.
    """

    data_class: type[VisionDataset]
    num_channels: int
    width: int
    height: int


IMAGE_DATASETS: dict[str, TorchVisionImageDataset] = dict(
    mnist=TorchVisionImageDataset(
        data_class=MNIST,
        num_channels=1,
        width=28,
        height=28,
    ),
    fashion_mnist=TorchVisionImageDataset(
        data_class=FashionMNIST,
        num_channels=1,
        width=28,
        height=28,
    ),
    cifar10=TorchVisionImageDataset(
        data_class=CIFAR10,
        num_channels=3,
        width=32,
        height=32,
    ),
)


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
            self.raw_dir, train=True, transform=transform, download=True
        )

        # Get the whole dataset as a single batch
        full_dataset_loader = TorchDataLoader(
            torch_dataset, batch_size=len(torch_dataset)
        )
        images, labels = next(iter(full_dataset_loader))

        # Select the classes we want for binary classification
        selected_classes = self.params.image_classification.selected_classes
        index = (labels == selected_classes[0]) | (labels == selected_classes[1])
        images = images[index]
        labels = labels[index]
        labels = (labels == selected_classes[1]).to(self.y_dtype)

        # Create the pixel features, which are all zeros
        x = torch.zeros(
            images.shape[0],
            *images.shape[-2:],
            self.params.max_message_rounds,
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
        selected_classes = self.params.image_classification.selected_classes
        return os.path.join(
            IC_DATA_DIR,
            self.params.dataset,
            (
                f"processed_{self.params.max_message_rounds}"
                f"_{selected_classes[0]},{selected_classes[1]}"
            ),
        )
