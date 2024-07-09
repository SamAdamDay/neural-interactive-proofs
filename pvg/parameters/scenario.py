"""Parameters specific to each scenario."""

from dataclasses import dataclass

from pvg.parameters.base import SubParameters


@dataclass
class ImageClassificationParameters(SubParameters):
    """Additional parameters for the image classification task.

    Parameters
    ----------
    num_block_groups : int
        The number of groups of building blocks (e.g. convolutional layers) in each
        agents's CNN.
    initial_num_channels : int
        The number of channels in the first building block in each agents's CNN.
    """

    num_block_groups: int = 1
    initial_num_channels: int = 16
