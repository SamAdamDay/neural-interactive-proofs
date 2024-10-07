"""Parameters specific to each scenario."""

from dataclasses import dataclass

from pvg.parameters.base import SubParameters, register_parameter_class


@register_parameter_class
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


@register_parameter_class
@dataclass
class CodeValidationParameters(SubParameters):
    """Additional parameters for the code validation task.

    Parameters
    ----------
    app_difficulty : str
        The difficulty level of the APPS dataset, if using.
    app_solution_number : int
        Each question in the APPS dataset has multiple solutions. This parameter
        specifies which solution to use.
    """

    apps_difficulty: str = "interview"
    apps_solution_number: int = 0
