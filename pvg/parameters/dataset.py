"""Parameters for the dataset."""

from dataclasses import dataclass
from typing import Optional

from pvg.parameters.base import SubParameters, register_parameter_class
from pvg.parameters.types import BinarificationMethodType


@register_parameter_class
@dataclass
class DatasetParameters(SubParameters):
    """Additional parameters for the dataset.

    Parameters
    ----------
    binarification_method : BinarificationMethodType
        The method to use to turn the multi-class classification task into a binary
        classification task.
    selected_classes : tuple[int, int], optional
        When selecting two classes from the original dataset, the indices of the classes
        to select. If not provided, the default for the dataset is used.
    binarification_seed : int, optional
        The seed used when doing a randomised binarification. If not provided, the
        default for the dataset is used.
    make_balanced : bool
        Whether to make sure the dataset is balanced.
    max_train_size : int, optional
        The size to reduce the training set to. If not provided, the dataset is not
        reduced, and the full training set is used.
    """

    binarification_method: BinarificationMethodType = BinarificationMethodType.MERGE
    selected_classes: Optional[tuple[int, int]] = None
    binarification_seed: Optional[int] = None
    make_balanced: bool = True

    max_train_size: Optional[int] = None
