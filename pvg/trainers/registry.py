"""Code for registering trainers.

Every trainer class registers itself here. This allows building the correct trainer
automatically based on the parameters.
"""

from typing import TypeVar

from pvg.parameters import TrainerType
from pvg.trainers.trainer_base import Trainer

TRAINER_REGISTRY: dict[TrainerType, type[Trainer]] = {}

T = TypeVar("T", bound=Trainer)


def register_trainer(trainer: TrainerType):
    """Register a trainer class. Used as a decorator."""

    def decorator(cls: type[T]) -> type[T]:
        TRAINER_REGISTRY[trainer] = cls
        return cls

    return decorator
