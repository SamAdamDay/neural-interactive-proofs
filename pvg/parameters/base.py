"""Base classes for parameters objects."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import dataclasses
import typing
from typing import Union
from types import UnionType

try:
    from enum import StrEnum
except ImportError:
    from pvg.utils.future import StrEnum


class ParameterValue(ABC):
    """Base class for things which can be used as parameter values."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Convert the parameter value to a dictionary."""


class BaseParameters(ParameterValue, ABC):
    """Base class for parameters objects."""

    def to_dict(self) -> dict:
        """Convert the parameters object to a dictionary.

        Turns enums into strings, and sub-parameters into dictionaries. Includes the
        is_random parameter if it exists.

        Returns
        -------
        params_dict : dict
            A dictionary of the parameters.
        """

        # Add all dataclass fields to the dictionary
        params_dict = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, StrEnum):
                value = value.value
            elif isinstance(value, ParameterValue):
                value = value.to_dict()
            params_dict[field.name] = value

        return params_dict

    @classmethod
    def construct_test_params(cls) -> "BaseParameters":
        """Construct a set of basic parameters for testing."""
        raise NotImplementedError

    def __post_init__(self):
        # Replace all Nones and all dictionaries with SubParameters objects
        for param in dataclasses.fields(self):
            param_value = getattr(self, param.name)
            if not isinstance(param_value, dict) and param_value is not None:
                continue
            origin_type = typing.get_origin(param.type)
            if origin_type is not Union and origin_type is not UnionType:
                continue
            for union_element in typing.get_args(param.type):
                try:
                    if issubclass(union_element, SubParameters):
                        param_class = union_element
                        break
                except TypeError:
                    continue
            else:
                continue
            if param_value is None:
                setattr(self, param.name, param_class())
            else:
                setattr(self, param.name, param_class(**param_value))


@dataclass
class SubParameters(BaseParameters, ABC):
    """Base class for sub-parameters objects."""
