"""Base classes for parameters objects."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import dataclasses
import typing
from typing import Union, Any, TypeVar
from types import UnionType

try:
    from enum import StrEnum
except ImportError:
    from pvg.utils.future import StrEnum


class ParameterValue(ABC):
    """Base class for things which can be used as parameter values."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialise the parameter value to a dictionary.

        The original parameter value must be able to be reconstructed from the
        dictionary using the from_dict method.

        The dictionary may contain a `_type` key, which is used to specify the exact
        class of the parameter value.
        """

    @classmethod
    @abstractmethod
    def from_dict(cls, params_dict: dict) -> "ParameterValue":
        """Create a parameter value from a serialised dictionary."""

    @classmethod
    def _get_param_class_from_dict(
        cls, param_dict: dict
    ) -> type["ParameterValue"] | None:
        """Try to get the parameter class from a dictionary of serialised parameters.

        Parameters
        ----------
        param_dict : dict
            A dictionary of parameters, which may have come from a `to_dict` method.
            This dictionary may contain a `_type` key, which is used to determine the
            class of the parameter.

        Returns
        -------
        param_class : type[ParameterValue] | None
            The class of the parameter, if it can be determined.

        Raises
        ------
        ValueError
            If the class specified in the dictionary is not a valid parameter class.
        """

        class_name = param_dict.get("_type", None)

        if class_name is None:
            return None

        # Get the class of the parameter
        return get_parameter_or_parameter_value_class(class_name)


class BaseHyperParameters(ParameterValue, ABC):
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

        params_dict["_type"] = type(self).__name__

        return params_dict

    @classmethod
    def from_dict(cls, params_dict: dict) -> "BaseHyperParameters":
        """Create a parameters object from a dictionary.

        Parameters
        ----------
        params_dict : dict
            A dictionary of the parameters.

        Returns
        -------
        hyper_params : BaseParameters
            The parameters object.
        """

        # Remove the `_type` key from the dictionary, checking that it is the correct
        # type
        param_class_name = params_dict.pop("_type", None)
        if param_class_name is not None:
            param_class = get_parameter_or_parameter_value_class(param_class_name)
            if not issubclass(param_class, cls):
                raise ValueError(
                    f"Invalid parameter class: {param_class_name!r} is not a subclass "
                    f"of {cls.__name__!r}"
                )

        # Call `from_dict` on all fields that are ParameterValues
        for param in dataclasses.fields(cls):

            param_value = params_dict.get(param.name, None)

            # Skip if the parameter is not a dictionary or is None
            if not isinstance(param_value, dict) or param_value is None:
                continue

            # Make sure the field is a Union and contains a `ParameterValue` subclass
            origin_type = typing.get_origin(param.type)
            if origin_type is not Union and origin_type is not UnionType:
                continue
            for union_element in typing.get_args(param.type):
                try:
                    if issubclass(union_element, ParameterValue):
                        param_base_class = union_element
                        break
                except TypeError:
                    continue
            else:
                continue

            # Get the specific class name of the parameter. TODO: It would be better if
            # this class could be inferred from the other parameters, rather than being
            # specified in the dictionary.
            param_class = cls._get_param_class_from_dict(param_value)

            if param_class is None:
                param_class = param_base_class

            else:

                # Make sure the class is a subclass of the base class
                if not issubclass(param_class, param_base_class):
                    raise ValueError(
                        f"Invalid parameter class: {type(param_class).__name__!r} is "
                        f"not a subclass of {type(param_base_class).__name__!r}"
                    )

            # Replace the dictionary with the `ParameterValue` subclass
            params_dict[param.name] = param_class.from_dict(param_value)

        # Create the parameters object from the modified dictionary
        return cls(**params_dict)

    def get(self, address: str) -> Any:
        """Get a value from the parameters object using a dot-separated address.

        Parameters
        ----------
        address : str
            The path to the value in the parameters object, separated by dots.

        Returns
        -------
        value : Any
            The value at the address.

        Raises
        ------
        KeyError
            If the address does not exist.
        """

        first_key, _, remainder = address.partition(".")

        for field in dataclasses.fields(self):
            if field.name == first_key:
                value = getattr(self, first_key)
                if isinstance(value, BaseHyperParameters):
                    try:
                        return value.get(remainder)
                    except KeyError:
                        raise KeyError(
                            f"Address {address!r} not found in parameters object."
                        )
                return value

        raise KeyError(f"Address {address!r} not found in parameters object.")

    @classmethod
    def construct_test_params(cls) -> "BaseHyperParameters":
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


PARAMETER_VALUE_CLASSES: dict[str, type[ParameterValue]] = {}
PARAMETER_CLASSES: dict[str, type[BaseHyperParameters]] = {}

V = TypeVar("V", bound=ParameterValue)
P = TypeVar("P", bound=BaseHyperParameters)


def register_parameter_value_class(cls: type[V]) -> type[V]:
    """Decorator to register a parameter value class."""

    PARAMETER_VALUE_CLASSES[cls.__name__] = cls
    return cls


def register_parameter_class(cls: type[P]) -> type[P]:
    """Decorator to register a parameter class."""

    PARAMETER_CLASSES[cls.__name__] = cls
    return cls


def get_parameter_or_parameter_value_class(
    name: str,
) -> type[ParameterValue] | type[BaseHyperParameters]:
    """Get a parameter class or parameter value class by name.

    If the class is not found in the parameter value classes, it is assumed to be a
    parameter class.

    Parameters
    ----------
    name : str
        The name of the class.

    Returns
    -------
    cls : type[ParameterValue] | type[BaseParameters]
        The class of the parameter.

    Raises
    ------
    ValueError
        If the class is not found.
    """
    try:
        return PARAMETER_VALUE_CLASSES[name]
    except KeyError:
        try:
            return PARAMETER_CLASSES[name]
        except KeyError:
            raise ValueError(f"Parameter class {name!r} not found.")


def get_parameter_class(name: str) -> type[BaseHyperParameters]:
    """Get a parameter class by name.

    Parameters
    ----------
    name : str
        The name of the class.

    Returns
    -------
    cls : type[BaseParameters]
        The class of the parameter.
    """
    return PARAMETER_CLASSES[name]


@dataclass
class SubParameters(BaseHyperParameters, ABC):
    """Base class for sub-parameters objects."""
