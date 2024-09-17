"""A nested dictionary of strings data structure."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Union, ClassVar
from textwrap import indent
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray, DTypeLike

from pvg.utils.types import NumpyStringDtype


class NestedArrayDict:
    """A nested dictionary of numpy arrays.

    A NestedDict behaves similarly to a TensorDict: it allows for nested dictionaries,
    expects each array have a shape and has a common batch size
    """

    def __init__(
        self,
        data: Optional[Any] = None,
        batch_size: Optional[int | Sequence[int]] = None,
    ):
        if data is None:
            data = {}
            if batch_size is None:
                batch_size = ()
        elif batch_size is None:
            raise ValueError("batch_size must be provided if data is not None")

        if isinstance(batch_size, int):
            batch_size = (batch_size,)
        elif batch_size is not None:
            batch_size = tuple(batch_size)

        self._batch_size = batch_size

        tuple_keys, arrays = self._create_arrays_from_data(data, batch_size=batch_size)
        self._arrays = arrays
        self._tuple_keys = tuple_keys

    @property
    def batch_size(self) -> tuple[int, ...] | None:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: tuple[int, ...] | None):
        if value is None:
            value = ()

        for key, array in zip(self._tuple_keys, self._arrays):
            if array.shape[: len(value)] != value:
                raise ValueError(
                    f"Batch size {value} is not compatible with array of shape "
                    f"{array.shape} with key {key}"
                )

        self._batch_size = value

    def keys(self, include_prefixes: bool = True) -> list[tuple[str, ...] | str]:
        """Return the all keys, optionally including prefixes.

        Top-level keys are returned as strings, while nested keys are returned as
        tuples.

        Parameters
        ----------
        include_prefixes : bool, default=True
            Whether to include prefixes of the leaf keys.

        Returns
        -------
        keys : list[tuple[str, ...] | str]
            The keys of the data. Each key is either a string or a a tuple of strings
            that represents the path to the data in the nested dictionary.
        """

        tuple_keys = set(self._tuple_keys)

        if include_prefixes:
            for key in self._tuple_keys:
                for i in range(1, len(key)):
                    tuple_keys.add(key[:i])

        # Convert single-element tuples to strings
        for key in self._tuple_keys:
            if len(key) == 1:
                tuple_keys.remove(key)
                tuple_keys.add(key[0])

        return tuple_keys

    def leaf_keys(self) -> list[tuple[str, ...]]:
        """Return the leaf keys as tuples.

        Returns
        -------
        leaf_keys : list[tuple[str, ...]]
            The leaf keys of the data. Each key is a tuple of strings that represents
            the path to the data in the nested dictionary to an array
        """
        return self._tuple_keys

    def key_is_leaf(self, key: tuple[str, ...]) -> bool:
        """Check if a key is a leaf, i.e. it is not a prefix of any other key.

        Leaf keys are keys that correspond to arrays of strings.

        Parameters
        ----------
        key : tuple[str, ...]
            The key to check.

        Returns
        -------
        is_terminal : bool
            Whether the key is terminal.
        """
        if key not in self._tuple_keys:
            raise ValueError(f"Key {key} not found in {type(self).__name__}")
        return all(key not in other_key for other_key in self._tuple_keys)

    def clone(self, recurse: bool = True) -> "NestedArrayDict":
        """Clone the NestedDict and create a new instance.

        Parameters
        ----------
        recurse : bool, default=True
            Whether to clone the arrays and scalars in the NestedDict. Otherwise, just
            clone the structure.

        Returns
        -------
        cloned_dict : NestedDict
            The cloned NestedDict.
        """
        cloned_dict = NestedArrayDict(self)
        if recurse:
            for i, array in enumerate(self._arrays):
                if isinstance(array, np.ndarray):
                    cloned_dict._arrays[i] = array.copy()
        return cloned_dict

    def update(self, other: Union["NestedArrayDict", dict]) -> "NestedArrayDict":
        """Update the NestedDict with the data from another dict.

        Parameters
        ----------
        other : NestedDict | dict
            The dict to update from. If a dict is provided, it is converted to a
            NestedDict with the same batch size as the current NestedDict.

        Returns
        -------
        self : NestedDict
            The updated NestedDict.
        """

        if isinstance(other, dict):
            other = NestedArrayDict(other, batch_size=self._batch_size)
        elif self._batch_size != other._batch_size:
            raise ValueError(
                f"Self batch size {self._batch_size} must match batch size of `other` "
                f"{other._batch_size}"
            )

        for key, value in other.items():
            self[key] = value

        return self

    def items(self):
        """Return an iterator over the items of the NestedDict.

        Yields
        ------
        key : tuple[str, ...]
            The key of the item.
        value : NDArray
            The value of the item.
        """
        for key in self.keys(include_prefixes=False):
            yield key, self[key]

    def __getitem__(self, index: Any) -> Union["NestedArrayDict", dict[str, str]]:

        if isinstance(index, str):
            index = (index,)

        if isinstance(index, tuple) and all(
            isinstance(sub_key, str) for sub_key in index
        ):
            # If the index matches a key, return the corresponding array
            if index in self._tuple_keys:
                key_index = self._tuple_keys.index(index)
                return self._arrays[key_index]

            # If the index is a prefix of a key, return a new NestedDict with the
            # selected data
            else:
                key_extensions = [
                    key[len(index) :]
                    for key in self._tuple_keys
                    if key[: len(index)] == index
                ]
                if len(key_extensions) == 0:
                    raise IndexError(f"Key {index} not found in {type(self).__name__}")
                selected = [
                    self._arrays[self._tuple_keys.index(index + key_extension)]
                    for key_extension in key_extensions
                ]
                return type(self).from_arrays_and_scalars(
                    selected, key_extensions, batch_size=self._batch_size
                )

        elif isinstance(index, tuple) and len(index) == 0:
            return self

        elif index is Ellipsis:
            return self

        # Try indexing each array
        try:
            selected = [self._arrays[i][index] for i in range(len(self._tuple_keys))]
        except IndexError as e:
            raise IndexError(
                f"Index {index} out of bounds in {type(self).__name__}"
            ) from e
        except TypeError as e:
            raise TypeError(
                f"Index {index} is not supported in {type(self).__name__}"
            ) from e

        # Build a new NestedDict from the selected arrays
        return type(self).from_arrays_and_scalars(
            selected, self._tuple_keys, batch_size=()
        )

    def __setitem__(self, index: Any, value: Any):
        if isinstance(index, str):
            index = (index,)

        if not isinstance(index, tuple):
            raise TypeError(
                f"Index must be a tuple of strings, but found {type(index)}"
            )

        if not all(isinstance(sub_key, str) for sub_key in index):
            raise TypeError(f"Index must be a tuple of strings, but got {index}")

        if not isinstance(value, (list, np.ndarray, NestedArrayDict, dict)):
            raise ValueError(
                f"Value for key {index} must be a valid array or dictionary. Got type "
                f"{type(value)}"
            )

        if isinstance(value, (list, np.ndarray)):
            value = np.array(value)

        # Check that arrays are compatible with the batch size
        if isinstance(value, np.ndarray):
            if not self._is_shape_compatible(value.shape):
                raise ValueError(
                    f"PyArrow arrays in {type(self).__name__} data must agree with "
                    f"batch size {self._batch_size[0]}, but found {len(value)}"
                )

        elif isinstance(value, dict):
            try:
                value = NestedArrayDict(value, batch_size=self._batch_size)
            except ValueError as e:
                raise ValueError(
                    f"Error creating {type(self).__name__} from dict to assign to key "
                    f"{index}"
                ) from e

        elif isinstance(value, NestedArrayDict):
            # Check that the value is a NestedDict with the correct batch size
            if value._batch_size != self._batch_size:
                raise ValueError(
                    f"Value must have the same batch size as {type(self).__name__}, "
                    f"but current batch size is {self._batch_size} while value batch "
                    f"size is {value._batch_size}."
                )

        # If any key is a strict prefix of the new key, raise an error
        for key in self._tuple_keys:
            if index[: len(key)] == key and len(index) > len(key):
                raise IndexError(
                    f"Cannot set the value of key {index} because it is a strict "
                    f"extension of the leaf key {key}."
                )

        # Remove the existing keys that are extensions of the new key
        index_key_extensions = [
            i for i, key in enumerate(self._tuple_keys) if key[: len(index)] == index
        ]
        self._tuple_keys = [
            key
            for i, key in enumerate(self._tuple_keys)
            if i not in index_key_extensions
        ]

        # For new arrays, add the new key and value
        if isinstance(value, np.ndarray):
            self._arrays.append(value)
            self._tuple_keys.append(index)

        # For new NestedDicts, add the new keys and values
        else:
            for key, sub_value in zip(value._tuple_keys, value._table.columns):
                new_tuple_key = index + key
                self._arrays.append(sub_value)
                self._tuple_keys.append(new_tuple_key)

    def __contains__(self, key: Any):
        raise NotImplementedError(
            f"Contains not yet implemented for {type(self).__name__}. Use "
            f"`key in string_dict.keys()` instead."
        )

    def __repr__(self):

        sorted_tuple_keys = sorted(self._tuple_keys)

        if len(sorted_tuple_keys) == 0:
            return f"{type(self).__name__}()"

        key_repr = self._make_key_repr(sorted_tuple_keys)
        key_repr = indent(key_repr, " " * 4)
        return (
            f"{type(self).__name__}(\n"
            + indent(f"fields={{\n{key_repr}}},\n", " " * 4)
            + indent(f"batch_size={self._batch_size})", " " * 4)
        )

    @classmethod
    def from_arrays_and_scalars(
        cls,
        array_and_scalars: list[ArrayLike | Any],
        tuple_keys: list[tuple[str, ...]],
        batch_size: tuple[int, ...],
    ) -> "NestedArrayDict":
        """Create a NestedDict from a list of arrays and scalars, and associated keys.

        Parameters
        ----------
        array_and_scalars : list[ArrayLike | Any]
            The list of arrays and scalars to store in the NestedDict.
        tuple_keys : list[tuple[str, ...]]
            The keys of the data. Each key is a tuple of strings that represents the
            path to the data in the nested dictionary.
        batch_size : tuple[int, ...]
            The batch size of the NestedDict, which must be the initial segment of the
            shape of the each array. If `()`, the arrays are scalars.

        Returns
        -------
        string_dict : NestedDict
            The NestedDict created from the arrays.
        """
        nested_dict = cls(batch_size=batch_size)
        nested_dict._arrays = array_and_scalars
        nested_dict._tuple_keys = tuple_keys
        return nested_dict

    def _create_arrays_from_data(
        self,
        data: dict[str, list | ArrayLike | Any] | Any,
        batch_size: tuple[int, ...],
    ) -> tuple[list[tuple[str, ...]], list[NDArray], tuple[int, ...]]:
        """Create a list of named arrays from the data.

        Returns a list of keys and a list of corresponding arrays.

        The data are validated and flattened, so that each key is a tuple of strings.

        Parameters
        ----------
        data : dict
            The data to convert. Can be a nested dictionary of lists or arrays. All keys
            must be strings.
        batch_size : tuple[int, ...]
            The batch size of the NestedDict, which must be the initial segment of the
            shape of the each array. If `()`, the arrays are scalars.

        Returns
        -------
        keys : list[tuple[str, ...]]
            The keys of the data. Each key is a tuple of strings that represents the
            path to the data in the nested dictionary.
        arrays : list[NDArray]
            The data as numpy arrays. The order of the arrays corresponds to the order
            of the keys.
        """

        if not isinstance(data, (dict, NestedArrayDict)):
            raise ValueError(
                f"{type(self).__name__} data must be a dict, not {type(data)}"
            )

        if len(data) == 0:
            return [], []

        tuple_keys: list[tuple[str, ...]] = []
        arrays: list[NDArray] = []

        for key, value in data.items():

            if not isinstance(key, str):
                raise ValueError(
                    f"Each key in {type(self).__name__} data must be a string, not "
                    f"{type(key)}"
                )

            # If the value is a nested dictionary, recursively create arrays
            if isinstance(value, (dict, NestedArrayDict)):
                sub_tuple_keys, sub_arrays = self._create_arrays_from_data(
                    value,
                    batch_size=batch_size,
                )
                for sub_key in sub_tuple_keys:
                    tuple_keys.append((key,) + sub_key)
                arrays.extend(sub_arrays)

            else:

                value = np.array(value)

                if not self._is_shape_compatible(value.shape):
                    raise ValueError(
                        f"All arrays in {type(self).__name__} data must have shape "
                        f"compatible with the batch size {self._batch_size}, but found "
                        f"{value.shape}"
                    )

                arrays.append(value)
                tuple_keys.append((key,))

        return tuple_keys, arrays

    def _make_key_repr(
        self, keys: list[tuple[str, ...]], prefix: tuple[str, ...] = ()
    ) -> str:
        """Make a string representation for a set of keys.

        Represents the keys as a nested NestedDict.

        Parameters
        ----------
        keys : list[tuple[str, ...]]
            The keys to represent.
        prefix : tuple[str, ...], optional
            The prefix to add to the keys, to get the full path.

        Returns
        -------
        key_repr : str
            The string representation of the keys.
        """

        # Select the first element of each key, while maintaining order
        top_level_keys = [key[0] for key in keys]
        top_level_keys = list(dict.fromkeys(top_level_keys))

        key_reprs = []
        for key in top_level_keys:
            sub_keys = [
                sub_key[1:]
                for sub_key in keys
                if sub_key[0] == key and len(sub_key) > 1
            ]
            if len(sub_keys) == 0:
                full_key = prefix + (key,)
                array = self._arrays[self._tuple_keys.index(full_key)]
                key_reprs.append(
                    f"{key!r}: Array(shape={array.shape}, dtype={array.dtype})"
                )
            else:
                sub_keys_repr = self._make_key_repr(sub_keys, prefix=prefix + (key,))
                sub_keys_repr = indent(sub_keys_repr, " " * 4)
                key_reprs.append(f"{key!r}: {{\n{sub_keys_repr}\n}}")

        return ",\n".join(key_reprs)

    def _is_shape_compatible(self, shape: tuple[int, ...]) -> bool:
        """Check whether a shape is compatible with the batch size.

        Parameters
        ----------
        shape : tuple[int, ...]
            The shape to check.

        Returns
        -------
        is_compatible : bool
            Whether the shape is compatible with the batch size.
        """
        return shape[: len(self._batch_size)] == self._batch_size


@dataclass
class NumpySpec(ABC):
    """Base class for numpy array specifications.

    A specification defines meta-data for arrays or nested dictionaries of arrays.
    """

    shape: tuple[int] | int
    dim_names: Optional[tuple[str] | str] = None

    @abstractmethod
    def zero(self) -> NDArray | NestedArrayDict:
        """Return a zero array or dictionary of zero arrays."""

    def __post_init__(self):
        if isinstance(self.shape, int):
            self.shape = (self.shape,)
        if isinstance(self.dim_names, str):
            self.dim_names = tuple(self.dim_names.split(" "))
        if self.dim_names is not None:
            if len(self.dim_names) != len(self.shape):
                raise ValueError(
                    f"Length of dim_names ({len(self.dim_names)}) must match the "
                    f"length of shape ({len(self.shape)})"
                )


class NumpyArraySpec(NumpySpec):
    """Specification for a single numpy array."""

    dtype: DTypeLike

    def zero(self) -> NDArray:
        return np.zeros(self.shape, dtype=self.dtype)


@dataclass
class IntArraySpec(NumpySpec):
    """Specification for a single integer numpy array."""

    dtype: ClassVar[DTypeLike] = np.int64


@dataclass
class FloatArraySpec(NumpySpec):
    """Specification for a single float numpy array."""

    dtype: ClassVar[DTypeLike] = np.float32


@dataclass
class BoolArraySpec(NumpySpec):
    """Specification for a single boolean numpy array."""

    dtype: ClassVar[DTypeLike] = np.bool


@dataclass
class StringArraySpec(NumpySpec):
    """Specification for a single string numpy array."""

    dtype: ClassVar[DTypeLike] = NumpyStringDtype

    def zero(self) -> NDArray:
        return np.full(self.shape, None, dtype=self.dtype)


class CompositeSpec(NumpySpec):
    """Specification for a nested dictionary of numpy arrays."""

    def __init__(
        self,
        shape: tuple[int],
        dim_names: Optional[tuple[str] | str] = None,
        **specs: dict[str, NumpySpec],
    ):
        self.specs = specs
        super().__init__(shape=shape, dim_names=dim_names)

        # Check that the shape and dim_names are consistent with the specs
        for key, spec in specs.items():
            if spec.shape[: len(shape)] != shape:
                raise ValueError(
                    f"The shape of the {type(self).__name__} ({shape}) must be a "
                    f"prefix of the shape of spec {key!r} ({spec.shape})"
                )
            if spec.dim_names is not None and dim_names is not None:
                if spec.dim_names[: len(shape)] != dim_names:
                    raise ValueError(
                        f"The dim_names of the {type(self).__name__} ({dim_names}) "
                        f"must be a prefix of the dim_names of spec {key!r} "
                        f"({spec.dim_names})"
                    )

    def zero(self) -> NestedArrayDict:
        return NestedArrayDict(
            {key: spec.zero() for key, spec in self.specs.items()},
            batch_size=self.shape,
        )
