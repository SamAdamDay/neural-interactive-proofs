"""A nested dictionary of strings data structure."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Union, ClassVar
from collections.abc import Iterator
from textwrap import indent
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray, DTypeLike

from pvg.utils.types import NumpyStringDtype


class NestedArrayDict:
    """A nested dictionary of numpy arrays.

    A NestedDict behaves similarly to a TensorDict: it allows for nested dictionaries
    and has a common batch size
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
            if isinstance(data, NestedArrayDict):
                batch_size = data.batch_size
            else:
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
        """The batch size of the NestedDict."""
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

    def keys(
        self, include_prefixes: bool = True, return_all_tuples: bool = False
    ) -> Iterator[tuple[str, ...] | str]:
        """Return an iterator over the all keys, optionally including prefixes.

        By default top-level keys are returned as strings, while nested keys are
        returned as tuples.

        Parameters
        ----------
        include_prefixes : bool, default=True
            Whether to include prefixes of the leaf keys.
        return_all_tuples : bool, default=False
            If true, all keys will be returned as tuples. Otherwise, single element
            tuples will be replaced by their items

        Yields
        ------
        key : tuple[str, ...] | str
            The keys of the data. Each key is either a string or a a tuple of strings
            that represents the path to the data in the nested dictionary.
        """

        tuple_keys = set(self._tuple_keys)

        if include_prefixes:
            for key in self._tuple_keys:
                for i in range(1, len(key)):
                    tuple_keys.add(key[:i])

        for key in self._tuple_keys:
            if len(key) == 1 and not return_all_tuples:
                yield key[0]
            else:
                yield key

    def items(
        self, include_prefixes: bool = True, return_all_tuples: bool = False
    ) -> Iterator[tuple[tuple[str, ...], NDArray]]:
        """Return an iterator over the items, optionally including prefixes.

        By default top-level keys are returned as strings, while nested keys are returned as
        tuples.

        Parameters
        ----------
        include_prefixes : bool, default=True
            Whether to include prefixes of the leaf keys.
        return_all_tuples : bool, default=False
            If true, all keys will be returned as tuples. Otherwise, single element
            tuples will be replaced by their items

        Yields
        ------
        key : tuple[str, ...]
            The key of the item.
        value : NDArray
            The value of the item.
        """
        for key in self.keys(
            include_prefixes=include_prefixes, return_all_tuples=return_all_tuples
        ):
            yield key, self[key]

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

        Leaf keys are keys that correspond to numpy arrays.

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

        for key, value in other.items(include_prefixes=True):
            self[key] = value

        return self

    @classmethod
    def zeros_like(self, other: "NestedArrayDict") -> "NestedArrayDict":
        """Create a NestedDict with zeros in the same structure as another NestedDict.

        Parameters
        ----------
        other : NestedArrayDict
            The NestedArrayDict to copy the structure from.

        Returns
        -------
        zero_dict : NestedArrayDict
            The NestedArrayDict with the same structure as `other`, but with zero
            arrays.
        """
        zero_dict = NestedArrayDict(batch_size=other.batch_size)
        for key in other.keys(include_prefixes=True):
            if other[key].dtype == NumpyStringDtype:
                zero_dict[key] = np.full_like(other[key], None)
            else:
                zero_dict[key] = np.zeros_like(other[key])
        return zero_dict

    def __getitem__(self, index: Any) -> Union["NestedArrayDict", NDArray]:

        original_index = index

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
                    raise IndexError(
                        f"Key {original_index!r} not found in {type(self).__name__} with "
                        f"keys {list(self.keys(include_prefixes=True))}"
                    )
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

        # Compute the batch size of the selected arrays. TODO: This is inefficient
        # because it requires allocating memory for an array of shape self.batch_size.
        # However, it means we can compute batch sizes for any index which numpy can
        # handle
        dummy_array = np.empty(self._batch_size, dtype=np.bool_)
        index_array = dummy_array[index]
        indexed_batch_size = index_array.shape

        # Build a new NestedDict from the selected arrays
        return type(self).from_arrays_and_scalars(
            selected, self._tuple_keys, batch_size=indexed_batch_size
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
                    f"Arrays in {type(self).__name__} data must agree with batch size "
                    f" {self._batch_size[0]}, but got shape {value.shape}"
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
        self._arrays = [
            array
            for i, array in enumerate(self._arrays)
            if i not in index_key_extensions
        ]

        # For new arrays, add the new key and value
        if isinstance(value, np.ndarray):
            self._arrays.append(value)
            self._tuple_keys.append(index)

        # For new NestedDicts, add the new keys and values
        else:
            for key, sub_value in value.items(
                include_prefixes=True, return_all_tuples=True
            ):
                new_tuple_key = index + key
                self._arrays.append(sub_value)
                self._tuple_keys.append(new_tuple_key)

    def __contains__(self, key: Any):
        raise NotImplementedError(
            f"Contains not yet implemented for {type(self).__name__}. Use "
            f"`key in string_dict.keys()` instead."
        )

    def __len__(self):
        return self.batch_size[0]

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

            # Validate and transform the key
            if isinstance(key, str):
                key = (key,)
            elif isinstance(key, tuple):
                if not all(isinstance(sub_key, str) for sub_key in key):
                    raise ValueError(
                        f"Any key in {type(self).__name__} data that is a tuple must "
                        f"contain only strings, but got {key}."
                    )
            else:
                raise ValueError(
                    f"Each key in {type(self).__name__} data must be a string or tuple "
                    f"of strings, not {type(key)}"
                )

            # If the value is a nested dictionary, recursively create arrays
            if isinstance(value, (dict, NestedArrayDict)):
                sub_tuple_keys, sub_arrays = self._create_arrays_from_data(
                    value,
                    batch_size=batch_size,
                )
                for sub_key in sub_tuple_keys:
                    tuple_keys.append(key + sub_key)
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
                tuple_keys.append(key)

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


def stack_nested_array_dicts(nds: Sequence[NestedArrayDict], dim=0) -> NestedArrayDict:
    """Stack a sequence of NestedArrayDicts along a new dimension.

    Parameters
    ----------
    nds : Sequence[NestedArrayDict]
        The NestedArrayDicts to stack. All NestedArrayDicts must have the same keys,
        shapes and batch sizes.
    dim : int, default=0
        The dimension to stack along. This must be a non-negative integer at most equal
        to the number of batch dimensions.

    Returns
    -------
    stacked : NestedArrayDict
        The NestedArrayDict with the stacked arrays. The batch size of the new
        NestedArrayDict is the batch size of the input NestedArrayDicts with the new
        dimension added.
    """

    # Check that all NestedArrayDicts have the same keys
    expected_keys = set(nds[0].keys(include_prefixes=True))
    for nd in nds[1:]:
        if set(nd.keys(include_prefixes=True)) != expected_keys:
            raise ValueError(
                f"All NestedArrayDicts must have the same keys, but found "
                f"{list(nd.keys(include_prefixes=True))} and {list(expected_keys)}"
            )

    # Stack the arrays
    nd_dict = {}
    for key in nds[0].keys(include_prefixes=True):
        nd_dict[key] = np.stack([nd[key] for nd in nds], axis=dim)

    common_batch_size = nds[0].batch_size
    stacked_batch_size = (*common_batch_size[:dim], len(nds), *common_batch_size[dim:])

    return NestedArrayDict(nd_dict, batch_size=stacked_batch_size)


def concatenate_nested_array_dicts(
    nds: Sequence[NestedArrayDict], dim=0
) -> NestedArrayDict:
    """Concatenate a sequence of NestedArrayDicts along an existing dimension.

    Parameters
    ----------
    nds : Sequence[NestedArrayDict]
        The NestedArrayDicts to concatenate. All NestedArrayDicts must have the same
        keys, shapes and batch sizes, except for the dimension to concatenate along.
    dim : int, default=0
        The dimension to concatenate along. This must be a non-negative integer at most
        equal to the number of batch dimensions.

    Returns
    -------
    concatenated : NestedArrayDict
        The NestedArrayDict with the concatenated arrays. The batch size of the new
        NestedArrayDict is the batch size of the input NestedArrayDicts with the
        concatenated dimension removed.
    """

    # Check that all NestedArrayDicts have the same keys
    for i, nd in enumerate(nds):
        if i != 0 and not set(nd.keys()) == set(nds[0].keys()):
            raise ValueError(
                f"All NestedArrayDicts must have the same keys but the first has keys "
                f"{set(nds[0].keys()) - set(nd.keys())} which the {i}th does not have, "
                f"and the {i}th has keys {set(nd.keys()) - set(nds[0].keys())} which "
                f"the first does not have."
            )

    # Concatenate the arrays
    nd_dict = {}
    for key in nds[0].keys(include_prefixes=True):
        nd_dict[key] = np.concatenate([nd[key] for nd in nds], axis=dim)

    common_batch_size = nds[0].batch_size
    dim_length = sum(nd.batch_size[dim] for nd in nds)
    concatenated_batch_size = (
        *common_batch_size[:dim],
        dim_length,
        *common_batch_size[dim + 1 :],
    )

    return NestedArrayDict(nd_dict, batch_size=concatenated_batch_size)


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
        """Return a zero array with the specified shape.

        Returns
        -------
        zero_array : NDArray
            The array of zeros with the specified shape.
        """
        return np.zeros(self.shape, dtype=self.dtype)


@dataclass
class IntArraySpec(NumpyArraySpec):
    """Specification for a single integer numpy array."""

    dtype: ClassVar[DTypeLike] = np.int64


@dataclass
class FloatArraySpec(NumpyArraySpec):
    """Specification for a single float numpy array."""

    dtype: ClassVar[DTypeLike] = np.float32


@dataclass
class BoolArraySpec(NumpyArraySpec):
    """Specification for a single boolean numpy array."""

    dtype: ClassVar[DTypeLike] = np.bool


@dataclass
class StringArraySpec(NumpyArraySpec):
    """Specification for a single string numpy array."""

    dtype: ClassVar[DTypeLike] = NumpyStringDtype

    def zero(self) -> NDArray:
        """Return a array of null strings with the specified shape.

        Returns
        -------
        zero_array : NDArray
            The array of null strings with the specified shape.
        """
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
                    f"The shape of the {type(self).__name__} {shape} must be a "
                    f"prefix of the shape of spec {key!r} {spec.shape}"
                )
            if spec.dim_names is not None and self.dim_names is not None:
                if spec.dim_names[: len(shape)] != self.dim_names:
                    raise ValueError(
                        f"The dim_names of the {type(self).__name__} {self.dim_names} "
                        f"must be a prefix of the dim_names of spec {key!r} "
                        f"{spec.dim_names}"
                    )

    def __getitem__(self, key: str) -> NumpySpec:
        return self.specs[key]

    def __setitem__(self, key: str, value: NumpySpec):
        self.specs[key] = value

    def zero(self) -> NestedArrayDict:
        """Return a dictionary of zero arrays with the specified shape.

        Returns
        -------
        zero_dict : NestedArrayDict
            The dictionary of zero arrays with the specified shape.
        """
        return NestedArrayDict(
            {key: spec.zero() for key, spec in self.specs.items()},
            batch_size=self.shape,
        )

    def keys(self, recurse=False):
        """Iterate over the keys of CompositeSpec, optionally recursing to sub-specs.

        Parameters
        ----------
        recurse : bool, default=False
            Whether to recurse to sub-specs. In this case, the keys are tuples of
            strings representing the path to the data in the nested dictionary.

        Yields
        ------
        key : str | tuple[str, ...]
            The key of the spec. If recurse is True, the key is a tuple of strings
            representing the path to the data in the nested dictionary. Otherwise, the
            key is a string.
        """

        if not recurse:
            for key in self.specs.keys():
                yield key

        else:
            for key, spec in self.specs.items():
                if isinstance(spec, CompositeSpec):
                    for sub_key in spec.keys(recurse=True):
                        yield (key,) + sub_key
                else:
                    yield (key,)
