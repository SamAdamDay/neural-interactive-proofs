"""A nested dictionary of strings data structure."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Union, ClassVar
from textwrap import indent
from functools import partial
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray, DTypeLike

import pyarrow as pa

from pvg.utils.types import NumpyStringDtype

ScalarType = Union[int, float, str, bool, pa.DataType]

type_to_pyarrow_type_factory = {
    int: pa.int64,
    float: pa.float64,
    str: pa.string,
    bool: pa.bool_,
    list: pa.list_,
}

type_checker_to_arrow_type_factory = {
    pa.types.is_boolean: pa.bool_,
    pa.types.is_string: pa.string,
    pa.types.is_int8: pa.int8,
    pa.types.is_int16: pa.int16,
    pa.types.is_int32: pa.int32,
    pa.types.is_int64: pa.int64,
    pa.types.is_float16: pa.float16,
    pa.types.is_float32: pa.float32,
    pa.types.is_float64: pa.float64,
}


def _convert_to_pyarrow_type_factory(
    dtype: Any, allow_callable: bool = True
) -> callable:
    """Convert a data type to a corresponding PyArrow factory function

    Parameters
    ----------
    dtype: Any
        The data type to be converted
    allow_callable: bool, default=True
        Whether to assume that a callable `dtype` is already a factory function if other
        methods do not succeed.
    """

    if dtype in type_to_pyarrow_type_factory:
        return type_to_pyarrow_type_factory[dtype]

    if isinstance(dtype, pa.DataType):
        for type_checker, factory in type_checker_to_arrow_type_factory.items():
            if type_checker(dtype):
                return factory

    if allow_callable and callable(dtype):
        return dtype

    raise ValueError(f"Could not determine PyArrow type factory for dtype {dtype}")


def _get_list_shape_and_dtype(
    data: Any, validate: bool = False
) -> tuple[tuple[int, ...], type | None]:
    """Get the shape and dtype of a multi-dimensional list.

    Parameters
    ----------
    data : Any
        The multi-dimensional list.
    validate : bool, default=False
        Whether to validate that the list is rectangular.

    Returns
    -------
    shape : tuple
        The shape of the list.
    dtype : type | None
        The type of the elements in the list.

    Raises
    ------
    ValueError
        If the list is not rectangular and `validate=True`.
    """

    if not isinstance(data, (list, pa.Array, pa.ListScalar)):
        if hasattr(data, "type"):
            return (), data.type
        else:
            return (), type(data)

    if len(data) == 0:
        return (0,), None

    sub_shape, dtype = _get_list_shape_and_dtype(data[0], validate=validate)
    shape = (len(data),) + sub_shape

    if validate:
        for sub_data in data[1:]:
            if _get_list_shape_and_dtype(sub_data)[0] != shape[1:]:
                raise ValueError("The list is not rectangular")

    return shape, dtype


def _create_multidim_pyarrow_type(
    shape: tuple[int, ...], dtype_factory: callable
) -> pa.DataType:
    """Create a multi-dimensional PyArrow type.

    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of the array.
    dtype_factory : callable
        The PyArrow type factory for the date type.

    Returns
    -------
    type : pa.DataType
        The PyArrow type of the array.
    """

    if len(shape) == 0:
        return dtype_factory()

    return pa.list_(_create_multidim_pyarrow_type(shape[1:], dtype_factory), shape[0])


def _create_constant_multidim_list(value: Any, shape: tuple[int, ...]) -> list:
    """Create a multi-dimensional list filled with a constant value.

    Parameters
    ----------
    value : Any
        The constant value to fill the list with.
    shape : tuple[int, ...]
        The shape of the list.

    Returns
    -------
    list : list
        The list filled with the constant value.
    """

    if len(shape) == 0:
        return value

    return [_create_constant_multidim_list(value, shape[1:]) for _ in range(shape[0])]


class Array:
    """A multi-dimensional array data structure backed by PyArrow.

    Parameters
    ----------
    data : Any, optional
        The data to store in the array. If `None`, the array is empty.
    dtype : ScalarType, optional
        The data type of the array. If `None`, the data type is inferred from the data.
    dim_names : Sequence[str] | str, optional
        Names for the dimensions of the array. This is purely for code documentation
        purposes. Can be either a tuple of strings or a single string of space-separated
        names.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def arrow_array(self) -> pa.Array:
        """The PyArrow array underlying the Array instance."""
        return self._data

    @property
    def dim_names(self) -> tuple[str, ...] | None:
        return self._dim_names

    @dim_names.setter
    def dim_names(self, dim_names: Sequence[str] | str | None):
        if dim_names is not None:
            if isinstance(dim_names, str):
                dim_names = tuple(dim_names.split())
            else:
                dim_names = tuple(dim_names)
            if len(dim_names) != len(self._shape):
                raise ValueError(
                    f"Number of dimension names {len(dim_names)} must match number of "
                    f"dimensions {len(self._shape)}"
                )
        self._dim_names = dim_names

    def __init__(
        self,
        data: Optional[Any] = None,
        dtype: Optional[ScalarType] = None,
        dim_names: Optional[Sequence[str] | str] = None,
    ):

        if data is None:

            if dtype is None:
                raise ValueError("Either `data` or `dtype` must be specified")

            data = []
            self._shape = (0,)

        # If the data is already an Array, we make a shallow copy
        elif isinstance(data, type(self)):
            data_dtype = data._dtype
            self._shape = data._shape
            data = data._data

        else:

            # If the data is a map generator, first make a PyArrow array out of it
            if isinstance(data, map):
                data = pa.array(data)

            # Determine the shape and dtype from the data
            self._shape, data_dtype = _get_list_shape_and_dtype(data, validate=False)

        if dtype is None:
            if data_dtype is None:
                raise ValueError("dtype must be specified for empty data")
            dtype = data_dtype

        self.dim_names = dim_names

        # Try to get the PyArrow type factory for the dtype
        self._dtype_factory = _convert_to_pyarrow_type_factory(
            dtype, allow_callable=False
        )
        self._dtype = self._dtype_factory()

        if self._shape != ():

            # Create the array type, which is a nested sequence of fixed list sizes
            self._array_type = _create_multidim_pyarrow_type(
                self._shape[1:], self._dtype_factory
            )

            # If the data is already a PyArrow array, we view it as the array type
            if isinstance(data, (pa.Array, pa.ChunkedArray)):
                self._data = data.view(self._array_type)

            # Otherwise, we create a new PyArrow array from the data
            else:
                self._data = pa.array(data, type=self._array_type)

        # For scalars, we just save the data as a scalar
        else:
            self._array_type = None
            self._data = pa.scalar(data)

    @classmethod
    def create_constant_array(
        cls,
        value: Any,
        shape: Sequence[int] | int,
        dtype: ScalarType,
        dim_names: Optional[Sequence[str] | str] = None,
    ) -> "Array":
        """Create a new Array filled with a constant value.

        Parameters
        ----------
        value : Any
            The constant value to fill the array with.
        shape : Sequence[int] | int
            The shape of the array.
        dtype : ScalarType
            The data type of the array.
        dim_names : Sequence[str] | str, optional
            Names for the dimensions of the array. This is purely for code documentation
            purposes. Can be either a tuple of strings or a single string of
            space-separated names.

        Returns
        -------
        array : Array
            The new array filled with zeros.
        """
        if isinstance(shape, int):
            shape = (shape,)
        return cls(
            _create_constant_multidim_list(value, shape), dtype, dim_names=dim_names
        )

    @classmethod
    def zeros(
        cls,
        shape: Sequence[int] | int,
        dtype: ScalarType,
        dim_names: Optional[Sequence[str] | str] = None,
    ) -> "Array":
        """Create a new Array filled with zeros.

        Parameters
        ----------
        shape : Sequence[int] | int
            The shape of the array.
        dtype : ScalarType
            The data type of the array.
        dim_names : Sequence[str] | str, optional
            Names for the dimensions of the array. This is purely for code documentation
            purposes. Can be either a tuple of strings or a single string of
            space-separated names.

        Returns
        -------
        array : Array
            The new array filled with zeros.
        """
        return cls.create_constant_array(0, shape, dtype, dim_names=dim_names)

    @classmethod
    def ones(
        cls,
        shape: Sequence[int] | int,
        dtype: ScalarType,
        dim_names: Optional[Sequence[str] | str] = None,
    ) -> "Array":
        """Create a new Array filled with ones.

        Parameters
        ----------
        shape : Sequence[int] | int
            The shape of the array.
        dtype : ScalarType
            The data type of the array.
        dim_names : Sequence[str] | str, optional
            Names for the dimensions of the array. This is purely for code documentation
            purposes. Can be either a tuple of strings or a single string of
            space-separated names.

        Returns
        -------
        array : Array
            The new array filled with ones.
        """
        return cls.create_constant_array(1, shape, dtype, dim_names=dim_names)

    @classmethod
    def empty_strings(
        cls,
        shape: Sequence[int] | int,
        dim_names: Optional[Sequence[str] | str] = None,
    ) -> "Array":
        """Create a new string Array filled with empty strings.

        Parameters
        ----------
        shape : Sequence[int] | int
            The shape of the array.
        dim_names : Sequence[str] | str, optional
            Names for the dimensions of the array. This is purely for code documentation
            purposes. Can be either a tuple of strings or a single string of
            space-separated names.

        Returns
        -------
        array : Array
            The new array filled with zeros.
        """
        return cls.create_constant_array(
            "", shape, dtype=pa.string(), dim_names=dim_names
        )

    @classmethod
    def nones(
        cls,
        shape: Sequence[int] | int,
        dtype: ScalarType,
        dim_names: Optional[Sequence[str] | str] = None,
    ) -> "Array":
        """Create a new Array filled with None.

        Parameters
        ----------
        shape : Sequence[int] | int
            The shape of the array.
        dtype : ScalarType
            The data type of the array.
        dim_names : Sequence[str] | str, optional
            Names for the dimensions of the array. This is purely for code documentation
            purposes. Can be either a tuple of strings or a single string of
            space-separated names.

        Returns
        -------
        array : Array
            The new array filled with None.
        """
        return cls.create_constant_array(None, shape, dtype, dim_names=dim_names)

    def get_item(self, index: Any, always_return_arrays: bool = True) -> Any:
        """Get an item or subarray from an index.

        Parameters
        ----------
        index : Any
            The index to get.
        always_return_arrays : bool, default=True
            Whether to always return arrays, even if the index is a single element.

        Returns
        -------
        item : Any
            The item or subarray at the index.
        """

        if self._shape == ():
            raise IndexError(f"Cannot index a scalar {type(self).__name__}")

        original_index = index

        if isinstance(index, (int, slice)) or index is Ellipsis:
            index = (index,)
        elif not isinstance(index, tuple):
            raise TypeError(
                f"Index must be an integer, slice, tuple, or Ellipsis, but got {index}"
            )

        # If the first item in the index is an ellipsis, replace it with `:` slices
        # until the length of the shape
        if index[0] is Ellipsis:
            index = (len(self._shape) - len(index) + 1) * (slice(None),) + index[1:]

        if len(index) > len(self._shape):
            raise IndexError(
                f"Index {original_index} is out of bounds for array of shape "
                f"{self._shape}"
            )

        has_slice = False
        all_colons = True
        for index_item in index:
            if index_item is Ellipsis:
                raise IndexError(
                    f"An ellipsis is only allowed at the start of the index, got"
                    f" {original_index!r}"
                )
            elif not isinstance(index_item, (int, slice)):
                raise TypeError(
                    f"Index must be tuple of integers or slices, but got "
                    f"{original_index!r}"
                )
            elif isinstance(index_item, slice):
                has_slice = True
            if index_item != slice(None):
                all_colons = False

        # If it's all colons, we just want `self`, so we can skip the rest of the logic
        if all_colons:
            return self

        item = self._get_tuple_item(index, self._data)

        # If the index contains a slice or is shorter than the shape, we are getting a
        # subarray, so return a new Array. Also do this if `always_return_arrays` is
        # True
        if has_slice or len(index) < len(self._shape) or always_return_arrays:
            item = type(self)(item, dtype=self._dtype)

        # Otherwise convert the item to a python object
        else:
            item = item.as_py()

        return item

    def __getitem__(self, index: Any) -> Any:
        return self.get_item(index)

    def __repr__(self) -> str:

        if self._shape == ():
            return (
                f"{type(self).__name__}({self.to_python()!r}, shape={self._shape}, "
                f"dtype={self._dtype})"
            )

        output = f"{type(self).__name__}(shape={self._shape}, dtype={self._dtype}"
        if self.dim_names is not None:
            output += f", dim_names={self.dim_names}"
        output += ")"
        return output

    def __iter__(self):
        if self._shape == ():
            raise TypeError("Cannot iterate over a scalar")
        for item in self._data:
            yield item.as_py()

    def to_list(self) -> list:
        """Get the array as a Python list

        Returns
        -------
        as_list : list
            The array converted to a list.
        """
        if self._shape == ():
            raise TypeError("Cannot convert a scalar to a list")
        return self._data.to_pylist()

    def to_python(self) -> Any:
        """Get the array as a Python object."""
        if self._shape == ():
            return self._data.as_py()
        return self._data.to_pylist()

    def _get_tuple_item(
        self, index: tuple[int | slice, ...], array: pa.Array | pa.FixedSizeListScalar
    ) -> Any:
        """Get an item or subarray from an index tuple recursively.

        If the index is a tuple of the same length as the shape of the array, the item
        at the index is returned. Otherwise, a subarray sliced using the index is
        returned.

        Parameters
        ----------
        index : index: tuple[int | slice, ...]
            The index tuple.
        array : pa.Array | pa.FixedSizeListScalar
            The array to index.

        Returns
        -------
        item : Any
            The item or subarray at the index.
        """

        if len(index) == 0:
            return array

        # If the first index item is an int, get the corresponding element
        if isinstance(index[0], int):
            return self._get_tuple_item(index[1:], array[index[0]])

        # Otherwise it is a slice. Slice the array and map the rest of the index
        else:
            if isinstance(array, (pa.Array, pa.ChunkedArray)):
                sliced_array = array[index[0]]
                return map(partial(self._get_tuple_item, index[1:]), sliced_array)
            else:
                start = index[0].start
                stop = index[0].stop
                step = index[0].step
                if start is None:
                    start = 0
                if step is None:
                    step = 1
                sliced_array = pa.compute.list_slice(array, start, stop, step)
                return [self._get_tuple_item(index[1:], item) for item in sliced_array]


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
        self._setup_from_arrays_and_scalars(tuple_keys, arrays)

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
        nested_dict._setup_from_arrays_and_scalars(tuple_keys, array_and_scalars)
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

            # If the value is a nested dictionary, recursively create pyarrow arrays
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

    def _setup_from_arrays_and_scalars(
        self,
        tuple_keys: list[tuple[str, ...]],
        array_and_scalars: list[ArrayLike | Any],
    ):
        """Set up the NestedDict from a list of arrays and keys.

        This method sets the internal pyarrow table and keys.

        Parameters
        ----------
        tuple_keys : tuple
            The keys of the data. Each key is a tuple of strings that represents the
            path to the data in the nested dictionary.
        array_and_scalars : list[ArrayLike | Any]
            The data as arrays and scalars. The order corresponds to the order of the
            keys.
        """

        self._arrays = array_and_scalars
        self._tuple_keys = tuple_keys

    # def _build_scalar_dict(
    #     self,
    #     keys: list[tuple[str, ...]],
    #     scalars: list,
    # ) -> dict:
    #     """Build a nested dictionary of scalars from a list of them.

    #     Parameters
    #     ----------
    #     keys : list[tuple[str, ...]]
    #         The keys to build the dictionary from.
    #     scalars : list
    #         The scalars to build the dictionary from, in the order of the tuple keys.

    #     Returns
    #     -------
    #     scalar_dict : dict
    #         The nested dictionary of scalars.
    #     """

    #     # Select the first element of each key
    #     top_level_keys = {key[0] for key in keys}

    #     scalar_dict = {}
    #     for key in top_level_keys:
    #         sub_keys = []
    #         sub_scalars = []
    #         for sub_key, scalar in zip(keys, scalars):
    #             if sub_key[0] == key and len(sub_key) > 1:
    #                 sub_keys.append(sub_key[1:])
    #                 sub_scalars.append(scalar)

    #         # If there are no subkeys, the key is a leaf and we add the scalar
    #         if len(sub_keys) == 0:
    #             scalar = scalars[keys.index((key,))]
    #             if isinstance(scalar, pa.Scalar):
    #                 scalar = scalar.as_py()
    #             scalar_dict[key] = scalar

    #         # If there are subkeys, the key is not terminal and we add the
    #         # sub-dictionary recursively
    #         else:
    #             sub_dict = self._build_scalar_dict(sub_keys, sub_scalars)
    #             scalar_dict[key] = sub_dict

    #     return scalar_dict

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
