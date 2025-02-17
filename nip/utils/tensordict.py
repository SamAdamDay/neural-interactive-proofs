"""Utilities for manipulating tensordicts."""

import torch

from tensordict import TensorDictBase
from tensordict.utils import NestedKey


def tensordict_add(
    td1: TensorDictBase, td2: TensorDictBase, *, inplace=False
) -> TensorDictBase:
    """Add two tensordicts together.

    Parameters
    ----------
    td1: TensorDictBase
        The first tensordict
    td2: TensorDictBase
        The second tensordict
    inplace: bool, default=False
        Whether to modify the first tensordict in place. Otherwise the first tensordict
        is cloned before adding the second tensordict.

    Returns
    -------
    td_sum: TensorDictBase
        The sum of the two tensordicts
    """

    if set(td1.keys(include_nested=True)) != set(td2.keys(include_nested=True)):
        raise ValueError("The keys of the two tensordicts must be the same to add.")

    if not inplace:
        td1 = td1.clone()

    keys = td1.keys(include_nested=True, leaves_only=True)

    return td1.update(
        {key: td1.get(key) + td2.get(key) for key in keys},
        clone=False,
        inplace=True,
    )


def tensordict_scalar_multiply(
    td: TensorDictBase, scalar: float | int, *, inplace=False
) -> TensorDictBase:
    """Multiply a tensordict by a scalar.

    Parameters
    ----------
    td: TensorDictBase
        The tensordict
    scalar: float | int
        The scalar to multiply every tensor in the tensordict by
    inplace: bool, default=False
        Whether to modify the tensordict in place. Otherwise the tensordict is cloned
        before multiplying by the scalar.

    Returns
    -------
    td_scaled: TensorDictBase
        The scaled tensordict
    """

    if not inplace:
        td = td.clone()

    keys = td.keys(include_nested=True, leaves_only=True)

    return td.update(
        {key: td.get(key) * scalar for key in keys},
        clone=False,
        inplace=True,
    )


def get_key_batch_size(td: TensorDictBase, key: NestedKey) -> torch.Size:
    """Get the batch_size of the inner-most tensordict containing the key.

    For instance, if the key is ("next", "agents", "done"), this function will return
    the batch size of the "agents" sub-tensordict of the "next" sub-tensordict.

    Parameters
    ----------
    td: TensorDictBase
        The tensordict containing the key
    key: NestedKey
        The key for which to get the batch size

    Returns
    -------
    batch_size: torch.Size
        The batch size of the inner-most tensordict containing the key
    """

    if isinstance(key, str):
        return td.batch_size
    return td.get(key[:-1]).batch_size
