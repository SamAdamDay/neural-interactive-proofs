import logging
from typing import Annotated as String, Any, get_origin, get_args, Union
from types import UnionType

import torch

from numpy.dtypes import StringDType

TorchDevice = torch.device | str | int

LoggingType = logging.Logger | logging.LoggerAdapter

NumpyStringDtype = StringDType(na_object=None)


def get_union_elements(tp: Any) -> list:
    """Get the elements of a union type

    If the type is not a union, returns a singleton list containing the type.

    Parameters
    ----------
    tp : Any
        The type, which may be a union type

    Returns
    -------
    type_list : list
        A list of types that are part of the union, or a singleton list if the type is
        not a union.
    """

    if get_origin(tp) is UnionType or get_origin(tp) is Union:
        return sum([get_union_elements(sub_tp) for sub_tp in get_args(tp)], [])
    else:
        return [tp]
