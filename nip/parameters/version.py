"""A function for transforming old hyper-parameter dicts to new ones.

The specification and meaning of the hyper-parameter fields can change over time.
However, sometimes we want to access hyper-parameter dictionaries that were created with
an older version of the package. For example, we may want resume an old run, or re-run
it with additional tests or logging.

The `converge_hyper_param_dict` function is used to transform an old hyper-parameter
dictionary to one which is compatible with the current version of the package. This is
based on the version number of the package that was used to create the hyper-parameter
dictionary, stored in the `_package_version` field of the dictionary.

The way this works is by stepping through the versions of the package, applying
conversion functions to the dictionary as needed. These conversion functions are stored
in this module, registered with the `register_conversion_function` decorator.

Each time a new version of the package is released, a new conversion function should be
added to this module.

Example
-------

>>> @register_conversion_function("2.1.5", "2.2")
>>> def _from_2_1_5_to_2_2(hyper_param_dict: dict) -> dict:
>>>     ...
"""

from typing import Callable
from warnings import warn

from nip.utils.version import (
    version_string_to_tuple,
    version_tuple_to_string,
    get_version,
    compare_versions,
    VersionTupleType,
)


class MultipleConversionFunctionsMatchError(Exception):
    """Exception raised when multiple conversion functions match a version number."""

    def __init__(self, dict_version: str, found_version_1: str, found_version_2: str):
        super().__init__(
            f"When converting hyper-parameter dictionary, multiple conversion "
            f"functions matched the version {dict_version}. Found {found_version_1} "
            f"and {found_version_2}."
        )


class ConversionFunctionNotFoundError(Exception):
    """Exception raised when no conversion function matches a version number."""

    def __init__(self, dict_version: str):
        super().__init(
            f"When converting hyper-parameter dictionary, no conversion function "
            f"matched the version {dict_version}."
        )


def convert_hyper_param_dict(hyper_param_dict: dict) -> dict:
    """Convert an old hyper-parameter dictionary to a new one.

    The specification and meaning of the hyper-parameter fields can change over time.
    However, sometimes we want to access hyper-parameter dictionaries that were created
    with an older version of the package. For example, we may want resume an old run, or
    re-run it with additional tests or logging.

    This function is used to transform an old hyper-parameter dictionary to one which is
    compatible with the current version of the package. This is based on the version
    number of the package that was used to create the hyper-parameter dictionary, stored
    in the `_package_version` field of the dictionary.

    The way this works is by stepping through the versions of the package, applying
    conversion functions to the dictionary as needed. These conversion functions are
    stored in the `nip.parameters.version` module, registered with the
    `register_conversion_function` decorator.

    Parameters
    ----------
    hyper_param_dict : dict
        The old hyper-parameter dictionary.

    Returns
    -------
    new_hyper_param_dict : dict
        The new hyper-parameter dictionary, compatible with the current version of the
        package.

    Raises
    ------
    MultipleConversionFunctionsMatchError
        If at some point in the conversion process, multiple conversion functions are
        found which match a version number.
    ConversionFunctionNotFoundError
        If at some point in the conversion process, no conversion function is found
        which matches a version number.
    """

    current_version = get_version(as_tuple=True)
    dict_version = hyper_param_dict.get("_package_version", None)

    while True:

        comparison, _ = compare_versions(current_version, dict_version)
        if comparison == "match":
            break

        # Find the next conversion function to apply
        conversion_func = None
        conversion_func_version = None
        next_version = None
        for (from_version, to_version), func in CONVERSION_FUNCTIONS.items():
            comparison, _ = compare_versions(from_version, dict_version)
            if comparison != "match":
                continue
            if conversion_func is None:
                conversion_func = func
                conversion_func_version = from_version
                next_version = to_version
            else:
                raise MultipleConversionFunctionsMatchError(
                    version_tuple_to_string(dict_version),
                    version_tuple_to_string(conversion_func_version),
                    version_tuple_to_string(from_version),
                )
        if conversion_func is None:
            raise ConversionFunctionNotFoundError(version_tuple_to_string(dict_version))

        # Apply the conversion function
        hyper_param_dict = conversion_func(hyper_param_dict)
        dict_version = next_version

    return hyper_param_dict


CONVERSION_FUNCTIONS: dict[
    tuple[VersionTupleType | None, VersionTupleType], Callable[[dict], dict]
] = {}


def register_conversion_function(
    from_version: str | None, to_version: str
) -> Callable[[Callable[[dict], dict]], Callable[[dict], dict]]:
    """Decorate a function to register it as a conversion function.

    A conversion function is used to transform hyper-parameter dictionaries from one
    version of the package to another. The function should take a dictionary as input
    and return a dictionary as output.

    Versions are represented as strings, e.g. "1.5.3". It is possible to omit the last
    elements of the version, e.g. "1.5" for version 1.5.x or "1" for version 1.x.x. In
    this case, it applies to any version which matches the pattern.

    The `from_version` parameter may be `None`. This deals with the case where the
    hyper-parameter dictionary does not have a `_package_version` field. This is the
    case for runs created prior to the introduction of this field.

    Parameters
    ----------
    from_version : str | None
        The version number from which the conversion function should be applied. See
        above for the format of the version number.
    to_version : str
        The version number to which the conversion function should be applied. See above
        for the format of the version number.
    """

    comparison, _ = compare_versions(from_version, to_version)
    if comparison != "less":
        raise ValueError(
            f"The `from_version` must be strictly less than the `to_version` in the "
            f"`register_conversion_function` decorator. Got {from_version} and "
            f"{to_version}."
        )

    if from_version is not None:
        from_version = version_string_to_tuple(from_version)
    to_version = version_string_to_tuple(to_version)

    def decorator(func: Callable[[dict], dict]) -> Callable[[dict], dict]:
        CONVERSION_FUNCTIONS[(from_version, to_version)] = func
        return func

    return decorator


@register_conversion_function(None, "0.1")
def _from_none_to_0_1(hyper_param_dict: dict) -> dict:

    warn(
        "Converting a hyper-parameter dict from before version numbers were tracked. "
        "The interpretation of some hyper-parameters may be incorrect."
    )

    # Nothing has changed
    return hyper_param_dict


@register_conversion_function("0.1", "1.0")
def _from_0_1_to_1_0(hyper_param_dict: dict) -> dict:

    # "pvg" was renamed to "nip"
    if hyper_param_dict.get("interaction_protocol", None) == "pvg":
        hyper_param_dict["interaction_protocol"] = "nip"
    if "pvg_protocol" in hyper_param_dict:
        hyper_param_dict["nip_protocol"] = hyper_param_dict.pop("pvg_protocol")
        hyper_param_dict["nip_protocol"]["_type"] = "NipProtocolParameters"

    # "abstract_decision_problem" was renamed to "adp"
    if hyper_param_dict.get("decision_problem", None) == "abstract_decision_problem":
        hyper_param_dict["decision_problem"] = "adp"

    return hyper_param_dict
