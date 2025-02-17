"""Utilities for dealing with this package's version number."""

import importlib
from typing import Literal, TypeAlias

VersionTupleType: TypeAlias = tuple[int, int, int] | tuple[int, int] | tuple[int]


def get_package_name() -> str:
    """Get the NIP experiments package name.

    Returns
    -------
    package_name : str
        The package name.
    """

    return __package__.partition(".")[0]


def get_version(as_tuple: bool = False) -> str | tuple[int, int, int]:
    """Get the package version number.

    Parameters
    ----------
    as_tuple : bool, optional
        If True, return the version number as a tuple of integers. If False, return the
        version number as a string. Default is False.

    Returns
    -------
    version : str | tuple[int, int, int]
        The package version number.
    """

    version_string = importlib.metadata.version(get_package_name())

    if as_tuple:
        return version_string_to_tuple(version_string)
    else:
        return version_string


def version_string_to_tuple(version_string: str) -> VersionTupleType:
    """Convert a version string to a tuple of integers.

    Parameters
    ----------
    version_string : str
        The version string to convert. Must be in the format 'x.y.z', 'x.y', or 'x'.

    Returns
    -------
    version_tuple : VersionTupleType
        The version string converted to a tuple of integers.

    Raises
    ------
    ValueError
        If the version string is not in the format 'x.y.z', 'x.y', or 'x'.
    """

    version_tuple = tuple(int(part) for part in version_string.split("."))

    if len(version_tuple) > 3:
        raise ValueError(
            f"Invalid version string: {version_string!r}. Must be in the format "
            f"'x.y.z' (3 parts), 'x.y' (2 parts), or 'x' (1 part)."
        )

    return version_tuple


def version_tuple_to_string(version_tuple: VersionTupleType | None) -> str:
    """Convert a version tuple to a string.

    Parameters
    ----------
    version_tuple : VersionTupleType | None
        The version tuple to convert. Must be a tuple of integers or None.

    Returns
    -------
    version_string : str
        The version tuple converted to a string.
    """

    if version_tuple is None:
        return "None"

    return ".".join(str(part) for part in version_tuple)


def compare_versions(
    version_1: str | VersionTupleType | None, version_2: str | VersionTupleType | None
) -> tuple[
    Literal["less", "greater", "match"], Literal["major", "minor", "patch", "none"]
]:
    """Compare two version strings, checking major, minor, and patch numbers.

    Less specific versions are treated a specifying a range of versions. E.g. "1.5" is
    treated as "1.5.x" and "1" is treated as "1.x.x". This means that "1.5" is less than
    "1.6.3" but matches "1.5.3".

    Versions can be specified as strings, tuples of integers or `None`. If `None` is
    passed, it is treated as older than any other version.

    Parameters
    ----------
    version_1 : str | VersionTupleType | None
        The first version to compare. Can be a version string, a tuple of integers or
        None.
    version_2 : str | VersionTupleType | None
        The second version to compare. Can be a version string, a tuple of integers or
        None.

    Returns
    -------
    comparison : Literal["less", "greater", "match"]
        The result of the comparison. If version_1 < version_2, returns "less". If
        version_1 > version_2, returns "greater". If the two versions compatible (i.e.
        they are the same up to the level of detail in both), returns "match".
    difference : Literal["major", "minor", "patch", "none"]
        The level of difference between the two versions. If the major version is
        different, returns "major". If the minor version is different, returns "minor".
        If the patch version is different, returns "patch". If all versions are the
        same, returns "none".
    """

    if version_1 is None:
        if version_2 is None:
            return "match", "none"
        else:
            return "less", "major"
    elif version_2 is None:
        return "greater", "major"

    if isinstance(version_1, str):
        version_1 = version_string_to_tuple(version_1)
    if isinstance(version_2, str):
        version_2 = version_string_to_tuple(version_2)

    for difference, part_1, part_2 in zip(
        ("major", "minor", "patch"), version_1, version_2
    ):
        if part_1 < part_2:
            return "less", difference
        if part_1 > part_2:
            return "greater", difference

    return "match", "none"
