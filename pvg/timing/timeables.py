"""Utilities for creating, registering, and timing timeable actions."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
from functools import partial

import torch
from torch.autograd.profiler_util import EventList


class Timeable(ABC):
    """Base class for an action that can be timed."""

    @abstractmethod
    def run(self, profiler: torch.profiler.profile):
        """Run the action.

        Parameters
        ----------
        profiler : torch.profiler.profile
            The PyTorch profiler which is being used to time the action.
        """


TIMEABLES: dict[str, Timeable] = {}


def register_timeable(
    _timeable: Optional[type[Timeable] | Callable] = None, *, name: Optional[str] = None
) -> type[Timeable]:
    """Decorator to register a timeable.

    If a callable is given, a subclass of Timeable is created with the callable as the
    run method.

    Parameters
    ----------
    timeable : Timeable | callable
        The timeable to register. Can a subclass of Timeable or a callable.
    name : str, optional
        The name to register the timeable under. If not given, the name of the timeable
        is used.

    Returns
    -------
    Timeable
        The timeable that was registered. If a callable was given, the subclass of
        Timeable that was created is returned.

    Examples
    --------
    Register a subclass of Timeable:
    >>> @register_timeable(name="my_timeable")
    ... class MyTimeable(Timeable):
    ...     def __init__(self):
    ...         pass
    ...     def run(self, profiler):
    ...         pass

    Register a callable:
    >>> @register_timeable
    ... def my_timeable(profiler):
    ...     pass
    """

    def _register_timeable(
        timeable: type[Timeable] | Callable, *, name: Optional[str] = None
    ) -> type[Timeable]:
        if name is None:
            name = timeable.__name__

        if not issubclass(timeable, Timeable):

            def init(self, **kwargs):
                pass

            timeable = type(name, (Timeable,), {"run": timeable, "__init__": init})

        TIMEABLES[name] = timeable
        return timeable

    if _timeable is None:
        return partial(_register_timeable, name=name)
    return _register_timeable(_timeable, name=name)


def time_timeable(name: str, print_results: bool = True, **kwargs) -> EventList:
    """Time a timeable by its name.

    Parameters
    ----------
    name : str
        The name of the timeable to time.
    print_results : bool, default=True
        Whether to print the results of the timing.
    **kwargs
        Any additional keyword arguments that are passed to the timeable constructor.

    Returns
    -------
    key_averages : torch.autograd.profiler.EventList
        The profiling results.
    """
    if name not in TIMEABLES:
        raise ValueError(f"No timeable with name '{name}' registered")
    timeable_instance = TIMEABLES[name](**kwargs)

    with torch.profiler.profile() as profiler:
        timeable_instance.run(profiler)

    if print_results:
        print(profiler.key_averages().table(sort_by="self_cpu_time_total"))

    return profiler.key_averages()


def time_all_timeables(
    print_results: bool = True,
    common_kwargs: dict[str, Any] = {},
    per_timeable_kwargs: dict[str, dict[str, Any]] = {},
) -> dict[str, EventList]:
    """Time all timeables.

    Parameters
    ----------
    print_results : bool, default=True
        Whether to print the results of the timing.
    common_kwargs : dict[str, Any], default={}
        A dictionary of keyword arguments that are passed to all timeables.
    timeable_kwargs : dict[str, dict[str, Any]], default={}
        A dictionary of dictionaries of keyword arguments that are passed to specific
        timeables, with the timeable name as the key.

    Returns
    -------
    results : dict[str, torch.autograd.profiler.EventList]
        The profiling results.
    """
    results = {}
    for name in TIMEABLES:
        if print_results:
            print(f"Timing '{name}'...")
        kwargs = common_kwargs.copy()
        kwargs.update(per_timeable_kwargs.get(name, {}))
        results[name] = time_timeable(name, print_results=False, **kwargs)
    if print_results:
        for name, result in results.items():
            print("-" * len(name))
            print(name)
            print("-" * len(name))
            print(result.table(sort_by="self_cpu_time_total"))
            print()
    return results


def list_timeables():
    """List all available timeables as tuples of (name, timeable).

    Returns
    -------
    timeables : list[tuple[str, Timeable]]
        The available timeables.
    """
    return list(TIMEABLES.items())
