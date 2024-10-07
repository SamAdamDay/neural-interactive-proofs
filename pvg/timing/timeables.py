"""Utilities for creating, registering, and timing timeable actions."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, TypeVar
from functools import partial

import torch
from torch.autograd.profiler_util import EventList

from pvg.constants import LOG_DIR
import os
import datetime


class Timeable(ABC):
    """Base class for an action that can be timed.

    Parameters
    ----------
    param_scale : float, default=1.0
        Key default parameters (if any) will be scaled by this factor.
    """

    def __init__(self, *, param_scale: float = 1.0):
        self.param_scale = param_scale

    @abstractmethod
    def run(self, profiler: torch.profiler.profile):
        """Run the action.

        Parameters
        ----------
        profiler : torch.profiler.profile
            The PyTorch profiler which is being used to time the action.
        """

    def _get_profiler_args(
        self,
        log_dir: Optional[str],
        record_shapes: bool,
        profile_memory: bool,
        with_stack: bool,
    ) -> dict:
        """Get the arguments for the PyTorch profiler.

        Parameters
        ----------
        log_dir : str or None
            The directory to save the profiling results to, if any.
        record_shapes : bool
            Whether to record tensor shapes. This introduces an additional overhead.
        profile_memory : bool
            Whether to profile memory usage.
        with_stack : bool
            Whether to record the stack trace. This introduces an additional overhead.

        Returns
        -------
        profiler_args : dict
            The arguments for the PyTorch profiler.
        """
        if log_dir is not None:
            on_trace_ready = torch.profiler.tensorboard_trace_handler(
                log_dir, use_gzip=True
            )
        else:
            on_trace_ready = None
        return dict(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=on_trace_ready,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
        )

    def time(
        self,
        log_dir: Optional[str] = None,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
    ) -> torch.profiler.profile:
        """Time the action.

        Parameters
        ----------
        log_dir : str, optional
            The directory to save the profiling results to, if any.
        record_shapes : bool
            Whether to record tensor shapes. This introduces an additional overhead.
        profile_memory : bool
            Whether to profile memory usage.
        with_stack : bool
            Whether to record the stack trace. This introduces an additional overhead.

        Returns
        -------
        profiler : torch.profiler.profile
            The PyTorch profiler containing the timing information.
        """

        profiler_args = self._get_profiler_args(
            log_dir=log_dir,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
        )
        with torch.profiler.profile(**profiler_args) as profiler:
            self.run(profiler)

        return profiler


class TrainingTimeable(Timeable, ABC):
    """Base class timeable which involves some kind of training.

    The schedule is as follows:

    1. For the first `wait` steps of training, do nothing.
    2. For each of the `repeat` cycles:
        a. For the first `warmup` steps of the cycle, run the profiler but don't record.
        b. For the next `active` steps of the cycle, run the profiler and record.

    Parameters
    ----------
    param_scale : float, default=1.0
        Key default parameters (if any) will be scaled by this factor.
    wait : int, default=2
        The number of training steps to wait before starting to profile.
    warmup : int, default=1
        The number of warmup steps in each cycle.
    active : int, default=3
        The number of steps to profile in each cycle.
    repeat : int, default=2
        The number of cycles to repeat.
    """

    def __init__(
        self,
        *,
        param_scale: float = 1.0,
        wait: int = 2,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 2,
    ):
        super().__init__(param_scale=param_scale)
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat

        self.num_steps = wait + (warmup + active) * repeat

    def _get_profiler_args(
        self,
        log_dir: Optional[str],
        record_shapes: bool,
        profile_memory: bool,
        with_stack: bool,
    ) -> dict:
        """Get the arguments for the PyTorch profiler.

        Parameters
        ----------
        log_dir : str, optional
            The directory to save the profiling results to, if any.
        record_shapes : bool
            Whether to record tensor shapes. This introduces an additional overhead.
        profile_memory : bool
            Whether to profile memory usage.
        with_stack : bool
            Whether to record the stack trace. This introduces an additional overhead.

        Returns
        -------
        profiler_args : dict
            The arguments for the PyTorch profiler.
        """
        profiler_args = super()._get_profiler_args(
            log_dir=log_dir,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
        )
        profiler_args.update(
            schedule=torch.profiler.schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
            )
        )
        return profiler_args


TIMEABLES: dict[str, Timeable] = {}

T = TypeVar[Timeable]


def register_timeable(
    _timeable: Optional[type[Timeable] | Callable] = None, *, name: Optional[str] = None
) -> Callable | type[Timeable]:
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
        timeable: type[T] | Callable, *, name: Optional[str] = None
    ) -> type[T] | type[Timeable]:
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


def time_timeable(
    name: str,
    log_tensorboard_results: bool = True,
    print_results: bool = False,
    record_shapes: bool = True,
    profile_memory: bool = True,
    with_stack: bool = False,
    *,
    param_scale: float = 1.0,
    **kwargs,
) -> torch.profiler.profile:
    """Time a timeable by its name.

    Parameters
    ----------
    name : str
        The name of the timeable to time.
    log_tensorboard_results : bool, default=True
        Whether to log the results to TensorBoard in the log directory.
    record_shapes : bool, default=True
        Whether to record tensor shapes. This introduces an additional overhead.
    profile_memory : bool, default=True
        Whether to profile memory usage.
    with_stack : bool, default=False
        Whether to record the stack trace. This introduces an additional overhead.
    print_results : bool, default=False
        Whether to print the results of the timing.
    param_scale : float, default=1.0
        The scale factor for key default parameters.
    **kwargs
        Any additional keyword arguments that are passed to the timeable constructor.

    Returns
    -------
    profiler : torch.profiler.profile
        The PyTorch profiler containing the timing information.
    """
    if name not in TIMEABLES:
        raise ValueError(f"No timeable with name '{name}' registered")
    timeable_instance = TIMEABLES[name](param_scale=param_scale, **kwargs)

    # Create a log directory for the profiler
    if log_tensorboard_results:
        time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(LOG_DIR, f"{name}_{time_now}")
        os.makedirs(log_dir, exist_ok=False)
    else:
        log_dir = None

    profiler = timeable_instance.time(
        log_dir=log_dir,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    )

    if print_results:
        print(  # noqa: T201
            profiler.key_averages().table(sort_by="self_cpu_time_total")
        )

    return profiler


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
            print(f"Timing '{name}'...")  # noqa: T201
        kwargs = common_kwargs.copy()
        kwargs.update(per_timeable_kwargs.get(name, {}))
        results[name] = time_timeable(name, print_results=False, **kwargs)
    if print_results:
        for name, result in results.items():
            print("-" * len(name))  # noqa: T201
            print(name)  # noqa: T201
            print("-" * len(name))  # noqa: T201
            print(result.table(sort_by="self_cpu_time_total"))  # noqa: T201
            print()  # noqa: T201
    return results


def list_timeables():
    """List all available timeables as tuples of (name, timeable).

    Returns
    -------
    timeables : list[tuple[str, Timeable]]
        The available timeables.
    """
    return list(TIMEABLES.items())
