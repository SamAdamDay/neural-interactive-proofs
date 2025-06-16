"""Utilities for async.io operations."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Coroutine, TypeVar

__all__ = [
    "run_coroutine_sync",
]

T = TypeVar("T")


def run_coroutine_sync(coroutine: Coroutine[Any, Any, T], timeout: float = 30) -> T:
    """Run a coroutine in a synchronous context.

    Copied from https://stackoverflow.com/a/78911765

    Parameters
    ----------
    coroutine : Coroutine[Any, Any, T]
        The coroutine to run synchronously.
    timeout : float, default=30
        The timeout for the coroutine execution, in seconds. If the coroutine does not
        complete within this time, a TimeoutError will be raised.

    Returns
    -------
    T
        The result of the coroutine.

    Raises
    ------
    TimeoutError
        If the coroutine does not complete within the specified timeout.
    """

    def run_in_new_loop():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(coroutine)
        finally:
            new_loop.close()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    if threading.current_thread() is threading.main_thread():
        if not loop.is_running():
            return loop.run_until_complete(coroutine)
        else:
            with ThreadPoolExecutor() as pool:
                future = pool.submit(run_in_new_loop)
                return future.result(timeout=timeout)
    else:
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result()
