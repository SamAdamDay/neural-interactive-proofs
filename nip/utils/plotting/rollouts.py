"""Utilities for processing raw rollouts for use in plotting."""

from numpy.typing import NDArray

from nip.utils.nested_array_dict import NestedArrayDict


def get_last_timestep_mask(rollouts: NestedArrayDict) -> NDArray:
    """Compute a mask for the last timestep of each rollout.

    The last timestep is defined as the timestep where the next done or next terminated
    flag is set to True, and the padding flag is not set.

    Shapes
    ------
    ``rollouts`` is a a nested array dict with the following keys and shapes:

    - ("next", "done") : (... round)
    - ("next", "terminated") : (... round)
    - ("padding") : (... round)

    The output ``last_timestep_mask`` is a boolean array with the same shape as the
    inputs: (... round)

    Parameters
    ----------
    rollouts : NestedArrayDict
        The rollouts to be analysed.

    Returns
    -------
    last_timestep_mask : NDArray
        A boolean array with the same shape as the inputs, where True indicates the last
        timestep of each rollout.
    """
    next_done = rollouts["next", "done"]
    next_terminated = rollouts["next", "terminated"]
    padding = rollouts["padding"]
    return (next_done | next_terminated) & ~padding
