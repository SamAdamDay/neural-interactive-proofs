"""Replacements for buggy parts of libraries we use."""

import torch
from torchrl.objectives.value.functional import (
    _geom_series_like,
    _custom_conv1d,
    _transpose_time,
    _get_num_per_traj,
    _split_and_pad_sequence,
    _inv_pad_sequence,
)
from torchrl.envs.transforms import Reward2GoTransform as Reward2GoTransformBuggy


@_transpose_time
def reward2go(
    reward,
    done,
    gamma,
    time_dim: int = -2,
):
    """Compute the discounted cumulative sum of rewards for multiple trajectories.

    THIS IS THE FIXED VERSION OF THE FUNCTION. The original version had a bug where the
    reward-to-go was reshaped rather than transposed.

    Parameters
    ----------
    reward : torch.Tensor
        A tensor containing the rewards received at each time step over multiple
        trajectories.
    done : torch.Tensor
        Boolean flag for end of episode. Differs from truncated, where the episode did
        not end but was interrupted.
    gamma : float, optional
        The discount factor to use for computing the discounted cumulative sum of
        rewards. Defaults to 1.0.
    time_dim : int, optional
        Dimension where the time is unrolled. Defaults to -2.

    Returns
    -------
    torch.Tensor
        A tensor of shape [B, T] containing the discounted cumulative sum of rewards
        (reward-to-go) at each time step.

    Examples
    --------
    >>> reward = torch.ones(1, 10)
    >>> done = torch.zeros(1, 10, dtype=torch.bool)
    >>> done[:, [3, 7]] = True
    >>> reward2go(reward, done, 0.99, time_dim=-1)
    tensor([[3.9404],
            [2.9701],
            [1.9900],
            [1.0000],
            [3.9404],
            [2.9701],
            [1.9900],
            [1.0000],
            [1.9900],
            [1.0000]])
    """
    shape = reward.shape
    if shape != done.shape:
        raise ValueError(
            f"reward and done must share the same shape, got {reward.shape} and {done.shape}"
        )
    # place time at dim -1
    reward = reward.transpose(-2, -1)
    done = done.transpose(-2, -1)
    # flatten if needed
    if reward.ndim > 2:
        reward = reward.flatten(0, -2)
        done = done.flatten(0, -2)

    num_per_traj = _get_num_per_traj(done)
    td0_flat = _split_and_pad_sequence(reward, num_per_traj)
    gammas = _geom_series_like(td0_flat[0], gamma, thr=1e-7)
    cumsum = _custom_conv1d(td0_flat.unsqueeze(1), gammas)
    cumsum = cumsum.squeeze(1)
    cumsum = _inv_pad_sequence(cumsum, num_per_traj)
    cumsum = cumsum.view_as(reward)

    # THIS IS THE PART THAT WAS FIXED
    cumsum = cumsum.transpose(-1, -2)

    return cumsum


class Reward2GoTransform(Reward2GoTransformBuggy):
    """Calculates the reward to go based on the episode reward and a discount factor.

    This is a fixed version of the ``Reward2GoTransform`` class from torchrl. The
    original version had a bug where the reward-to-go was reshaped rather than
    transposed.

    See :external+torchrl:class:`torchrl.envs.transforms.Reward2GoTransform` for more
    information.
    """

    def _inv_apply_transform(
        self, reward: torch.Tensor, done: torch.Tensor
    ) -> torch.Tensor:
        return reward2go(reward, done, self.gamma)
