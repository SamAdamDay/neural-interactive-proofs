"""Utilities for useful mathematical operations."""

from typing import Tuple, Union, Sequence, Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from jaxtyping import Float, Int

from pvg.parameters import IhvpVariant


def dot_td(td1, td2):
    """
    Calculate the dot product between two (parameter) dictionaries.

    Parameters:
    td1 (dict): The first dictionary.
    td2 (dict): The second dictionary.

    Returns:
    float: The dot product of the two dictionaries.

    Raises:
    ValueError: If td1 and td2 do not have the same keys.
    """
    if td1.keys() != td2.keys():
        raise ValueError("td1 and td2 must have the same keys.")
    else:
        return sum((td1[k] * td2[k]).sum() for k in td1.keys())


def sum_td(td1, td2):
    """
    Calculate the sum of two (parameter) dictionaries.

    Parameters:
    td1 (dict): The first dictionary.
    td2 (dict): The second dictionary.

    Returns:
    dict: The sum product of the two dictionaries.

    Raises:
    ValueError: If td1 and td2 do not have the same keys.
    """
    if td1.keys() != td2.keys():
        raise ValueError("td1 and td2 must have the same keys.")
    else:
        return {k: td1[k] + td2[k] for k in td1.keys()}


def mul_td(td, c):
    """
    Calculate a scalar multiple of a (parameter) dictionaries.

    Parameters:
    td (dict): The dictionary.
    c (float): The scalar.

    Returns:
    float: The scalar multiple of the dictionary.
    """
    return {k: td[k] * c for k in td.keys()}


def compute_sos_update(xi, H_0_xi, chi, a, b):
    """
    Compute the update for the Stable Opponent Shaping (SOS) algorithm. See Algorithm 1 in the paper "Stable Opponent Shaping in Differentiable Games" by Letcher et al.

    Args:
        xi (dict): The vanilla individual updates.
        H_0_xi (dict): See the original paper for a definition of this term.
        chi (dict): See the original paper for a definition of this term.
        a (float): A scaling factor (between 0 and 1).
        b (float): A threshold value (between 0 and 1).

    Returns:
        dict: The update to be made to the parameters.

    """

    xi_0 = {}
    for k in xi:
        xi_0[k] = xi[k] - H_0_xi[k]
    denom = -dot_td(chi, xi_0)
    if denom >= 0.0:
        p_1 = 1.0
    else:
        p_1 = min(1.0, -a * dot_td(xi_0, xi_0) / denom)
    xi_norm_squared = dot_td(xi, xi)
    if xi_norm_squared < b * b:
        p_2 = xi_norm_squared
    else:
        p_2 = 1.0
    p = min(p_1, p_2)
    # Compute xi_p from xi_0
    for k in xi_0:
        xi_0[k] -= p * chi[k]

    return xi_0


def conjugate_gradient(
    f_loss: Tensor,
    l_loss: Tensor,
    f_params: Tuple[Tensor, ...],
    l_params: Tuple[Tensor, ...],
    num_iterations: int,
    lr: float,
) -> Tuple[Tensor, ...]:
    pass  # TODO


def neumann(
    f_loss: Tensor,
    l_loss: Tensor,
    f_params: Tuple[Tensor, ...],
    l_params: Tuple[Tensor, ...],
    num_iterations: int,
    lr: float,
) -> Tuple[Tensor, ...]:
    pass  # TODO


# Note that this is adapted from https://github.com/moskomule/hypergrad/blob/main/hypergrad/approx_hypergrad.py
def nystrom(
    f_loss: Tensor,
    l_loss: Tensor,
    f_params: Tuple[Tensor, ...],
    l_params: Tuple[Tensor, ...],
    rank: int,
    rho: float,
) -> Tuple[Tensor, ...]:
    """Nystrom method to approximate inverse Hessian vector product

    Args:
        f_loss: Follower objective
        l_loss: Leader objective
        f_params: Follower parameters
        l_params: Leader parameters
        rank: Rank of low-rank approximation
        rho: additive constant to improve numerical stability

    Returns: approximated implicit gradients
    """

    device = f_params[0].device

    # Compute gradients
    f_grads = torch.autograd.grad(
        f_loss, f_params, retain_graph=True, create_graph=True
    )
    indices = torch.randperm(sum([p.numel() for p in f_params]), device=device)[:rank]
    f_grads = torch.cat([g.reshape(-1) for g in f_grads])

    # Compute rank rows of the Hessian
    hessian_rows = [
        torch.autograd.grad(f_grads[i], f_params, retain_graph=True) for i in indices
    ]
    c = torch.cat(
        [
            torch.stack(tuple(r[i].flatten() for r in hessian_rows))
            for i in range(len(hessian_rows[0]))
        ],
        dim=1,
    )

    # Compute more gradients
    for p in f_params:
        p.grad.zero_()
    l_grads = torch.autograd.grad(l_loss, f_params, retain_graph=True)

    # Woodbury matrix identity
    m = c.take_along_dim(indices[None], dim=1)
    v = torch.cat([v.view(-1) for v in l_grads])
    x = 1 / rho * v - 1 / (rho**2) * c.t() @ torch.linalg.solve(
        0.1 * rho * torch.eye(len(indices), device=device) + m + 1 / rho * c @ c.t(),
        c @ v,
    )  # We use a small extra identity matrix in case the matrix to be inversed is singular

    # Reformat (this is vector_to_params from https://github.com/moskomule/hypergrad/blob/main/hypergrad/utils.py)
    pointer = 0
    ihvp = []
    for p in l_grads:
        size = p.numel()
        ihvp.append(x[pointer : pointer + size].view_as(p))
        pointer += size

    return ihvp


def ihvp(f_loss, l_loss, f_params, l_params, variant, num_iterations, rank, rho):
    """
    Compute the inverse Hessian-vector product (IHVP) based on the specified approximation method.

    Args:
        f_loss: The follower loss.
        l_loss: The leader loss.
        f_params: The parameters of the follower.
        l_params: The parameters of the leader.
        variant: The approximation method for computing the IHVP.
        num_iterations: The number of iterations for the conujugate gradient or Neumann approximation methods.
        rank: The rank parameter for the Nystrom approximation method.
        rho: The rho parameter for the Nystrom approximation method.

    Returns:
        The computed IHVP.
    """

    if variant == IhvpVariant.CONJ_GRAD:  # TODO not yet implemented
        ihvp = conjugate_gradient(
            f_loss,
            l_loss,
            tuple(f_params.values()),
            tuple(l_params.values()),
            num_iterations,
            lr=1.0,
        )
    elif variant == IhvpVariant.NEUMANN:  # TODO not yet implemented
        ihvp = neumann(
            f_loss,
            l_loss,
            tuple(f_params.values()),
            tuple(l_params.values()),
            num_iterations,
            lr=1.0,
        )
    elif variant == IhvpVariant.NYSTROM:
        ihvp = nystrom(
            f_loss,
            l_loss,
            tuple(f_params.values()),
            tuple(l_params.values()),
            rank,
            rho,
        )

    for param in f_params.values():
        param.grad.zero_()
    for param in l_params.values():
        param.grad.zero_()

    return dict(zip(f_params.keys(), ihvp))


def logit_entropy(logits: Float[Tensor, "... logits"]) -> Float[Tensor, "..."]:
    """
    Compute the entropy of a set of logits.

    Parameters
    ----------
    logits : Float[Tensor, "... logits"]
        The logits.

    Returns
    -------
    Float[Tensor, "..."]
        The entropy of the logits.
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1)


def logit_or_2(
    a: Float[Tensor, "... logits"], b: Float[Tensor, "... logits"]
) -> Float[Tensor, "... logits"]:
    """
    Computes the logit OR operation for two input tensors using the log-sum-exp trick.

    The logit OR operation is defined as:
        max_logit + log1p(exp(min_logit - max_logit))
    where max_logit is the element-wise maximum of the inputs,
    and min_logit is the element-wise minimum of the inputs.

    Args:
        a (torch.Tensor): The first input tensor.
        b (torch.Tensor): The second input tensor.

    Returns:
        torch.Tensor: The result of the logit OR operation applied element-wise to the input tensors.
    """

    max_logit = torch.maximum(a, b)
    min_logit = torch.minimum(a, b)
    return max_logit + torch.log1p(torch.exp(min_logit - max_logit))


def logit_or_n(logits: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """
    Compute the logit of the OR of n events given their logits.

    Args:
    logits (torch.Tensor): A tensor of logit values.
    dim (int, optional): The dimension along which to apply the OR operation.
                         If None, the operation is applied to all elements.

    Returns:
    torch.Tensor: The logit of the OR of input events along the specified dimension.
    """
    if dim is None:
        max_logit = torch.max(logits)
        return max_logit + torch.log1p(torch.sum(torch.exp(logits - max_logit)) - 1)
    else:
        max_logit = torch.max(logits, dim=dim, keepdim=True).values
        exp_sum = torch.sum(torch.exp(logits - max_logit), dim=dim, keepdim=False)
        return max_logit + torch.log1p(exp_sum - 1)


def mean_episode_reward(
    reward: Float[Tensor, "... step"], done_mask: Float[Tensor, "... step"]
) -> float:
    """Compute the mean total episode reward for a batch of concatenated episodes.

    The `done_mask` tensor specifies episode boundaries. The mean total reward per
    episode is computed by summing the rewards within each episode and dividing by the
    number of episodes.

    Note that the first episode is ignored, because it could be partly included in the
    previous batch.

    Parameters
    ----------
    reward : Float["... step"]
        The reward tensor. Multiple episodes are concatenated along the last dimension.
    done_mask : Float["... step"]
        A mask indicating the end of each episode.

    Returns
    -------
    mean_total_reward : float
        The mean total reward per episode.

    Examples
    --------
    >>> reward = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    >>> mask = torch.tensor([[True, True, False, True, False]])
    >>> mean_episode_reward(reward, mask)
    4.5
    """

    # Take the cumulative sum of the rewards throughout concatenated episodes
    episode_rewards = torch.cumsum(reward, dim=-1)

    # Select the cumsum rewards for the done steps
    episode_rewards = episode_rewards[done_mask]

    # A mask indicating the done steps which are not in the first episode
    not_first_episode_mask, _ = torch.cummax(done_mask, dim=-1)
    not_first_episode_mask = not_first_episode_mask.roll(shifts=1, dims=-1)
    not_first_episode_mask[..., 0] = False
    not_first_done_mask = not_first_episode_mask[done_mask]

    # Take the difference between consecutive done steps to get the total rewards per
    # episode, plus some junk corresponding to the first done step
    episode_rewards = torch.diff(
        episode_rewards, dim=-1, prepend=torch.tensor([0.0], device=reward.device)
    )

    # Remove the junk corresponding to the first done step
    episode_rewards = episode_rewards[not_first_done_mask]

    return episode_rewards.mean().item()


def minstd_generate_pseudo_random_sequence(
    seed: Int[Tensor, "..."], length: int
) -> Int[Tensor, "... length"]:
    """Generate a pseudo-random sequence of numbers using the MINSTD algorithm.

    The MINSTD algorithm is a simple linear congruential generator (LCG) that is defined
    by the following recurrence relation:

        x_{n+1} = (48271 * x_n) % 2147483647

    where x_0 is the seed value.

    Parameters
    ----------
    seed : Int[Tensor, ...]
        The seed value for the pseudo-random number generator. This is a tensor of
        arbitrary shape.
    length : int
        The length of the pseudo-random sequence to generate.

    Returns
    -------
    pseudo_random_sequence : Int[Tensor, "... length"]
        The pseudo-random sequence of numbers (x_1, \ldots, x_{length}) generated using
        the MINSTD algorithm. An extra dimension is added to the output tensor to
        represent the sequence length.
    """

    pseudo_random_sequence = torch.empty(
        (*seed.shape, length), dtype=seed.dtype, device=seed.device
    )

    for i in range(length):
        seed = (48271 * seed) % 2147483647
        pseudo_random_sequence[..., i] = seed

    return pseudo_random_sequence
