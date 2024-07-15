"""Utilities for useful mathematical operations."""

from typing import Tuple, Literal, Optional
from itertools import product
import random

import torch
from torch import Tensor
from torch.nn import functional as F

import numpy as np
from numpy.typing import NDArray

from einops import rearrange

from sklearn.utils.extmath import randomized_svd

from jaxtyping import Float

from pvg.parameters import IhvpVariant
from pvg.utils.types import TorchDevice


def manual_seed(seed: int):
    """Set the random seed for PyTorch, NumPy, and Python's random module.

    Parameters
    ----------
    seed : int
        The random seed to use.

    Returns
    -------
    rng = torch.Generator
        The PyTorch random number generator.
    """
    np.random.seed(seed)
    random.seed(seed)
    return torch.manual_seed(seed)


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


def product_range(sizes: tuple[int]):
    """Yeilds an iterator over the Cartesian product of ranges of the specified sizes.

    Yeilds
    ------
    tuple[int]
        A tuple of indices into the Cartesian product.
    """

    for indices in product(*(range(size) for size in sizes)):
        yield indices


def pca_project(
    A: Float[Tensor, "... n m"],
    num_components: int,
    centre: bool = True,
    method: Literal["torch_svd", "sklearn_randomized_svd"] = "torch_svd",
    concat_batch_dims: bool = False,
    pca_sample_prop: float = 1.0,
    device: Optional[TorchDevice] = None,
    tqdm_func: Optional[callable] = None,
    tqdm_desc: str = "Computing SVD",
) -> Float[Tensor, "... n num_components"]:
    """Project a matrix onto its principal components

    Parameters
    ----------
    A : Float[Tensor, "... n m"]
        The input matrix.
    num_components : int
        The number of principal components to compute.
    centre : bool, default=True
        Whether to centre the data.
    method : Literal["torch_svd", "scipy_randomized_svd"], default="torch_svd"
        The method to use for computing the principal components.
            - "torch_svd": Use `torch.linalg.svd` from PyTorch.
            - "sklearn_randomized_svd": Use `sklearn.utils.extmath.randomized_svd`,
              first converting the input matrix to a NumPy array. For this method we
              loop through the batch dimensions with a *Python loop*, because
              `randomized_svd` does not support batch dimensions. :(
    concat_batch_dims : bool, default=False
        Whether to rearrange `A` so that the batch dimensions are concatenated into the
        feature dim. This means that instead of doing independent PCAs on each batch
        element, we do a single PCA on all the batch elements together, which
        concatenated features.
    pca_sample_prop : float, default=1.0
        The proportion of samples to use for the PCA. This is useful for speeding up the
        computation by using a subset of the samples.
    device : Optional[TorchDevice], default=None
        The device to use for the computations. Only applicable to PyTorch methods.
    tqdm_func : callable, optional
        A function to use for displaying progress bars. Not applicable to all methods.
    tqdm_desc : str, default="Computing SVD"
        The description to use for the progress bar.

    Returns
    -------
    projected_matrix : Float[Tensor, "... n num_components"]
        The matrix projected onto its principal components.
    """

    if centre:
        A = A - A.mean(dim=-2, keepdim=True)

    original_device = A.device
    original_batch_shape = A.shape[:-2]

    if concat_batch_dims:
        A = rearrange(A, "... n m -> n (... m)")

    if pca_sample_prop != 1.0:
        select_indices = torch.randperm(A.shape[-2], device=A.device)[
            : int(pca_sample_prop * A.shape[-2])
        ]
        A_full = A.clone()
        A = A_full[..., select_indices, :]
    else:
        A_full = A

    if method == "torch_svd":

        if device is not None:
            A = A.to(device)
            A_full = A_full.to(device)

        _, _, Vh = torch.linalg.svd(A, full_matrices=False)
        projected_matrix = torch.matmul(A_full, Vh[..., :num_components, :].mT)

        if device is not None:
            projected_matrix = projected_matrix.to(original_device)

    elif method == "sklearn_randomized_svd":

        A_np: NDArray = A.detach().cpu().numpy()

        # Make sure the matrix has at least 1 batch dimension
        if len(A_np.shape) == 2:
            A_np = np.expand_dims(A_np, 0)

        batch_shape = A_np.shape[:-2]

        if tqdm_func is not None:
            pbar = tqdm_func(total=np.prod(batch_shape), desc=tqdm_desc)

        # Loop through the batch dimensions and compute the SVD for each batch
        Vh = np.empty(batch_shape + (num_components, A_np.shape[-1]), dtype=A_np.dtype)
        for indices in product(*(range(size) for size in batch_shape)):
            _, _, Vh[indices] = randomized_svd(
                A_np[indices], n_components=num_components
            )
            if tqdm_func is not None:
                pbar.update(1)

        if tqdm_func is not None:
            pbar.close()

        Vh = torch.tensor(Vh, device=A.device)

        projected_matrix = torch.matmul(A_full, Vh.mT)

    else:
        raise ValueError(f"Unknown method: {method!r}")

    if concat_batch_dims:
        dim_str = " ".join(f"dim{i}" for i in range(len(original_batch_shape)))
        projected_matrix = rearrange(
            projected_matrix,
            f"n ({dim_str} m) -> {dim_str} n m",
            **{f"dim_{i}": size for i, size in enumerate(original_batch_shape)},
        )

    if pca_sample_prop != 1.0:
        del A_full

    return projected_matrix
