"""Utilities for useful mathematical operations."""

from typing import Tuple, Union, Sequence, Optional
import random

import torch
from torch import Tensor
from torch.nn import functional as F, Parameter

import numpy as np

from jaxtyping import Float, Int

from nip.parameters import IhvpVariantType


def set_seed(seed: int):
    """Set the seed in Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dict_dot_product(dict_1: dict[Tensor], dict_2: dict[Tensor]) -> float:
    """Calculate the dot product between two dictionaries of tensors.

    Parameters
    ----------
    dict_1 : dict[Tensor]
        The first dictionary.
    dict_2 : dict[Tensor]
        The second dictionary.

    Returns
    -------
    dot_product : float
        The dot product of the two dictionaries.

    Raises
    ------
    ValueError:
        If the dictionaries do not have the same keys.
    """

    if set(dict_1.keys()) != set(dict_2.keys()):
        raise ValueError(
            f"dict_1 and dict_2 must have the same keys. "
            f"Got {dict_1.keys()} and {dict_2.keys()}."
        )
    else:
        return sum((dict_1[key] * dict_2[key]).sum() for key in dict_1.keys())


def dict_sum(dict_1: dict[Tensor], dict_2: dict[Tensor]) -> dict[Tensor]:
    """Calculate the sum of two dictionaries of tensors, element-wise.

    Parameters
    ----------
    dict_1 : dict[Tensor]
        The first dictionary.
    dict_2 : dict[Tensor]
        The second dictionary.

    Returns
    -------
    sum_dict : dict[Tensor]
        The sum of the two dictionaries.

    Raises
    ------
    ValueError:
        If the dictionaries do not have the same keys.
    """

    if set(dict_1.keys()) != set(dict_2.keys()):
        raise ValueError(
            f"dict_1 and dict_2 must have the same keys. "
            f"Got {dict_1.keys()} and {dict_2.keys()}."
        )
    else:
        return {key: dict_1[key] + dict_2[key] for key in dict_1.keys()}


def dict_scalar_multiple(dictionary: dict[Tensor], scalar: float) -> dict[Tensor]:
    """Calculate a scalar multiple of a dictionary of tensors.

    Parameters
    ----------
    dictionary : dict[Tensor]
        The dictionary of tensors.
    scalar : float
        The scalar to multiply the dictionary by.

    Returns
    -------
    scaled_dict : dict[Tensor]
        The scaled dictionary.
    """

    return {key: dictionary[key] * scalar for key in dictionary.keys()}


def compute_sos_update(
    simultaneous_grad: dict[str, Tensor],
    hessian_grad_product: dict[str, Tensor],
    opponent_shaping: dict[str, Tensor],
    scaling_factor: float,
    threshold_factor: float,
) -> dict[str, Tensor]:
    r"""Compute the update for the Stable Opponent Shaping (SOS) algorithm.

    See Algorithm 1 in :cite:t:`Letcher2019`. Effectively, this update interpolates between the LOLA update (:cite:t:`Foerster2018`) and the LookAhead update (:cite:t:`Zhang2010`) by computing a coefficient p between 0 and 1 where p = 0 corresponds to LookAhead and p = 1 corresponds to LOLA.

    Parameters
    ----------
    simultaneous_grad : dict[str, Tensor]
        The vanilla individual updates. Named $\xi$ in Letcher et al.
    hessian_grad_product : dict[str, Tensor]
        The product of the anti-diagonal of the Hessian matrix with the vector $\xi$.
    opponent_shaping : dict[str, Tensor]
        The opponent shaping term for each parameter. Named $\chi$ in Letcher et al.
    scaling_factor : float
        A scaling factor (between 0 and 1). Named $a$ in Letcher et al.
    threshold_factor : float
        A threshold value (between 0 and 1). Named $b$ in Letcher et al.

    Returns
    -------
    update : dict[str, Tensor]
        The update to be made to the parameters.
    """

    # First we compute $\xi_0 = (I - \alpha H_o)\xi$, where $\alpha$ is the learning rate (which has already been applied) and $H_o \xi$ is the product of the anti-diagonal of the Hessian matrix with the vector $\xi$.
    xi_0 = {}
    for k in simultaneous_grad:
        xi_0[k] = simultaneous_grad[k] - hessian_grad_product[k]

    # Next, we compute the coefficient p. First, we ensure that the eventual gradient update "points in the same direction" as LookAhead
    denom = -dict_dot_product(opponent_shaping, xi_0)
    if denom >= 0.0:
        p_1 = 1.0
    else:
        p_1 = min(1.0, -scaling_factor * dict_dot_product(xi_0, xi_0) / denom)

    # After that, we ensure local convergence by scaling up if the magnitude of the simultaneous gradient is small
    xi_norm_squared = dict_dot_product(simultaneous_grad, simultaneous_grad)
    if xi_norm_squared < threshold_factor * threshold_factor:
        p_2 = xi_norm_squared
    else:
        p_2 = 1.0

    # The actual coefficient p is the minimum of the two computed coefficients
    p = min(p_1, p_2)

    # Finally, we compute the gradient update by applying the p coefficient to the opponent shaping term
    for k in xi_0:
        xi_0[k] -= p * opponent_shaping[k]
    return xi_0


def compute_conjugate_gradient_ihvp(
    follower_loss: Tensor,
    leader_loss: Tensor,
    follower_params: dict[str, Tensor],
    leader_params: dict[str, Tensor],
    num_iterations: int,
    lr: float,
) -> dict[str, Tensor]:
    """Approximate the inverse Hessian vector product with conjugate gradient."""
    raise NotImplementedError("Conjugate gradient is not yet implemented.")  # TODO


def compute_neumann_ihvp(
    follower_loss: Tensor,
    leader_loss: Tensor,
    follower_params: dict[str, Tensor],
    leader_params: dict[str, Tensor],
    num_iterations: int,
    lr: float,
) -> dict[str, Tensor]:
    """Approximate the inverse Hessian vector product with the Neumann method."""
    raise NotImplementedError("Neumann approximation is not yet implemented.")  # TODO


def compute_nystrom_ihvp(
    follower_loss: Tensor,
    leader_loss: Tensor,
    follower_params: dict[str, Tensor | Parameter],
    leader_params: dict[str, Tensor | Parameter],
    rank: int = 5,
    rho: float = 0.1,
    retain_graph: bool = True,
    generator: Optional[torch.Generator] = None,
) -> dict[str, Tensor]:
    r"""Approximate the inverse Hessian vector product with the Nystrom method.

    This function approximates the inverse Hessian of the follower's loss with respect
    to the its parameters and multiplies it by the gradients of the leader's loss with
    respect to the follower's parameters. See :cite:t:`Hataya2023` for more details.

    The function computes:

    .. math::

        (H_k + \rho I) \frac{\partial g}{\partial \theta}

    where:

    - $f$ is the follower's loss
    - $g$ is the leader's loss
    - $\theta$ are the follower's parameters
    - $\phi$ are the leader's parameters
    - $H_k$ is the $k$-rank approximation of the Hessian of $f$ with respect to $\theta$

    Parameters
    ----------
    follower_loss : Tensor
        Follower objective
    leader_loss : Tensor
        Leader objective
    follower_params : dict[str, Tensor | Parameter]
        The parameters of the follower agent
    leader_params : dict[str, Tensor | Parameter]
        The parameters of the leader agent
    rank: int, default=5
        Rank of low-rank approximation
    rho : float, default=0.1
        Additive constant to improve numerical stability
    retain_graph : bool, default=True
        Whether to retain the computation graph for use in computing higher-order
        derivatives.
    generator : torch.Generator, optional
        The PyTorch random number generator, used for sampling the rank columns of the
        Hessian matrix.

    Returns
    -------
    ihvp : dict[str, Tensor]
        A dictionary where the keys are the follower parameter names and the values are
        the inverse Hessian-vector product, i.e. the inverse Hessian multiplied by the
        leader gradients

    Notes
    -----
    Adapted from
    https://github.com/moskomule/hypergrad/blob/main/hypergrad/approx_hypergrad.py
    """

    follower_param_values = list(follower_params.values())
    device = follower_param_values[0].device

    # Compute gradients
    follower_grads = torch.autograd.grad(
        follower_loss, follower_param_values, retain_graph=True, create_graph=True
    )
    follower_grads = torch.cat([grad.reshape(-1) for grad in follower_grads])

    # Randomly sample ``rank`` parameters to compute the rank approximation
    num_follower_params = sum(param.numel() for param in follower_param_values)
    follower_param_indices = torch.randperm(
        num_follower_params, device=device, generator=generator
    )[:rank]

    # Compute ``rank`` rows of the Hessian
    hessian_rows = []
    for i in follower_param_indices:
        hessian_rows.append(
            torch.autograd.grad(
                follower_grads[i], follower_param_values, retain_graph=True
            )
        )
    partial_hessian = torch.cat(
        [
            torch.stack([row[i].flatten() for row in hessian_rows])
            for i in range(len(hessian_rows[0]))
        ],
        dim=1,
    )

    # Compute the gradients of the follower parameters with respect to the leader loss
    leader_grads = torch.autograd.grad(
        leader_loss, follower_param_values, retain_graph=retain_graph
    )

    # Select the columns of the partial Hessian matrix according to the parameter
    # indices to yield a square matrix
    square_hessian = partial_hessian.take_along_dim(follower_param_indices[None], dim=1)

    # The vector of leader gradients, which the inverse Hessian will be multiplied by
    leader_grad_vector = torch.cat([grad.view(-1) for grad in leader_grads])

    # Use the Woodbury matrix identity to compute the inverse Hessian multiplied by the
    # leader gradients. We use a small extra identity matrix in case the matrix to be
    # inverted is singular
    to_solve = (
        0.1 * rho * torch.eye(len(follower_param_indices), device=device)
        + square_hessian
        + 1 / rho * partial_hessian @ partial_hessian.t()
    )
    solved_part = torch.linalg.solve(
        to_solve,
        partial_hessian @ leader_grad_vector,
    )
    ihvp_cat = 1 / rho * leader_grad_vector
    ihvp_cat = ihvp_cat - 1 / (rho**2) * partial_hessian.t() @ solved_part

    # Convert the concatenated vector into a dictionary of tensors corresponding to the
    # follower parameters
    pointer = 0
    ihvp = {}
    for param_name, param in follower_params.items():
        size = param.numel()
        ihvp[param_name] = ihvp_cat[pointer : pointer + size].view_as(param)
        pointer += size

    return ihvp


def inverse_hessian_vector_product(
    follower_loss: Tensor,
    leader_loss: Tensor,
    follower_params: dict[str, Tensor | Parameter],
    leader_params: dict[str, Tensor | Parameter],
    variant: IhvpVariantType,
    num_iterations: int,
    rank: int = 5,
    rho: float = 0.1,
    retain_graph: bool = True,
    generator: Optional[torch.Generator] = None,
) -> dict[str, Tensor]:
    """Compute the inverse Hessian-vector product using specified approximation method.

    Note that this method zeros the gradients of the leader and follower parameters.

    Parameters
    ----------
    follower_loss : Tensor
        The follower loss.
    leader_loss : Tensor
        The leader loss.
    follower_params : dict[str, Tensor | Parameter]
        The parameters of the follower.
    leader_params : dict[str, Tensor | Parameter]
        The parameters of the leader.
    variant : IhvpVariantType
        The approximation method for computing the IHVP.
    num_iterations : int
        The number of iterations for the conujugate gradient or Neumann approximation
        methods.
    rank : int, default=5
        The rank parameter for the Nystrom approximation method.
    rho : float, default=0.1
        The rho parameter for the Nystrom approximation method.
    retain_graph : bool, default=True
        Whether to retain the computation graph for use in computing higher-order
        derivatives.
    generator : torch.Generator, optional
        The PyTorch random number generator.

    Returns
    -------
    ihvp : dict[str, Tensor]
        The computed IHVP.
    """

    if variant == "conj_grad":
        ihvp = compute_conjugate_gradient_ihvp(
            follower_loss,
            leader_loss,
            follower_params,
            leader_params,
            num_iterations,
            lr=1.0,
        )
    elif variant == "neumann":
        ihvp = compute_neumann_ihvp(
            follower_loss,
            leader_loss,
            follower_params,
            leader_params,
            num_iterations,
            lr=1.0,
        )
    elif variant == "nystrom":
        ihvp = compute_nystrom_ihvp(
            follower_loss,
            leader_loss,
            follower_params,
            leader_params,
            rank,
            rho,
            retain_graph=retain_graph,
            generator=generator,
        )
    else:
        raise ValueError(f"Invalid IHVP variant: {variant!r}")

    # Zero the gradients of the parameters
    _zero_grad(follower_params)
    _zero_grad(leader_params)

    return ihvp


def logit_entropy(logits: Float[Tensor, "... logits"]) -> Float[Tensor, "..."]:
    """Compute the entropy of a set of logits.

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


def logit_or_dual(
    a: Float[Tensor, "... logits"], b: Float[Tensor, "... logits"]
) -> Float[Tensor, "... logits"]:
    r"""Compute the logit OR operation for two input tensors with the log-sum-exp trick.

    The logit OR operation is defined as:

    .. math::

        \max(a, b) + \log(1 + \exp(\min(a, b) - \max(a, b)))

    where $\max(a, b)$ is the element-wise maximum of the inputs, and $\min(a, b)$ is
    the element-wise minimum of the inputs.

    Parameters
    ----------
    a : Float[Tensor, "... logits"]
        The first input tensor.
    b : Float[Tensor, "... logits"]
        The second input tensor.

    Returns
    -------
    torch.Tensor
        The result of the logit OR operation applied element-wise to the input tensors.
    """

    max_logit = torch.maximum(a, b)
    min_logit = torch.minimum(a, b)
    return max_logit + torch.log1p(torch.exp(min_logit - max_logit))


def logit_or(logits: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    r"""Compute the logit of the OR of n events given their logits.

    The logit OR operation is defined as:

    .. math::

        \max_d(\ell) + \log(1 + \exp(\min_d(\ell) - \max_d(\ell)))

    where $\max(\ell)$ is the element-wise maximum of the logits along the specified
    dimension, and $\min(\ell)$ is the element-wise minimum of the logits along the
    specified dimension.

    Parameters
    ----------
    logits : torch.Tensor
        A tensor of logit values.
    dim : int, optional
        The dimension along which to apply the OR operation. If None, the operation is
        applied to all elements.

    Returns
    -------
    torch.Tensor:
        The logit of the OR of input events along the specified dimension.
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

    The ``done_mask`` tensor specifies episode boundaries. The mean total reward per
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


def aggregate_mean_grouped_by_class(
    values: Float[Tensor | np.ndarray, "batch"],
    classes: Int[Tensor | np.ndarray, "batch"],
    num_classes: Optional[int] = None,
) -> Float[Tensor | np.ndarray, "class"]:
    """Compute the mean of values grouped by class.

    ``values`` is a 1D tensor of values to aggregate, and ``classes`` is a 1D tensor of
    class labels for each value. The function computes the mean of the values for each
    class.

    It returns a 1D tensor of mean values for each class. If any class has no values
    associated with it, the mean for that class is set to NaN.

    Parameters
    ----------
    values : Float[Tensor | numpy.ndarray, "batch"]
        The values to aggregate.
    classes : Int[Tensor | numpy.ndarray, "batch"]
        The class labels for each value.
    num_classes : int, optional
        The number of classes. If not provided, it is inferred from the class labels.

    Returns
    -------
    mean_values : Float[Tensor | numpy.ndarray, "class"]
        The mean of the values for each class. If either of the arguments is a numpy
        array, this will be one too.
    """

    if values.ndim != 1:
        raise ValueError(f"`values` must be a 1D tensor, but got shape {values.shape}")
    if classes.ndim != 1:
        raise ValueError(
            f"`classes` must be a 1D tensor, but got shape {classes.shape}"
        )
    if values.shape != classes.shape:
        raise ValueError(
            f"`values` and `classes` must have the same shape, but got {values.shape} "
            f"and {classes.shape}"
        )

    was_numpy = False
    if isinstance(values, np.ndarray):
        values = torch.from_numpy(values)
        was_numpy = True
    if isinstance(classes, np.ndarray):
        classes = torch.from_numpy(classes)
        was_numpy = True

    if num_classes is None:
        num_classes = classes.max().item() + 1

    class_counts = torch.bincount(classes, minlength=num_classes)
    sum_per_class = torch.bincount(classes, values, minlength=num_classes)

    mean_values = sum_per_class / class_counts.float()

    if was_numpy:
        mean_values = mean_values.cpu().detach().numpy()

    return mean_values


def minstd_generate_pseudo_random_sequence(
    seed: Int[Tensor, "..."], length: int
) -> Int[Tensor, "... length"]:
    r"""Generate a pseudo-random sequence of numbers using the MINSTD algorithm.

    The MINSTD algorithm is a simple linear congruential generator (LCG) that is defined
    by the following recurrence relation:

    .. math::

        x_{n+1} = (48271 * x_n) % 2147483647

    where $x_0$ is the seed value.

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
        The pseudo-random sequence of numbers $(x_1, \ldots, x_{\text{length}})$
        generated using the MINSTD algorithm. An extra dimension is added to the output
        tensor to represent the sequence length.
    """

    pseudo_random_sequence = torch.empty(
        (*seed.shape, length), dtype=seed.dtype, device=seed.device
    )

    for i in range(length):
        seed = (48271 * seed) % 2147483647
        pseudo_random_sequence[..., i] = seed

    return pseudo_random_sequence


def is_broadcastable(shape_1: tuple, shape_2: tuple) -> bool:
    """Check if two shapes are broadcastable.

    Two shapes are broadcastable if when they are aligned from the right, corresponding
    dimensions are either equal or one of them is 1.

    Parameters
    ----------
    shape_1 : tuple
        The shape of the first array.
    shape_2 : tuple
        The shape of the second array.

    Returns
    -------
    is_broadcastable : bool
        True if the shapes are broadcastable, False otherwise.
    """

    # Adapted from https://stackoverflow.com/a/24769712
    for a, b in zip(shape_1[::-1], shape_2[::-1]):
        if a != 1 and b != 1 and a != b:
            return False
    return True


def mean_for_unique_keys(
    data: np.ndarray, key: np.ndarray, axis: int = 0
) -> np.ndarray:
    """Compute the mean of values grouped by unique keys.

    The two input arrays ``data`` and ``key`` should have the same shape. It is assumed
    that when two elements of ``key`` are equal, the corresponding elements of ``data``
    should be equal. The function selects the unique keys from ``key`` and computes the
    mean of the corresponding values in ``data``.

    Parameters
    ----------
    data : numpy.ndarray
        The values to aggregate.
    key : numpy.ndarray
        The keys for each value. Must be broadcastable with ``data``.
    axis : int, default=0
        The axis along which to compute the mean.

    Returns
    -------
    mean_values : numpy.ndarray
        The mean of the values for each unique key.
    """

    if not is_broadcastable(data.shape, key.shape):
        raise ValueError(
            f"`data` and `key` must be broadcastable, but got shapes {data.shape} "
            f"and {key.shape}"
        )

    def get_unique_mask(array: np.ndarray) -> np.ndarray:
        """Create a mask for unique elements in a 1D array."""
        mask = np.zeros_like(array, dtype=bool)
        mask[np.unique(array, return_index=True)[1]] = True
        return mask

    keys_unique_mask = np.apply_along_axis(get_unique_mask, axis, key)

    return np.mean(data, axis=axis, where=keys_unique_mask)


def _zero_grad(params: dict[str, Tensor | Parameter]):
    """Zero the gradients of the parameters in a dictionary.

    Parameters
    ----------
    params : dict[str, Tensor | Parameter]
        The dictionary of parameters.
    """
    for param in params.values():
        if param.grad is not None:
            param.grad.zero_()
