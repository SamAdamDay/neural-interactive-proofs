"""Utilities for useful mathematical operations."""


from pvg.parameters import IhvpVariant
from typing import Tuple
import torch
from torch import Tensor


def dot(td1, td2):
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
