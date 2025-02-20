"""Tests for the maths utility module."""

from itertools import product

import torch
from torch import Tensor
from torch.nn import Linear, Parameter

from nip.utils.maths import compute_nystrom_ihvp


def _single_nystrom_ihvp_test(
    dim: int, rank: int, rho: float = 0.1, seed: int = 0
) -> dict[str, Tensor]:
    """Do one test of the Nystrom IHVP computation.

    Creates simple agents and data, runs the Nystrom IHVP implementation, and
    analytically computes the expected IHVP.

    We use simple linear layers with diagonal weight matrices, and no biases. Under this
    setup the Hessian of the follower's loss with respect its parameters is a block
    diagonal matrix, where each block is the outer product of the input data with
    itself. The gradient of the leader's loss with respect to the follower's parameters
    can also be computed analytically.

    Parameters
    ----------
    dim : int
        The dimensionality of the input and output.
    rank : int
        The rank of the Nystrom approximation.
    rho : float, default=0.1
        The additive constant to improve numerical stability, used in the Nystrom
        approximation.
    seed : int, default=0
        The seed for the random number generator.

    Returns
    -------
    results : dict[str, Tensor]
        A dictionary containing the expected IHVP, the approximated IHVP, the Hessian,
        the gradient of the leader's loss with respect to the follower's parameters, the
        matrix to invert, and the inverted matrix.
    """

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    # Create simple linear layers as leader and follower, where the weight matrices are
    # diagonal and there are no biases
    leader = Linear(dim, dim, bias=False)
    follower = Linear(dim, dim, bias=False)
    leader_diag = torch.arange(1, dim + 1, dtype=torch.float32)
    follower_diag = 2 * torch.arange(1, dim + 1, dtype=torch.float32)
    leader.weight = Parameter(torch.diagflat(leader_diag))
    follower.weight = Parameter(torch.diagflat(follower_diag))

    # Create input data
    x = torch.arange(1, dim + 1, dtype=torch.float32)

    # Run the follower on the data, and the leader on the follower's output
    follower_out = follower(x)
    follower_loss = torch.sum(follower_out**2)
    leader_loss = torch.sum(leader(follower_out) ** 2)

    # Compute the ihvp approximation
    ihvp_approx = compute_nystrom_ihvp(
        follower_loss=follower_loss,
        leader_loss=leader_loss,
        leader_params=dict(leader.named_parameters()),
        follower_params=dict(follower.named_parameters()),
        rank=rank,
        rho=rho,
        retain_graph=False,
        generator=generator,
    )

    # Analytically compute the Hessian
    input_block = 2 * torch.outer(x, x)
    hessian = torch.block_diag(*[input_block for _ in range(dim)])

    # Analytically compute the gradient of the leader loss with respect to the
    # follower's parameters
    leader_grad = (input_block * (leader_diag * leader_diag * follower_diag)).T
    leader_grad = leader_grad.reshape(dim * dim)

    # Compute the expected IHVP
    to_invert = hessian + rho * torch.eye(dim**2)
    inverted = torch.inverse(to_invert)
    ihvp_expected = inverted @ leader_grad
    ihvp_expected = ihvp_expected.reshape(dim, dim)

    return dict(
        ihvp_expected=ihvp_expected,
        ihvp_approx=ihvp_approx,
        hessian=hessian,
        leader_grad=leader_grad,
        to_invert=to_invert,
        inverted=inverted,
    )


def test_nystrom_ihvp():
    """Test the implementation of the Nystrom IHVP computation.

    Constructs a basic scenario and checks the Nystrom IHVP implementation against
    analytically computed values over a range of dimensions and for full rank and
    under-rank approximations. An addition, each test is repeated several times with a
    different seed.
    """

    for seed, dim in product(range(5), range(2, 5)):
        rank_range = range(dim * dim - dim + 1, dim * dim + 1)
        for rank_num, rank in enumerate(rank_range):
            results = _single_nystrom_ihvp_test(dim, rank, seed=seed)
            atol = 4 ** (-rank_num)
            assert torch.allclose(
                results["ihvp_expected"], results["ihvp_approx"]["weight"], atol=atol
            ), f"Nystrom IHVP not close to expected for dim={dim}, rank={rank}."
