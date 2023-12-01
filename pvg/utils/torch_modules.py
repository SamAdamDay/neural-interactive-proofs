from typing import Callable, Optional, Iterable

import torch
from torch import Tensor
import torch.nn as nn

from tensordict import TensorDictBase, TensorDict

import einops

from jaxtyping import Float, Bool


class GlobalMaxPool(nn.Module):
    """Global max pooling layer over a dimension.

    Parameters
    ----------
    dim : int, default=-1
        The dimension to pool over.
    keepdim : bool, default=False
        Whether to keep the pooled dimension or not.
    """

    def __init__(self, dim: int = -1, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: Tensor) -> Tensor:
        return x.max(dim=self.dim, keepdim=self.keepdim)[0]


class CatGraphPairDim(nn.Module):
    """Concatenate the two node sets for each graph pair.

    Parameters
    ----------
    cat_dim : int
        The dimension to concatenate over (i.e. the node dimension).
    pair_dim : int, default=0
        The graph pair dimension.
    """

    def __init__(self, cat_dim: int, pair_dim: int = 0):
        super().__init__()
        self.cat_dim = cat_dim
        self.pair_dim = pair_dim

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat(
            [x.select(self.pair_dim, 0), x.select(self.pair_dim, 1)],
            dim=self.cat_dim - 1,
        )


class PairedGaussianNoise(nn.Module):
    """Add Gaussian noise copied across the graph pair dimension.

    Parameters
    ----------
    sigma : float
        The relative standard deviation of the Gaussian noise. This will be multiplied
        by the magnitude of the input to get the standard deviation for the noise.
    pair_dim : int, default=0
        The graph pair dimension.
    train_sigma : bool, default=False
        Whether the `sigma` parameter should be trained or not.

    Notes
    -----
    Adapted from
    https://discuss.pytorch.org/t/where-is-the-noise-layer-in-pytorch/2887/4
    """

    def __init__(
        self,
        sigma: float,
        pair_dim: int = 0,
        train_sigma: bool = False,
        dtype=torch.float32,
    ):
        super().__init__()
        if train_sigma:
            self.sigma = nn.Parameter(torch.tensor(sigma))
        else:
            self.sigma = sigma
        self.train_sigma = train_sigma
        self.pair_dim = pair_dim
        self._noise = torch.tensor(0, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.sigma != 0:
            # If we're not training sigma, we need to detach `x` when computing the
            # scale so that the gradient doesn't propagate to sigma
            if self.train_sigma:
                scale = self.sigma * x.detach()
            else:
                scale = self.sigma * x

            # Sample the noise once and repeat it across the graph pair dimension
            size = list(x.size())
            size[self.pair_dim] = 1
            sampled_noise = self._noise.repeat(*size).normal_() * scale

            # Add the noise to the input
            x = x + sampled_noise
        return x

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self._noise = self._noise.to(*args, **kwargs)
        return self

    def __repr__(self):
        return (
            f"PairedGaussianNoise(sigma={self.sigma}, pair_dim={self.pair_dim}, "
            f"train_sigma={self.train_sigma})"
        )


class PairInvariantizer(nn.Module):
    """Transform the input to be invariant to the order of the graphs in a pair.

    Works by taking the mean of the pair and half the absolute difference between the
    graphs.

    Parameters
    ----------
    pair_dim : int, default=0
        The graph pair dimension.
    """

    def __init__(self, pair_dim: int = 0):
        super().__init__()
        self.pair_dim = pair_dim

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=self.pair_dim)
        abs_diff = 0.5 * torch.abs(
            x.select(self.pair_dim, 0) - x.select(self.pair_dim, 1)
        )
        return torch.stack((mean, abs_diff), dim=0)


class GIN(nn.Module):
    """A graph isomorphism network (GIN) layer.

    This is a message-passing layer that aggregates the features of the neighbours as
    follows:
    $$
        x_i' = MLP((1 + \epsilon) x_i + \sum_{j \in \mathcal{N}(i)} x_j)
    $$
    where $x_i$ is the feature vector of node $i$, $\mathcal{N}(i)$ is the set of
    neighbours of node $i$, and $\epsilon$ is a (possibly learnable) parameter.

    From the paper "How Powerful are Graph Neural Networks?" by Keyulu Xu et al.
    (https://arxiv.org/abs/1810.00826).

    The difference between this implementation and the one in PyTorch Geometric is that
    this one takes as input a TensorDict with dense representations of the graphs and
    features.

    Parameters
    ----------
    mlp
        The MLP to apply to the aggregated features.
    eps
        The initial value of $\epsilon$.
    train_eps
        Whether to train $\epsilon$ or keep it fixed.

    Shapes
    ------
    Takes as input a TensorDict with the following keys:
    * `x` - Float["... max_nodes feature"] - The features of the nodes.
    * `adjacency` - Float["... max_nodes max_nodes"] - The adjacency matrix of the
      graph.
    * `node_mask` - Bool["... max_nodes"] - A mask indicating which nodes exist
    """

    def __init__(self, mlp: nn.Module, eps: float = 0.0, train_eps: bool = False):
        super().__init__()
        self.mlp = mlp
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)

    def forward(
        self,
        tensordict: TensorDictBase,
        key_map: Optional[dict[str, str] | Callable[[str], str]] = None,
    ) -> torch.Tensor:
        # Map the keys of the input to the correct ones
        def _key_map(key: str) -> str:
            if key_map is None:
                return key
            elif isinstance(key_map, dict):
                return key_map[key]
            else:
                return key_map(key)

        # Extract the features, adjacency matrix and node mask from the input
        x: Float[Tensor, "... max_nodes feature"] = tensordict[_key_map("x")]
        adjacency: Float[Tensor, "... max_nodes max_nodes"] = tensordict[
            _key_map("adjacency")
        ]
        if _key_map("node_mask") in tensordict.keys():
            node_mask: Bool[Tensor, "... max_nodes"] = tensordict[_key_map("node_mask")]
        else:
            node_mask = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)

        # Aggregate the features of the neighbours using summation
        x_expanded = einops.rearrange(
            x, "... max_nodes feature -> ... max_nodes 1 feature"
        )
        adjacency = einops.rearrange(
            adjacency,
            "... max_nodes_a max_nodes_b -> ... max_nodes_a max_nodes_b 1",
        )
        # (..., max_nodes, feature)
        x_aggregated = einops.reduce(
            x_expanded * adjacency,
            "... max_nodes_a max_nodes_b feature -> ... max_nodes_b feature",
            "sum",
        )

        # Apply the MLP to the aggregated features plus a contribution from the node
        # itself. We do this only according to the node mask, putting zeros elsewhere.
        new_x_flat = self.mlp((1 + self.eps) * x[node_mask] + x_aggregated[node_mask])
        new_x = torch.zeros(
            (*x.shape[:-1], new_x_flat.shape[-1]), dtype=x.dtype, device=x.device
        )
        new_x[node_mask] = new_x_flat

        out = TensorDict(tensordict)
        out[_key_map("x")] = new_x

        return out


class TensorDictize(nn.Module):
    """Convert a module to one which works on a key of a TensorDict.

    Parameters
    ----------
    module : nn.Module
        The module to apply to TensorDictize.
    key : str
        The key of the TensorDict to apply the module to.
    """

    def __init__(self, module: nn.Module, key: str):
        super().__init__()
        self.module = module
        self.key = key

    def forward(
        self,
        tensordict: TensorDictBase,
        key_map: Optional[dict[str, str] | Callable[[str], str]],
    ) -> TensorDictBase:
        # Map the keys of the input to the correct ones
        def _key_map(key: str) -> str:
            if key_map is None:
                return key
            elif isinstance(key_map, dict):
                return key_map[key]
            else:
                return key_map(key)

        out = TensorDict(tensordict)
        out[_key_map(self.key)] = self.module(tensordict[_key_map(self.key)])
        return out


class SequentialKwargs(nn.Sequential):
    """A sequential module which passes keyword arguments during forward pass."""

    def forward(self, input, **kwargs):
        for module in self:
            input = module(input, **kwargs)
        return input


class Print(nn.Module):
    """Print the shape or value of a tensor.

    Parameters
    ----------
    name : str, default=None
        The name of the tensor.
    print_value : bool, default=False
        Whether to print the value or the shape.
    transform : Callable, default=None
        A function to apply to the tensor before printing.
    """

    def __init__(
        self,
        name: str = None,
        print_value: bool = False,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.name = name
        self.print_value = print_value
        self.transform = transform

    def forward(self, x: Tensor) -> Tensor:
        if self.name is not None:
            print(f"{self.name}:")
        if self.print_value:
            if self.transform is not None:
                x = self.transform(x)
            print(x)
        else:
            print(x.shape)
        return x
