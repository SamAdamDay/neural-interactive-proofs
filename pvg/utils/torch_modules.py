"""Handy PyTorch modules."""

from typing import Callable, Optional, Iterable

import torch
from torch import Tensor
import torch.nn as nn

from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey

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
        return torch.stack((mean, abs_diff), dim=self.pair_dim)


class GIN(TensorDictModuleBase):
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
    mlp : nn.Module
        The MLP to apply to the aggregated features.
    eps : float, default=0.0
        The initial value of $\epsilon$.
    train_eps : bool, default=False
        Whether to train $\epsilon$ or keep it fixed.
    feature_in_key : NestedKey, default="x"
        The key of the input features in the input TensorDict.
    feature_out_key : NestedKey, default="x"
        The key of the output features in the output TensorDict.
    vmap_compatible : bool, default=False
        Whether the module is compatible with `vmap` or not. If `True`, the node mask
        is only applied after the MLP, which is less efficient but allows for the use
        of `vmap`.

    Shapes
    ------
    Takes as input a TensorDict with the following keys:
    * `x` - Float["... max_nodes feature"] - The features of the nodes.
    * `adjacency` - Float["... max_nodes max_nodes"] - The adjacency matrix of the
      graph.
    * `node_mask` - Bool["... max_nodes"] - A mask indicating which nodes exist
    """

    @property
    def in_keys(self) -> Iterable[str]:
        return (self.feature_in_key, "adjacency", "node_mask")

    @property
    def out_keys(self) -> Iterable[str]:
        return (self.feature_out_key, "adjacency", "node_mask")

    def __init__(
        self,
        mlp: nn.Module,
        eps: float = 0.0,
        train_eps: bool = False,
        feature_in_key: NestedKey = "x",
        feature_out_key: NestedKey = "x",
        vmap_compatible: bool = False,
    ):
        super().__init__()
        self.mlp = mlp
        self.initial_eps = eps
        self.feature_in_key = feature_in_key
        self.feature_out_key = feature_out_key
        self.vmap_compatible = vmap_compatible
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
    ) -> torch.Tensor:
        # Extract the features, adjacency matrix and node mask from the input
        x: Float[Tensor, "... max_nodes feature"] = tensordict[self.feature_in_key]
        adjacency: Float[Tensor, "... max_nodes max_nodes"] = tensordict["adjacency"]
        if "node_mask" in tensordict.keys():
            node_mask: Bool[Tensor, "... max_nodes"] = tensordict["node_mask"]
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
        if self.vmap_compatible:
            new_x = self.mlp((1 + self.eps) * x + x_aggregated)
            new_x = new_x * node_mask[..., None]
        else:
            new_x_flat = self.mlp(
                (1 + self.eps) * x[node_mask] + x_aggregated[node_mask]
            )
            new_x = torch.zeros(
                (*x.shape[:-1], new_x_flat.shape[-1]), dtype=x.dtype, device=x.device
            )
            new_x[node_mask] = new_x_flat

        out = TensorDict(tensordict)
        out[self.feature_out_key] = new_x

        return out


class Squeeze(nn.Module):
    """Squeeze a dimension.

    Parameters
    ----------
    dim : int, default=-1
        The dimension to squeeze.
    """

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.squeeze(self.dim)


class BatchNorm1dBatchDims(nn.BatchNorm1d):
    """Batch normalization layer with arbitrary batch dimensions.

    See `torch.nn.BatchNorm1d` for documentation.
    """

    def forward(self, x: Tensor) -> Tensor:
        # Get the shape of the batch dims
        batch_shape = x.shape[:-1]

        # Flatten the batch dims
        x = x.reshape(-1, x.shape[-1])

        # Apply the batch normalization
        x = super().forward(x)

        # Reshape the output to have the same shape as the input
        return x.reshape(*batch_shape, x.shape[-1])


class TensorDictCat(TensorDictModuleBase):
    """Concatenate the keys of a TensorDict.

    Parameters
    ----------
    in_keys : NestedKey | Iterable[NestedKey]
        The keys to concatenate.
    out_key : NestedKey
        The key of the concatenated tensor.
    dim : int, default=0
        The dimension to concatenate over.
    """

    def __init__(
        self, in_keys: NestedKey | Iterable[NestedKey], out_key: NestedKey, dim=0
    ):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = (out_key,)
        self.dim = dim

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict.update(
            {
                self.out_keys[0]: torch.cat(
                    [tensordict[key] for key in self.in_keys], dim=self.dim
                )
            }
        )


class ParallelTensorDictModule(TensorDictModuleBase):
    """Apply a module to each key of a TensorDict.

    Parameters
    ----------
    module : nn.Module
        The module to apply.
    in_keys : NestedKey | Iterable[NestedKey]
        The keys to apply the module to.
    out_keys : NestedKey | Iterable[NestedKey]
        The keys to store the output in.
    """

    def __init__(
        self,
        module: nn.Module,
        in_keys: NestedKey | Iterable[NestedKey],
        out_keys: NestedKey | Iterable[NestedKey],
    ):
        super().__init__()
        self.module = module
        self.in_keys = in_keys
        self.out_keys = out_keys
        if len(list(self.in_keys)) != len(list(self.out_keys)):
            raise ValueError(
                f"The number of input keys must be the same as the number of output "
                f"keys. Got {len(list(self.in_keys))} input keys and "
                f"{len(list(self.out_keys))} output keys."
            )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict.update(
            {
                out_key: self.module(tensordict[in_key])
                for in_key, out_key in zip(self.in_keys, self.out_keys)
            }
        )


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
