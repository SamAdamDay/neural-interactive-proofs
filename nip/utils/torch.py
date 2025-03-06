"""Handy PyTorch classes and utilities, including modules."""

from typing import Callable, Optional, Iterable, Iterator
from abc import abstractmethod
from math import prod

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import BatchSampler, Sampler

from torchvision.models.resnet import (
    BasicBlock as BasicResNetBlock,
    Bottleneck as BottleneckResNetBlock,
)

from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey

import einops

from jaxtyping import Float, Bool


ACTIVATION_CLASSES = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def flatten_batch_dims(x: Tensor, num_batch_dims: int) -> Tensor:
    """Return a new view of a tensor with the batch dimensions flattened.

    Parameters
    ----------
    x : Tensor
        The input tensor. Has shape ``(B1, B2, ..., Bn, D1, D2, ..., Dm)``, where n is
        the number of batch dimensions ``num_batch_dims``.
    num_batch_dims : int
        The number of batch dimensions to flatten.

    Returns
    -------
    x_flattened : Tensor
        The input tensor with the batch dimensions flattened. Has shape
        ``(B, D1, D2, ..., Dm)``, where ``B = B1 * B2 * ... * Bn``.
    """
    return x.flatten(0, num_batch_dims - 1)


def apply_orthogonal_initialisation(module: nn.Module, gain: float):
    """Apply orthogonal initialisation to a module's weights and set the biases to 0.

    Parameters
    ----------
    module : nn.Module
        The module to which to apply the initialisation.
    gain : float
        The orthogonal initialisation gain.
    """

    def init_weights(sub_module: nn.Module):
        if hasattr(sub_module, "weight"):
            if sub_module.weight.dim() >= 2:
                torch.nn.init.orthogonal_(sub_module.weight, gain=gain)
        if hasattr(sub_module, "bias") and sub_module.bias is not None:
            torch.nn.init.constant_(sub_module.bias, 0.0)

    module.apply(init_weights)


class DummyOptimizer(torch.optim.Optimizer):
    """A dummy optimizer which does nothing."""

    def __init__(self, *args, **kwargs):  # noqa: D102
        pass

    def step(self, *args, **kwargs):  # noqa: D102
        pass

    def zero_grad(self, *args, **kwargs):  # noqa: D102
        pass


class SimulateBatchDimsMixin:
    """A mixin for simulating multiple batch dimensions.

    Used for modules that don't support multiple batch dimensions, but can be simulated
    by flattening the batch dimensions and then unflattening them after applying the
    module.

    Classes that use this mixin should implement the ``feature_dims`` property.
    """

    @property
    @abstractmethod
    def feature_dims(self) -> int:
        """The number of non-batch dimensions."""
        pass

    def forward(self, x: Tensor) -> Tensor:
        """Apply the module to the input tensor, simulating multiple batch dimensions.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        out : Tensor
            The output tensor after applying the module.
        """

        # Get the shape of the batch dims
        batch_shape = x.shape[: -self.feature_dims]

        # Flatten the batch dims
        x = x.reshape(-1, *x.shape[-self.feature_dims :])

        # Apply the batch normalization
        x = super().forward(x)

        # Reshape the output to have the same shape as the input
        return x.reshape(*batch_shape, *x.shape[-self.feature_dims :])


class BatchNorm1dSimulateBatchDims(SimulateBatchDimsMixin, nn.BatchNorm1d):
    """Batch normalization layer with arbitrary batch dimensions.

    See :external+torch:class:`torch.nn.BatchNorm1d` for documentation.

    Assumes an input of shape (... features).
    """

    feature_dims = 1


class UpsampleSimulateBatchDims(SimulateBatchDimsMixin, nn.Upsample):
    """Upsample layer with arbitrary batch dimensions.

    See :external+torch:class:`torch.nn.Upsample` for documentation.

    Assumes an input of shape (... channels height width).
    """

    feature_dims = 3


class Conv2dSimulateBatchDims(SimulateBatchDimsMixin, nn.Conv2d):
    """2D convolutional layer with arbitrary batch dimensions.

    See :external+torch:class:`torch.nn.Conv2d` for documentation.

    Assumes an input of shape (... channels height width).
    """

    feature_dims = 3


class MaxPool2dSimulateBatchDims(SimulateBatchDimsMixin, nn.MaxPool2d):
    """2D max pool layer with arbitrary batch dimensions.

    See :external+torch:class:`torch.nn.MaxPool2d` for documentation.

    Assumes an input of shape (... channels height width).
    """

    feature_dims = 3


class ResNetBasicBlockSimulateBatchDims(SimulateBatchDimsMixin, BasicResNetBlock):
    """ResNet basic block with arbitrary batch dimensions.

    This is a subclass of the ``BasicBlock`` module from torchvision's ResNet
    implementation: :external+torchvision:doc:`models/resnet`. The ``BasicBlock`` module
    is not documented, but you can see the source code `here
    <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_.

    Assumes an input of shape (... channels height width).
    """

    feature_dims = 3


class ResNetBottleneckBlockSimulateBatchDims(
    SimulateBatchDimsMixin, BottleneckResNetBlock
):
    """ResNet bottleneck block with arbitrary batch dimensions.

    This is a subclass of the ``Bottleneck`` module from torchvision's ResNet
    implementation: :external+torchvision:doc:`models/resnet`. The ``Bottleneck`` module
    is not documented, but you can see the source code `here
    <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_.

    Assumes an input of shape (... channels height width).
    """

    feature_dims = 3


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
        """Apply global max pooling to the input tensor.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The output tensor after global max pooling.
        """
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
        """Concatenate the two node sets for each graph pair.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        x_cat : Tensor
            The input tensor with the two node sets concatenated.
        """
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
        Whether the ``sigma`` parameter should be trained or not.

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
        """Add Gaussian noise to the input tensor.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        x_noisy : Tensor
            The input tensor with Gaussian noise added.
        """
        if self.training and self.sigma != 0:
            # If we're not training sigma, we need to detach ``x`` when computing the
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

    def to(self, *args, **kwargs) -> "PairedGaussianNoise":
        """Move the module to a new device or dtype.

        Parameters
        ----------
        *args
            Arguments to pass to the ``to`` method of the superclass.
        **kwargs
            Keyword arguments to pass to the ``to`` method of the superclass.

        Returns
        -------
        self : PairedGaussianNoise
            The module itself.
        """
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
        """Transform the input to be invariant to the order of the graphs in a pair.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        transformed_x : Tensor
            The input tensor transformed to be invariant to the order of the graphs in a
            pair
        """
        mean = x.mean(dim=self.pair_dim)
        abs_diff = 0.5 * torch.abs(
            x.select(self.pair_dim, 0) - x.select(self.pair_dim, 1)
        )
        return torch.stack((mean, abs_diff), dim=self.pair_dim)


class GIN(TensorDictModuleBase):
    r"""A graph isomorphism network (GIN) layer :cite:`Xu2018`.

    This is a message-passing layer that aggregates the features of the neighbours as
    follows:

    .. math::

        x_i' = \text{MLP}((1 + \epsilon) x_i + \sum_{j \in \mathcal{N}(i)} x_j)

    where $x_i$ is the feature vector of node $i$, $\mathcal{N}(i)$ is the set of
    neighbours of node $i$, and $\epsilon$ is a (possibly learnable) parameter.

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
    adjacency_key : NestedKey, default="adjacency"
        The key of the adjacency matrix in the input TensorDict.
    node_mask_key : NestedKey, default="node_mask"
        The key of the node mask in the input TensorDict.
    vmap_compatible : bool, default=False
        Whether the module is compatible with ``vmap`` or not. If ``True``, the node
        mask is only applied after the MLP, which is less efficient but allows for the
        use of ``vmap``.

    Shapes
    ------
    Takes as input a TensorDict with the following keys:

    - "x" - Float["... max_nodes feature"] - The features of the nodes.
    - "adjacency" - Float["... max_nodes max_nodes"] - The adjacency matrix of the
      graph.
    - "node_mask" - Bool["... max_nodes"] - A mask indicating which nodes exist
    """

    @property
    def in_keys(self) -> Iterable[str]:
        """The keys of the input TensorDict."""
        return (self.feature_in_key, self.adjacency_key, self.node_mask_key)

    @property
    def out_keys(self) -> Iterable[str]:
        """The keys of the output TensorDict."""
        return (self.feature_out_key, self.adjacency_key, self.node_mask_key)

    def __init__(
        self,
        mlp: nn.Module,
        eps: float = 0.0,
        train_eps: bool = False,
        feature_in_key: NestedKey = "x",
        feature_out_key: NestedKey = "x",
        adjacency_key: NestedKey = "adjacency",
        node_mask_key: NestedKey = "node_mask",
        vmap_compatible: bool = False,
    ):
        super().__init__()
        self.mlp = mlp
        self.initial_eps = eps
        self.feature_in_key = feature_in_key
        self.feature_out_key = feature_out_key
        self.adjacency_key = adjacency_key
        self.node_mask_key = node_mask_key
        self.vmap_compatible = vmap_compatible
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of the layer."""
        self.eps.data.fill_(self.initial_eps)

    def forward(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDict:
        """Apply the GIN layer to the input TensorDict.

        Parameters
        ----------
        tensordict : TensorDictBase
            The input TensorDict with a dense representation of the graph. It should
            have a key for the features, adjacency matrix and (optionally) node mask.

        Returns
        -------
        out : TensorDict
            The input TensorDict with the GIN layer applied. This includes the updated
            features.
        """

        # Extract the features, adjacency matrix and node mask from the input
        x: Float[Tensor, "... max_nodes feature"] = tensordict[self.feature_in_key]
        adjacency: Float[Tensor, "... max_nodes max_nodes"] = tensordict[
            self.adjacency_key
        ]
        if self.node_mask_key in tensordict.keys():
            node_mask: Bool[Tensor, "... max_nodes"] = tensordict[self.node_mask_key]
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
        """Squeeze the input tensor.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        x_squeezed : Tensor
            The input tensor with the specified dimension squeezed.
        """
        return x.squeeze(self.dim)


class TensorDictCat(TensorDictModuleBase):
    """Concatenate the keys of a TensorDict.

    Parameters
    ----------
    in_keys : Iterable[NestedKey]
        The keys to concatenate.
    out_key : NestedKey
        The key of the concatenated tensor.
    dim : int, default=0
        The dimension to concatenate over.
    """

    def __init__(self, in_keys: Iterable[NestedKey], out_key: NestedKey, dim=0):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = (out_key,)
        self.dim = dim

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Concatenate the keys of the input TensorDict.

        Parameters
        ----------
        tensordict : TensorDictBase
            The input TensorDict.

        Returns
        -------
        concatenated_tensordict : TensorDictBase
            The input TensorDict with the keys concatenated.
        """
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
        """Apply the module to each key of the input TensorDict.

        Parameters
        ----------
        tensordict : TensorDictBase
            The input TensorDict.

        Returns
        -------
        transformed_tensordict : TensorDictBase
            The input TensorDict with the module applied to each key.
        """
        return tensordict.update(
            {
                out_key: self.module(tensordict[in_key])
                for in_key, out_key in zip(self.in_keys, self.out_keys)
            }
        )


class TensorDictCloneKeys(TensorDictModuleBase):
    """Clone the keys of a TensorDict.

    Parameters
    ----------
    in_keys : NestedKey | Iterable[NestedKey]
        The keys to clone.
    out_keys : NestedKey | Iterable[NestedKey]
        The keys to store the cloned values in.
    """

    def __init__(
        self,
        in_keys: NestedKey | Iterable[NestedKey],
        out_keys: NestedKey | Iterable[NestedKey],
    ):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys
        if len(list(self.in_keys)) != len(list(self.out_keys)):
            raise ValueError(
                f"The number of input keys must be the same as the number of output "
                f"keys. Got {len(list(self.in_keys))} input keys and "
                f"{len(list(self.out_keys))} output keys."
            )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Clone the keys of the input TensorDict.

        Parameters
        ----------
        tensordict : TensorDictBase
            The input TensorDict.

        Returns
        -------
        cloned_tensordict : TensorDictBase
            The input TensorDict with the keys cloned.
        """
        return tensordict.update(
            {
                out_key: tensordict[in_key]
                for in_key, out_key in zip(self.in_keys, self.out_keys)
            }
        )


class OneHot(nn.Module):
    """One-hot encode a tensor.

    Parameters
    ----------
    num_classes : int, default=-1
        The number of classes to one-hot encode.
    """

    def __init__(self, num_classes: int = -1):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        """One-hot encode the input tensor.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        x_one_hot : Tensor
            The one-hot encoded tensor.
        """
        return torch.nn.functional.one_hot(x, self.num_classes).float()


class NormalizeOneHotMessageHistory(TensorDictModuleBase):
    """Normalize the history of one-hot message exchanges.

    Normalizes each component to have zero mean and unit variance, giving each possible
    length of messages the same weight.

    The input is assumed to have some number of batch dimensions followed some number of
    structure dimensions, followed by the round dimension (these two are reversed when
    ``round_dim_last`` is False). The 'structure' dimensions are those that specify the
    structure of a data point, e.g. the height and width of an image. The input is
    assumed to be one-hot encoded across all the structure dimensions for each round
    where a message has been exchanged.

    Shapes
    ------
    Takes as input a TensorDict with key:

    - "x" with shape one of:
      - ``Float["... structure_dim_1 ... structure_dim_k round"]``
      - ``Float["... round structure_dim_1 ... structure_dim_k"]``

    Parameters
    ----------
    max_message_rounds : int
        The maximum length of the message history.
    message_in_key : NestedKey, default="x"
        The key containing the message history.
    message_out_key : NestedKey, default="x_normalized"
        The key to store the normalized message history.
    num_structure_dims : int, default=1
        The number of feature dimensions to normalize over (see above).
    round_dim_last : bool, default=True
        Whether the round dimension is the last dimension or whether it is located just
        before the structure dimensions.
    """

    @property
    def in_keys(self) -> Iterable[str]:
        """The keys of the input TensorDict."""
        return (self.message_in_key,)

    @property
    def out_keys(self) -> Iterable[str]:
        """The keys of the output TensorDict."""
        return (self.message_out_key,)

    def __init__(
        self,
        max_message_rounds: int,
        message_in_key: NestedKey = "x",
        message_out_key: NestedKey = "x_normalized",
        num_structure_dims: int = 1,
        round_dim_last: bool = True,
    ):
        super().__init__()
        self.max_message_rounds = max_message_rounds
        self.message_in_key = message_in_key
        self.message_out_key = message_out_key
        self.num_structure_dims = num_structure_dims
        self.round_dim_last = round_dim_last

        self._cached_mean: Optional[Tensor] = None
        self._cached_std: Optional[Tensor] = None
        self._cached_structure_shape: Optional[torch.Size] = None

    def _get_mean_and_std(self, x: Tensor) -> tuple[Tensor, Tensor]:
        r"""Get the mean and standard deviation for the structure shape of ``x``.

        These are computed based only on the shape of the structure dimensions, so they
        can be cached and reused for tensors with the same structure shape.

        Let ``n`` be the total size of the structure dimensions and ``m`` be the maximum
        number of message rounds. Then the mean and standard deviation are computed as
        follows:

        .. math::

            \text{mean} = \frac 1 {n m} (m - 1, m - 2, \ldots, 0) \\
            \text{std} = \frac 1 {n m} \sqrt{ 
                ((m - 1) (n m - m + 1), (m - 2) (n m - m + 2), \ldots, 0)
            }

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        mean : Tensor
            The mean for message histories with the structure shape of ``x``.
        std : Tensor
            The standard deviation for message histories with the structure shape of 
            ``x``.
        """
        # Get the shape of the structure dimensions
        if self.round_dim_last:
            structure_shape = x.shape[-(self.num_structure_dims + 1) : -1]
        else:
            structure_shape = x.shape[-self.num_structure_dims :]

        # Check if the mean and standard deviation are already cached
        if self._cached_mean is not None and self._cached_std is not None:
            if structure_shape == self._cached_structure_shape:
                return self._cached_mean, self._cached_std

        self._cached_structure_shape = structure_shape
        structure_size = prod(structure_shape)

        # Compute the mean, assuming that each possible length of messages is equally
        # likely.
        self._cached_mean = torch.arange(
            self.max_message_rounds - 1, -1, -1, dtype=x.dtype, device=x.device
        )
        self._cached_mean = self._cached_mean / (
            structure_size * self.max_message_rounds
        )

        # Compute the standard deviation, assuming that each possible length of messages
        # is equally likely.
        self._cached_std = torch.arange(
            self.max_message_rounds - 1, -1, -1, dtype=x.dtype, device=x.device
        )
        self._cached_std = self._cached_std * (
            structure_size * self.max_message_rounds
            - self.max_message_rounds
            + 1
            + torch.arange(
                self.max_message_rounds,
                dtype=x.dtype,
                device=x.device,
            )
        )
        self._cached_std = torch.sqrt(self._cached_std)
        self._cached_std = self._cached_std / (structure_size * self.max_message_rounds)
        self._cached_std[-1] = 1.0  # Avoid division by zero

        # Add singleton dimensions to the mean and standard deviation to match the shape
        # of the input tensor when the round dimension is not the last dimension
        if not self.round_dim_last:
            self._cached_mean = self._cached_mean.reshape(
                (self.max_message_rounds,) + (1,) * self.num_structure_dims
            )
            self._cached_std = self._cached_std.reshape(
                (self.max_message_rounds,) + (1,) * self.num_structure_dims
            )

        return self._cached_mean, self._cached_std

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Normalize the message history.

        Parameters
        ----------
        tensordict : TensorDictBase
            The input tensordict.

        Returns
        -------
        normalized_tensordict : TensorDictBase
            The input tensordict with the message history normalized.
        """

        x = tensordict[self.message_in_key]

        # Get the mean and standard deviation for the structure shape of ``x``
        mean, std = self._get_mean_and_std(x)

        # Normalize the message history
        x = (x - mean) / std

        # Store the normalized message history in the output TensorDict
        return tensordict.update({self.message_out_key: x})

    def to(self, *args, **kwargs) -> "NormalizeOneHotMessageHistory":
        """Move the module to a new device or dtype.

        Parameters
        ----------
        *args
            Positional arguments to pass to the ``to``
        **kwargs
            Keyword arguments to pass to the ``to``

        Returns
        -------
        self : NormalizeOneHotMessageHistory
            The module, moved to the new device or dtype.
        """
        super().to(*args, **kwargs)
        if self._cached_mean is not None:
            self._cached_mean = self._cached_mean.to(*args, **kwargs)
        if self._cached_std is not None:
            self._cached_std = self._cached_std.to(*args, **kwargs)
        return self


class Print(nn.Module):
    """Print information about an input tensor.

    Parameters
    ----------
    name : str, default=None
        The name of the tensor.
    mode : str, default="shape"
        The mode to print. One of the following:

        - "shape": Print the shape of the tensor.
        - "value": Print the value of the tensor.
        - "nan": Print the fraction of NaN values in the tensor.

    transform : Callable, default=None
        A function to apply to the tensor before printing.
    """

    def __init__(
        self,
        name: str = None,
        mode: bool = False,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.name = name
        self.mode = mode
        self.transform = transform

    def forward(self, x: Tensor) -> Tensor:
        """Print the information about the input tensor.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        x : Tensor
            The input tensor, unchanged.
        """
        if self.name is not None:
            print(f"{self.name}:")  # noqa: T201
        if self.mode == "value":
            if self.transform is not None:
                x = self.transform(x)
            print(x)  # noqa: T201
        elif self.mode == "nan":
            print(x.isnan().float().mean())  # noqa: T201
        else:
            print(x.shape)  # noqa: T201
        return x


class TensorDictPrint(TensorDictModuleBase):
    """Print information about an input tensordict.

    Parameters
    ----------
    keys : NestedKey | Iterable[NestedKey]
        The keys to print.
    name : str, default=None
        The name of the tensordict, which will be printed before the keys.
    print_nan_proportion : bool, default=False
        Whether to print the proportion of NaN values in the tensors.
    """

    def __init__(
        self,
        keys: NestedKey | Iterable[NestedKey],
        name: Optional[str] = None,
        print_nan_proportion: bool = False,
    ):
        super().__init__()
        self.name = name
        if isinstance(keys, str) or (
            isinstance(keys, tuple) and isinstance(keys[0], str)
        ):
            keys = (keys,)
        self.in_keys = keys
        self.out_keys = keys
        self.print_nan_proportion = print_nan_proportion

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Print the information about the tensors in the input tensordict.

        Parameters
        ----------
        tensordict : TensorDictBase
            The input tensordict.

        Returns
        -------
        tensordict : TensorDictBase
            The input tensordict, unchanged.
        """
        if self.name is not None:
            print(f"{type(self).__name__} {self.name!r}:")  # noqa: T201
        for key in self.in_keys:
            to_print = f"{key}: ({tensordict[key].shape})"
            if self.print_nan_proportion:
                to_print += (
                    f", NaN proportion: {tensordict[key].isnan().float().mean()!s}"
                )
            print(to_print)  # noqa: T201
        return tensordict


class FastForwardableBatchSampler(BatchSampler):
    """A batch sampler which can skip an initial number of items.

    See the docs for PyTorch's ``BatchSampler`` for details.

    Parameters
    ----------
    sampler : Sampler[int] | Iterable[int]
        Base sampler. Can be any iterable object
    batch_size : int
        The size of the mini-batch
    drop_last : bool
        If ``True``, the sampler will drop the last batch if its size would be less than
        ``batch_size``
    initial_skip : int, default=0
        The number of items to skip at the start of the sampler.
    """

    def __init__(
        self,
        sampler: Sampler[int] | Iterable[int],
        batch_size: int,
        drop_last: bool,
        initial_skip: int = 0,
    ):
        super().__init__(sampler, batch_size, drop_last)
        self.initial_skip = initial_skip

    def __iter__(self) -> Iterator[list[int]]:
        # Adapted from ``torch.utils.data.sampler.BatchSampler.__iter__``.
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            for _ in range(self.initial_skip):
                next(sampler_iter)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for i, idx in enumerate(self.sampler):
                if i < self.initial_skip:
                    continue
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]
