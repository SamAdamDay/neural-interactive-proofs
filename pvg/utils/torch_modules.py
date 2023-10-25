from typing import Callable, Optional

import torch
from torch import Tensor
import torch.nn as nn


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
