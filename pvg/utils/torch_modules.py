from torch import Tensor
import torch.nn as nn

class GlobalMaxPool(nn.Module):
    """Global max pooling layer over a dimension."""

    def __init__(self, dim: int = -1, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: Tensor) -> Tensor:
        return x.max(dim=self.dim, keepdim=self.keepdim)[0]