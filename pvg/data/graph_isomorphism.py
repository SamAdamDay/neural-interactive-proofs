from abc import ABC
from typing import Any, Optional

import torch
from torch import Tensor

from torch_geometric.data import Data as GeometricData

from pvg.data.base import Dataset, DataLoader


class GraphIsomorphismDataset(Dataset, ABC):
    """A dataset for the graph isomorphism experiments."""

    pass


class GraphIsomorphismData(GeometricData):
    """A data object consisting of two graphs their W-L score.

    Parameters
    ----------
    edge_index_a : torch.Tensor, optional
        The edge indices of graph A.
    edge_index_b : torch.Tensor, optional
        The edge indices of graph B.
    wl_score : torch.Tensor, optional
        A tensor with a single element, which is the number of rounds of the
        Weisfeiler-Lehman algorithm needed to distinguish the graphs, or -1 if the
        graphs cannot be distinguished with the algorithm.
    x_a : torch.Tensor, optional
        The node features of graph A, which are one-hot encodings of whether the node is
        part of the i-th message sent.
    x_b : torch.Tensor, optional
        The node features of graph B, which are one-hot encodings of whether the node is
        part of the i-th message sent.
    num_nodes_a : int, optional
        The number of nodes in graph A, for use when `x_a` is not given.
    num_nodes_b : int, optional
        The number of nodes in graph B, for use when `x_b` is not given.
    d_features : int, optional
        The dimension of the node features, for use when `x_a` and `x_b` are not given.
    """

    def __init__(
        self,
        edge_index_a: Optional[Tensor] = None,
        edge_index_b: Optional[Tensor] = None,
        wl_score: Optional[Tensor] = None,
        x_a: Optional[Tensor] = None,
        x_b: Optional[Tensor] = None,
        num_nodes_a: Optional[int] = None,
        num_nodes_b: Optional[int] = None,
        d_features: Optional[int] = None,
    ):
        if x_a is None and d_features is not None and num_nodes_a is not None:
            x_a = torch.zeros(num_nodes_a, d_features)
        if x_b is None and d_features is not None and num_nodes_b is not None:
            x_b = torch.zeros(num_nodes_b, d_features)
        super().__init__(
            edge_index_a=edge_index_a,
            edge_index_b=edge_index_b,
            wl_score=wl_score,
            x_a=x_a,
            x_b=x_b,
        )

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_a":
            return self.x_a.size(0)
        if key == "edge_index_b":
            return self.x_b.size(0)
        return super().__inc__(key, value, *args, **kwargs)
