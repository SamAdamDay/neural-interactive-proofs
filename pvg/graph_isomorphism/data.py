from abc import ABC
import os
from typing import Optional, Callable

import torch
from torch import Tensor

from torch_geometric.data import (
    Data as GeometricData,
    InMemoryDataset as GeometricInMemoryDataset,
)

from pvg.parameters import Parameters
from pvg.base import Dataset, DataLoader
from pvg.constants import GI_DATA_DIR


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


class GraphIsomorphismDataset(GeometricInMemoryDataset, Dataset):
    """A dataset for the graph isomorphism experiments."""

    def __init__(
        self,
        params: Parameters,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.params = params
        self.root = os.path.join(GI_DATA_DIR, params.dataset)
        super().__init__(self.root, transform, pre_transform, pre_filter)

        data, self.slices, self.sizes = torch.load(self.processed_paths[0])
        if isinstance(data, dict):
            self.data = GraphIsomorphismData.from_dict(data)
        else:
            self.data = data

    @property
    def processed_dir(self) -> str:
        return os.path.join(
            self.root, f"processed_{self.params.max_message_rounds}"
        )
    
    @property
    def num_node_features(self) -> int:
        return self.params.max_message_rounds

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, "raw")

    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt"]

    def process(self):
        # Copied from https://github.com/pyg-team/pytorch_geometric/blob/f71ead8ade8a67be23982114cfff649b7d074cfb/torch_geometric/datasets/tu_dataset.py#L193
        self.data, self.slices, self.sizes = _read_gi_data(
            self.raw_dir, self.num_node_features
        )

        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.data, self.slices = self.collate(data_list)
            self._data_list = None

        torch.save(
            (self._data.to_dict(), self.slices, self.sizes), self.processed_paths[0]
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.params.dataset!r}, "
            f"num_features={self.num_node_features}, "
            f"num_pairs={len(self)})"
        )


def _read_gi_data(
    raw_dir: str, d_features: int
) -> tuple[GraphIsomorphismData, dict[str, Tensor], dict[str, Tensor]]:
    """Reads the graph isomorphism dataset from the raw directory.

    Parameters
    ----------
    raw_dir : str
        The path to the raw directory.
    d_features : int
        The dimension of the node features (which will be filled with zeros).

    Returns
    -------
    data : GraphIsomorphismData
        The data object containing the whole dataset.
    slices : dict[str, torch.Tensor]
        A dictionary mapping the attributes to the their slices, for cutting the data
        up into individual graph pairs.
    sizes : dict[str, torch.Tensor]
        The sizes of the graphs in the dataset. Has keys "a" and "b".
    """
    data_dict = torch.load(os.path.join(raw_dir, "data.pt"))

    num_graphs = len(data_dict["wl_scores"])
    total_num_nodes_a = data_dict["sizes_a"].sum().item()
    total_num_nodes_b = data_dict["sizes_b"].sum().item()

    data = GraphIsomorphismData(
        edge_index_a=data_dict["edge_index_a"],
        edge_index_b=data_dict["edge_index_b"],
        wl_score=data_dict["wl_scores"],
        num_nodes_a=total_num_nodes_a,
        num_nodes_b=total_num_nodes_b,
        d_features=d_features,
    )

    slices = dict(
        edge_index_a=data_dict["slices_a"],
        edge_index_b=data_dict["slices_b"],
        wl_score=torch.arange(num_graphs + 1),
        x_a=torch.cat(
            (torch.zeros(1, dtype=int), torch.cumsum(data_dict["sizes_a"], dim=0))
        ),
        x_b=torch.cat(
            (torch.zeros(1, dtype=int), torch.cumsum(data_dict["sizes_b"], dim=0))
        ),
    )

    sizes = dict(
        a=data_dict["sizes_a"],
        b=data_dict["sizes_b"],
    )

    return data, slices, sizes
