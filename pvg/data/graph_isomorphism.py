from abc import ABC

from torch_geometric.data import Data as GeometricData

from pvg.data.base import Dataset, DataLoader


class GraphIsomorphismDataset(Dataset, ABC):
    """A dataset for the graph isomorphism experiments."""


class GraphIsomorphismData(GeometricData):
    """A data object consisting of two graphs and bit for if they are isomorphic.
    
    Attributes
    ----------
    x_a : torch.Tensor
        The node features of graph A.
    x_b : torch.Tensor
        The node features of graph B.
    edge_index_a : torch.Tensor
        The edge indices of graph A.
    edge_index_b : torch.Tensor
        The edge indices of graph B.
    y : torch.Tensor
        A tensor with a single element, which is 1 if the graphs are isomorphic and 0
        otherwise.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_a':
            return self.x_a.size(0)
        if key == 'edge_index_b':
            return self.x_b.size(0)
        return super().__inc__(key, value, *args, **kwargs)