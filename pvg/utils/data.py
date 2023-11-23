from tensordict import TensorDict

from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.data import Batch as GeometricBatch

from pvg.graph_isomorphism.data import GraphIsomorphismData


def gi_data_to_tensordict(data: GraphIsomorphismData | GeometricBatch) -> TensorDict:
    """Convert a GraphIsomorphismData object to a TensorDict with dense representations.
    
    The TensorDict contains the following keys:
    - x_a: The node features of graph A.
    - adjacency_a: The adjacency matrix of graph A.
    - node_mask_a: The node mask of graph A.
    - x_b: The node features of graph B.
    - adjacency_b: The adjacency matrix of graph B.
    - node_mask_b: The node mask of graph B.

    Parameters
    ----------
    data : GraphIsomorphismData | GeometricBatch
        The data to convert.

    Returns
    -------
    data_tensordict : TensorDict
        The converted data.
    """
    adjacency_a = to_dense_adj(data.edge_index_a, data.x_a_batch)
    x_a, node_mask_a = to_dense_batch(data.x_a, data.x_a_batch)
    adjacency_b = to_dense_adj(data.edge_index_b, data.x_b_batch)
    x_b, node_mask_b = to_dense_batch(data.x_b, data.x_b_batch)
    return TensorDict(
        dict(
            x_a=x_a,
            adjacency_a=adjacency_a,
            node_mask_a=node_mask_a,
            x_b=x_b,
            adjacency_b=adjacency_b,
            node_mask_b=node_mask_b,
        ),
        batch_size=x_a.shape[0],
    )
