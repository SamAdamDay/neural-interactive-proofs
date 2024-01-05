"""Utilities for data handling."""

from typing import Optional

import torch
import torch.nn.functional as F

from tensordict import TensorDict

from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.data import Batch as GeometricBatch


def gi_data_to_tensordict(
    data: GeometricBatch,
    node_dim_size: Optional[int] = None,
    adjacency_dtype: torch.dtype = torch.int,
) -> TensorDict:
    """Convert a GraphIsomorphismData object to a TensorDict with dense representations.

    The TensorDict contains the following keys:

    - "x" (batch, graph, node, feature): The node features of both graphs.
    - "adjacency" (batch, graph, node, node): The adjacency matrix of both graphs.
    - "node_mask" (batch, graph, node): A mask indicating which nodes are are present in
      the graph.

    Parameters
    ----------
    data : GraphIsomorphismData | GeometricBatch
        The data to convert.
    node_dim_size : int, optional
        The size of the node dimension. If None, the maximum number of nodes in the
        batch is used.
    adjacency_dtype : torch.dtype, default=torch.int
        The dtype of the adjacency matrix.

    Returns
    -------
    data_tensordict : TensorDict
        The converted data.
    """

    adjacency_a = to_dense_adj(data.edge_index_a, data.x_a_batch)
    x_a, node_mask_a = to_dense_batch(data.x_a, data.x_a_batch)
    adjacency_b = to_dense_adj(data.edge_index_b, data.x_b_batch)
    x_b, node_mask_b = to_dense_batch(data.x_b, data.x_b_batch)

    # Convert to the desired dtype
    adjacency_a = adjacency_a.to(adjacency_dtype)
    adjacency_b = adjacency_b.to(adjacency_dtype)

    # Compute the maximum number of nodes in the batch and make sure that the node
    # dimension is large enough
    max_num_nodes = max(x_a.shape[1], x_b.shape[1])
    if node_dim_size is None:
        node_dim_size = max_num_nodes
    elif node_dim_size < max_num_nodes:
        raise ValueError(
            f"node_dim_size ({node_dim_size}) must be at least as large as the maximum "
            f"number of nodes in the batch ({max_num_nodes})"
        )

    # Make sure that the node axes are the same for both graphs
    adjacency_a = F.pad(adjacency_a, (0, node_dim_size - adjacency_a.shape[-1]) * 2)
    x_a = F.pad(x_a, (0, 0, 0, node_dim_size - x_a.shape[-2]))
    node_mask_a = F.pad(node_mask_a, (0, node_dim_size - node_mask_a.shape[-1]))
    adjacency_b = F.pad(adjacency_b, (0, node_dim_size - adjacency_b.shape[-1]) * 2)
    x_b = F.pad(x_b, (0, 0, 0, node_dim_size - x_b.shape[-2]))
    node_mask_b = F.pad(node_mask_b, (0, node_dim_size - node_mask_b.shape[-1]))

    # Stack the pairs of in a new batch dimension
    adjacency = torch.stack((adjacency_a, adjacency_b), dim=1)
    x = torch.stack((x_a, x_b), dim=1)
    node_mask = torch.stack((node_mask_a, node_mask_b), dim=1)

    return TensorDict(
        dict(
            adjacency=adjacency,
            x=x,
            node_mask=node_mask,
        ),
        batch_size=x.shape[:2],
    )
