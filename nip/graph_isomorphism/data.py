"""Dataset classes for the graph isomorphism experiments."""

import os

import torch

from tensordict import TensorDict

from torch_geometric.utils import to_dense_adj

from einops import repeat

from nip.scenario_base import Dataset, TensorDictDataset
from nip.parameters import ScenarioType
from nip.factory import register_scenario_class
from nip.constants import GI_DATA_DIR


@register_scenario_class("graph_isomorphism", Dataset)
class GraphIsomorphismDataset(TensorDictDataset):
    """A dataset for the graph isomorphism experiments.

    Uses the a pre-generated set of graphs.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters for the experiment.
    settings : ExperimentSettings
        The settings for the experiment.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    train : bool
        Whether to load the training or test set.
    """

    instance_keys = ("adjacency", "x", "node_mask", "wl_score")

    adjacency_dtype = torch.int32
    x_dtype = torch.float32
    y_dtype = torch.int64

    @property
    def raw_dir(self) -> str:
        """The path to the directory containing the raw data."""
        sub_dir = "train" if self.train else "test"
        return os.path.join(GI_DATA_DIR, self.hyper_params.dataset, "raw", sub_dir)

    @property
    def processed_dir(self) -> str:
        """The path to the directory containing the processed data."""

        sub_dir = "train" if self.train else "test"

        if self.train and self.hyper_params.dataset_options.max_train_size is not None:
            max_size_suffix = f"_{self.hyper_params.dataset_options.max_train_size}"
        else:
            max_size_suffix = ""

        return os.path.join(
            GI_DATA_DIR,
            self.hyper_params.dataset,
            f"processed"
            f"_{self.protocol_handler.max_message_rounds}"
            f"_{self.protocol_handler.num_message_channels}"
            f"_{self.hyper_params.message_size}"
            f"{max_size_suffix}",
            sub_dir,
        )

    def build_tensor_dict(self) -> TensorDict:
        """Build the tensor dict dataset from the raw data.

        Returns
        -------
        dataset : TensorDict
            The dataset as a tensor dict.
        """
        data_dict = torch.load(os.path.join(self.raw_dir, "data.pt"))

        num_graph_pairs = len(data_dict["wl_scores"])
        max_num_nodes = max(data_dict["sizes_a"].max(), data_dict["sizes_b"].max())

        def make_components(graph_index: str):
            sizes = data_dict[f"sizes_{graph_index}"]

            # Create the mask which indicates which nodes are present in the graph
            node_indices = repeat(
                torch.arange(0, sizes.max()),
                "node -> graph node",
                graph=num_graph_pairs,
            )
            node_mask = node_indices < sizes[:, None]

            # Create batch vector, which assigns each node to a graph
            graph_indices = repeat(
                torch.arange(0, num_graph_pairs),
                "graph -> graph node",
                node=sizes.max(),
            )
            batch = graph_indices[node_mask]

            # Shift the edge indices up according to ``slices``
            num_edges_per_graph = (
                data_dict[f"slices_{graph_index}"][1:]
                - data_dict[f"slices_{graph_index}"][:-1]
            )
            edge_index_shift = repeat(
                torch.cat((torch.tensor([0]), torch.cumsum(sizes, dim=0)[:-1])),
                "graph -> graph edge",
                edge=num_edges_per_graph.max(),
            )
            graph_edge_indices = repeat(
                torch.arange(0, num_edges_per_graph.max()),
                "edge -> graph edge",
                graph=num_graph_pairs,
            )
            shift_mask = graph_edge_indices < num_edges_per_graph[:, None]
            edge_index_shift = edge_index_shift[shift_mask]
            edge_index = data_dict[f"edge_index_{graph_index}"] + edge_index_shift

            # Turn the sparse adjacency matrix into a dense one
            adjacency = to_dense_adj(
                edge_index, batch=batch, max_num_nodes=max_num_nodes
            ).to(self.adjacency_dtype)

            # Create the node features, which are all zeros
            x = torch.zeros(
                num_graph_pairs,
                self.protocol_handler.max_message_rounds,
                self.protocol_handler.num_message_channels,
                self.hyper_params.message_size,
                max_num_nodes,
                dtype=self.x_dtype,
            )

            return adjacency, x, node_mask

        adjacency_a, x_a, node_mask_a = make_components("a")
        adjacency_b, x_b, node_mask_b = make_components("b")

        # Stack the pairs of in a new batch dimension
        adjacency = torch.stack((adjacency_a, adjacency_b), dim=1)
        x = torch.stack((x_a, x_b), dim=-2)
        node_mask = torch.stack((node_mask_a, node_mask_b), dim=1)

        y = (data_dict["wl_scores"] == -1).to(self.y_dtype)

        return TensorDict(
            dict(
                adjacency=adjacency,
                x=x,
                node_mask=node_mask,
                y=y,
                wl_score=data_dict["wl_scores"],
            ),
            batch_size=num_graph_pairs,
        )
