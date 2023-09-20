import torch
from torch.nn import ReLU, Linear, MultiheadAttention, Sequential
from torch import Tensor

from torch_geometric.nn import GCNConv
from torch_geometric.nn import Sequential as GeometricSequential
from torch_geometric.utils import to_dense_batch

from jaxtyping import Float

from pvg.scenarios.base import Prover, Verifier, Scenario
from pvg.parameters import Parameters
from pvg.data import Dataset, GraphIsomorphismData


class GraphIsomorphismProver(Prover):
    """The prover for the graph isomorphism task.
    
    Takes as input a pair of graphs and outputs for each node a logit for the
    probability that this node should be sent as a message to the verifier.
    """

    def __init__(self, parameters: Parameters, device: str | torch.device):
        super().__init__(parameters, device)

        # Build up the GNN
        gnn_layers = []
        gnn_layers.append(
            (
                GCNConv(
                    parameters.max_num_messages * 2 + 1,
                    parameters.graph_isomorphism.prover_d_gnn,
                ),
                "x, edge_index -> x",
            )
        )
        for _ in range(parameters.graph_isomorphism.prover_num_layers - 1):
            gnn_layers.append(ReLU(inplace=True))
            gnn_layers.append(
                (
                    GCNConv(
                        parameters.graph_isomorphism.prover_d_gnn,
                        parameters.graph_isomorphism.prover_d_gnn,
                    ),
                    "x, edge_index -> x",
                )
            )
        self.gnn = GeometricSequential("x, edge_index", gnn_layers)

        self.attention = MultiheadAttention(
            embed_dim=parameters.graph_isomorphism.prover_d_attn,
            num_heads=parameters.graph_isomorphism.prover_num_heads,
            batch_first=True,
        )

        self.final_mlp = Sequential(
            ReLU(inplace=True),
            Linear(
                in_features=parameters.graph_isomorphism.prover_d_attn,
                out_features=parameters.graph_isomorphism.prover_d_final_mlp,
            ),
            ReLU(inplace=True),
            Linear(
                in_features=parameters.graph_isomorphism.prover_d_final_mlp,
                out_features=1,
            ),
        )

    def forward(
        self, data
    ) -> Float[Tensor, "batch_size, max_num_nodes_a + max_num_nodes_b"]:
        # Run the GNN on the two graphs
        # (total_num_nodes, d_gnn)
        gnn_output_a = self.gnn(x=data.x_a, edge_index=data.edge_index_a)
        gnn_output_b = self.gnn(x=data.x_b, edge_index=data.edge_index_b)

        # Convert from a sparse to a dense representation
        # (batch_size, max_num_nodes, d_gnn), (batch_size, max_num_nodes)
        gnn_output_a, mask_a = to_dense_batch(gnn_output_a, data.x_a_batch)
        gnn_output_b, mask_b = to_dense_batch(gnn_output_b, data.x_b_batch)

        # Concatenate the two graph representations and masks
        # (batch_size, max_num_nodes_a + max_num_nodes_b, d_gnn)
        gnn_output = torch.cat([gnn_output_a, gnn_output_b], dim=1)
        # (batch_size, max_num_nodes_a + max_num_nodes_b)
        mask = torch.cat([mask_a, mask_b], dim=1)
        mask = ~mask

        # Compute the attention output
        # (batch_size, max_num_nodes_a + max_num_nodes_b, d_attn)
        attention_output = self.attention(
            query=gnn_output,
            key=gnn_output,
            value=gnn_output,
            attn_mask=mask,
            needs_weights=False,
        )

        # Compute the final output
        # (batch_size, max_num_nodes_a + max_num_nodes_b)
        final_output = self.final_mlp(attention_output).squeeze(-1)

        return final_output

    def to(self, device: str | torch.device):
        self.device = device
        self.gnn.to(device)
        self.attention.to(device)
