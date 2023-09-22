from abc import ABC
from typing import Optional
from dataclasses import dataclass

import torch
from torch.nn import ReLU, Linear, MultiheadAttention, Sequential
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Categorical

from torch_geometric.nn import (
    GCNConv,
    Sequential as GeometricSequential,
    Linear as GeometricLinear,
)
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Batch as GeometricBatch

from einops import rearrange

from jaxtyping import Float, Bool

from pvg.scenarios.base import (
    Agent,
    Prover,
    Verifier,
    Scenario,
    Message,
    MessageExchange,
)
from pvg.parameters import Parameters
from pvg.data import Dataset, GraphIsomorphismData
from pvg.utils import GlobalMaxPool


class GraphIsomorphismAgent(Agent, ABC):
    """Mixin for the graph isomorphism agents.

    Provides some utility methods for constructing and running the various modules.
    """

    def __init__(self, parameters: Parameters, device: str | torch.device):
        super().__init__(parameters, device)
        self.gnn: GeometricSequential
        self.attention: MultiheadAttention

    def _build_gnn_and_attention(
        self,
        d_input: int,
        d_gnn: int,
        num_layers: int,
        num_heads: int,
    ) -> tuple[GeometricSequential, MultiheadAttention]:
        """Builds the GNN and attention modules for a prover or verifier.

        Parameters
        ----------
        d_input : int
            The dimensionality of the input features.
        d_gnn : int
            The dimensionality of the GNN hidden layers and of the attention embedding.
        num_layers : int
            The number of GNN layers.
        num_heads : int
            The number of attention heads.

        Returns
        -------
        gnn : torch_geometric.nn.Sequential
            The GNN module.
        attention : MultiheadAttention
            The attention module.
        """
        # Build up the GNN
        gnn_layers = []
        gnn_layers.append(
            (
                GeometricLinear(
                    d_input,
                    d_gnn,
                ),
                "x -> x",
            )
        )
        for _ in range(num_layers):
            gnn_layers.append(ReLU(inplace=True))
            gnn_layers.append(
                (
                    GCNConv(
                        d_gnn,
                        d_gnn,
                    ),
                    "x, edge_index -> x",
                )
            )
        gnn = GeometricSequential("x, edge_index", gnn_layers)

        attention = MultiheadAttention(
            embed_dim=d_gnn,
            num_heads=num_heads,
            batch_first=True,
        )

        return gnn, attention

    def _build_node_selector(
        self,
        d_gnn: int,
        d_node_selector: int,
        d_out: int,
    ) -> Sequential:
        """Builds the MLP which selects a node to send as a message.

        The MLP has one hidden layer and ReLU activations.

        Parameters
        ----------
        d_gnn : int
            The dimensionality of the attention embedding (also of the GNN hidden
            layers).
        d_node_selector : int
            The dimensionality of the MLP hidden layer.
        d_out : int
            The dimensionality of the output.

        Returns
        -------
        node_selector : torch.nn.Sequential
            The node selector module, which is an MLP.
        """
        return Sequential(
            ReLU(inplace=True),
            Linear(
                in_features=d_gnn,
                out_features=d_node_selector,
            ),
            ReLU(inplace=True),
            Linear(
                in_features=d_node_selector,
                out_features=d_out,
            ),
        )

    def _build_decider(
        self,
        d_gnn: int,
        d_decider: int,
    ) -> Sequential:
        """Builds the module which decides whether to continue exchanging messages

        Outputs a single triple of logits for the three options: continue exchanging
        messages, guess that the graphs are isomorphic, or guess that the graphs are not
        isomorphic.

        The module consists of a linear layer, a global max pooling layer, a final
        linear layer, and ReLU activations.

        Parameters
        ----------
        d_gnn : int
            The dimensionality of the attention embedding (also of the GNN hidden
            layers).
        d_decider : int
            The dimensionality of the final MLP hidden layers.
        d_out : int
            The dimensionality of the output.

        Returns
        -------
        decider : torch.nn.Sequential
            The decider module.
        """
        return Sequential(
            ReLU(inplace=True),
            Linear(
                in_features=d_gnn,
                out_features=d_decider,
            ),
            ReLU(inplace=True),
            GlobalMaxPool(dim=1),
            Linear(
                in_features=d_decider,
                out_features=3,
            ),
        )

    def _run_gnn_and_attention(
        self,
        data: GraphIsomorphismData | GeometricBatch,
        gnn: Optional[GeometricSequential] = None,
        attention: Optional[MultiheadAttention] = None,
    ) -> tuple[
        Float[Tensor, "batch_size max_nodes_a+max_nodes_b d_gnn"],
        Float[Tensor, "batch_size max_nodes_a+max_nodes_b d_gnn"],
        Bool[Tensor, "batch_size max_nodes_a+max_nodes_b"],
    ]:
        """Runs the GNN and attention modules.

        Parameters
        ----------
        data : GraphIsomorphismData | GraphIsomorphismDataBatch
            The data to run the GNN and attention on.
        gnn : Optional[GeometricSequential], optional
            The GNN module to use. If None, uses the module stored in the class.
        attention : Optional[MultiheadAttention], optional
            The attention module to use. If None, uses the module stored in the class.

        Returns
        -------
        gnn_output : Float[Tensor, "batch_size max_nodes_a+max_nodes_b d_gnn"]
            The output of the GNN.
        attention_output : Float[Tensor "batch_size max_nodes_a+max_nodes_b
        d_gnn"]
            The output of the attention.
        node_mask : Bool[Tensor, "batch_size max_nodes_a+max_nodes_b"]
            A mask indicating which nodes are present in the graphs.
        """
        if gnn is None:
            gnn = self.gnn
        if attention is None:
            attention = self.attention

        # Run the GNN on the two graphs
        # (total_num_nodes, d_gnn)
        gnn_output_a = gnn(x=data.x_a, edge_index=data.edge_index_a)
        gnn_output_b = gnn(x=data.x_b, edge_index=data.edge_index_b)

        # Convert from a sparse to a dense representation
        # (batch_size, max_nodes, d_gnn), (batch_size, max_nodes)
        gnn_output_a, mask_a = to_dense_batch(gnn_output_a, data.x_a_batch)
        gnn_output_b, mask_b = to_dense_batch(gnn_output_b, data.x_b_batch)

        # Concatenate the two graph representations and masks
        # (batch_size, max_nodes_a+max_nodes_b, d_gnn)
        gnn_output = torch.cat([gnn_output_a, gnn_output_b], dim=1)
        # (batch_size, max_nodes_a+max_nodes_b)
        node_mask = torch.cat([mask_a, mask_b], dim=1)

        # Turn the mask into batch of 2D attention masks
        attn_mask = ~node_mask
        # (batch_size, max_nodes_a+max_nodes_b, max_nodes_a+max_nodes_b)
        attn_mask = rearrange(attn_mask, "b n -> b n 1") * rearrange(
            attn_mask, "b n -> b 1 n"
        )

        # Compute the attention output
        # (batch_size, max_nodes_a+max_nodes_b, d_gnn)
        attention_output, _ = attention(
            query=gnn_output,
            key=gnn_output,
            value=gnn_output,
            attn_mask=attn_mask,
            need_weights=False,
        )

        return gnn_output, attention_output, node_mask


class GraphIsomorphismProver(Prover, GraphIsomorphismAgent):
    """The prover for the graph isomorphism task.

    Takes as input a pair of graphs and outputs for each node a logit for the
    probability that this node should be sent as a message to the verifier.
    """

    def __init__(self, parameters: Parameters, device: str | torch.device):
        super().__init__(parameters, device)

        # Build up the GNN and attention modules
        self.gnn, self.attention = self._build_gnn_and_attention(
            d_input=parameters.max_message_rounds,
            d_gnn=parameters.graph_isomorphism.prover_d_gnn,
            num_layers=parameters.graph_isomorphism.prover_num_layers,
            num_heads=parameters.graph_isomorphism.prover_num_heads,
        )

        self.node_selector = self._build_node_selector(
            d_gnn=parameters.graph_isomorphism.prover_d_gnn,
            d_node_selector=parameters.graph_isomorphism.prover_d_node_selector,
            d_out=1,
        )

    def forward(
        self, data: GraphIsomorphismData | GeometricBatch
    ) -> [
        Float[Tensor, "batch_size max_nodes_a+max_nodes_b"],
        Bool[Tensor, "batch_size max_nodes_a+max_nodes_b"],
    ]:
        """Runs the prover on the given data.

        Parameters
        ----------
        data : GraphIsomorphismData | GraphIsomorphismDataBatch
            The data to run the prover on.

        Returns
        -------
        node_logits : Float[Tensor, "batch_size max_nodes_a+max_nodes_b"]
            A logit for each node, indicating the probability that this node should be
            sent as a message to the verifier.
        node_mask : Bool[Tensor, "batch_size max_nodes_a+max_nodes_b"]
            A mask indicating which nodes are present in the graphs.
        """
        # (batch_size, max_nodes_a+max_nodes_b, d_gnn), (batch_size,
        # max_nodes_a+max_nodes_b)
        _, attention_output, node_mask = self._run_gnn_and_attention(data)

        # (batch_size, max_nodes_a+max_nodes_b)
        node_logits = self.node_selector(attention_output).squeeze(-1)

        return node_logits, node_mask

    def to(self, device: str | torch.device):
        self.device = device
        self.gnn.to(device)
        self.attention.to(device)


class GraphIsomorphismVerifier(Verifier, GraphIsomorphismAgent):
    """The verifier for the graph isomorphism task.

    Takes as input a pair of graphs and outputs for each node a logit for the
    probability that this node should be sent as a message to the verifier, as well as
    a single triple of logits for the three options: continue exchanging messages,
    guess that the graphs are isomorphic, or guess that the graphs are not isomorphic.
    """

    def __init__(self, parameters: Parameters, device: str | torch.device):
        super().__init__(parameters, device)

        # Build up the GNN and attention modules
        self.gnn, self.attention = self._build_gnn_and_attention(
            d_input=parameters.max_message_rounds,
            d_gnn=parameters.graph_isomorphism.verifier_d_gnn,
            num_layers=parameters.graph_isomorphism.verifier_num_layers,
            num_heads=parameters.graph_isomorphism.verifier_num_heads,
        )

        self.node_selector = self._build_node_selector(
            d_gnn=parameters.graph_isomorphism.verifier_d_gnn,
            d_node_selector=parameters.graph_isomorphism.verifier_d_node_selector,
            d_out=1,
        )

        self.decider = self._build_decider(
            d_gnn=parameters.graph_isomorphism.verifier_d_gnn,
            d_decider=parameters.graph_isomorphism.verifier_d_decider,
        )

    def forward(
        self, data: GraphIsomorphismData | GeometricBatch
    ) -> tuple[
        Float[Tensor, "batch_size max_nodes_a+max_nodes_b"],
        Float[Tensor, "batch_size 3"],
        Bool[Tensor, "batch_size max_nodes_a+max_nodes_b"],
    ]:
        """Runs the verifier on the given data.

        Parameters
        ----------
        data : GraphIsomorphismData | GraphIsomorphismDataBatch
            The data to run the verifier on.

        Returns
        -------
        node_logits : Float[Tensor, "batch_size max_nodes_a+max_nodes_b"]
            A logit for each node, indicating the probability that this node should be
            sent as a message to the prover.
        decider_logits : Float[Tensor, "batch_size 3"]
            A logit for each of the three options: continue exchanging messages, guess
            that the graphs are isomorphic, or guess that the graphs are not isomorphic.
        node_mask : Bool[Tensor, "batch_size max_nodes_a+max_nodes_b"]
            A mask indicating which nodes are present in the graphs.
        """
        # (batch_size, max_nodes_a+max_nodes_b, d_gnn), (batch_size,
        # max_nodes_a+max_nodes_b)
        _, attention_output, node_mask = self._run_gnn_and_attention(data)

        # (batch_size, max_nodes_a+max_nodes_b)
        node_logits = self.node_selector(attention_output).squeeze(-1)

        # (batch_size, 3)
        decider_logits = self.decider(attention_output)

        return node_logits, decider_logits, node_mask

    def to(self, device: str | torch.device):
        self.device = device
        self.gnn.to(device)
        self.attention.to(device)
        self.decider.to(device)


@dataclass
class GraphIsomorphismMessage(Message):
    """A message sent between the prover and the verifier."""

    from_verifier: bool
    message: Float[Tensor, "batch_size"]
    verifier_guess: Float[Tensor, "batch_size"] = None


class GraphIsomorphismScenario(Scenario):
    """The graph isomorphism scenario."""

    def __init__(self, parameters: Parameters, device: str | torch.device):
        super().__init__(parameters, device)
        self.prover = GraphIsomorphismProver(parameters, device)
        self.verifier = GraphIsomorphismVerifier(parameters, device)

    def rollout(self, data: GraphIsomorphismData | GeometricBatch) -> MessageExchange:
        message_exchange = MessageExchange()
        max_nodes_a = torch.bincount(data.x_a_batch).max().item()
        max_nodes_b = torch.bincount(data.x_b_batch).max().item()
        for round in range(self.parameters.max_message_rounds):
            ## Verifier sends a message
            # (batch_size, max_nodes_a+max_nodes_b), (batch_size, 3), (batch_size,
            # max_nodes_a+max_nodes_b)
            node_logits, decider_logits, node_mask = self.verifier(data)

            # (batch_size, max_nodes_a+max_nodes_b)
            node_probs = F.softmax(node_logits, dim=-1)
            node_probs[~node_mask] = 0

            # (batch_size)
            verifier_node = Categorical(probs=node_probs).sample()

            # (batch_size)
            verifier_guess = Categorical(logits=decider_logits).sample()

            message_exchange.append(
                GraphIsomorphismMessage(
                    from_verifier=True,
                    message=verifier_node,
                    verifier_guess=verifier_guess,
                )
            )

            # One-hot encode the verifier's message in the node features
            # Python loop because batches are small (hopefully) and doing it with tensor
            # operations is a pain
            x_a_idx = 0
            x_b_idx = 0
            x_a_ptr = data.x_a_ptr.tolist()
            x_b_ptr = data.x_b_ptr.tolist()
            for node_idx, len_a, len_b in zip(verifier_node.tolist(), x_a_ptr, x_b_ptr):
                if node_idx < max_nodes_a:
                    data.x_a[x_a_idx, round] = 1
                else:
                    data.x_b[x_b_idx, round] = 1
                x_a_idx += len_a
                x_b_idx += len_b

            ## Prover sends a message
            # (batch_size, max_nodes_a+max_nodes_b), (batch_size,
            # max_nodes_a+max_nodes_b)
            node_logits, node_mask = self.prover(data)

            # (batch_size, max_nodes_a+max_nodes_b)
            node_probs = F.softmax(node_logits, dim=-1)
            node_probs[~node_mask] = 0

            # Make sure that the prover chooses a node in the opposite graph to the one
            # chosen by the verifier
            node_probs = torch.where(
                verifier_node < max_nodes_a,
                torch.cat(
                    (
                        torch.zeros(node_probs.shape[0], max_nodes_a),
                        node_probs[:, max_nodes_a:],
                    ),
                    dim=-1,
                ),
                torch.cat(
                    (
                        node_probs[:, :max_nodes_a],
                        torch.zeros(node_probs.shape[0], max_nodes_b),
                    ),
                    dim=-1,
                ),
            )

            # (batch_size)
            prover_node = Categorical(probs=node_probs).sample()

            message_exchange.append(
                GraphIsomorphismMessage(
                    from_verifier=False,
                    message=prover_node,
                )
            )
