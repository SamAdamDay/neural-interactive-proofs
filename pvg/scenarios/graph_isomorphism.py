from abc import ABC
from typing import Optional, Callable
from dataclasses import dataclass
from math import sqrt
from functools import partial
from collections import OrderedDict

import torch
from torch.nn import ReLU, Linear, MultiheadAttention, Sequential
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Categorical

from torch_geometric.nn import (
    GCNConv,
    GINConv,
    Sequential as GeometricSequential,
    Linear as GeometricLinear,
)
from torch_geometric.utils import to_dense_batch, to_networkx
from torch_geometric.data import Batch as GeometricBatch, Data as GeometricData

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

from jaxtyping import Float, Bool

import plotly.graph_objects as go
import plotly.express as px

import networkx as nx

from pvg.scenarios.base import (
    Agent,
    Prover,
    Verifier,
    Rollout,
    Scenario,
    Message,
    MessageExchange,
)
from pvg.parameters import Parameters
from pvg.data import GraphIsomorphismData, GraphIsomorphismDataset
from pvg.utils.torch_modules import CatGraphPairDim, Print


class GraphIsomorphismAgent(Agent, ABC):
    """Mixin for the graph isomorphism agents.

    Provides some utility methods for constructing and running the various modules.
    """

    def __init__(self, params: Parameters, device: str | torch.device):
        super().__init__(params, device)
        self.gnn: GeometricSequential
        self.attention: MultiheadAttention

    def _build_gnn_and_attention(
        self,
        d_input: int,
        d_gnn: int,
        d_gin_mlp: int,
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
        d_gin_mlp: int
            The dimensionality of the hidden layers in the Graph Isomorphism Network
            MLP.
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
        gnn_layers = OrderedDict()
        gnn_layers["input"] = (
            GeometricLinear(
                d_input,
                d_gnn,
            ),
            "x -> x",
        )
        for i in range(num_layers):
            gnn_layers[f"ReLU_{i}"] = ReLU(inplace=True)
            gnn_layers[f"GNN_layer_{i}"] = (
                GINConv(
                    Sequential(
                        Linear(
                            d_gnn,
                            d_gin_mlp,
                        ),
                        ReLU(inplace=True),
                        Linear(
                            d_gin_mlp,
                            d_gnn,
                        ),
                    )
                ),
                "x, edge_index -> x",
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
            # (2, batch_size, max_nodes, d_gnn)
            CatGraphPairDim(cat_dim=2),
            # (batch_size, max_nodes_a+max_nodes_b, d_gnn)
            ReLU(inplace=True),
            Linear(
                in_features=d_gnn,
                out_features=d_node_selector,
            ),
            # (batch_size, max_nodes_a+max_nodes_b, d_node_selector)
            ReLU(inplace=True),
            Linear(
                in_features=d_node_selector,
                out_features=d_out,
            ),
            # (batch_size, max_nodes_a+max_nodes_b, d_node_out)
        )

    def _build_decider(
        self,
        d_gnn: int,
        d_decider: int,
        d_out: int = 3,
    ) -> Sequential:
        """Builds the module which produces a graph-pair level output.

        By default it is used to decide whether to continue exchanging messages. In this
        case it outputs a single triple of logits for the three options: continue
        exchanging messages, guess that the graphs are isomorphic, or guess that the
        graphs are not isomorphic.

        The module consists of an MLP, a global max pooling layer per graph, and a final
        MLP. The MLPs have one hidden layer and ReLU activations.

        Parameters
        ----------
        d_gnn : int
            The dimensionality of the attention embedding (also of the GNN hidden
            layers).
        d_decider : int
            The dimensionality of the final MLP hidden layers.
        d_out : int, default=3
            The dimensionality of the output.

        Returns
        -------
        decider : torch.nn.Sequential
            The decider module.
        """
        return Sequential(
            Linear(
                in_features=d_gnn,
                out_features=d_decider,
            ),
            ReLU(inplace=True),
            Reduce(
                "pair batch_size max_nodes d_decider -> pair batch_size d_decider",
                "sum",
            ),
            Rearrange("pair batch_size d_decider -> batch_size (pair d_decider)"),
            Linear(
                in_features=2 * d_decider,
                out_features=d_decider,
            ),
            ReLU(inplace=True),
            Linear(
                in_features=d_decider,
                out_features=d_decider,
            ),
            ReLU(inplace=True),
            Linear(
                in_features=d_decider,
                out_features=d_out,
            ),
        )

    def _run_gnn_and_attention(
        self,
        data: GraphIsomorphismData | GeometricBatch,
        gnn: Optional[GeometricSequential] = None,
        attention: Optional[MultiheadAttention] = None,
    ) -> tuple[
        Float[Tensor, "2 batch_size max_nodes d_gnn"],
        Float[Tensor, "2 batch_size max_nodes d_gnn"],
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
        gnn_output_stacked : Float[Tensor, "2 batch_size max_nodes d_gnn"]
            The output of the GNN with the two graphs stacked in a new batch dimension.
        attention_output_stacked : Float[Tensor "2 batch_size max_nodes d_gnn"]
            The output of the attention module with the two graphs stacked in a new
            batch dimension.
        node_mask : Bool[Tensor, "batch_size max_nodes_a+max_nodes_b"]
            A mask indicating which nodes are present in the graphs, with the two graphs
            concatenated along the node dimension.
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

        # Get the maximum number of nodes in each graph
        max_nodes_a = gnn_output_a.shape[1]
        max_nodes_b = gnn_output_b.shape[1]

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

        # Separate the graph representations back out
        # (batch_size, max_nodes_a, d_gnn)
        gnn_output_a = gnn_output[:, :max_nodes_a]
        attention_output_a = attention_output[:, :max_nodes_a]
        # (batch_size, max_nodes_b, d_gnn)
        gnn_output_b = gnn_output[:, max_nodes_a:]
        attention_output_b = attention_output[:, max_nodes_a:]

        # Put these together in a new batch dimension
        # (2, batch_size, max_nodes, d_gnn)
        gnn_output_stacked = torch.stack((gnn_output_a, gnn_output_b))
        attention_output_stacked = torch.stack((attention_output_a, attention_output_b))

        return gnn_output_stacked, attention_output_stacked, node_mask


class GraphIsomorphismProver(Prover, GraphIsomorphismAgent):
    """The prover for the graph isomorphism task.

    Takes as input a pair of graphs and outputs for each node a logit for the
    probability that this node should be sent as a message to the verifier.
    """

    def __init__(self, params: Parameters, device: str | torch.device):
        super().__init__(params, device)

        # Build up the GNN and attention modules
        self.gnn, self.attention = self._build_gnn_and_attention(
            d_input=params.max_message_rounds,
            d_gnn=params.graph_isomorphism.prover_d_gnn,
            d_gin_mlp=params.graph_isomorphism.prover_d_gin_mlp,
            num_layers=params.graph_isomorphism.prover_num_layers,
            num_heads=params.graph_isomorphism.prover_num_heads,
        )

        self.node_selector = self._build_node_selector(
            d_gnn=params.graph_isomorphism.prover_d_gnn,
            d_node_selector=params.graph_isomorphism.prover_d_node_selector,
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
        # (2, batch_size, max_nodes, d_gnn),
        # (2, batch_size, max_nodes, d_gnn),
        # (batch_size, max_nodes_a+max_nodes_b)
        gnn_output, attention_output, node_mask = self._run_gnn_and_attention(data)

        # (2, batch_size, max_nodes, d_gnn)
        gnn_attn_output = gnn_output + attention_output

        # (batch_size, max_nodes_a+max_nodes_b)
        node_logits = self.node_selector(gnn_attn_output).squeeze(-1)

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

    def __init__(self, params: Parameters, device: str | torch.device):
        super().__init__(params, device)

        # Build up the GNN and attention modules
        self.gnn, self.attention = self._build_gnn_and_attention(
            d_input=params.max_message_rounds,
            d_gnn=params.graph_isomorphism.verifier_d_gnn,
            d_gin_mlp=params.graph_isomorphism.verifier_d_gin_mlp,
            num_layers=params.graph_isomorphism.verifier_num_layers,
            num_heads=params.graph_isomorphism.verifier_num_heads,
        )

        self.node_selector = self._build_node_selector(
            d_gnn=params.graph_isomorphism.verifier_d_gnn,
            d_node_selector=params.graph_isomorphism.verifier_d_node_selector,
            d_out=1,
        )

        self.decider = self._build_decider(
            d_gnn=params.graph_isomorphism.verifier_d_gnn,
            d_decider=params.graph_isomorphism.verifier_d_decider,
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
        # (batch_size, max_nodes_a+max_nodes_b, d_gnn),
        # (batch_size, max_nodes_a+max_nodes_b, d_gnn),
        # (batch_size, max_nodes_a+max_nodes_b)
        gnn_output, attention_output, node_mask = self._run_gnn_and_attention(data)

        # (batch_size, max_nodes_a+max_nodes_b, d_gnn)
        gnn_attn_output = gnn_output + attention_output

        # (batch_size, max_nodes_a+max_nodes_b)
        node_logits = self.node_selector(gnn_attn_output).squeeze(-1)

        # (batch_size, 3)
        decider_logits = self.decider(gnn_attn_output)

        return node_logits, decider_logits, node_mask

    def to(self, device: str | torch.device):
        self.device = device
        self.gnn.to(device)
        self.attention.to(device)
        self.decider.to(device)


class GraphIsomorphismRollout(Rollout):
    """A message exchange in the graph isomorphism task."""

    def __init__(
        self,
        message_exchange: MessageExchange,
        batch: GraphIsomorphismData | GeometricBatch,
    ):
        super().__init__(message_exchange, batch)

    def visualise(
        self,
        graph_layout_function: Optional[Callable] = None,
        graph_layout_seed: Optional[int] = None,
        colour_sequence: str = "Dark24",
        node_text_colour: str = "white",
    ):
        """Visualize the rollout as a plotly graph.

        Parameters
        ----------
        graph_layout_function : Callable, default=None
            A function which takes a networkx graph and returns a dictionary of node
            positions. Best to use a function from networkx.layout, possibly partially
            applied with some arguments. If None, uses networkx.spring_layout with
            `k=4/sqrt(n)`, where `n` is the number of nodes in the graph.
        graph_layout_seed : int, default=None
            The seed to use for the graph layout function. If None, the random number
            generator is the `RandomState` instance used by `numpy.random`.
        colour_sequence : str, default="Dark24"
            The name of the colour sequence to use to colour the nodes. Must be one of
            the colour sequences from plotly.express.colors.qualitative.
        node_text_colour : str, default="white"
            The colour of the node labels.
        """

        if graph_layout_function is None:

            def graph_layout_function(graph, *args, **kwargs):
                return nx.spring_layout(graph, k=4 / sqrt(len(graph)), *args, **kwargs)

        graph_layout_function = partial(graph_layout_function, seed=graph_layout_seed)

        # Get the colour sequence
        colour_list = px.colors.qualitative.__getattribute__(colour_sequence)

        def get_colour(i):
            return colour_list[i % len(colour_list)]

        # Add titles for the two graphs
        graph_title_trace = go.Scatter(
            text=["Graph A", "Graph B"],
            x=[0, 2.2],
            y=[1.2, 1.2],
            textfont=dict(size=20),
            mode="text",
            visible=True,
        )

        # Add labels for the exchange of messages at the bottom
        message_label_trace = go.Scatter(
            text=["Verifier messages:", "Prover messages:"],
            x=[-0.15, -0.15],
            y=[-1.3, -1.6],
            mode="text",
            textposition="middle left",
            visible=True,
        )

        traces = [
            graph_title_trace,
            message_label_trace,
        ]
        buttons = []

        for idx in range(len(self.batch)):
            # Only show the first batch item initially
            traces_visible = idx == 0

            # Get the data for this batch item as networkx graphs
            data = self.batch[idx]

            for k in ["a", "b"]:
                # Get the graph as a networkx graph
                if k == "a":
                    graph = to_networkx(
                        GeometricData(edge_index=data.edge_index_a, x=data.x_a),
                        to_undirected=True,
                    )
                else:
                    graph = to_networkx(
                        GeometricData(edge_index=data.edge_index_b, x=data.x_b),
                        to_undirected=True,
                    )

                # Generate the layouts for the graph
                graph_pos = graph_layout_function(graph)

                # Add an offset to the x coordinates of the nodes in graph B so that it
                # is to the right of graph A
                x_add = 0 if k == "a" else 2.2

                # Add the trace for the edges
                edge_x = []
                edge_y = []
                for edge in graph.edges():
                    x0, y0 = graph_pos[edge[0]]
                    x1, y1 = graph_pos[edge[1]]
                    edge_x.append(x0 + x_add)
                    edge_x.append(x1 + x_add)
                    edge_x.append(None)
                    edge_y.append(y0)
                    edge_y.append(y1)
                    edge_y.append(None)
                traces.append(
                    go.Scatter(
                        x=edge_x,
                        y=edge_y,
                        line=dict(width=0.5, color="#888"),
                        hoverinfo="none",
                        mode="lines",
                        visible=traces_visible,
                    )
                )

                # Add the trace for the nodes
                node_x = []
                node_y = []
                node_colour = []
                node_size = []
                node_text = []
                for node in graph.nodes():
                    x, y = graph_pos[node]
                    x += x_add
                    if k == "a":
                        selected_indices = (
                            (data.x_a[node] == 1).nonzero().squeeze(-1).tolist()
                        )
                    else:
                        selected_indices = (
                            (data.x_b[node] == 1).nonzero().squeeze(-1).tolist()
                        )
                    if len(selected_indices) == 0:
                        node_x.append(x)
                        node_y.append(y)
                        node_colour.append("grey")
                        node_size.append(20)
                        node_text.append(str(node))
                    else:
                        for i, j in reversed(list(enumerate(selected_indices))):
                            node_x.append(x)
                            node_y.append(y)
                            node_colour.append(get_colour(j))
                            node_size.append(10 * i + 20)
                            node_text.append(str(node))
                traces.append(
                    go.Scatter(
                        x=node_x,
                        y=node_y,
                        text=node_text,
                        textfont=dict(color=node_text_colour),
                        mode="markers+text",
                        hoverinfo="text",
                        marker=dict(
                            color=node_colour,
                            size=node_size,
                            line_width=0,
                            opacity=1,
                        ),
                        visible=traces_visible,
                    )
                )

            # Add the trace for the timeline of the messages exchanged
            timeline_node_x = []
            timeline_node_y = []
            timeline_node_text = []
            timeline_node_colour = []
            for i, message in enumerate(self.message_exchange):
                round = i // 2
                x = round * 0.25
                y = -1.3 if message.from_verifier else -1.6
                timeline_node_x.append(x)
                timeline_node_y.append(y)
                graph_letter = "A" if message.message[idx, 0].item() == 1 else "B"
                timeline_node_text.append(
                    f"{graph_letter}{message.message[idx, 1].item()}"
                )
                timeline_node_colour.append(get_colour(round))
            traces.append(
                go.Scatter(
                    x=timeline_node_x,
                    y=timeline_node_y,
                    text=timeline_node_text,
                    textfont=dict(color=node_text_colour),
                    mode="markers+text",
                    hoverinfo="text",
                    marker=dict(
                        color=timeline_node_colour,
                        size=30,
                        line_width=0,
                        opacity=1,
                    ),
                    visible=traces_visible,
                )
            )

            # Add a button to show this batch item
            buttons.append(
                dict(
                    label=f"Batch item {idx}",
                    method="update",
                    args=[
                        {
                            "visible": [True, True]
                            + [False] * (5 * idx)
                            + [True] * 5
                            + [False] * (5 * (len(self.batch) - idx - 1))
                        }
                    ],
                )
            )

        layout = go.Layout(
            title="Prover-Verifier messages",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=5, l=5, r=5, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            updatemenus=[dict(buttons=buttons, active=0, showactive=True)],
        )

        fig = go.Figure(data=traces, layout=layout)
        fig.show()


@dataclass
class GraphIsomorphismMessage(Message):
    """A message sent between the prover and the verifier."""

    from_verifier: bool
    message: Float[Tensor, "batch_size 2"]
    verifier_guess: Float[Tensor, "batch_size"] = None


class GraphIsomorphismScenario(Scenario):
    """The graph isomorphism scenario."""

    def __init__(self, params: Parameters, device: str | torch.device):
        super().__init__(params, device)
        self.prover = GraphIsomorphismProver(params, device)
        self.verifier = GraphIsomorphismVerifier(params, device)
        self.dataset = GraphIsomorphismDataset(params)

    def rollout(
        self, data: GraphIsomorphismData | GeometricBatch
    ) -> GraphIsomorphismRollout:
        message_exchange = MessageExchange()
        max_nodes_a = torch.bincount(data.x_a_batch).max().item()
        max_nodes_b = torch.bincount(data.x_b_batch).max().item()
        for round in range(self.params.max_message_rounds):
            ## Verifier sends a message
            # (batch_size, max_nodes_a+max_nodes_b), (batch_size, 3), (batch_size,
            # max_nodes_a+max_nodes_b)
            node_logits, decider_logits, node_mask = self.verifier(data)

            # (batch_size, max_nodes_a+max_nodes_b)
            node_probs = F.softmax(node_logits, dim=-1)
            node_probs[~node_mask] = 0

            # (batch_size)
            verifier_node = Categorical(probs=node_probs).sample()

            # Extract which graph the verifier node is from and the index of the node
            # within that graph
            verifier_node_graph = (verifier_node < max_nodes_a).to(torch.int64)
            verifier_node = torch.where(
                verifier_node_graph == 1,
                verifier_node,
                verifier_node - max_nodes_a,
            )
            verifier_message = torch.stack((verifier_node_graph, verifier_node), dim=-1)

            # (batch_size)
            verifier_guess = Categorical(logits=decider_logits).sample()

            message_exchange.append(
                GraphIsomorphismMessage(
                    from_verifier=True,
                    message=verifier_message,
                    verifier_guess=verifier_guess,
                )
            )

            # One-hot encode the verifier's message in the node features
            node_index_a = verifier_node + data.x_a_ptr[:-1]
            node_index_a = node_index_a[verifier_node_graph == 1]
            node_index_b = verifier_node + data.x_b_ptr[:-1]
            node_index_b = node_index_b[verifier_node_graph == 0]
            data.x_a[node_index_a, round] = 1
            data.x_b[node_index_b, round] = 1

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
                (verifier_node_graph == 1).unsqueeze(-1),
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

            # Extract which graph the prover node is from and the index of the node
            # within that graph
            prover_node_graph = (prover_node < max_nodes_a).to(torch.int64)
            prover_node = torch.where(
                prover_node_graph == 1,
                prover_node,
                prover_node - max_nodes_a,
            )
            prover_message = torch.stack((prover_node_graph, prover_node), dim=-1)

            message_exchange.append(
                GraphIsomorphismMessage(
                    from_verifier=False,
                    message=prover_message,
                )
            )

            # One-hot encode the prover's message in the node features
            node_index_a = prover_node + data.x_a_ptr[:-1]
            node_index_a = node_index_a[prover_node_graph == 1]
            node_index_b = prover_node + data.x_b_ptr[:-1]
            node_index_b = node_index_b[prover_node_graph == 0]
            data.x_a[node_index_a, round] = 1
            data.x_b[node_index_b, round] = 1

        return GraphIsomorphismRollout(message_exchange, data)
