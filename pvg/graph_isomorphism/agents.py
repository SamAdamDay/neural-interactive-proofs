from abc import ABC
from typing import Optional, Callable
from dataclasses import dataclass
from math import sqrt
from functools import partial
from collections import OrderedDict

import torch
from torch.nn import (
    ReLU,
    Linear,
    TransformerEncoder,
    TransformerEncoderLayer,
    Sequential,
    BatchNorm1d,
)
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Categorical

from tensordict import TensorDictBase, TensorDict

from torch_geometric.utils import to_networkx
from torch_geometric.data import Batch as GeometricBatch, Data as GeometricData

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

from jaxtyping import Float, Bool

import plotly.graph_objects as go
import plotly.express as px

import networkx as nx

from pvg.base import (
    AgentPart,
    AgentBody,
    AgentHead,
    AgentPolicyHead,
    AgentCriticHead,
)
from pvg.parameters import Parameters, GraphIsomorphismAgentParameters
from pvg.graph_isomorphism.data import GraphIsomorphismData, GraphIsomorphismDataset
from pvg.utils.torch_modules import (
    PairedGaussianNoise,
    PairInvariantizer,
    GIN,
    TensorDictize,
    SelectTensorDictValue,
    SequentialKwargs,
    Print,
)
from pvg.utils.data import gi_data_to_tensordict


class GraphIsomorphismAgentPart(AgentPart, ABC):
    """Base class for all graph isomorphism agent parts.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    agent_name : str
        The name of the agent. Must be either "prover" or "verifier".
    device : str or torch.device, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    def __init__(
        self,
        params: Parameters,
        agent_name: str,
        device: Optional[str | torch.device] = None,
    ):
        super().__init__(params, device)
        self.agent_name = agent_name

        if agent_name == "prover":
            self._agent_params = params.graph_isomorphism.prover
        elif agent_name == "verifier":
            self._agent_params = params.graph_isomorphism.verifier
        else:
            raise ValueError(f"Unknown agent name: {agent_name}")


class GraphIsomorphismAgentBody(GraphIsomorphismAgentPart, AgentBody):
    """Agent body for the graph isomorphism task.

    Takes as input a pair of graphs and outputs node-level and graph-level
    representations.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    agent_name : str
        The name of the agent. Must be either "prover" or "verifier".
    device : str or torch.device, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    def __init__(
        self,
        params: Parameters,
        agent_name: str,
        device: Optional[str | torch.device] = None,
    ):
        super().__init__(params, agent_name, device)

        # Build up the GNN module
        self.gnn = self._build_gnn(
            agent_params=self._agent_params,
            d_input=self.params.max_message_rounds,
        )

        # Build the global pooling module, which computes the graph-level representation
        # from the GNN output
        self.global_pooling = self._build_global_pooling(
            agent_params=self._agent_params
        )

        # Build the encoder going from the GNN to the transformer
        self.gnn_transformer_encoder = self._build_gnn_transformer_encoder(
            agent_params=self._agent_params
        )

        # Build the transformer
        self.transformer = self._build_transformer(agent_params=self._agent_params)

    @staticmethod
    def _build_gnn(
        agent_params: GraphIsomorphismAgentParameters, d_input: int
    ) -> SequentialKwargs:
        """Builds the GNN module for a prover or verifier.

        Parameters
        ----------
        agent_params : GraphIsomorphismAgentParameters
            The parameters used for constructing the agent
        d_input : int
            The dimensionality of the input features.

        Returns
        -------
        gnn : torch.nn.Sequential
            The GNN module, which takes as input a TensorDict with keys "x", "adjacency"
            and "node_mask".
        transformer : torch.nn.TransformerEncoder
            The transformer module.
        """
        # Build up the GNN
        gnn_layers = OrderedDict()
        gnn_layers["input"] = TensorDictize(
            Linear(d_input, agent_params.d_gnn), key="x"
        )
        for i in range(agent_params.num_gnn_layers):
            gnn_layers[f"ReLU_{i}"] = TensorDictize(ReLU(inplace=True), key="x")
            gnn_layers[f"GNN_layer_{i}"] = GIN(
                Sequential(
                    Linear(
                        agent_params.d_gnn,
                        agent_params.d_gin_mlp,
                    ),
                    ReLU(inplace=True),
                    Linear(
                        agent_params.d_gin_mlp,
                        agent_params.d_gnn,
                    ),
                )
            )
        gnn = SequentialKwargs(gnn_layers)

        return gnn

    @staticmethod
    def _build_gnn_transformer_encoder(
        agent_params: GraphIsomorphismAgentParameters,
    ) -> Linear:
        """Build the encoder layer which translates the GNN output to transformer input

        This is a simple linear layer, where the number of input features is `d_gnn` +
        3, where the extra features encode which graph-level representation the token
        is, if any and whether a node is in the most recent message from the other
        agent.

        Parameters
        ----------
        agent_params : GraphIsomorphismAgentParameters
            The parameters used for constructing the agent

        Returns
        -------
        gnn_transformer_encoder : torch.nn.Linear
            The encoder module

        """
        return Linear(agent_params.d_gnn + 3, agent_params.d_transformer)

    @staticmethod
    def _build_transformer(
        agent_params: GraphIsomorphismAgentParameters,
    ) -> TransformerEncoder:
        """Builds the transformer module for a prover or verifier.

        Parameters
        ----------
        agent_params : GraphIsomorphismAgentParameters
            The parameters used for constructing the agent

        Returns
        -------
        transformer : torch.nn.TransformerEncoder
            The transformer module.
        """

        transformer = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=agent_params.d_transformer,
                nhead=agent_params.num_heads,
                batch_first=True,
                dropout=agent_params.transformer_dropout,
            ),
            num_layers=agent_params.num_transformer_layers,
        )

        return transformer

    @staticmethod
    def _build_global_pooling(
        agent_params: GraphIsomorphismAgentParameters,
    ) -> Sequential:
        """Builds a pooling layer which computes the graph-level representation.

        The module consists of a global sum pooling layer, an optional batch norm layer,
        a paired Gaussian noise layer and an optional pair invariant pooling layer.

        Parameters
        ----------
        agent_params : GraphIsomorphismAgentParameters
            The parameters used for constructing the agent

        Returns
        -------
        global_pooling : torch.nn.Sequential
            The global pooling module.
        """
        layers = [
            Reduce(
                "batch pair max_nodes d_gnn -> batch pair d_gnn",
                "sum",
            ),
        ]
        if agent_params.use_batch_norm:
            layers.extend(
                [
                    Rearrange("batch pair d_gnn -> (batch pair) d_gnn"),
                    BatchNorm1d(num_features=agent_params.d_gnn),
                    Rearrange(
                        "(batch pair) d_gnn -> batch pair d_gnn",
                        pair=2,
                    ),
                ]
            )
        layers.append(
            PairedGaussianNoise(sigma=agent_params.noise_sigma, pair_dim=1),
        )
        if agent_params.use_pair_invariant_pooling:
            layers.append(PairInvariantizer(pair_dim=1))
        return Sequential(*layers)

    def forward(
        self,
        data: TensorDictBase | GraphIsomorphismData | GeometricBatch,
    ) -> TensorDict:
        """Obtain graph-level and node-level representations by running components

        Runs the GNN, pools the output, puts everything through a linear encoder, then
        runs the transformer on this.

        Parameters
        ----------
        data : TensorDictBase | GraphIsomorphismData | GraphIsomorphismDataBatch
            The data to run the GNN and transformer on. Either a TensorDictBase with
            keys:

            - "x" (batch pair node feature): The graph node features (message history)
            - "adjacency" (batch pair node node): The graph adjacency matrices
            - "message" (batch): (optional) (optional) The most recent message from the
              other agent
            - "node_mask" (batch pair node): Which nodes actually exist
            or a GraphIsomorphism data object.

        Returns
        -------
        out : TensorDict
            A tensor dict with keys:

            - "graph_level_repr" (batch 2 d_transformer): The output graph-level
              representations.
            - "node_level_repr" (batch 2 max_nodes d_transformer): The output node-level
              representations.
            - "node_mask" : (batch 2*max_nodes): A mask indicating which nodes are
              present in the graphs.
        """

        # Convert the data to a TensorDict with dense representations
        if not isinstance(data, TensorDictBase):
            data = gi_data_to_tensordict(data)

        # The size of the node dimension
        max_num_nodes = data["x"].shape[2]

        # Run the GNN on the graphs
        # (batch, pair, max_nodes, d_gnn)
        gnn_output = self.gnn(data)["x"]

        # Obtain the graph-level representations by pooling
        # (batch, pair, d_gnn)
        pooled_gnn_output = self.global_pooling(gnn_output)

        # Flatten the two batch dimensions in the graph representation and mask
        gnn_output_flatter = rearrange(
            gnn_output, "batch pair node feature -> batch (pair node) feature"
        )
        node_mask_flatter = rearrange(
            data["node_mask"], "batch pair node -> batch (pair node)"
        )

        # Add the graph-level representations to the transformer input with an extra
        # features which distinguishes them, and add (optional) the most recent message
        # (batch, 2 + 2 * node, d_gnn + 3)
        transformer_input = torch.cat((pooled_gnn_output, gnn_output_flatter), dim=1)
        pooled_feature = torch.zeros(
            *transformer_input.shape[:-1],
            2,
            device=transformer_input.device,
            dtype=transformer_input.dtype,
        )
        pooled_feature[:, 0, 0] = 1
        pooled_feature[:, 1, 1] = 1
        if "message" in data.keys():
            message_feature = F.one_hot(
                data["message"] + 2, num_classes=2 + 2 * max_num_nodes
            )
            message_feature = rearrange(message_feature, "batch token -> batch token 1")
        else:
            message_feature = torch.zeros(
                *transformer_input.shape[:-1],
                1,
                device=transformer_input.device,
                dtype=transformer_input.dtype,
            )
        transformer_input = torch.cat(
            (transformer_input, pooled_feature, message_feature), dim=-1
        )

        # Run the transformer input through the encoder first
        # (batch, 2 + 2 * node, d_transformer)
        transformer_input = self.gnn_transformer_encoder(transformer_input)

        # Create the padding mask so that the transformer only attends to the actual
        # nodes (and the pooled representations)
        # (batch, 2 + 2 * node)
        padding_mask = ~node_mask_flatter
        padding_mask = torch.cat(
            (
                torch.ones(
                    (padding_mask.shape[0], 2), device=padding_mask.device, dtype=bool
                ),
                padding_mask,
            ),
            dim=-1,
        )

        # The attention mask applied to all batch elements, which makes sure that nodes
        # only attend to nodes in the other graph and to the pooled representations.
        src_mask = torch.ones(
            (2 + 2 * max_num_nodes,) * 2, device=padding_mask.device, dtype=bool
        )
        src_mask[2 : 2 + max_num_nodes, 2 : 2 + max_num_nodes] = 0
        src_mask[2 + max_num_nodes :, 2 + max_num_nodes :] = 0

        # Compute the transformer output
        # (batch, 2 + 2 * max_nodes, d_transformer)
        transformer_output_flatter = self.transformer(
            transformer_input,
            mask=src_mask,
            src_key_padding_mask=padding_mask,
            is_causal=False,
        )

        # Extract the graph-level representations and rearrange the rest to have two
        # batch dims
        graph_level_repr = transformer_output_flatter[:, :2]
        node_level_repr = rearrange(
            transformer_output_flatter[:, 2:],
            "batch (pair node) feature -> batch pair node feature",
            pair=2,
        )

        return TensorDict(
            dict(
                graph_level_repr=graph_level_repr,
                node_level_repr=node_level_repr,
                node_mask_flatter=node_mask_flatter,
            ),
            batch_size=graph_level_repr.shape[0],
        )

    def to(self, device: Optional[str | torch.device] = None):
        super().to(device)
        self.device = device
        self.gnn.to(device)
        self.transformer.to(device)
        self.global_pooling.to(device)
        self.global_pooling[-1].to(device)
        return self


class GraphIsomorphismAgentHead(GraphIsomorphismAgentPart, AgentHead, ABC):
    """Base class for all graph isomorphism agent heads.

    This class provides some utility methods for constructing and running the various
    modules.
    """

    @staticmethod
    def _build_node_level_mlp(
        d_in: int,
        d_hidden: int,
        d_out: int,
        num_layers: int,
    ) -> Sequential:
        """Builds an MLP which acts on the node-level representations.

        This takes as input a TensorDict with key "node_level_repr" and outputs a
        Tensor.

        Parameters
        ----------
        d_in : int
            The dimensionality of the input.
        d_hidden : int
            The dimensionality of the hidden layers.
        d_out : int
            The dimensionality of the output.
        num_layers : int
            The number of hidden layers in the MLP.

        Returns
        -------
        node_level_mlp : torch.nn.Sequential
            The node-level MLP.
        """
        layers = []

        # Select the node-level representations
        layers.append(SelectTensorDictValue(key="node_level_repr"))

        # The layers of the MLP
        layers.append(Linear(d_in, d_hidden))
        layers.append(ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(Linear(d_hidden, d_hidden))
            layers.append(ReLU(inplace=True))
        layers.append(Linear(d_hidden, d_out))

        # Concatenate the pair and node dimensions
        layers.append(Rearrange("batch pair node d_out -> batch (pair node) d_out"))

        return Sequential(*layers)

    @staticmethod
    def _build_graph_level_mlp(
        d_in: int,
        d_hidden: int,
        d_out: int,
        num_layers: int,
    ) -> Sequential:
        """Builds an MLP which acts on the node-level representations.

        This takes as input a TensorDict with key "graph_level_repr" and outputs a
        Tensor.

        Parameters
        ----------
        d_in : int
            The dimensionality of the input.
        d_hidden : int
            The dimensionality of the hidden layers.
        d_out : int
            The dimensionality of the output.
        num_layers : int
            The number of hidden layers in the MLP.

        Returns
        -------
        node_level_mlp : torch.nn.Sequential
            The node-level MLP.
        """
        layers = []

        # Select the graph-level representations
        layers.append(SelectTensorDictValue(key="graph_level_repr"))

        # Concatenate the two graph-level representations
        layers.append(Rearrange("batch pair d_in -> batch (pair d_in)"))

        # The layers of the MLP
        layers.append(Linear(d_in, d_hidden))
        layers.append(ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(Linear(d_hidden, d_hidden))
            layers.append(ReLU(inplace=True))
        layers.append(Linear(d_hidden, d_out))

        return Sequential(*layers)


class GraphIsomorphismAgentPolicyHead(GraphIsomorphismAgentHead, AgentPolicyHead):
    """Agent policy head for the graph isomorphism task.

    Takes as input the output of the agent body and outputs a policy distribution over
    the actions. Both agents select a node to send as a message, and the verifier also
    decides whether to continue exchanging messages or to guess that the graphs are
    isomorphic or not.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    agent_name : str
        The name of the agent. Must be either "prover" or "verifier".
    device : str or torch.device, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    def __init__(
        self,
        params: Parameters,
        agent_name: str,
        device: Optional[str | torch.device] = None,
    ):
        super().__init__(params, agent_name, device)

        # Build the node selector module
        self.node_selector = self._build_node_selector(agent_params=self._agent_params)

        # Build the decider module if necessary
        if agent_name == "verifier":
            self.decider = self._build_decider(agent_params=self._agent_params)
        else:
            self.decider = None

    @classmethod
    def _build_node_selector(
        cls,
        agent_params: GraphIsomorphismAgentParameters,
    ) -> Sequential:
        """Builds the module which selects which node to send as a message.

        Parameters
        ----------
        agent_params : GraphIsomorphismAgentParameters
            The parameters used for constructing the agent

        Returns
        -------
        node_selector : torch.nn.Sequential
            The node selector module.
        """
        return cls._build_node_level_mlp(
            d_in=agent_params.d_transformer,
            d_hidden=agent_params.d_node_selector,
            d_out=1,
            num_layers=agent_params.num_node_selector_layers,
        )

    @classmethod
    def _build_decider(
        cls,
        agent_params: GraphIsomorphismAgentParameters,
        d_out: int = 3,
    ) -> Sequential:
        """Builds the module which produces a graph-pair level output.

        By default it is used to decide whether to continue exchanging messages. In this
        case it outputs a single triple of logits for the three options: guess that the
        graphs are not isomorphic, guess that the graphs are isomorphic, or continue
        exchanging messages.

        Parameters
        ----------
        agent_params : GraphIsomorphismAgentParameters
            The parameters used for constructing the agent
        d_out : int, default=3
            The dimensionality of the output.

        Returns
        -------
        decider : torch.nn.Sequential
            The decider module.
        """
        return cls._build_graph_level_mlp(
            d_in=agent_params.d_transformer,
            d_hidden=agent_params.d_decider,
            d_out=d_out,
            num_layers=agent_params.num_decider_layers,
        )

    def forward(self, body_output: TensorDict) -> TensorDict:
        """Runs the policy head on the given body output.

        Runs the node selector module and the decider module if present.

        Parameters
        ----------
        body_output : TensorDict
            The output of the body module. A tensor dict with keys:

            - "graph_level_repr" (batch 2 d_transformer): The output graph-level
              representations.
            - "node_level_repr" (batch 2 max_nodes d_transformer): The output node-level
              representations.
            - (optional) "node_mask" : (batch 2*max_nodes): A mask indicating which
              nodes are present in the graphs.

        Returns
        -------
        out : TensorDict
            A tensor dict with keys:

            - "node_selected_logits" (batch 2*max_nodes): A logit for each node,
              indicating the probability that this node should be sent as a message to
              the verifier.
            - (optional) "decider_logits" (batch 3): A logit for each of the three
              options: continue exchanging messages, guess that the graphs are
              isomorphic, or guess that the graphs are not isomorphic.
            - (optional) "node_mask" : (batch 2*max_nodes): The node mask from the body
              output.
        """

        out_dict = {}

        out_dict["node_selected_logits"] = self.node_selector(body_output).squeeze(-1)

        if self.decider is not None:
            out_dict["decider_logits"] = self.decider(body_output)

        if "node_mask" in body_output.keys():
            out_dict["node_mask"] = body_output["node_mask"]

        return TensorDict(
            out_dict, batch_size=out_dict["node_selected_logits"].shape[0]
        )


class GraphIsomorphismAgentCriticHead(GraphIsomorphismAgentHead, AgentCriticHead):
    """Critic head for the graph isomorphism task.

    Takes as input the output of the agent body and outputs a value function.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    agent_name : str
        The name of the agent. Must be either "prover" or "verifier".
    device : str or torch.device, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    def __init__(
        self,
        params: Parameters,
        agent_name: str,
        device: Optional[str | torch.device] = None,
    ):
        super().__init__(params, agent_name, device)

        self.critic_mlp = self._build_critic_mlp(agent_params=self._agent_params)

    @classmethod
    def _build_critic_mlp(
        cls,
        agent_params: GraphIsomorphismAgentParameters,
    ) -> Sequential:
        """Builds the module which computes the value function.

        Parameters
        ----------
        agent_params : GraphIsomorphismAgentParameters
            The parameters used for constructing the agent
        d_out : int, default=3
            The dimensionality of the output.

        Returns
        -------
        critic_mlp : torch.nn.Sequential
            The critic module.
        """
        return cls._build_graph_level_mlp(
            d_in=agent_params.d_transformer,
            d_hidden=agent_params.d_critic,
            d_out=1,
            num_layers=agent_params.num_critic_layers,
        )


# class GraphIsomorphismRollout(Rollout):
#     """A message exchange in the graph isomorphism task."""

#     def __init__(
#         self,
#         message_exchange: MessageExchange,
#         batch: GraphIsomorphismData | GeometricBatch,
#     ):
#         super().__init__(message_exchange, batch)

#     def visualise(
#         self,
#         graph_layout_function: Optional[Callable] = None,
#         graph_layout_seed: Optional[int] = None,
#         colour_sequence: str = "Dark24",
#         node_text_colour: str = "white",
#     ):
#         """Visualize the rollout as a plotly graph.

#         Parameters
#         ----------
#         graph_layout_function : Callable, default=None
#             A function which takes a networkx graph and returns a dictionary of node
#             positions. Best to use a function from networkx.layout, possibly partially
#             applied with some arguments. If None, uses networkx.spring_layout with
#             `k=4/sqrt(n)`, where `n` is the number of nodes in the graph.
#         graph_layout_seed : int, default=None
#             The seed to use for the graph layout function. If None, the random number
#             generator is the `RandomState` instance used by `numpy.random`.
#         colour_sequence : str, default="Dark24"
#             The name of the colour sequence to use to colour the nodes. Must be one of
#             the colour sequences from plotly.express.colors.qualitative.
#         node_text_colour : str, default="white"
#             The colour of the node labels.
#         """

#         if graph_layout_function is None:

#             def graph_layout_function(graph, *args, **kwargs):
#                 return nx.spring_layout(graph, k=4 / sqrt(len(graph)), *args, **kwargs)

#         graph_layout_function = partial(graph_layout_function, seed=graph_layout_seed)

#         # Get the colour sequence
#         colour_list = px.colors.qualitative.__getattribute__(colour_sequence)

#         def get_colour(i):
#             return colour_list[i % len(colour_list)]

#         # Add titles for the two graphs
#         graph_title_trace = go.Scatter(
#             text=["Graph A", "Graph B"],
#             x=[0, 2.2],
#             y=[1.2, 1.2],
#             textfont=dict(size=20),
#             mode="text",
#             visible=True,
#         )

#         # Add labels for the exchange of messages at the bottom
#         message_label_trace = go.Scatter(
#             text=["Verifier messages:", "Prover messages:"],
#             x=[-0.15, -0.15],
#             y=[-1.3, -1.6],
#             mode="text",
#             textposition="middle left",
#             visible=True,
#         )

#         traces = [
#             graph_title_trace,
#             message_label_trace,
#         ]
#         buttons = []

#         for idx in range(len(self.batch)):
#             # Only show the first batch item initially
#             traces_visible = idx == 0

#             # Get the data for this batch item as networkx graphs
#             data = self.batch[idx]

#             for k in ["a", "b"]:
#                 # Get the graph as a networkx graph
#                 if k == "a":
#                     graph = to_networkx(
#                         GeometricData(edge_index=data.edge_index_a, x=data.x_a),
#                         to_undirected=True,
#                     )
#                 else:
#                     graph = to_networkx(
#                         GeometricData(edge_index=data.edge_index_b, x=data.x_b),
#                         to_undirected=True,
#                     )

#                 # Generate the layouts for the graph
#                 graph_pos = graph_layout_function(graph)

#                 # Add an offset to the x coordinates of the nodes in graph B so that it
#                 # is to the right of graph A
#                 x_add = 0 if k == "a" else 2.2

#                 # Add the trace for the edges
#                 edge_x = []
#                 edge_y = []
#                 for edge in graph.edges():
#                     x0, y0 = graph_pos[edge[0]]
#                     x1, y1 = graph_pos[edge[1]]
#                     edge_x.append(x0 + x_add)
#                     edge_x.append(x1 + x_add)
#                     edge_x.append(None)
#                     edge_y.append(y0)
#                     edge_y.append(y1)
#                     edge_y.append(None)
#                 traces.append(
#                     go.Scatter(
#                         x=edge_x,
#                         y=edge_y,
#                         line=dict(width=0.5, color="#888"),
#                         hoverinfo="none",
#                         mode="lines",
#                         visible=traces_visible,
#                     )
#                 )

#                 # Add the trace for the nodes
#                 node_x = []
#                 node_y = []
#                 node_colour = []
#                 node_size = []
#                 node_text = []
#                 for node in graph.nodes():
#                     x, y = graph_pos[node]
#                     x += x_add
#                     if k == "a":
#                         selected_indices = (
#                             (data.x_a[node] == 1).nonzero().squeeze(-1).tolist()
#                         )
#                     else:
#                         selected_indices = (
#                             (data.x_b[node] == 1).nonzero().squeeze(-1).tolist()
#                         )
#                     if len(selected_indices) == 0:
#                         node_x.append(x)
#                         node_y.append(y)
#                         node_colour.append("grey")
#                         node_size.append(20)
#                         node_text.append(str(node))
#                     else:
#                         for i, j in reversed(list(enumerate(selected_indices))):
#                             node_x.append(x)
#                             node_y.append(y)
#                             node_colour.append(get_colour(j))
#                             node_size.append(10 * i + 20)
#                             node_text.append(str(node))
#                 traces.append(
#                     go.Scatter(
#                         x=node_x,
#                         y=node_y,
#                         text=node_text,
#                         textfont=dict(color=node_text_colour),
#                         mode="markers+text",
#                         hoverinfo="text",
#                         marker=dict(
#                             color=node_colour,
#                             size=node_size,
#                             line_width=0,
#                             opacity=1,
#                         ),
#                         visible=traces_visible,
#                     )
#                 )

#             # Add the trace for the timeline of the messages exchanged
#             timeline_node_x = []
#             timeline_node_y = []
#             timeline_node_text = []
#             timeline_node_colour = []
#             for i, message in enumerate(self.message_exchange):
#                 round = i // 2
#                 x = round * 0.25
#                 y = -1.3 if message.from_verifier else -1.6
#                 timeline_node_x.append(x)
#                 timeline_node_y.append(y)
#                 graph_letter = "A" if message.message[idx, 0].item() == 1 else "B"
#                 timeline_node_text.append(
#                     f"{graph_letter}{message.message[idx, 1].item()}"
#                 )
#                 timeline_node_colour.append(get_colour(round))
#             traces.append(
#                 go.Scatter(
#                     x=timeline_node_x,
#                     y=timeline_node_y,
#                     text=timeline_node_text,
#                     textfont=dict(color=node_text_colour),
#                     mode="markers+text",
#                     hoverinfo="text",
#                     marker=dict(
#                         color=timeline_node_colour,
#                         size=30,
#                         line_width=0,
#                         opacity=1,
#                     ),
#                     visible=traces_visible,
#                 )
#             )

#             # Add a button to show this batch item
#             buttons.append(
#                 dict(
#                     label=f"Batch item {idx}",
#                     method="update",
#                     args=[
#                         {
#                             "visible": [True, True]
#                             + [False] * (5 * idx)
#                             + [True] * 5
#                             + [False] * (5 * (len(self.batch) - idx - 1))
#                         }
#                     ],
#                 )
#             )

#         layout = go.Layout(
#             title="Prover-Verifier messages",
#             showlegend=False,
#             hovermode="closest",
#             margin=dict(b=5, l=5, r=5, t=50),
#             xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#             updatemenus=[dict(buttons=buttons, active=0, showactive=True)],
#         )

#         fig = go.Figure(data=traces, layout=layout)
#         fig.show()


# class GraphIsomorphismScenario(Scenario):
#     """The graph isomorphism scenario."""

#     def __init__(self, params: Parameters, device: Optional[str | torch.device] = None):
#         super().__init__(params, device)
#         self.prover = GraphIsomorphismProverBody(params, device)
#         self.verifier = GraphIsomorphismVerifierBody(params, device)
#         self.dataset = GraphIsomorphismDataset(params)

#     def rollout(
#         self, data: GraphIsomorphismData | GeometricBatch
#     ) -> GraphIsomorphismRollout:
#         message_exchange = MessageExchange()
#         max_nodes_a = torch.bincount(data.x_a_batch).max().item()
#         max_nodes_b = torch.bincount(data.x_b_batch).max().item()
#         for round in range(self.params.max_message_rounds):
#             ## Verifier sends a message
#             # (batch, 2*max_nodes), (batch, 3), (batch,
#             # 2*max_nodes)
#             node_selected_logits, decider_logits, node_mask = self.verifier(data)

#             # (batch, 2*max_nodes)
#             node_probs = F.softmax(node_selected_logits, dim=-1)
#             node_probs[~node_mask] = 0

#             # (batch)
#             verifier_node = Categorical(probs=node_probs).sample()

#             # Extract which graph the verifier node is from and the index of the node
#             # within that graph
#             verifier_node_graph = (verifier_node < max_nodes_a).to(torch.int64)
#             verifier_node = torch.where(
#                 verifier_node_graph == 1,
#                 verifier_node,
#                 verifier_node - max_nodes_a,
#             )
#             verifier_message = torch.stack((verifier_node_graph, verifier_node), dim=-1)

#             # (batch)
#             verifier_guess = Categorical(logits=decider_logits).sample()

#             message_exchange.append(
#                 GraphIsomorphismMessage(
#                     from_verifier=True,
#                     message=verifier_message,
#                     verifier_guess=verifier_guess,
#                 )
#             )

#             # One-hot encode the verifier's message in the node features
#             node_index_a = verifier_node + data.x_a_ptr[:-1]
#             node_index_a = node_index_a[verifier_node_graph == 1]
#             node_index_b = verifier_node + data.x_b_ptr[:-1]
#             node_index_b = node_index_b[verifier_node_graph == 0]
#             data.x_a[node_index_a, round] = 1
#             data.x_b[node_index_b, round] = 1

#             ## Prover sends a message
#             # (batch, 2*max_nodes), (batch,
#             # 2*max_nodes)
#             node_selected_logits, node_mask = self.prover(data)

#             # (batch, 2*max_nodes)
#             node_probs = F.softmax(node_selected_logits, dim=-1)
#             node_probs[~node_mask] = 0

#             # Make sure that the prover chooses a node in the opposite graph to the one
#             # chosen by the verifier
#             node_probs = torch.where(
#                 (verifier_node_graph == 1).unsqueeze(-1),
#                 torch.cat(
#                     (
#                         torch.zeros(node_probs.shape[0], max_nodes_a),
#                         node_probs[:, max_nodes_a:],
#                     ),
#                     dim=-1,
#                 ),
#                 torch.cat(
#                     (
#                         node_probs[:, :max_nodes_a],
#                         torch.zeros(node_probs.shape[0], max_nodes_b),
#                     ),
#                     dim=-1,
#                 ),
#             )

#             # (batch)
#             prover_node = Categorical(probs=node_probs).sample()

#             # Extract which graph the prover node is from and the index of the node
#             # within that graph
#             prover_node_graph = (prover_node < max_nodes_a).to(torch.int64)
#             prover_node = torch.where(
#                 prover_node_graph == 1,
#                 prover_node,
#                 prover_node - max_nodes_a,
#             )
#             prover_message = torch.stack((prover_node_graph, prover_node), dim=-1)

#             message_exchange.append(
#                 GraphIsomorphismMessage(
#                     from_verifier=False,
#                     message=prover_message,
#                 )
#             )

#             # One-hot encode the prover's message in the node features
#             node_index_a = prover_node + data.x_a_ptr[:-1]
#             node_index_a = node_index_a[prover_node_graph == 1]
#             node_index_b = prover_node + data.x_b_ptr[:-1]
#             node_index_b = node_index_b[prover_node_graph == 0]
#             data.x_a[node_index_a, round] = 1
#             data.x_b[node_index_b, round] = 1

#         return GraphIsomorphismRollout(message_exchange, data)
