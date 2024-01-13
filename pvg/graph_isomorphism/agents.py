"""Graph isomorphism agents components.

Contains classes for building agent bodies and heads for the graph isomorphism task.
"""

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
)
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Categorical

from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

from torch_geometric.utils import to_networkx

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from jaxtyping import Float, Bool, Int

import plotly.graph_objects as go
import plotly.express as px

import networkx as nx

from pvg.constants import PROVER_AGENT_NUM, VERIFIER_AGENT_NUM
from pvg.scenario_base import (
    AgentPart,
    AgentBody,
    AgentHead,
    DummyAgentBody,
    AgentPolicyHead,
    RandomAgentPolicyHead,
    AgentValueHead,
    ConstantAgentValueHead,
    AgentCriticHead,
    SoloAgentHead,
    CombinedBody,
    CombinedPolicyHead,
    CombinedValueHead,
)
from pvg.parameters import Parameters, GraphIsomorphismAgentParameters
from pvg.utils.torch_modules import (
    PairedGaussianNoise,
    PairInvariantizer,
    GIN,
    Squeeze,
    BatchNorm1dBatchDims,
    Print,
)
from pvg.utils.types import TorchDevice


class GraphIsomorphismAgentPart(AgentPart, ABC):
    """Base class for all graph isomorphism agent parts.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    agent_name : str
        The name of the agent.
    device : TorchDevice, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    def __init__(
        self,
        params: Parameters,
        agent_name: str,
        device: Optional[TorchDevice] = None,
    ):
        super().__init__(params, device)
        self.agent_name = agent_name

        self._agent_params: GraphIsomorphismAgentParameters = params.agents[agent_name]
        for i, _agent_name in enumerate(params.agents):
            if _agent_name == agent_name:
                self.agent_index = i
                break

    @classmethod
    def _run_masked_transformer(
        cls,
        transformer: TransformerEncoder,
        transformer_input: Float[Tensor, "... 2+2*node d_transformer"],
        node_mask: Float[Tensor, "... pair node"],
    ) -> Float[Tensor, "... 2+2*node d_transformer"]:
        """Run a transformer on graph and node representations, with masking.

        The input is expected to be the concatenation of the two graph-level
        representations and the node-level representations.

        Attention is masked so that nodes only attend to nodes in the other graph and
        to the pooled representations. We also make sure that the transformer only
        attends to the actual nodes (and the pooled representations).

        Parameters
        ----------
        transformer : torch.nn.TransformerEncoder
            The transformer module.
        transformer_input : Float[Tensor, "... 2+2*node d_transformer"]
            The input to the transformer. This is expected to be the concatenation of
            the two graph-level representations and the node-level representations.
        node_mask : Float[Tensor, "... pair node"]
            Which nodes actually exist.

        Returns
        -------
        transformer_output_flatter : Float[Tensor, "... 2+2*node d_transformer"]
            The output of the transformer.
        """

        # The batch size and the size of the node dimension
        batch_shape = transformer_input.shape[:-2]
        max_num_nodes = node_mask.shape[-1]

        # Flatten the node mask to concatenate the two graphs
        node_mask_flatter = rearrange(node_mask, "... pair node -> ... (pair node)")

        # Create the padding mask so that the transformer only attends to the actual
        # nodes (and the pooled representations)
        # (..., 2 + 2 * node)
        padding_mask = ~node_mask_flatter
        padding_mask = torch.cat(
            (
                torch.zeros((*batch_shape, 2), device=padding_mask.device, dtype=bool),
                padding_mask,
            ),
            dim=-1,
        )

        # The attention mask applied to all batch elements, which makes sure that nodes
        # only attend to nodes in the other graph and to the pooled representations.
        src_mask = torch.zeros(
            (2 + 2 * max_num_nodes,) * 2, device=padding_mask.device, dtype=bool
        )
        src_mask[2 : 2 + max_num_nodes, 2 : 2 + max_num_nodes] = 1
        src_mask[2 + max_num_nodes :, 2 + max_num_nodes :] = 1

        # Flatten the batch dimensions in the transformer input and padding mask
        transformer_input_flatter = transformer_input.reshape(
            -1, *transformer_input.shape[-2:]
        )
        padding_mask_flatter = padding_mask.reshape(-1, *padding_mask.shape[-1:])

        # Compute the transformer output
        # (..., 2 + 2 * max_nodes, d_transformer)
        transformer_output_flatter = transformer(
            transformer_input_flatter,
            mask=src_mask,
            src_key_padding_mask=padding_mask_flatter,
            is_causal=False,
        )

        # Expand out the batch dimensions
        transformer_output = transformer_output_flatter.reshape(
            *transformer_input.shape[:-2], *transformer_output_flatter.shape[-2:]
        )

        return transformer_output


class GraphIsomorphismAgentBody(GraphIsomorphismAgentPart, AgentBody):
    """Agent body for the graph isomorphism task.

    Takes as input a pair of graphs and outputs node-level and graph-level
    representations.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    agent_name : str
        The name of the agent.
    device : TorchDevice, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    in_keys = ("x", "adjacency", "message", "node_mask", "ignore_message")
    out_keys = ("graph_level_repr", "node_level_repr")

    def __init__(
        self,
        params: Parameters,
        agent_name: str,
        device: Optional[TorchDevice] = None,
    ):
        super().__init__(params, agent_name, device)

        # Build up the GNN module
        self.gnn = self._build_gnn(
            d_input=self.params.max_message_rounds,
        )

        # Build the global pooling module, which computes the graph-level representation
        # from the GNN output
        self.global_pooling = self._build_global_pooling()

        # Build the encoder going from the GNN to the transformer
        self.gnn_transformer_encoder = self._build_gnn_transformer_encoder()

        # Build the transformer
        self.transformer = self._build_transformer()

    def _build_gnn(self, d_input: int) -> TensorDictSequential:
        """Builds the GNN module for an agent.

        Parameters
        ----------
        d_input : int
            The dimensionality of the input features.

        Returns
        -------
        gnn : TensorDictSequential
            The GNN module, which takes as input a TensorDict with keys "gnn_repr",
            "adjacency" and "node_mask".
        """
        # Build up the GNN
        gnn_layers = []
        gnn_layers.append(
            TensorDictModule(
                Linear(d_input, self._agent_params.d_gnn),
                in_keys=("x",),
                out_keys=("gnn_repr",),
            )
        )
        for _ in range(self._agent_params.num_gnn_layers):
            gnn_layers.append(
                TensorDictModule(
                    ReLU(inplace=True), in_keys=("gnn_repr",), out_keys=("gnn_repr",)
                )
            )
            gnn_layers.append(
                GIN(
                    Sequential(
                        Linear(
                            self._agent_params.d_gnn,
                            self._agent_params.d_gin_mlp,
                        ),
                        ReLU(inplace=True),
                        Linear(
                            self._agent_params.d_gin_mlp,
                            self._agent_params.d_gnn,
                        ),
                    ),
                    feature_in_key="gnn_repr",
                    feature_out_key="gnn_repr",
                    vmap_compatible=True,
                )
            )
        gnn = TensorDictSequential(*gnn_layers)

        gnn = gnn.to(self.device)

        return gnn

    def _build_gnn_transformer_encoder(
        self,
    ) -> Linear:
        """Build the encoder layer which translates the GNN output to transformer input

        This is a simple linear layer, where the number of input features is `d_gnn` +
        3, where the extra features encode which graph-level representation the token
        is, if any and whether a node is in the most recent message from the other
        agent.

        Returns
        -------
        gnn_transformer_encoder : torch.nn.Linear
            The encoder module

        """
        return Linear(
            self._agent_params.d_gnn + 3,
            self._agent_params.d_transformer,
            device=self.device,
        )

    def _build_transformer(self) -> TransformerEncoder:
        """Builds the transformer module for an agent.

        Returns
        -------
        transformer : torch.nn.TransformerEncoder
            The transformer module.
        """

        transformer = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=self._agent_params.d_transformer,
                nhead=self._agent_params.num_heads,
                batch_first=True,
                dropout=self._agent_params.transformer_dropout,
            ),
            num_layers=self._agent_params.num_transformer_layers,
        )

        transformer = transformer.to(self.device)

        return transformer

    def _build_global_pooling(self) -> Sequential:
        """Builds a pooling layer which computes the graph-level representation.

        The module consists of a global sum pooling layer, an optional batch norm layer,
        a paired Gaussian noise layer and an optional pair invariant pooling layer.

        Returns
        -------
        global_pooling : torch.nn.Sequential
            The global pooling module.
        """

        layers = [
            Reduce(
                "... pair max_nodes d_gnn -> ... pair d_gnn",
                "sum",
            ),
        ]

        if self._agent_params.use_batch_norm:
            layers.append(BatchNorm1dBatchDims(num_features=self._agent_params.d_gnn))

        layers.append(
            PairedGaussianNoise(sigma=self._agent_params.noise_sigma, pair_dim=-2),
        )

        if self._agent_params.use_pair_invariant_pooling:
            layers.append(PairInvariantizer(pair_dim=-2))

        global_pooling = Sequential(*layers)

        global_pooling = global_pooling.to(self.device)

        return global_pooling

    def forward(
        self,
        data: TensorDictBase,
    ) -> TensorDict:
        """Obtain graph-level and node-level representations by running components

        Runs the GNN, pools the output, puts everything through a linear encoder, then
        runs the transformer on this.

        Parameters
        ----------
        data : TensorDictBase
            The data to run the GNN and transformer on. A TensorDictBase with
            keys:

            - "x" (... pair node feature): The graph node features (message history)
            - "adjacency" (... pair node node): The graph adjacency matrices
            - "message" (...): The most recent message from the other agent
            - "node_mask" (... pair node): Which nodes actually exist or a
              GraphIsomorphism data object.
            - "ignore_message" (...): Whether to ignore any values in "message". For
              example, in the first round the there is no message, and the "message"
              field is set to a dummy value.


        Returns
        -------
        out : TensorDict
            A tensor dict with keys:

            - "graph_level_repr" (... 2 d_transformer): The output graph-level
              representations.
            - "node_level_repr" (... 2 max_nodes d_transformer): The output node-level
              representations.
        """

        # The size of the node dimension
        max_num_nodes = data["x"].shape[-2]

        # If the data is empty, return empty outputs
        if data.batch_size[0] == 0:
            return TensorDict(
                dict(
                    graph_level_repr=torch.empty(
                        (*data.batch_size, 2, self._agent_params.d_transformer),
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    node_level_repr=torch.empty(
                        (
                            *data.batch_size,
                            2,
                            max_num_nodes,
                            self._agent_params.d_transformer,
                        ),
                        device=self.device,
                        dtype=torch.float32,
                    ),
                ),
                batch_size=data.batch_size,
            )

        # Run the GNN on the graphs
        # (batch, pair, max_nodes, d_gnn)
        gnn_output = self.gnn(data)["gnn_repr"]

        # Obtain the graph-level representations by pooling
        # (batch, pair, d_gnn)
        pooled_gnn_output = self.global_pooling(gnn_output)

        # Flatten the two batch dimensions in the graph representation and mask
        gnn_output_flatter = rearrange(
            gnn_output, "... pair node feature -> ... (pair node) feature"
        )

        # Add the graph-level representations to the transformer input
        # (..., 2 + 2 * node, d_gnn + 3)
        transformer_input = torch.cat((pooled_gnn_output, gnn_output_flatter), dim=-2)

        # Add extra features to distinguish the pooled representations from the
        # node-level representations
        pooled_feature = torch.zeros(
            *transformer_input.shape[:-1],
            2,
            device=transformer_input.device,
            dtype=transformer_input.dtype,
        )
        pooled_feature[..., 0, 0] = 1
        pooled_feature[..., 1, 1] = 1

        # Add the most recent message as a new one-hot feature
        message_feature = F.one_hot(
            data["message"] + 2, num_classes=2 + 2 * max_num_nodes
        )
        message_feature = torch.where(
            data["ignore_message"][..., None], 0, message_feature
        )
        message_feature = rearrange(message_feature, "... token -> ... token 1")

        # Concatenate everything together
        transformer_input = torch.cat(
            (transformer_input, pooled_feature, message_feature), dim=-1
        )

        # Run the transformer input through the encoder first
        # (..., 2 + 2 * node, d_transformer)
        transformer_input = self.gnn_transformer_encoder(transformer_input)

        # Run the transformer
        # (..., 2 + 2 * node, d_transformer)
        transformer_output_flatter = self._run_masked_transformer(
            transformer=self.transformer,
            transformer_input=transformer_input,
            node_mask=data["node_mask"],
        )

        # Extract the graph-level representations and rearrange the rest to have two
        # batch dims
        graph_level_repr = transformer_output_flatter[..., :2, :]
        node_level_repr = rearrange(
            transformer_output_flatter[..., 2:, :],
            "... (pair node) feature -> ... pair node feature",
            pair=2,
        )

        return TensorDict(
            dict(
                graph_level_repr=graph_level_repr,
                node_level_repr=node_level_repr,
            ),
            batch_size=graph_level_repr.shape[:-2],
        )

    def to(self, device: Optional[TorchDevice] = None):
        super().to(device)
        self.device = device
        self.gnn.to(device)
        self.global_pooling.to(device)
        self.global_pooling[-1].to(device)
        self.gnn_transformer_encoder.to(device)
        self.transformer.to(device)
        return self


class GraphIsomorphismDummyAgentBody(GraphIsomorphismAgentPart, DummyAgentBody):
    """Dummy agent body for the graph isomorphism task."""

    in_keys = ("x",)
    out_keys = ("graph_level_repr", "node_level_repr")

    def forward(self, data: TensorDictBase) -> TensorDict:
        """Returns dummy outputs.

        Parameters
        ----------
        data : TensorDictBase
            The data to run the GNN and transformer on. A TensorDictBase with
            keys:

            - "x" (... pair node feature): The graph node features (message history)

        Returns
        -------
        out : TensorDict
            A tensor dict with keys:

            - "graph_level_repr" (... 2 1): The output graph-level
              representations.
            - "node_level_repr" (... 2 max_nodes 1): The output node-level
              representations.
        """

        # The size of the node dimension
        max_num_nodes = data["x"].shape[-2]

        # The dummy graph-level representations
        graph_level_repr = torch.zeros(
            *data.batch_size,
            2,
            1,
            device=self.device,
            dtype=torch.float32,
        )

        # The dummy node-level representations
        node_level_repr = torch.zeros(
            *data.batch_size,
            2,
            max_num_nodes,
            1,
            device=self.device,
            dtype=torch.float32,
        )

        # Multiply the outputs by the dummy parameter, so that the gradients PyTorch
        # doesn't complain about not having any gradients
        graph_level_repr = graph_level_repr * self.dummy_parameter
        node_level_repr = node_level_repr * self.dummy_parameter

        return data.update(
            dict(graph_level_repr=graph_level_repr, node_level_repr=node_level_repr)
        )


class GraphIsomorphismAgentHead(GraphIsomorphismAgentPart, AgentHead, ABC):
    """Base class for all graph isomorphism agent heads.

    This class provides some utility methods for constructing and running the various
    modules.
    """

    def _build_node_level_mlp(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        num_layers: int,
        out_key: str = "node_level_mlp_output",
    ) -> TensorDictModule:
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
        out_key : str, default="node_level_mlp_output"
            The tensordict key to use for the output of the MLP.

        Returns
        -------
        node_level_mlp : TensorDictModule
            The node-level MLP.
        """
        layers = []

        # The layers of the MLP
        layers.append(Linear(d_in, d_hidden))
        layers.append(ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(Linear(d_hidden, d_hidden))
            layers.append(ReLU(inplace=True))
        layers.append(Linear(d_hidden, d_out))

        # Concatenate the pair and node dimensions
        layers.append(Rearrange("batch pair node d_out -> batch (pair node) d_out"))

        # Make the layers into a sequential module and wrap it in a TensorDictModule
        sequential = Sequential(*layers)
        tensor_dict_sequential = TensorDictModule(
            sequential, in_keys=("node_level_repr",), out_keys=(out_key,)
        )

        tensor_dict_sequential = tensor_dict_sequential.to(self.device)

        return tensor_dict_sequential

    def _build_graph_level_mlp(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        num_layers: int,
        out_key: str = "graph_level_mlp_output",
        squeeze: bool = False,
    ) -> TensorDictModule:
        """Builds an MLP which acts on the node-level representations.

        This takes as input a TensorDict with key "graph_level_repr" and outputs a
        Tensor.

        Parameters
        ----------
        d_in : int
            The dimensionality of the graph-level representations. This will be
            multiplied by two, as the MLP takes as input the concatenation of the two
            graph-level representations.
        d_hidden : int
            The dimensionality of the hidden layers.
        d_out : int
            The dimensionality of the output.
        num_layers : int
            The number of hidden layers in the MLP.
        out_key : str, default="graph_level_mlp_output"
            The tensordict key to use for the output of the MLP.
        squeeze : bool, default=False
            Whether to squeeze the output dimension. Only use this if the output
            dimension is 1.

        Returns
        -------
        node_level_mlp : TensorDictModule
            The node-level MLP.
        """
        layers = []

        # Concatenate the two graph-level representations
        layers.append(Rearrange("... pair d_in -> ... (pair d_in)"))

        # The layers of the MLP
        layers.append(Linear(2 * d_in, d_hidden))
        layers.append(ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(Linear(d_hidden, d_hidden))
            layers.append(ReLU(inplace=True))
        layers.append(Linear(d_hidden, d_out))

        # Squeeze the output dimension if necessary
        if squeeze:
            layers.append(Squeeze())

        # Make the layers into a sequential module, and wrap it in a TensorDictModule
        sequential = Sequential(*layers)
        tensor_dict_sequential = TensorDictModule(
            sequential, in_keys=("graph_level_repr",), out_keys=(out_key,)
        )

        tensor_dict_sequential = tensor_dict_sequential.to(self.device)

        return tensor_dict_sequential

    def _build_decider(self, d_out: int = 3) -> TensorDictModule:
        """Builds the module which produces a graph-pair level output.

        By default it is used to decide whether to continue exchanging messages. In this
        case it outputs a single triple of logits for the three options: guess that the
        graphs are not isomorphic, guess that the graphs are isomorphic, or continue
        exchanging messages.

        Parameters
        ----------
        d_out : int, default=3
            The dimensionality of the output.

        Returns
        -------
        decider : TensorDictModule
            The decider module.
        """
        return self._build_graph_level_mlp(
            d_in=self._agent_params.d_transformer,
            d_hidden=self._agent_params.d_decider,
            d_out=d_out,
            num_layers=self._agent_params.num_decider_layers,
            out_key="decision_logits",
        )


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
        The name of the agent.
    device : TorchDevice, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    in_keys = ("graph_level_repr", "node_level_repr")

    @property
    def out_keys(self):
        if self.decider is None:
            return ("node_selected_logits",)
        else:
            return ("node_selected_logits", "decision_logits")

    def __init__(
        self,
        params: Parameters,
        agent_name: str,
        device: Optional[TorchDevice] = None,
    ):
        super().__init__(params, agent_name, device)

        # Build the node selector module
        self.node_selector = self._build_node_selector()

        # Build the decider module if necessary
        if agent_name == "verifier":
            self.decider = self._build_decider()
        else:
            self.decider = None

    def _build_node_selector(self) -> TensorDictModule:
        """Builds the module which selects which node to send as a message.

        Returns
        -------
        node_selector : TensorDictModule
            The node selector module.
        """
        return self._build_node_level_mlp(
            d_in=self._agent_params.d_transformer,
            d_hidden=self._agent_params.d_node_selector,
            d_out=1,
            num_layers=self._agent_params.num_node_selector_layers,
            out_key="node_selected_logits",
        )

    def forward(self, body_output: TensorDict) -> TensorDict:
        """Runs the policy head on the given body output.

        Runs the node selector module and the decider module if present.

        Parameters
        ----------
        body_output : TensorDict
            The output of the body module. A tensor dict with keys:

            - "graph_level_repr" (... 2 d_transformer): The output graph-level
              representations.
            - "node_level_repr" (... 2 max_nodes d_transformer): The output node-level
              representations.

        Returns
        -------
        out : TensorDict
            A tensor dict with keys:

            - "node_selected_logits" (... 2*max_nodes): A logit for each node,
              indicating the probability that this node should be sent as a message to
              the verifier.
            - "decision_logits" (... 3): A logit for each of the three options: continue
              exchanging messages, guess that the graphs are isomorphic, or guess that
              the graphs are not isomorphic. Set to zeros when the decider is not
              present.
        """

        out_dict = {}

        if body_output.batch_size[0] == 0:
            out_dict["node_selected_logits"] = torch.empty(
                (*body_output.batch_size, 2 * body_output["node_level_repr"].shape[2]),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            out_dict["node_selected_logits"] = self.node_selector(body_output)[
                "node_selected_logits"
            ].squeeze(-1)

        if body_output.batch_size[0] == 0:
            out_dict["decision_logits"] = torch.empty(
                (*body_output.batch_size, 3),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            if self.decider is not None:
                out_dict["decision_logits"] = self.decider(body_output)[
                    "decision_logits"
                ]
            else:
                out_dict["decision_logits"] = torch.zeros(
                    (*body_output.batch_size, 3),
                    device=self.device,
                    dtype=torch.float32,
                )

        return TensorDict(out_dict, batch_size=body_output.batch_size)

    def to(self, device: Optional[TorchDevice] = None):
        super().to(device)
        self.device = device
        self.node_selector.to(device)
        if self.decider is not None:
            self.decider.to(device)


class GraphIsomorphismRandomAgentPolicyHead(
    GraphIsomorphismAgentPart, RandomAgentPolicyHead
):
    """Policy head for the graph isomorphism task yielding a uniform distribution."""

    in_keys = ("graph_level_repr", "node_level_repr")

    @property
    def out_keys(self):
        if self.decider:
            return ("node_selected_logits",)
        else:
            return ("node_selected_logits", "decision_logits")

    def __init__(
        self,
        params: Parameters,
        agent_name: str,
        device: Optional[TorchDevice] = None,
    ):
        super().__init__(params, agent_name, device)

        # Determine if we should output a decision too
        self.decider = agent_name == "verifier"

    def forward(self, body_output: TensorDict) -> TensorDict:
        """Outputs a uniform distribution.

        Parameters
        ----------
        body_output : TensorDict
            The output of the body module. A tensor dict with keys:

            - "graph_level_repr" (... 2 1): The output graph-level
              representations.
            - "node_level_repr" (... 2 max_nodes 1): The output node-level
              representations.

        Returns
        -------
        out : TensorDict
            A tensor dict with keys (all zeros):

            - "node_selected_logits" (... 2*max_nodes): A logit for each node,
              indicating the probability that this node should be sent as a message to
              the verifier.
            - "decision_logits" (... 3): A logit for each of the three options: continue
              exchanging messages, guess that the graphs are isomorphic, or guess that
              the graphs are not isomorphic. Set to zeros when the decider is not
              present.
        """

        max_num_nodes = body_output["node_level_repr"].shape[-2]

        node_selected_logits = torch.zeros(
            *body_output.batch_size,
            2 * max_num_nodes,
            device=self.device,
            dtype=torch.float32,
        )
        decision_logits = torch.zeros(
            *body_output.batch_size,
            3,
            device=self.device,
            dtype=torch.float32,
        )

        # Multiply the outputs by the dummy parameter, so that the gradients PyTorch
        # doesn't complain about not having any gradients
        node_selected_logits = node_selected_logits * self.dummy_parameter
        decision_logits = decision_logits * self.dummy_parameter

        return body_output.update(
            dict(
                node_selected_logits=node_selected_logits,
                decision_logits=decision_logits,
            )
        )


class GraphIsomorphismAgentValueHead(GraphIsomorphismAgentHead, AgentValueHead):
    """Value head for the graph isomorphism task.

    Takes as input the output of the agent body and outputs a value function.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    agent_name : str
        The name of the agent.
    device : TorchDevice, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    in_keys = ("graph_level_repr",)
    out_keys = ("value",)

    def __init__(
        self,
        params: Parameters,
        agent_name: str,
        device: Optional[TorchDevice] = None,
    ):
        super().__init__(params, agent_name, device)

        self.value_mlp = self._build_mlp()

    def _build_mlp(self) -> TensorDictModule:
        """Builds the module which computes the value function.

        Returns
        -------
        value_mlp : TensorDictModule
            The value module.
        """
        return self._build_graph_level_mlp(
            d_in=self._agent_params.d_transformer,
            d_hidden=self._agent_params.d_value,
            d_out=1,
            num_layers=self._agent_params.num_value_layers,
            out_key="value",
            squeeze=True,
        )

    def forward(self, body_output: TensorDict) -> TensorDict:
        """Runs the value head on the given body output.

        Parameters
        ----------
        body_output : TensorDict
            The output of the body module. A tensor dict with keys:

            - "graph_level_repr" (... 2 d_transformer): The output graph-level
              representations.

        Returns
        -------
        value_out : TensorDict
            A tensor dict with keys:

            - "value" (...): The estimated value for each batch item
        """

        if body_output.batch_size[0] == 0:
            return TensorDict(
                dict(
                    value=torch.empty(
                        (*body_output.batch_size,),
                        device=self.device,
                        dtype=torch.float32,
                    )
                ),
                batch_size=body_output.batch_size,
            )

        return self.value_mlp(body_output)

    def to(self, device: Optional[TorchDevice] = None):
        super().to(device)
        self.device = device
        self.value_mlp.to(device)


class GraphIsomorphismConstantAgentValueHead(
    GraphIsomorphismAgentHead, ConstantAgentValueHead
):
    """A constant value head for the graph isomorphism task."""

    in_keys = ("graph_level_repr", "node_level_repr")
    out_keys = ("value",)

    def forward(self, body_output: TensorDict) -> TensorDict:
        """Returns a constant value.

        Parameters
        ----------
        body_output : TensorDict
            The output of the body module. A tensor dict with keys:

            - "graph_level_repr" (... 2 1): The output graph-level
              representations.
            - "node_level_repr" (... 2 max_nodes 1): The output node-level
              representations.

        Returns
        -------
        value_out : TensorDict
            A tensor dict with keys:

            - "value" (...): The 'value' for each batch item, which is a constant zero.
        """

        value = torch.zeros(
            *body_output.batch_size,
            device=self.device,
            dtype=torch.float32,
        )

        # Multiply the output by the dummy parameter, so that the gradients PyTorch
        # doesn't complain about not having any gradients
        value = value * self.dummy_parameter

        return body_output.update(dict(value=value))


class GraphIsomorphismAgentCriticHead(GraphIsomorphismAgentHead, AgentCriticHead):
    """Critic head for the graph isomorphism task.

    Takes as input the output of the agent body and the actions taken and outputs a
    value function.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    agent_name : str
        The name of the agent.
    device : TorchDevice, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    in_keys = (
        "graph_level_repr",
        "node_level_repr",
        ("agents", "node_selected"),
        ("agents", "decision"),
        "node_mask",
    )
    out_keys = ("value",)

    def __init__(
        self,
        params: Parameters,
        agent_name: str,
        device: Optional[TorchDevice] = None,
    ):
        super().__init__(params, agent_name, device)

        self.transformer_encoder = self._build_transformer_encoder()
        self.transformer = self._build_transformer()
        self.critic_mlp = self._build_mlp()

    def _get_agent_level_tensordict_value(
        self, key: str, tensordict: TensorDict
    ) -> Tensor:
        """Get a value from a TensorDict with agent-level keys.

        Selects the value ["agents", key] from the tensordict, then selects the tensor
        corresponding to the agent index.

        Parameters
        ----------
        key : str
            The key to get.
        tensordict : TensorDict
            The TensorDict to get the value from.

        Returns
        -------
        value : Tensor
            The value.
        """
        return tensordict["agents", key][..., self.agent_index]

    def _build_transformer_encoder(self) -> Linear:
        """Build the encoder layer from the hidden representations to the transformer

        This is a simple linear layer, where the number of input features is
        `d_transformer` + 1, where the extra feature encodes whether a node is in the
        next message from to other agent, and the decision made (if any) one-hot
        encoded, repeated across all tokens.

        Returns
        -------
        transformer_encoder : torch.nn.Linear
            The encoder module

        """
        return Linear(
            self._agent_params.d_transformer + 4,
            self._agent_params.d_transformer,
            device=self.device,
        )

    def _build_transformer(self) -> TransformerEncoder:
        """Builds the transformer which processes the next message.

        Returns
        -------
        transformer : torch.nn.TransformerEncoder
            The transformer module.
        """

        transformer = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=self._agent_params.d_transformer,
                nhead=self._agent_params.num_heads,
                batch_first=True,
                dropout=self._agent_params.transformer_dropout,
            ),
            num_layers=self._agent_params.num_critic_transformer_layers,
        )

        transformer = transformer.to(self.device)

        return transformer

    def _build_mlp(self) -> TensorDictModule:
        """Builds the module which computes the value function.

        Returns
        -------
        critic_mlp : TensorDictModule
            The critic module.
        """
        return self._build_graph_level_mlp(
            d_in=self._agent_params.d_transformer,
            d_hidden=self._agent_params.d_critic,
            d_out=1,
            num_layers=self._agent_params.num_critic_layers,
            out_key="value",
        )

    def forward(self, critic_input: TensorDict) -> TensorDict:
        """Runs the critic head on the given body output.

        Parameters
        ----------
        critic_input : TensorDict
            The output of the body module. A tensor dict with keys:

            - "graph_level_repr" (batch 2 d_transformer): The output graph-level
              representations.
            - "node_level_repr" (batch 2 max_nodes d_transformer): The output node-level
              representations.
            - "node_selected" (batch 2 max_nodes): Which node was selected to send as a
              message.
            - "decision" (batch): The decision made by the verifier.
            - "node_mask" (batch 2 max_nodes): Which nodes actually exist.

        Returns
        -------
        critic_out : TensorDict
            A tensor dict with keys:

            - "value" (batch): The estimated value for each batch item
        """

        if critic_input.batch_size[0] == 0:
            return TensorDict(
                dict(
                    value=torch.empty(
                        (0,),
                        device=self.device,
                        dtype=torch.float32,
                    )
                ),
                batch_size=0,
            )

        # The size of the node dimension
        max_num_nodes = critic_input["node_level_repr"].shape[2]

        # Concatenate the two graph-level representations and the node-level
        # representations
        # (batch, 2 + 2 * node, d_transformer)
        transformer_input = torch.cat(
            (
                critic_input["graph_level_repr"],
                rearrange(
                    critic_input["node_level_repr"],
                    "... pair node d_transformer -> ... (pair node) d_transformer",
                ),
            ),
            dim=-2,
        )

        # Add the most recent message as a new one-hot feature
        message_feature = F.one_hot(
            self._get_agent_level_tensordict_value("node_selected", critic_input) + 2,
            num_classes=2 + 2 * max_num_nodes,
        )
        message_feature = rearrange(message_feature, "... token -> ... token 1")

        # Add the decision as new one-hot features
        decision_features = F.one_hot(
            self._get_agent_level_tensordict_value("decision", critic_input),
            num_classes=3,
        )
        decision_features = repeat(
            decision_features,
            "... decision -> ... token decision",
            token=2 + 2 * max_num_nodes,
        )

        # Concatenate everything together
        transformer_input = torch.cat(
            (transformer_input, message_feature, decision_features), dim=-1
        )

        # Run the transformer input through the encoder first
        # (batch, 2 + 2 * max_nodes, d_transformer)
        transformer_input = self.transformer_encoder(transformer_input)

        # Run the transformer
        # (batch, 2 + 2 * node, d_transformer)
        transformer_output_flatter = self._run_masked_transformer(
            transformer=self.transformer,
            transformer_input=transformer_input,
            node_mask=critic_input["node_mask"],
        )

        # Extract the graph-level representations, and feed them through the MLP
        graph_level_repr = transformer_output_flatter[:, :2]
        mlp_input = TensorDict(
            dict(graph_level_repr=graph_level_repr),
            batch_size=critic_input.batch_size,
        )
        critic_out = self.critic_mlp(mlp_input)
        critic_out["value"] = critic_out["value"].squeeze(-1)

        return critic_out

    def to(self, device: Optional[TorchDevice] = None):
        super().to(device)
        self.device = device
        self.transformer_encoder.to(device)
        self.transformer.to(device)
        self.critic_mlp.to(device)


class GraphIsomorphismSoloAgentHead(GraphIsomorphismAgentHead, SoloAgentHead):
    """Solo agent head for the graph isomorphism task.

    Solo agents try to solve the task on their own, without interacting with another
    agents.
    """

    in_keys = ("graph_level_repr",)
    out_keys = ("decision_logits",)

    def __init__(
        self,
        params: Parameters,
        agent_name: str,
        device: Optional[TorchDevice] = None,
    ):
        super().__init__(params, agent_name, device)

        self.decider = self._build_decider(d_out=2)

    def forward(self, body_output: TensorDict) -> TensorDict:
        """Runs the solo agent head on the given body output.

        Parameters
        ----------
        body_output : TensorDict
            The output of the body module. A tensor dict with keys:

            - "graph_level_repr" (... 2 d_transformer): The output graph-level
              representations.

        Returns
        -------
        out : TensorDict
            A tensor dict with keys:

            - "decision_logits" (... 2): A logit for each of the two options: guess that
              the graphs are isomorphic, or guess that the graphs are not isomorphic.
        """

        return self.decider(body_output)

    def to(self, device: Optional[TorchDevice] = None):
        super().to(device)
        self.device = device
        self.decider.to(device)


class GraphIsomorphismCombinedBody(CombinedBody):
    """A module which combines the agent bodies for the graph isomorphism task.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    bodies : dict[str, GraphIsomorphismAgentBody]
        The agent bodies to combine.
    """

    in_keys = ("round", "x", "adjacency", "message", "node_mask")
    out_keys = ("round", ("agents", "node_level_repr"), ("agents", "graph_level_repr"))

    def __init__(
        self,
        params: Parameters,
        bodies: dict[str, GraphIsomorphismAgentBody],
    ) -> None:
        super().__init__(params, bodies)

    def forward(self, data: TensorDictBase) -> TensorDict:
        round: Int[Tensor, "batch"] = data["round"]

        # Run the agent bodies
        body_outputs: dict[str, TensorDict] = {}
        for agent_name in self.params.agents:
            # Build the input dict for the agent body
            input_dict = {}
            for key in self.bodies[agent_name].in_keys:
                if key == "ignore_message":
                    input_dict[key] = round == 0
                else:
                    input_dict[key] = data[key]
            input_td = TensorDict(
                input_dict,
                batch_size=data.batch_size,
            )

            # Run the agent body
            body_outputs[agent_name] = self.bodies[agent_name](input_td)

        # Stack the outputs
        node_level_repr = torch.stack(
            [body_outputs[name]["node_level_repr"] for name in self.params.agents],
            dim=-3,
        )
        graph_level_repr = torch.stack(
            [body_outputs[name]["graph_level_repr"] for name in self.params.agents],
            dim=-2,
        )

        return data.update(
            dict(
                agents=dict(
                    node_level_repr=node_level_repr,
                    graph_level_repr=graph_level_repr,
                )
            )
        )


class GraphIsomorphismCombinedPolicyHead(CombinedPolicyHead):
    """A module which combines the agent policy heads for the graph isomorphism task.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    policy_heads : dict[str, GraphIsomorphismAgentPolicyHead]
        The agent policy heads to combine.
    """

    in_keys = (("agents", "node_level_repr"), ("agents", "graph_level_repr"))
    out_keys = (("agents", "node_selected_logits"), ("agents", "decision_logits"))

    def __init__(
        self,
        params: Parameters,
        policy_heads: dict[str, GraphIsomorphismAgentPolicyHead],
    ):
        super().__init__(params, policy_heads)

    def forward(self, head_output: TensorDictBase) -> TensorDict:
        """Run the agent policy heads and combine their outputs.

        Parameters
        ----------
        tensordict : TensorDictBase
            The input to the value heads. Should contain the keys:

            - ("agents", "node_level_repr"): The node-level representation from the
              body.
            - ("agents", "graph_level_repr"): The node-level representation from the
              body.

        Returns
        -------
        tensordict: TensorDict
            The tensordict update in place with the output of the value heads.
        """

        # Run the policy heads to obtain the probability distributions
        policy_outputs: dict[str, TensorDict] = {}
        for i, agent_name in enumerate(self.params.agents):
            input_td = TensorDict(
                dict(
                    node_level_repr=head_output["agents", "node_level_repr"][
                        ..., i, :, :, :
                    ],
                    graph_level_repr=head_output["agents", "graph_level_repr"][
                        ..., i, :, :
                    ],
                ),
                batch_size=head_output.batch_size,
            )
            policy_outputs[agent_name] = self.policy_heads[agent_name](input_td)

        # Stack the outputs
        node_selected_logits = torch.stack(
            [
                policy_outputs[name]["node_selected_logits"]
                for name in self.params.agents
            ],
            dim=-2,
        )
        decision_logits = torch.stack(
            [policy_outputs[name]["decision_logits"] for name in self.params.agents],
            dim=-2,
        )

        return head_output.update(
            dict(
                agents=TensorDict(
                    dict(
                        node_selected_logits=node_selected_logits,
                        decision_logits=decision_logits,
                    ),
                    batch_size=head_output.batch_size,
                )
            )
        )


class GraphIsomorphismCombinedValueHead(CombinedValueHead):
    """A module which combines the agent value heads for the graph isomorphism task.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    value_heads : dict[str, GraphIsomorphismAgentValueHead]
        The agent value heads to combine.
    """

    in_keys = (("agents", "graph_level_repr"),)
    out_keys = (("agents", "value"),)

    def __init__(
        self,
        params: Parameters,
        value_heads: dict[str, GraphIsomorphismAgentValueHead],
    ):
        super().__init__(params, value_heads)

    def forward(self, head_output: TensorDictBase) -> TensorDict:
        """Run the agent value heads and combine their values.

        Parameters
        ----------
        tensordict : TensorDictBase
            The input to the value heads. Should contain the keys:

            - ("agents", "graph_level_repr"): The node-level representation from the
              body.

        Returns
        -------
        tensordict: TensorDict
            The tensordict update in place with the output of the value heads.
        """

        # Run the policy heads to obtain the value estimates
        value_outputs: dict[str, TensorDict] = {}
        for i, agent_name in enumerate(self.params.agents):
            input_td = TensorDict(
                dict(
                    node_level_repr=head_output["agents", "node_level_repr"][
                        ..., i, :, :, :
                    ],
                    graph_level_repr=head_output["agents", "graph_level_repr"][
                        ..., i, :, :
                    ],
                ),
                batch_size=head_output.batch_size,
            )
            value_outputs[agent_name] = self.value_heads[agent_name](input_td)

        # Stack the outputs
        value = torch.stack(
            [value_outputs[name]["value"] for name in self.params.agents], dim=-1
        )

        return head_output.update(
            dict(
                agents=TensorDict(
                    dict(value=value),
                    batch_size=head_output.batch_size,
                )
            ),
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

#     def __init__(self, params: Parameters, device: Optional[TorchDevice] = None):
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
#             node_selected_logits, decision_logits, node_mask = self.verifier(data)

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
#             verifier_guess = Categorical(logits=decision_logits).sample()

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
