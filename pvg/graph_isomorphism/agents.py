"""Graph isomorphism agents components.

Contains classes for building agent bodies and heads for the graph isomorphism task.

The structure of all agent bodies is the same:

- A GNN module which takes as input the two graphs and the message history and outputs
  node-level representations for each graph.
- A global pooling module which takes as input the node-level representations and
  outputs graph-level representations for each graph.
- A transformer module which takes as input the graph-level representations and the
  node-level representations for both graphs together with the most recent message and
  outputs graph-level and node-level representations.
- A representation encoder which takes as input the graph-level and node-level
  representations and outputs the final representations.
"""

from abc import ABC
from typing import Optional
from dataclasses import dataclass

import torch
from torch.nn import (
    Linear,
    TransformerEncoder,
    TransformerEncoderLayer,
    Sequential,
)
from torch import Tensor
import torch.nn.functional as F

from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from jaxtyping import Float, Int

from pvg.scenario_base import (
    AgentHooks,
    AgentPart,
    AgentBody,
    AgentHead,
    DummyAgentBody,
    AgentPolicyHead,
    RandomAgentPolicyHead,
    AgentValueHead,
    ConstantAgentValueHead,
    SoloAgentHead,
    CombinedBody,
    CombinedPolicyHead,
    CombinedValueHead,
)
from pvg.scenario_instance import register_scenario_class
from pvg.parameters import (
    Parameters,
    GraphIsomorphismAgentParameters,
    RandomAgentParameters,
    ScenarioType,
)
from pvg.protocols import ProtocolHandler
from pvg.utils.torch_modules import (
    ACTIVATION_CLASSES,
    PairedGaussianNoise,
    PairInvariantizer,
    GIN,
    Squeeze,
    BatchNorm1dSimulateBatchDims,
    OneHot,
    TensorDictCat,
    NormalizeOneHotMessageHistory,
    Print,
    TensorDictPrint,
)
from pvg.utils.types import TorchDevice

GI_SCENARIO = ScenarioType.GRAPH_ISOMORPHISM


@dataclass
class GraphIsomorphismAgentHooks(AgentHooks):
    """Holder for hooks to run at various points in the agent forward pass."""

    gnn_output: Optional[callable] = None
    gnn_output_rounded: Optional[callable] = None
    pooled_gnn_output: Optional[callable] = None
    gnn_output_flatter: Optional[callable] = None
    transformer_input_initial: Optional[callable] = None
    pooled_feature: Optional[callable] = None
    message_feature: Optional[callable] = None
    transformer_input_pre_encoder: Optional[callable] = None
    transformer_input: Optional[callable] = None
    transformer_output_flatter: Optional[callable] = None
    graph_level_repr_pre_encoder: Optional[callable] = None
    node_level_repr_pre_encoder: Optional[callable] = None
    graph_level_repr: Optional[callable] = None
    node_level_repr: Optional[callable] = None


class GraphIsomorphismAgentPart(AgentPart, ABC):
    """Base class for all graph isomorphism agent parts.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    agent_name : str
        The name of the agent.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    device : TorchDevice, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    _agent_params: GraphIsomorphismAgentParameters | RandomAgentParameters

    def __init__(
        self,
        params: Parameters,
        agent_name: str,
        protocol_handler: ProtocolHandler,
        *,
        device: Optional[TorchDevice] = None,
    ):
        super().__init__(params, agent_name, protocol_handler, device=device)

        self._agent_params = params.agents[agent_name]
        self.agent_index = self.protocol_handler.agent_names.index(agent_name)

        if isinstance(self._agent_params, GraphIsomorphismAgentParameters):
            self.activation_function = ACTIVATION_CLASSES[
                self._agent_params.activation_function
            ]

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


@register_scenario_class(GI_SCENARIO, AgentBody)
class GraphIsomorphismAgentBody(GraphIsomorphismAgentPart, AgentBody):
    """Agent body for the graph isomorphism task.

    Takes as input a pair of graphs, message history and the most recent message and
    outputs node-level and graph-level representations.

    Shapes
    ------
    Input:
        - "x" (... pair node feature): The graph node features (message history)
        - "adjacency" (... pair node node): The graph adjacency matrices
        - "message" (... pair node), optional: The most recent message from the other
          agent
        - "node_mask" (... pair node): Which nodes actually exist
        - "ignore_message" (...), optional: Whether to ignore any values in "message".
          For example, in the first round the there is no message, and the "message"
          field is set to a dummy value.

    Output:
        - "graph_level_repr" (... 2 d_representation): The output graph-level
          representations.
        - "node_level_repr" (... 2 max_nodes d_representation): The output node-level
          representations.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    agent_name : str
        The name of the agent.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    device : TorchDevice, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    in_keys = ("x", "adjacency", "message", "node_mask", "ignore_message")
    out_keys = ("graph_level_repr", "node_level_repr")

    def __init__(
        self,
        params: Parameters,
        agent_name: str,
        protocol_handler: ProtocolHandler,
        *,
        device: Optional[TorchDevice] = None,
    ):
        super().__init__(params, agent_name, protocol_handler, device=device)

        # Build the message history normalizer if necessary
        if self._agent_params.normalize_message_history:
            self.message_history_normalizer = NormalizeOneHotMessageHistory(
                max_message_rounds=self.protocol_handler.max_message_rounds,
                message_out_key="gnn_repr",
                num_structure_dims=2,
            )

        # Build up the GNN module
        self.gnn = self._build_gnn(
            d_input=self.protocol_handler.max_message_rounds,
        )

        # Build the global pooling module, which computes the graph-level representation
        # from the GNN output
        self.global_pooling = self._build_global_pooling()

        # Build the encoder going from the GNN to the transformer
        self.gnn_transformer_encoder = self._build_gnn_transformer_encoder()

        # Build the transformer
        self.transformer = self._build_transformer()

        # Build the encoder going from the transformer output to the representation
        # space
        self.representation_encoder = self._build_representation_encoder()

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
                in_keys=("gnn_repr",),
                out_keys=("gnn_repr",),
            )
        )
        for _ in range(self._agent_params.num_gnn_layers):
            gnn_layers.append(
                TensorDictModule(
                    self.activation_function(),
                    in_keys=("gnn_repr",),
                    out_keys=("gnn_repr",),
                )
            )
            gnn_layers.append(
                GIN(
                    Sequential(
                        Linear(
                            self._agent_params.d_gnn,
                            self._agent_params.d_gin_mlp,
                        ),
                        self.activation_function(),
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

        if self._agent_params.num_transformer_layers == 0:
            return None

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
            layers.append(
                BatchNorm1dSimulateBatchDims(num_features=self._agent_params.d_gnn)
            )

        layers.append(
            PairedGaussianNoise(sigma=self._agent_params.noise_sigma, pair_dim=-2),
        )

        if self._agent_params.use_pair_invariant_pooling:
            layers.append(PairInvariantizer(pair_dim=-2))

        global_pooling = Sequential(*layers)

        global_pooling = global_pooling.to(self.device)

        return global_pooling

    def _build_representation_encoder(self) -> Linear:
        """Builds the encoder layer which translates to the representation space.

        This is a simple linear layer ensures that the number of output features is
        `params.d_representation`.

        Returns
        -------
        representation_encoder : torch.nn.Linear
            The encoder module
        """
        return Linear(
            self._agent_params.d_transformer,
            self.params.d_representation,
            device=self.device,
        )

    def forward(
        self,
        data: TensorDictBase,
        hooks: Optional[GraphIsomorphismAgentHooks] = None,
    ) -> TensorDict:
        """Obtain graph-level and node-level representations by running components

        Runs the GNN, pools the output, puts everything through a linear encoder, then
        runs the transformer on this.

        Parameters
        ----------
        data : TensorDictBase
            The data to run the GNN and transformer on. A TensorDictBase with keys:

            - "x" (... pair node feature): The graph node features (message history)
            - "adjacency" (... pair node node): The graph adjacency matrices
            - "message" (... pair node), optional: The most recent message from the
              other agent
            - "node_mask" (... pair node), optional: Which nodes actually exist or a
              GraphIsomorphism data object.
            - "ignore_message" (...), optional: Whether to ignore any values in
              "message". For example, in the first round the there is no message, and
              the "message" field is set to a dummy value.
        hooks : GraphIsomorphismAgentHooks, optional
            Hooks to run at various points in the agent forward pass.

        Returns
        -------
        out : TensorDict
            A tensor dict with keys:

            - "graph_level_repr" (... 2 d_representation): The output graph-level
              representations.
            - "node_level_repr" (... 2 max_nodes d_representation): The output
              node-level representations.
        """

        # The size of the node dimension
        max_num_nodes = data["x"].shape[-2]

        # Normalize the message history if necessary
        if self._agent_params.normalize_message_history:
            data = self.message_history_normalizer(data)
        else:
            data = data.update(dict(gnn_repr=data["x"]))

        # Run the GNN on the graphs
        # (batch, pair, max_nodes, d_gnn)
        gnn_output = self.gnn(data)["gnn_repr"]

        self._run_recorder_hook(hooks, "gnn_output", gnn_output)

        if self._agent_params.gnn_output_digits is not None:
            gnn_output = torch.round(
                gnn_output, decimals=self._agent_params.gnn_output_digits
            )

        self._run_recorder_hook(hooks, "gnn_output_rounded", gnn_output)

        # Obtain the graph-level representations by pooling
        # (batch, pair, d_gnn)
        pooled_gnn_output = self.global_pooling(gnn_output)

        self._run_recorder_hook(hooks, "pooled_gnn_output", pooled_gnn_output)

        # Flatten the two batch dimensions in the graph representation and mask
        gnn_output_flatter = rearrange(
            gnn_output, "... pair node feature -> ... (pair node) feature"
        )

        self._run_recorder_hook(hooks, "gnn_output_flatter", gnn_output_flatter)

        # Add the graph-level representations to the transformer input
        # (..., 2 + 2 * node, d_gnn + 3)
        transformer_input = torch.cat((pooled_gnn_output, gnn_output_flatter), dim=-2)

        self._run_recorder_hook(hooks, "transformer_input_initial", transformer_input)

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

        self._run_recorder_hook(hooks, "pooled_feature", pooled_feature)

        # Turn the most recent message into a feature
        # (..., 2 + 2 * node, 1)
        if "message" in data.keys():
            message_feature = rearrange(
                data["message"], "... pair node -> ... (pair node)"
            )
            message_feature = torch.cat(
                [
                    torch.zeros(
                        (*message_feature.shape[:-1], 2),
                        device=message_feature.device,
                        dtype=message_feature.dtype,
                    ),
                    message_feature,
                ],
                dim=-1,
            )
            message_feature = torch.where(
                data["ignore_message"][..., None], 0, message_feature
            )
            message_feature = rearrange(message_feature, "... token -> ... token 1")
        else:
            message_feature = torch.zeros(
                (*transformer_input.shape[:-1], 1),
                device=transformer_input.device,
                dtype=transformer_input.dtype,
            )

        self._run_recorder_hook(hooks, "message_feature", message_feature)

        # Concatenate everything together
        transformer_input = torch.cat(
            (transformer_input, pooled_feature, message_feature), dim=-1
        )

        self._run_recorder_hook(
            hooks, "transformer_input_pre_encoder", transformer_input
        )

        # Run the transformer input through the encoder first
        # (..., 2 + 2 * node, d_transformer)
        transformer_input = self.gnn_transformer_encoder(transformer_input)

        self._run_recorder_hook(hooks, "transformer_input", transformer_input)

        if self.transformer is not None:
            # Run the transformer
            # (..., 2 + 2 * node, d_transformer)
            transformer_output_flatter = self._run_masked_transformer(
                transformer=self.transformer,
                transformer_input=transformer_input,
                node_mask=data["node_mask"],
            )
        else:
            transformer_output_flatter = transformer_input

        self._run_recorder_hook(
            hooks, "transformer_output_flatter", transformer_output_flatter
        )

        # Extract the graph-level representations and rearrange the rest to have two
        # batch dims
        graph_level_repr = transformer_output_flatter[..., :2, :]
        node_level_repr = rearrange(
            transformer_output_flatter[..., 2:, :],
            "... (pair node) feature -> ... pair node feature",
            pair=2,
        )

        self._run_recorder_hook(hooks, "graph_level_repr_pre_encoder", graph_level_repr)
        self._run_recorder_hook(hooks, "node_level_repr_pre_encoder", node_level_repr)

        # Run the node-level representations through the representation encoder
        graph_level_repr = self.representation_encoder(graph_level_repr)
        node_level_repr = self.representation_encoder(node_level_repr)

        self._run_recorder_hook(hooks, "graph_level_repr", graph_level_repr)
        self._run_recorder_hook(hooks, "node_level_repr", node_level_repr)

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
        if self._agent_params.normalize_message_history:
            self.message_history_normalizer.to(device)
        self.gnn.to(device)
        self.global_pooling.to(device)
        self.global_pooling[-1].to(device)
        self.gnn_transformer_encoder.to(device)
        if self.transformer is not None:
            self.transformer.to(device)
        self.representation_encoder.to(device)
        return self


@register_scenario_class(GI_SCENARIO, DummyAgentBody)
class GraphIsomorphismDummyAgentBody(GraphIsomorphismAgentPart, DummyAgentBody):
    """Dummy agent body for the graph isomorphism task."""

    in_keys = ("x",)
    out_keys = ("graph_level_repr", "node_level_repr")

    def forward(
        self,
        data: TensorDictBase,
        hooks: Optional[GraphIsomorphismAgentHooks] = None,
    ) -> TensorDict:
        """Returns dummy outputs.

        Parameters
        ----------
        data : TensorDictBase
            A TensorDictBase with keys:

            - "x" (... pair node feature): The graph node features (message history)

        hooks : GraphIsomorphismAgentHooks, optional
            Hooks to run at various points in the agent forward pass.

        Returns
        -------
        out : TensorDict
            A tensor dict with keys:

            - "graph_level_repr" (... 2 d_representation): The output graph-level
              representations.
            - "node_level_repr" (... 2 max_nodes d_representation): The output
              node-level representations.
        """

        # The size of the node dimension
        max_num_nodes = data["x"].shape[-2]

        # The dummy graph-level representations
        graph_level_repr = torch.zeros(
            *data.batch_size,
            2,
            self.params.d_representation,
            device=self.device,
            dtype=torch.float32,
        )

        # The dummy node-level representations
        node_level_repr = torch.zeros(
            *data.batch_size,
            2,
            max_num_nodes,
            self.params.d_representation,
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

        Shapes
        ------
        Input:
            - "node_level_repr": (... 2 max_nodes d_in)

        Output:
            - "node_level_mlp_output": (... 2 max_nodes d_out)

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
        layers.append(self.activation_function())
        for _ in range(num_layers - 2):
            layers.append(Linear(d_hidden, d_hidden))
            layers.append(self.activation_function())
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
        include_round: bool = False,
        out_key: str = "graph_level_mlp_output",
        squeeze: bool = False,
    ) -> TensorDictSequential:
        """Builds an MLP which acts on the node-level representations.

        Shapes
        ------
        Input:
            - "graph_level_repr": (... 2 d_in)

        Output:
            - "graph_level_mlp_output": (... 2 d_out)

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
        include_round : bool, default=False
            Whether to include the round number as a (one-hot encoded) input to the MLP.
        out_key : str, default="graph_level_mlp_output"
            The tensordict key to use for the output of the MLP.
        squeeze : bool, default=False
            Whether to squeeze the output dimension. Only use this if the output
            dimension is 1.

        Returns
        -------
        node_level_mlp : TensorDictSequential
            The node-level MLP.
        """
        mlp_layers = []

        # The layers of the MLP
        updated_d_in = 2 * d_in
        if include_round:
            updated_d_in += self.protocol_handler.max_message_rounds + 1
        mlp_layers.append(Linear(updated_d_in, d_hidden))
        mlp_layers.append(self.activation_function())
        for _ in range(num_layers - 2):
            mlp_layers.append(Linear(d_hidden, d_hidden))
            mlp_layers.append(self.activation_function())
        mlp_layers.append(Linear(d_hidden, d_out))

        # Squeeze the output dimension if necessary
        if squeeze:
            mlp_layers.append(Squeeze())

        # Make the layers into a sequential module, and wrap it in a TensorDictModule
        mlp = Sequential(*mlp_layers)
        mlp = TensorDictModule(
            mlp, in_keys=("graph_level_mlp_input",), out_keys=(out_key,)
        )

        # The final module includes one or two more things before the MLP
        td_sequential_layers = []

        # Concatenate the two graph-level representations
        td_sequential_layers.append(
            TensorDictModule(
                Rearrange("... pair d_in -> ... (pair d_in)"),
                in_keys=("graph_level_repr",),
                out_keys=("graph_level_mlp_input",),
            )
        )

        if include_round:
            # Add the round number as an input to the MLP
            td_sequential_layers.append(
                TensorDictModule(
                    OneHot(num_classes=self.protocol_handler.max_message_rounds + 1),
                    in_keys=("round"),
                    out_keys=("round_one_hot",),
                )
            )
            td_sequential_layers.append(
                TensorDictCat(
                    in_keys=("graph_level_mlp_input", "round_one_hot"),
                    out_key="graph_level_mlp_input",
                    dim=-1,
                ),
            )

        td_sequential_layers.append(mlp)

        return TensorDictSequential(*td_sequential_layers).to(self.device)

    def _build_decider(
        self, d_out: int = 3, include_round: Optional[bool] = None
    ) -> TensorDictModule:
        """Builds the module which produces a graph-pair level output.

        By default it is used to decide whether to continue exchanging messages. In this
        case it outputs a single triple of logits for the three options: guess that the
        graphs are not isomorphic, guess that the graphs are isomorphic, or continue
        exchanging messages.

        Parameters
        ----------
        d_out : int, default=3
            The dimensionality of the output.
        include_round : bool, optional
            Whether to include the round number as a (one-hot encoded) input to the MLP.
            If not given, the value from the agent parameters is used.

        Returns
        -------
        decider : TensorDictModule
            The decider module.
        """

        if include_round is None:
            include_round = self._agent_params.include_round_in_decider

        return self._build_graph_level_mlp(
            d_in=self.params.d_representation,
            d_hidden=self._agent_params.d_decider,
            d_out=d_out,
            num_layers=self._agent_params.num_decider_layers,
            include_round=include_round,
            out_key="decision_logits",
        )


@register_scenario_class(GI_SCENARIO, AgentPolicyHead)
class GraphIsomorphismAgentPolicyHead(GraphIsomorphismAgentHead, AgentPolicyHead):
    """Agent policy head for the graph isomorphism task.

    Takes as input the output of the agent body and outputs a policy distribution over
    the actions. Both agents select a node to send as a message, and the verifier also
    decides whether to guess that the graphs are isomorphic or not or to continue
    exchanging messages.

    Shapes
    ------
    Input:
        - "graph_level_repr" (... 2 d_representation): The output graph-level
          representations.
        - "node_level_repr" (... 2 max_nodes d_representation): The output node-level
          representations.
        - "round" (optional) (...): The current round number.

    Output:
        - "node_selected_logits" (... 2*max_nodes): A logit for each node, indicating
          the probability that this node should be sent as a message to the verifier.
        - "decision_logits" (optional) (... 3): A logit for each of the three options:
          guess that the graphs are isomorphic,  guess that the graphs are not
          isomorphic, or continue exchanging messages. Set to zeros when the decider is
          not present.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    agent_name : str
        The name of the agent.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    device : TorchDevice, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    @property
    def in_keys(self):
        if self.decider is not None and self._agent_params.include_round_in_decider:
            return ("graph_level_repr", "node_level_repr", "round")
        else:
            return ("graph_level_repr", "node_level_repr")

    @property
    def in_keys(self):
        if self.decider is not None and self._agent_params.include_round_in_decider:
            return ("graph_level_repr", "node_level_repr", "round")
        else:
            return ("graph_level_repr", "node_level_repr")

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
        protocol_handler: ProtocolHandler,
        *,
        device: Optional[TorchDevice] = None,
    ):
        super().__init__(params, agent_name, protocol_handler, device=device)

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
            d_in=self.params.d_representation,
            d_hidden=self._agent_params.d_node_selector,
            d_out=1,
            num_layers=self._agent_params.num_node_selector_layers,
            out_key="node_selected_logits",
        )

    def forward(
        self,
        body_output: TensorDict,
        hooks: Optional[GraphIsomorphismAgentHooks] = None,
    ) -> TensorDict:
        """Runs the policy head on the given body output.

        Runs the node selector module and the decider module if present.

        Parameters
        ----------
        body_output : TensorDict
            The output of the body module. A tensor dict with keys:

            - "graph_level_repr" (... 2 d_representation): The output graph-level
              representations.
            - "node_level_repr" (... 2 max_nodes d_representation): The output
              node-level representations.

        hooks : GraphIsomorphismAgentHooks, optional
            Hooks to run at various points in the agent forward pass.

        Returns
        -------
        out : TensorDict
            A tensor dict with keys:

            - "node_selected_logits" (... 2*max_nodes): A logit for each node,
              indicating the probability that this node should be sent as a message to
              the verifier.
            - "decision_logits" (... 3): A logit for each of the three options: guess
              that the graphs are isomorphic,  guess that the graphs are not isomorphic,
              or continue exchanging messages. Set to zeros when the decider is not
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


@register_scenario_class(GI_SCENARIO, RandomAgentPolicyHead)
class GraphIsomorphismRandomAgentPolicyHead(
    GraphIsomorphismAgentPart, RandomAgentPolicyHead
):
    """Policy head for the graph isomorphism task yielding a uniform distribution.

    Shapes
    ------
    Input:
        - "graph_level_repr" (... 2 d_representation): The output graph-level
          representations.
        - "node_level_repr" (... 2 max_nodes d_representation): The output node-level
          representations.

    Output:
        - "node_selected_logits" (... 2*max_nodes): A logit for each node, indicating
          the probability that this node should be sent as a message to the verifier.
        - "decision_logits" (... 3): A logit for each of the three options: guess that
          the graphs are isomorphic, guess that the graphs are not isomorphic, or
          continue exchanging messages.
    """

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
        protocol_handler: ProtocolHandler,
        *,
        device: Optional[TorchDevice] = None,
    ):
        super().__init__(params, agent_name, protocol_handler, device=device)

        # Determine if we should output a decision too
        self.decider = agent_name == "verifier"

    def forward(
        self,
        body_output: TensorDict,
        hooks: Optional[GraphIsomorphismAgentHooks] = None,
    ) -> TensorDict:
        """Outputs a uniform distribution.

        Parameters
        ----------
        body_output : TensorDict
            The output of the body module. A tensor dict with keys:

            - "graph_level_repr" (... 2 d_representation): The output graph-level
              representations.
            - "node_level_repr" (... 2 max_nodes d_representation): The output node-level
              representations.

        hooks : GraphIsomorphismAgentHooks, optional
            Hooks to run at various points in the agent forward pass.

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


@register_scenario_class(GI_SCENARIO, AgentValueHead)
class GraphIsomorphismAgentValueHead(GraphIsomorphismAgentHead, AgentValueHead):
    """Value head for the graph isomorphism task.

    Takes as input the output of the agent body and outputs a value function.

    Shapes
    ------
    Input:
        - "graph_level_repr" (... 2 d_representation): The output graph-level
          representations.
        - "round" (optional) (...): The current round number.

    Output:
        - "value" (...): The estimated value for each batch item

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    agent_name : str
        The name of the agent.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    device : TorchDevice, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    out_keys = ("value",)

    @property
    def in_keys(self):
        if self._agent_params.include_round_in_value:
            return ("graph_level_repr", "round")
        else:
            return "graph_level_repr"

    def __init__(
        self,
        params: Parameters,
        agent_name: str,
        protocol_handler: ProtocolHandler,
        *,
        device: Optional[TorchDevice] = None,
    ):
        super().__init__(params, agent_name, protocol_handler, device=device)

        self.value_mlp = self._build_mlp()

    def _build_mlp(self) -> TensorDictModule:
        """Builds the module which computes the value function.

        Returns
        -------
        value_mlp : TensorDictModule
            The value module.
        """
        return self._build_graph_level_mlp(
            d_in=self.params.d_representation,
            d_hidden=self._agent_params.d_value,
            d_out=1,
            num_layers=self._agent_params.num_value_layers,
            include_round=self._agent_params.include_round_in_value,
            out_key="value",
            squeeze=True,
        )

    def forward(
        self,
        body_output: TensorDict,
        hooks: Optional[GraphIsomorphismAgentHooks] = None,
    ) -> TensorDict:
        """Runs the value head on the given body output.

        Parameters
        ----------
        body_output : TensorDict
            The output of the body module. A tensor dict with keys:

            - "graph_level_repr" (... 2 d_representation): The output graph-level
              representations.

        hooks : GraphIsomorphismAgentHooks, optional
            Hooks to run at various points in the agent forward pass.

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


@register_scenario_class(GI_SCENARIO, ConstantAgentValueHead)
class GraphIsomorphismConstantAgentValueHead(
    GraphIsomorphismAgentHead, ConstantAgentValueHead
):
    """A constant value head for the graph isomorphism task.

    Shapes
    ------
    Input:
        - "graph_level_repr" (... 2 d_representation): The output graph-level
          representations.
        - "node_level_repr" (... 2 max_nodes d_representation): The output node-level
          representations.

    Output:
        - "value" (...): The 'value' for each batch item, which is a constant zero.
    """

    in_keys = ("graph_level_repr", "node_level_repr")
    out_keys = ("value",)

    def forward(
        self,
        body_output: TensorDict,
        hooks: Optional[GraphIsomorphismAgentHooks] = None,
    ) -> TensorDict:
        """Returns a constant value.

        Parameters
        ----------
        body_output : TensorDict
            The output of the body module. A tensor dict with keys:

            - "graph_level_repr" (... 2 1): The output graph-level
              representations.
            - "node_level_repr" (... 2 max_nodes 1): The output node-level
              representations.

        hooks : GraphIsomorphismAgentHooks, optional
            Hooks to run at various points in the agent forward pass.

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


@register_scenario_class(GI_SCENARIO, SoloAgentHead)
class GraphIsomorphismSoloAgentHead(GraphIsomorphismAgentHead, SoloAgentHead):
    """Solo agent head for the graph isomorphism task.

    Solo agents try to solve the task on their own, without interacting with another
    agents.

    Shapes
    ------
    Input:
        - "graph_level_repr" (... 2 d_representation): The output graph-level
          representations.

    Output:
        - "decision_logits" (... 2): A logit for each of the two options: guess that the
          graphs are isomorphic, or guess that the graphs are not isomorphic.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    agent_name : str
        The name of the agent.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    device : TorchDevice, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    in_keys = ("graph_level_repr",)
    out_keys = ("decision_logits",)

    def __init__(
        self,
        params: Parameters,
        agent_name: str,
        protocol_handler: ProtocolHandler,
        *,
        device: Optional[TorchDevice] = None,
    ):
        super().__init__(params, agent_name, protocol_handler, device=device)

        self.decider = self._build_decider(d_out=2, include_round=False)

    def forward(
        self,
        body_output: TensorDict,
        hooks: Optional[GraphIsomorphismAgentHooks] = None,
    ) -> TensorDict:
        """Runs the solo agent head on the given body output.

        Parameters
        ----------
        body_output : TensorDict
            The output of the body module. A tensor dict with keys:

            - "graph_level_repr" (... 2 d_representation): The output graph-level
              representations.

        hooks : GraphIsomorphismAgentHooks, optional
            Hooks to run at various points in the agent forward pass.

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


@register_scenario_class(GI_SCENARIO, CombinedBody)
class GraphIsomorphismCombinedBody(CombinedBody):
    """A module which combines the agent bodies for the graph isomorphism task.

    Shapes
    ------
    Input:
        - "round" (...): The current round number.
        - "x" (... pair node feature): The graph node features (message history)
        - "adjacency" (... pair node node): The adjacency matrices.
        - "message" (... pair node), optional: The most recent message.
        - "node_mask" (... pair node): Which nodes actually exist.

    Output:
        - ("agents", "node_level_repr") (... agents max_nodes d_representation): The
          output node-level representations.
        - ("agents", "graph_level_repr") (... agents d_representation): The output
          graph-level representations.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    bodies : dict[str, GraphIsomorphismAgentBody]
        The agent bodies to combine.
    """

    in_keys = ("round", "x", "adjacency", "message", "node_mask")
    out_keys = (("agents", "node_level_repr"), ("agents", "graph_level_repr"))

    def __init__(
        self,
        params: Parameters,
        protocol_handler: ProtocolHandler,
        bodies: dict[str, GraphIsomorphismAgentBody],
    ) -> None:
        super().__init__(params, protocol_handler, bodies)

    def forward(
        self,
        data: TensorDictBase,
        hooks: Optional[GraphIsomorphismAgentHooks] = None,
    ) -> TensorDict:
        round: Int[Tensor, "batch"] = data["round"]

        # Run the agent bodies
        body_outputs: dict[str, TensorDict] = {}
        for agent_name in self._agent_names:
            # Build the input dict for the agent body
            input_dict = {}
            for key in self.bodies[agent_name].in_keys:
                if key == "ignore_message":
                    input_dict[key] = round == 0
                else:
                    if key == "message" and "message" not in data.keys():
                        continue
                    input_dict[key] = data[key]
            input_td = TensorDict(
                input_dict,
                batch_size=data.batch_size,
            )

            # Run the agent body
            body_outputs[agent_name] = self.bodies[agent_name](input_td, hooks=hooks)

        # Stack the outputs
        node_level_repr = torch.stack(
            [body_outputs[name]["node_level_repr"] for name in self._agent_names],
            dim=-4,
        )
        graph_level_repr = torch.stack(
            [body_outputs[name]["graph_level_repr"] for name in self._agent_names],
            dim=-3,
        )

        return data.update(
            dict(
                agents=dict(
                    node_level_repr=node_level_repr,
                    graph_level_repr=graph_level_repr,
                )
            )
        )


@register_scenario_class(GI_SCENARIO, CombinedPolicyHead)
class GraphIsomorphismCombinedPolicyHead(CombinedPolicyHead):
    """A module which combines the agent policy heads for the graph isomorphism task.

    Shapes
    ------
    Input:
        - ("agents", "node_level_repr") (... agents 2*max_nodes d_representation): The
          output node-level representations.
        - ("agents", "graph_level_repr") (... agents d_representation): The output
          graph-level representations.
        - "round" (...): The current round number.
        - "node_mask" (... pair node): Which nodes actually exist.
        - "message" (... pair node): The most recent message.
        - "ignore_message" (...): Whether to ignore the message
        - "decision_restriction" (...): The restriction on what decisions are allowed.

    Output:
        - ("agents", "node_selected_logits") (... agents 2*max_nodes): A logit for each
          node, indicating the probability that this node should be sent as a message to
          the verifier.
        - ("agents", "decision_logits") (... agents 3): A logit for each of the three
          options: guess that the graphs are isomorphic, guess that the graphs are not
          isomorphic, or continue exchanging messages.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    policy_heads : dict[str, GraphIsomorphismAgentPolicyHead]
        The agent policy heads to combine.
    """

    in_keys = (
        ("agents", "node_level_repr"),
        ("agents", "graph_level_repr"),
        "round",
        "node_mask",
        "message",
        "ignore_message",
        "decision_restriction",
    )
    out_keys = (
        ("agents", "node_selected_logits"),
        ("agents", "decision_logits"),
    )

    def __init__(
        self,
        params: Parameters,
        protocol_handler: ProtocolHandler,
        policy_heads: dict[str, GraphIsomorphismAgentPolicyHead],
    ):
        super().__init__(params, protocol_handler, policy_heads)

    def forward(
        self,
        head_output: TensorDictBase,
        hooks: Optional[GraphIsomorphismAgentHooks] = None,
    ) -> TensorDict:
        """Run the agent policy heads and combine their outputs.

        Parameters
        ----------
        tensordict : TensorDictBase
            The input to the value heads.

        hooks : GraphIsomorphismAgentHooks, optional
            Hooks to run at various points in the agent forward pass.

        Returns
        -------
        tensordict: TensorDict
            The tensordict update in place with the output of the value heads.
        """

        # Run the policy heads to obtain the probability distributions
        policy_outputs: dict[str, TensorDict] = {}
        for i, agent_name in enumerate(self._agent_names):
            input_td = TensorDict(
                dict(
                    node_level_repr=head_output["agents", "node_level_repr"][
                        ..., i, :, :, :
                    ],
                    graph_level_repr=head_output["agents", "graph_level_repr"][
                        ..., i, :, :
                    ],
                    round=head_output["round"],
                ),
                batch_size=head_output.batch_size,
            )
            policy_outputs[agent_name] = self.policy_heads[agent_name](
                input_td, hooks=hooks
            )

            # Make sure the provers only selects nodes in the opposite graph to the most
            # recent message
            if agent_name in self.protocol_handler.prover_names:
                message: Tensor = head_output["message"]
                max_num_nodes = head_output["agents", "node_level_repr"].shape[-2]
                other_graph = (message[..., 0, :].max(dim=-1)[0] > 0).long()
                node_ok_mask = F.one_hot(other_graph, num_classes=2)
                node_ok_mask = node_ok_mask.bool()
                node_ok_mask = repeat(
                    node_ok_mask, "... pair -> ... (pair node)", node=max_num_nodes
                )
                policy_outputs[agent_name]["node_selected_logits"] = torch.where(
                    node_ok_mask,
                    policy_outputs[agent_name]["node_selected_logits"],
                    torch.full_like(
                        policy_outputs[agent_name]["node_selected_logits"], -1e9
                    ),
                )

        # Stack the outputs
        node_selected_logits = torch.stack(
            [
                policy_outputs[name]["node_selected_logits"]
                for name in self._agent_names
            ],
            dim=-2,
        )
        decision_logits = torch.stack(
            [policy_outputs[name]["decision_logits"] for name in self._agent_names],
            dim=-2,
        )

        # Make sure the agents only select nodes which exist
        node_mask_flatter = rearrange(
            head_output["node_mask"], "... pair node -> ... 1 (pair node)"
        )
        node_selected_logits = torch.where(
            node_mask_flatter,
            node_selected_logits,
            torch.full_like(node_selected_logits, -1e9),
        )

        # Make sure the verifier only selects decisions which are allowed
        decision_logits = self._restrict_decisions(
            head_output["decision_restriction"], decision_logits
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


@register_scenario_class(GI_SCENARIO, CombinedValueHead)
class GraphIsomorphismCombinedValueHead(CombinedValueHead):
    """A module which combines the agent value heads for the graph isomorphism task.

    Shapes
    ------
    Input:
        - ("agents", "graph_level_repr") (... agents d_representation): The output
          graph-level representations.
        - "round" (...): The current round number.

    Output:
        - ("agents", "value") (... agents): The estimated value for each batch item

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    value_heads : dict[str, GraphIsomorphismAgentValueHead]
        The agent value heads to combine.
    """

    in_keys = (("agents", "graph_level_repr"), "round")
    out_keys = (("agents", "value"),)

    def __init__(
        self,
        params: Parameters,
        protocol_handler: ProtocolHandler,
        value_heads: dict[str, GraphIsomorphismAgentValueHead],
    ):
        super().__init__(params, protocol_handler, value_heads)

    def forward(
        self,
        head_output: TensorDictBase,
        hooks: Optional[GraphIsomorphismAgentHooks] = None,
    ) -> TensorDict:
        """Run the agent value heads and combine their values.

        Parameters
        ----------
        tensordict : TensorDictBase
            The input to the value heads. Should contain the keys:

            - ("agents", "graph_level_repr"): The node-level representation from the
              body.

        hooks : GraphIsomorphismAgentHooks, optional
            Hooks to run at various points in the agent forward pass.

        Returns
        -------
        tensordict: TensorDict
            The tensordict update in place with the output of the value heads.
        """

        # Run the policy heads to obtain the value estimates
        value_outputs: dict[str, TensorDict] = {}
        for i, agent_name in enumerate(self._agent_names):
            input_td = TensorDict(
                dict(
                    node_level_repr=head_output["agents", "node_level_repr"][
                        ..., i, :, :, :
                    ],
                    graph_level_repr=head_output["agents", "graph_level_repr"][
                        ..., i, :, :
                    ],
                    round=head_output["round"],
                ),
                batch_size=head_output.batch_size,
            )
            value_outputs[agent_name] = self.value_heads[agent_name](
                input_td, hooks=hooks
            )

        # Stack the outputs
        value = torch.stack(
            [value_outputs[name]["value"] for name in self._agent_names], dim=-1
        )

        return head_output.update(
            dict(
                agents=TensorDict(
                    dict(value=value),
                    batch_size=head_output.batch_size,
                )
            ),
        )
