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
from typing import Optional, Any, Iterable, ClassVar
from dataclasses import dataclass
from functools import partial
import re

import torch
from torch.nn import (
    Linear,
    TransformerEncoder,
    TransformerEncoderLayer,
    Sequential,
)
from torch import Tensor
import torch.nn.functional as F
from torch.linalg import vector_norm
from torch.nn.parameter import Parameter as TorchParameter

from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.utils import NestedKey

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from jaxtyping import Float, Int

from nip.scenario_base import (
    AgentHooks,
    TensorDictAgentPartMixin,
    AgentBody,
    DummyAgentBody,
    TensorDictDummyAgentPartMixin,
    AgentHead,
    AgentPolicyHead,
    RandomAgentPolicyHead,
    AgentValueHead,
    ConstantAgentValueHead,
    SoloAgentHead,
    CombinedBody,
    CombinedPolicyHead,
    CombinedValueHead,
    Agent,
)
from nip.factory import register_scenario_class
from nip.parameters import (
    HyperParameters,
    GraphIsomorphismAgentParameters,
    RandomAgentParameters,
    ScenarioType,
    InteractionProtocolType,
)
from nip.experiment_settings import ExperimentSettings
from nip.protocols import ProtocolHandler
from nip.utils.torch import (
    ACTIVATION_CLASSES,
    PairedGaussianNoise,
    PairInvariantizer,
    GIN,
    Squeeze,
    BatchNorm1dSimulateBatchDims,
    OneHot,
    TensorDictCat,
    Print,
    TensorDictPrint,
)
from nip.utils.types import TorchDevice

GI_SCENARIO = "graph_isomorphism"


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


class GraphIsomorphismAgentPart(TensorDictAgentPartMixin, ABC):
    """Base class for all graph isomorphism agent parts.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The settings of the experiment.
    agent_name : str
        The name of the agent.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    device : TorchDevice, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    agent_params: GraphIsomorphismAgentParameters

    def __init__(
        self,
        hyper_params: HyperParameters,
        settings: ExperimentSettings,
        agent_name: str,
        protocol_handler: ProtocolHandler,
    ):
        super().__init__(
            hyper_params=hyper_params,
            settings=settings,
            agent_name=agent_name,
            protocol_handler=protocol_handler,
        )

        self.activation_function = ACTIVATION_CLASSES[
            self.agent_params.activation_function
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


class GraphIsomorphismDummyAgentPart(TensorDictDummyAgentPartMixin, ABC):
    """Base class for all graph isomorphism dummy agent parts."""

    agent_params: RandomAgentParameters


@register_scenario_class(GI_SCENARIO, AgentBody)
class GraphIsomorphismAgentBody(GraphIsomorphismAgentPart, AgentBody):
    """Agent body for the graph isomorphism task.

    Takes as input a pair of graphs, message history and the most recent message and
    outputs node-level and graph-level representations.

    Shapes
    ------
    Input:

    - "x" (... round channel position pair node): The graph node features (message
      history)
    - "adjacency" (... pair node node): The graph adjacency matrices
    - "message" (... channel position pair node), optional: The most recent message from
      the other agent
    - "node_mask" (... pair node): Which nodes actually exist
    - "ignore_message" (...), optional: Whether to ignore any values in "message". For
      example, in the first round the there is no message, and the "message" field is
      set to a dummy value.
    - "linear_message_history" : (... round channel position linear_message), optional:
      The linear message history, if using

    Output:

    - "graph_level_repr" (... 2 d_representation): The output graph-level
      representations.
    - "node_level_repr" (... 2 max_nodes d_representation): The output node-level
      representations.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The settings of the experiment.
    agent_name : str
        The name of the agent.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    device : TorchDevice, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    agent_level_in_keys = ("ignore_message",)

    @property
    def env_level_in_keys(self) -> tuple[str, ...]:
        """The environment-level input keys for the agent body."""

        env_level_in_keys = ("x", "adjacency", "message", "node_mask")

        if self.hyper_params.include_linear_message_space:
            env_level_in_keys = (*env_level_in_keys, "linear_message_history")

        return env_level_in_keys

    agent_level_out_keys = ("graph_level_repr", "node_level_repr")

    @property
    def d_gnn_out(self) -> int:
        """The dimensionality of the GNN output after the stream and feature dims."""
        if (
            self.agent_params.use_dual_gnn
            and not self.agent_params.use_manual_architecture
        ):
            return 2 * self.agent_params.d_gnn
        else:
            return self.agent_params.d_gnn

    def __init__(
        self,
        hyper_params: HyperParameters,
        settings: ExperimentSettings,
        agent_name: str,
        protocol_handler: ProtocolHandler,
    ):
        super().__init__(
            hyper_params=hyper_params,
            settings=settings,
            agent_name=agent_name,
            protocol_handler=protocol_handler,
        )

        if self.agent_params.normalize_message_history:
            raise NotImplementedError(
                "Message history normalization is implemented any more."
            )

        # Build up the GNN module
        self.gnn = self._build_gnn()

        if self.agent_params.use_manual_architecture:
            if hyper_params.interaction_protocol != "nip":
                raise NotImplementedError(
                    "Manual graph isomorphism agent architecture is only supported for "
                    "the NIP interaction protocol."
                )

            # Build the encoder going from the gnn output to the representation space
            self.representation_encoder = self._build_representation_encoder(
                self.agent_params.d_gnn
            )

        else:
            # Build the global pooling module, which computes the graph-level
            # representation from the GNN output
            self.global_pooling = self._build_global_pooling()

            # Build the encoder going from the GNN to the transformer
            self.gnn_transformer_encoder = self._build_gnn_transformer_encoder()

            # Build the transformer
            self.transformer = self._build_transformer()

            # Build the encoder going from the transformer output to the representation
            # space
            self.representation_encoder = self._build_representation_encoder(
                self.agent_params.d_transformer
            )

        self._init_weights()

    def _build_gnn(self) -> TensorDictSequential:
        """Build the GNN module for an agent.

        Shapes
        ------
        Input:

        - "gnn_repr" (... stream pair node feature): The input graph node features
        - "adjacency" (... stream pair node node): The graph adjacency matrices

        Output:

        - "gnn_repr" (... stream pair node feature): The output graph node features

        Returns
        -------
        gnn : TensorDictSequential
            The GNN module, which takes as input a TensorDict with keys "gnn_repr",
            "adjacency" and "node_mask".
        """

        d_input = (
            self.max_message_rounds
            * self.num_visible_message_channels
            * self.hyper_params.message_size
        )

        # Build up the GNN
        gnn_layers = []
        gnn_layers.append(
            TensorDictModule(
                Linear(d_input, self.agent_params.d_gnn),
                in_keys=("gnn_repr",),
                out_keys=("gnn_repr",),
            )
        )

        for _ in range(self.agent_params.num_gnn_layers):
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
                            self.agent_params.d_gnn,
                            self.agent_params.d_gin_mlp,
                        ),
                        self.activation_function(),
                        Linear(
                            self.agent_params.d_gin_mlp,
                            self.agent_params.d_gnn,
                        ),
                    ),
                    feature_in_key="gnn_repr",
                    feature_out_key="gnn_repr",
                    adjacency_key="adjacency_stream",
                    node_mask_key="node_mask_stream",
                    vmap_compatible=True,
                )
            )

        gnn = TensorDictSequential(*gnn_layers)

        gnn = gnn.to(self.device)

        return gnn

    def _build_gnn_transformer_encoder(
        self,
    ) -> Linear:
        """Build the encoder layer which translates the GNN output to transformer input.

        This is a simple linear layer, where the number of input features is normally
        ``d_gnn + 3``, where the extra features encode which graph-level representation
        the token is, if any and whether a node is in the most recent message from the
        other agent. When we are using a linear message space, the number of input
        features is increased by the number of rounds times the number of message
        features.

        Returns
        -------
        gnn_transformer_encoder : torch.nn.Linear
            The encoder module

        """

        in_features = (
            self.d_gnn_out
            + 2
            + self.num_visible_message_channels * self.hyper_params.message_size
        )
        if self.hyper_params.include_linear_message_space:
            in_features += (
                self.protocol_handler.max_message_rounds
                * self.num_visible_message_channels
                * self.hyper_params.message_size
                * self.hyper_params.d_linear_message_space
            )

        return Linear(
            in_features,
            self.agent_params.d_transformer,
            device=self.device,
        )

    def _build_transformer(self) -> TransformerEncoder:
        """Build the transformer module for an agent.

        Returns
        -------
        transformer : torch.nn.TransformerEncoder
            The transformer module.
        """

        if self.agent_params.num_transformer_layers == 0:
            return None

        transformer = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=self.agent_params.d_transformer,
                nhead=self.agent_params.num_heads,
                batch_first=True,
                dropout=self.agent_params.transformer_dropout,
                dim_feedforward=self.agent_params.d_transformer_mlp,
            ),
            num_layers=self.agent_params.num_transformer_layers,
        )

        transformer = transformer.to(self.device)

        return transformer

    def _build_global_pooling(self) -> Sequential:
        """Build a pooling layer which computes the graph-level representation.

        The module consists of a global sum pooling layer, an optional batch norm layer,
        a paired Gaussian noise layer and an optional pair invariant pooling layer.

        Shapes
        ------
        Input:

        - "gnn_repr" (... pair node feature*stream): The input graph node features

        Output:

        - "pooled_gnn_output" (... pair feature*stream): The output graph-level
          representation

        Returns
        -------
        global_pooling : torch.nn.Sequential
            The global pooling module.
        """

        layers = [
            Reduce(
                "... pair max_nodes d_gnn_out -> ... pair d_gnn_out",
                "sum",
            ),
        ]

        if self.agent_params.use_batch_norm:
            layers.append(
                BatchNorm1dSimulateBatchDims(
                    num_features=self.d_gnn_out,
                    track_running_stats=self.hyper_params.functionalize_modules,
                )
            )

        layers.append(
            PairedGaussianNoise(sigma=self.agent_params.noise_sigma, pair_dim=-2),
        )

        if self.agent_params.use_pair_invariant_pooling:
            layers.append(PairInvariantizer(pair_dim=-2))

        global_pooling = Sequential(*layers)

        global_pooling = global_pooling.to(self.device)

        return global_pooling

    def _build_representation_encoder(self, d_input: int) -> Linear:
        """Build the encoder layer which translates to the representation space.

        This is a simple linear layer ensures that the number of output features is
        ``hyper_params.d_representation``.

        Parameters
        ----------
        d_input : int
            The number of input features.

        Returns
        -------
        representation_encoder : torch.nn.Linear
            The encoder module
        """
        return Linear(
            d_input,
            self.hyper_params.d_representation,
            device=self.device,
        )

    def forward(
        self,
        data: TensorDictBase,
        hooks: Optional[GraphIsomorphismAgentHooks] = None,
    ) -> TensorDict:
        """Obtain graph-level and node-level representations by running components.

        Runs the GNN, pools the output, puts everything through a linear encoder, then
        runs the transformer on this.

        Parameters
        ----------
        data : TensorDictBase
            The data to run the GNN and transformer on. A TensorDictBase with keys:

            - "x" (... round channel position pair node): The graph node features
              (message history)
            - "adjacency" (... pair node node): The graph adjacency matrices
            - "message" (... channel position pair node), optional: The most recent
              message from the other agent
            - "node_mask" (... pair node), optional: Which nodes actually exist or a
              GraphIsomorphism data object.
            - "ignore_message" (...), optional: Whether to ignore any values in
              "message". For example, in the first round the there is no message, and
              the "message" field is set to a dummy value.
            - "linear_message_history" : (... round channel position linear_message),
              optional: The linear message history, if using.

        hooks : GraphIsomorphismAgentHooks, optional
            Hooks to run at various points in the agent forward pass.

        Returns
        -------
        out : TensorDict
            A tensor dict with keys:

            - "graph_level_repr" (... pair d_representation): The output graph-level
              representations.
            - "node_level_repr" (... pair max_nodes d_representation): The output
              node-level representations.
        """

        batch_size = data.batch_size
        max_num_nodes = data["x"].shape[-1]

        # Put the round dimension at the end
        data["x_rearranged"] = rearrange(
            data["x"],
            "... round channel position pair node "
            "-> ... pair node (round channel position)",
        )

        # If we are using the manual architecture, skip everything else and run that
        if self.agent_params.use_manual_architecture:
            return self._run_manual_architecture(data, hooks)

        data = data.update(dict(gnn_repr=data["x_rearranged"]))

        # Add the stream dimension, with a vector of zeros when we're using a dual GNN
        gnn_repr = data["gnn_repr"]
        if self.agent_params.use_dual_gnn:
            gnn_repr = torch.stack(
                (gnn_repr, torch.zeros_like(gnn_repr)),
                dim=-4,
            )
        else:
            gnn_repr = rearrange(
                gnn_repr,
                "... pair node feature -> ... 1 pair node feature",
            )
        data = data.update(
            dict(
                gnn_repr=gnn_repr,
                adjacency_stream=rearrange(
                    data["adjacency"],
                    "... pair node1 node2 -> ... 1 pair node1 node2",
                ),
                node_mask_stream=rearrange(
                    data["node_mask"],
                    "... pair node -> ... 1 pair node",
                ),
            )
        )

        # Run the GNN on the graphs
        # (batch, stream, pair, max_nodes, d_gnn)
        gnn_output = self.gnn(data)["gnn_repr"]

        self._run_recorder_hook(hooks, "gnn_output", gnn_output)

        if self.agent_params.gnn_output_digits is not None:
            gnn_output = torch.round(
                gnn_output, decimals=self.agent_params.gnn_output_digits
            )

        self._run_recorder_hook(hooks, "gnn_output_rounded", gnn_output)

        # Combine the stream and feature dimensions
        gnn_output = rearrange(
            gnn_output,
            "... stream pair node feature -> ... pair node (feature stream)",
        )

        # Obtain the graph-level representations by pooling
        # (batch, pair, stream * d_gnn)
        pooled_gnn_output = self.global_pooling(gnn_output)

        self._run_recorder_hook(hooks, "pooled_gnn_output", pooled_gnn_output)

        # Merge the pair and node dimensions
        # (batch, pair * node, stream * d_gnn)
        gnn_output_flatter = rearrange(
            gnn_output, "... pair node feature -> ... (pair node) feature"
        )

        self._run_recorder_hook(hooks, "gnn_output_flatter", gnn_output_flatter)

        # Add the graph-level representations to the transformer input
        # (..., 2 + 2 * node, stream * d_gnn)
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
        # (..., 2 + 2 * node, channel*position)
        if "message" in data.keys():
            message_feature = rearrange(
                data["message"],
                "... channel position pair node -> ... channel position (pair node)",
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
                data["ignore_message"][..., None, None, None], 0, message_feature
            )
            message_feature = rearrange(
                message_feature,
                "... channel position token -> ... token (channel position)",
            )
        else:
            message_feature = torch.zeros(
                (
                    *transformer_input.shape[:-1],
                    self.num_visible_message_channels * self.hyper_params.message_size,
                ),
                device=transformer_input.device,
                dtype=transformer_input.dtype,
            )

        self._run_recorder_hook(hooks, "message_feature", message_feature)

        # Turn the linear message history into a feature, if using
        if (
            self.hyper_params.include_linear_message_space
            and "linear_message_history" in data.keys()
        ):
            linear_message_feature = repeat(
                data["linear_message_history"],
                "... round channel position linear_message "
                "-> ... token (round channel position linear_message)",
                token=2 + 2 * max_num_nodes,
            )
            linear_message_feature = torch.where(
                data["ignore_message"][..., None, None], 0, linear_message_feature
            )
        elif self.hyper_params.include_linear_message_space:
            num_linear_message_features = (
                self.protocol_handler.max_message_rounds
                * self.num_visible_message_channels
                * self.hyper_params.message_size
                * self.hyper_params.d_linear_message_space
            )
            linear_message_feature = torch.zeros(
                (*transformer_input.shape[:-1], num_linear_message_features),
                device=transformer_input.device,
                dtype=transformer_input.dtype,
            )
        else:
            linear_message_feature = torch.zeros(
                (*transformer_input.shape[:-1], 0),
                device=transformer_input.device,
                dtype=transformer_input.dtype,
            )

        # Concatenate everything together
        # (..., 2 + 2 * node, streams * d_gnn + 2 + channel*position + linear_message)
        transformer_input = torch.cat(
            (
                transformer_input,
                pooled_feature,
                message_feature,
                linear_message_feature,
            ),
            dim=-1,
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
            batch_size=batch_size,
        )

    def _run_manual_architecture(
        self,
        data: TensorDictBase,
        hooks: Optional[GraphIsomorphismAgentHooks] = None,
    ) -> TensorDict:
        """Run the body part of the manual architecture.

        The verifier symmetrises the message history so that the information in round
        ``2i`` is the same as the information in round ``2i + 1``, while the prover
        ignores the message history completely.

        The message history is then run through a linear layer to make it the right
        size, then run through a GNN to get the node-level representations. The
        graph-level representations are then obtained by summing the node-level
        representations.

        Parameters
        ----------
        data : TensorDictBase
            The data to run the GNN and transformer on. A TensorDictBase with keys:

            - "x_rearranged" (... pair node round): The graph node features (message
              history) rearranged to but the round dimension at the end
            - "adjacency" (... pair node node): The graph adjacency matrices
            - "message" (...): The most recent message from the other agent
            - "node_mask" (... pair node): Which nodes actually exist or a
              GraphIsomorphism data object.
            - "ignore_message" (...): Whether to ignore any values in "message". For
              example, in the first round the there is no message, and the "message"
              field is set to a dummy value.
        hooks : GraphIsomorphismAgentHooks, optional
            Hooks to run at various points in the agent forward pass.

        Returns
        -------
        out : TensorDict
            A tensor dict with keys:

            - "graph_level_repr" (... pair d_representation): The output graph-level
              representations.
            - "node_level_repr" (... pair max_nodes d_representation): The output
              node-level representations.
        """

        if self.is_verifier:
            # Symmetrise the message history
            rounded_rounds = (self.protocol_handler.max_message_rounds // 2) * 2
            gnn_repr = data["x_rearranged"].clone()
            gnn_repr[..., range(0, rounded_rounds, 2)] += data["x_rearranged"][
                ..., range(1, rounded_rounds, 2)
            ]
            gnn_repr[..., range(1, rounded_rounds, 2)] += data["x_rearranged"][
                ..., range(0, rounded_rounds, 2)
            ]
        else:
            gnn_repr = torch.zeros_like(data["x_rearranged"])

        data = data.update(
            dict(
                gnn_repr=rearrange(
                    gnn_repr,
                    "... pair node feature -> ... 1 pair node feature",
                ),
                adjacency_stream=rearrange(
                    data["adjacency"],
                    "... pair node1 node2 -> ... 1 pair node1 node2",
                ),
                node_mask_stream=rearrange(
                    data["node_mask"],
                    "... pair node -> ... 1 pair node",
                ),
            )
        )

        # Run the GNN on the graphs
        # (batch, stream, pair, max_nodes, d_gnn)
        gnn_output = self.gnn(data)["gnn_repr"]

        self._run_recorder_hook(hooks, "gnn_output", gnn_output)

        # Remove the stream dimension
        gnn_output = gnn_output.squeeze(-4)

        # Run the GNN output through the representation encoder to get the node-level
        # representations
        node_level_repr = self.representation_encoder(gnn_output)

        # Obtain the graph-level representations by pooling
        graph_level_repr = node_level_repr.sum(dim=-2)

        return TensorDict(
            dict(
                graph_level_repr=graph_level_repr,
                node_level_repr=node_level_repr,
            ),
            batch_size=data.batch_size,
        )

    def to(self, device: Optional[TorchDevice] = None):
        """Move the agent body to a new device.

        Parameters
        ----------
        device : TorchDevice, optional
            The device to move the agent body to. If not given, the CPU is used.

        Returns
        -------
        self : GraphIsomorphismAgentBody
            The agent body on the new device.
        """
        super().to(device)
        self.device = device
        if self.agent_params.normalize_message_history:
            self.message_history_normalizer.to(device)
        self.gnn.to(device)
        self.representation_encoder.to(device)
        if not self.agent_params.use_manual_architecture:
            self.global_pooling.to(device)
            self.global_pooling[-1].to(device)
            self.gnn_transformer_encoder.to(device)
            if self.transformer is not None:
                self.transformer.to(device)
        return self


@register_scenario_class(GI_SCENARIO, DummyAgentBody)
class GraphIsomorphismDummyAgentBody(GraphIsomorphismDummyAgentPart, DummyAgentBody):
    """Dummy agent body for the graph isomorphism task."""

    env_level_in_keys = ("x",)
    agent_level_out_keys = ("graph_level_repr", "node_level_repr")

    def forward(
        self,
        data: TensorDictBase,
        hooks: Optional[GraphIsomorphismAgentHooks] = None,
    ) -> TensorDict:
        """Return dummy outputs.

        Parameters
        ----------
        data : TensorDictBase
            A TensorDictBase with keys:

            - "x" (... round channel position pair node): The graph node features
              (message history)

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
        max_num_nodes = data["x"].shape[-1]

        # The dummy graph-level representations
        graph_level_repr = torch.zeros(
            *data.batch_size,
            2,
            self.hyper_params.d_representation,
            device=self.device,
            dtype=torch.float32,
        )

        # The dummy node-level representations
        node_level_repr = torch.zeros(
            *data.batch_size,
            2,
            max_num_nodes,
            self.hyper_params.d_representation,
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
        """Build an MLP which acts on the node-level representations.

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
        """Build an MLP which acts on the node-level representations.

        Shapes
        ------
        Input:

        - "graph_level_repr": (... 2 d_in)

        Output:

        - "graph_level_mlp_output": (... d_out)

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
        """Build the module which produces a graph-pair level output.

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
            include_round = self.agent_params.include_round_in_decider

        return self._build_graph_level_mlp(
            d_in=self.hyper_params.d_representation,
            d_hidden=self.agent_params.d_decider,
            d_out=d_out,
            num_layers=self.agent_params.num_decider_layers,
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

    - "node_selected_logits" (... channel position 2*max_nodes): A logit for each node,
      indicating the probability that this node should be sent as a message.
    - "decision_logits" (optional) (... 3): A logit for each of the three options: guess
      that the graphs are isomorphic,  guess that the graphs are not isomorphic, or
      continue exchanging messages. Set to zeros when the decider is not present.
    - "linear_message_selected_logits" (... channel position linear_message) (optional):
      A logit for each linear message, indicating the probability that this linear
      message should be sent as a message.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The settings of the experiment.
    agent_name : str
        The name of the agent.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    device : TorchDevice, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    agent_level_in_keys = ("graph_level_repr", "node_level_repr")

    @property
    def env_level_in_keys(self) -> tuple[str, ...]:
        """The environment-level input keys."""
        if self.has_decider is not None and self.agent_params.include_round_in_decider:
            return ("message", "round")
        else:
            return ("message",)

    @property
    def agent_level_out_keys(self) -> tuple[str, ...]:
        """The agent-level output keys.

        These are the keys that are returned by the forward pass of the agent policy
        head.
        """

        agent_level_out_keys = ("node_selected_logits", "decision_logits")

        if self.hyper_params.include_linear_message_space:
            agent_level_out_keys = (
                *agent_level_out_keys,
                "linear_message_selected_logits",
            )

        return agent_level_out_keys

    def __init__(
        self,
        hyper_params: HyperParameters,
        settings: ExperimentSettings,
        agent_name: str,
        protocol_handler: ProtocolHandler,
    ):
        super().__init__(
            hyper_params=hyper_params,
            settings=settings,
            agent_name=agent_name,
            protocol_handler=protocol_handler,
        )

        self.decider = None

        if self.agent_params.use_manual_architecture:
            if hyper_params.interaction_protocol != "nip":
                raise NotImplementedError(
                    "Manual graph isomorphism agent architecture is only supported for "
                    "the NIP interaction protocol."
                )
        else:
            # Build the node selector module
            self.node_selector = self._build_node_selector()

            # Build the decider module if necessary
            if self.has_decider:
                self.decider = self._build_decider()

        # Build the linear message selector if necessary
        if self.hyper_params.include_linear_message_space:
            self.linear_message_selector = self._build_linear_message_selector()
        else:
            self.linear_message_selector = None

        self._init_weights()

    def _build_node_selector(self) -> TensorDictModule:
        """Build the module which selects which node to send as a message.

        Returns
        -------
        node_selector : TensorDictModule
            The node selector module.
        """
        return self._build_node_level_mlp(
            d_in=self.hyper_params.d_representation,
            d_hidden=self.agent_params.d_node_selector,
            d_out=self.num_visible_message_channels * self.hyper_params.message_size,
            num_layers=self.agent_params.num_node_selector_layers,
            out_key="node_selected_logits",
        )

    def _build_linear_message_selector(self) -> TensorDictModule:
        """Build the module which selects which linear message to send.

        Returns
        -------
        linear_message_selector : TensorDictModule
            The linear message selector module.
        """
        return TensorDictSequential(
            self._build_graph_level_mlp(
                d_in=self.hyper_params.d_representation,
                d_hidden=self.agent_params.d_linear_message_selector,
                d_out=self.num_visible_message_channels
                * self.hyper_params.message_size
                * self.hyper_params.d_linear_message_space,
                num_layers=self.agent_params.num_linear_message_selector_layers,
                include_round=False,
                out_key="linear_message_selected_logits",
            ),
            TensorDictModule(
                Rearrange(
                    "... (channel position linear_message) "
                    "-> ... channel position linear_message",
                    channel=self.num_visible_message_channels,
                    linear_message=self.hyper_params.d_linear_message_space,
                ),
                in_keys="linear_message_selected_logits",
                out_keys="linear_message_selected_logits",
            ),
        )

    def forward(
        self,
        body_output: TensorDict,
        hooks: Optional[GraphIsomorphismAgentHooks] = None,
    ) -> TensorDict:
        """Run the policy head on the given body output.

        Runs the node selector module and the decider module if present.

        Parameters
        ----------
        body_output : TensorDict
            The output of the body module. A tensor dict with keys:

            - "graph_level_repr" (... 2 d_representation): The output graph-level
              representations.
            - "node_level_repr" (... 2 max_nodes d_representation): The output
              node-level representations.
            - "message" (...): The most recent message from the other agent.
            - "round" (optional) (...): The current round number.

        hooks : GraphIsomorphismAgentHooks, optional
            Hooks to run at various points in the agent forward pass.

        Returns
        -------
        out : TensorDict
            A tensor dict with keys:

            - "node_selected_logits" (... channel position 2*max_nodes): A logit for
              each node, indicating the probability that this node should be sent as a
              message to the verifier.
            - "decision_logits" (... 3): A logit for each of the three options: guess
              that the graphs are isomorphic,  guess that the graphs are not isomorphic,
              or continue exchanging messages. Set to zeros when the decider is not
              present.
            - "linear_message_selected_logits" (... channel position linear_message)
              (optional): A logit for each linear message, indicating the probability
              that this linear message should be sent as a message.
        """

        if self.agent_params.use_manual_architecture:
            return self._run_manual_architecture(body_output, hooks)

        out_dict = {}

        node_selected_logits = self.node_selector(body_output)["node_selected_logits"]

        out_dict["node_selected_logits"] = rearrange(
            node_selected_logits,
            "... node (channel position) -> ... channel position node",
            channel=self.num_visible_message_channels,
        )

        if self.decider is not None:
            out_dict["decision_logits"] = self.decider(body_output)["decision_logits"]
        else:
            out_dict["decision_logits"] = torch.zeros(
                (*body_output.batch_size, 3),
                device=self.device,
                dtype=torch.float32,
            )

        if self.hyper_params.include_linear_message_space:
            out_dict["linear_message_selected_logits"] = self.linear_message_selector(
                body_output
            )["linear_message_selected_logits"]

        return TensorDict(out_dict, batch_size=body_output.batch_size)

    def _run_manual_architecture(
        self,
        body_output: TensorDict,
        hooks: Optional[GraphIsomorphismAgentHooks] = None,
    ) -> TensorDict:
        """Run the manually specified algorithm for the agent and environment.

        The verifier waits until the last round, selecting nodes at random. In the last
        round it guesses that the graphs are isomorphic if the graph-level
        representations are close enough, and that they are not isomorphic otherwise.
        When the round number is not provided, it guesses with probability 0.5.

        Without shared reward, the prover selects the node according to its
        representation's similarity to the representation the node selected by the
        verifier in the previous round.

        With shared reward, the prover does this when its graph-level representations
        are close (in which case it believes the graphs are isomorphic) and selects a
        node whose representation is most dissimilar to the representation the node
        selected by the verifier in the previous round when its graph-level
        representations are far apart (in which case it believes the graphs are not
        isomorphic).

        Parameters
        ----------
        body_output : TensorDict
            The output of the body module. A tensor dict with keys:

            - "graph_level_repr" (... 2 d_representation): The output graph-level
              representations.
            - "node_level_repr" (... 2 max_nodes d_representation): The output
              node-level representations.
            - "message" (... channel position 2 max_nodes): The most recent message in
              the channel.
            - "round" (optional) (...): The current round number.

        hooks : GraphIsomorphismAgentHooks, optional
            Hooks to run at various points in the agent forward pass.

        Returns
        -------
        out : TensorDict
            A tensor dict with keys:

            - "node_selected_logits" (... channel position 2*max_nodes): A logit for
              each node, indicating the probability that this node should be sent as a
              message to the verifier.
            - "decision_logits" (... 3): A logit for each of the three
              options: guess that the graphs are isomorphic,  guess that the graphs are
              not isomorphic, or continue exchanging messages. Set to zeros when the
              decider is not present.
        """

        if self.hyper_params.message_size != 1:
            raise NotImplementedError(
                "Manual architecture is only supported for message size 1."
            )

        if self.num_visible_message_channels != 1:
            raise NotImplementedError(
                "Manual architecture is only supported for one message channel."
            )

        graph_level_repr: Float[Tensor, "... 2 feature"] = body_output[
            "graph_level_repr"
        ]
        node_level_repr: Float[Tensor, "... 2 node feature"] = body_output[
            "node_level_repr"
        ]
        message: Float[Tensor, "... channel position 2 node"] = body_output["message"]
        batch_size = body_output.batch_size
        max_num_nodes = node_level_repr.shape[-2]
        d_repr = graph_level_repr.shape[-1]

        out_dict = {}

        # Remove the message channel and position dims (which we assume are singletons)
        message = rearrange(message, "... 1 1 pair node -> ... pair node")

        # The agent's best guess about whether the graphs are isomorphic
        # (batch)
        isomorphic_guess = (
            torch.abs((graph_level_repr[..., 0, :] - graph_level_repr[..., 1, :])).mean(
                dim=-1
            )
            < 1e-6
        )

        if self.is_verifier:
            # Whether to make a guess in this round
            if "round" in body_output.keys():
                make_guess = (
                    body_output["round"] >= self.protocol_handler.max_message_rounds - 2
                )
            else:
                make_guess = torch.rand(*batch_size, device=self.device) < 0.5

            # Turn the guesses into logits. The first two are the guesses, the last is
            # to continue exchanging messages. We set the logits to -1e6 for the
            # impossible options, so that the softmax will give a probability of 0.
            decision_logits = torch.ones(*batch_size, 3, device=self.device) * -1e6
            decision_logits[make_guess, 0] = torch.where(
                ~isomorphic_guess[make_guess], 0, -1e6
            )
            decision_logits[make_guess, 1] = torch.where(
                isomorphic_guess[make_guess], 0, -1e6
            )
            decision_logits[~make_guess, 2] = 0
            out_dict["decision_logits"] = decision_logits

            # The verifier selects nodes at random
            out_dict["node_selected_logits"] = torch.zeros(
                *batch_size,
                self.num_visible_message_channels,
                self.hyper_params.message_size,
                2 * max_num_nodes,
                device=self.device,
                dtype=torch.float32,
            )

        elif self.is_prover:
            message_flattened = rearrange(message, "... pair node -> ... (pair node)")

            # Get the index of the node selected by the verifier in the previous round.
            # This only works when there is only one message.
            # (... channel position)
            message_index = torch.argmax(message_flattened, dim=-1)

            # Compute the similarity between the node sent as a message and the other
            # nodes
            node_level_repr_flattened = rearrange(
                node_level_repr, "... pair node feature -> ... (pair node) feature"
            )
            message_node_repr = torch.gather(
                node_level_repr_flattened,
                -2,
                repeat(
                    message_index, "... -> ... 1 representation", representation=d_repr
                ),
            )
            node_distance = vector_norm(
                message_node_repr - node_level_repr_flattened, dim=-1
            )

            # With shared reward, the prover selects the most dissimilar nodes when the
            # graph-level representations are far apart. Otherwise, it selects the most
            # similar nodes.
            if self.hyper_params.protocol_common.shared_reward:
                select_similar_mask = isomorphic_guess
            else:
                select_similar_mask = torch.ones(
                    *batch_size, device=self.device, dtype=bool
                )
            node_selected_logits = torch.where(
                select_similar_mask[..., None],
                -node_distance * 1e3,
                node_distance * 1e3,
            )
            out_dict["node_selected_logits"] = rearrange(
                node_selected_logits, "... pair_node -> ... 1 1 pair_node"
            )

            out_dict["decision_logits"] = torch.zeros(
                *batch_size, 3, device=self.device, dtype=torch.float32
            )

        return TensorDict(out_dict, batch_size=batch_size)

    def to(self, device: Optional[TorchDevice] = None):
        """Move the agent head to the given device.

        Parameters
        ----------
        device : TorchDevice, optional
            The device to move the agent head to. If not given, the CPU is used.
        """
        super().to(device)
        self.device = device
        if not self.agent_params.use_manual_architecture:
            self.node_selector.to(device)
            if self.decider is not None:
                self.decider.to(device)


@register_scenario_class(GI_SCENARIO, RandomAgentPolicyHead)
class GraphIsomorphismRandomAgentPolicyHead(
    GraphIsomorphismDummyAgentPart, RandomAgentPolicyHead
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

    - "node_selected_logits" (... channel position 2*max_nodes): A logit for each node,
      indicating the probability that this node should be sent as a message to the
      verifier.
    - "decision_logits" (... 3): A logit for each of the three options: guess that the
      graphs are isomorphic, guess that the graphs are not isomorphic, or continue
      exchanging messages.
    - "linear_message_selected_logits" (... channel position d_linear_message_space)
      (optional): A logit for each linear message, indicating the probability that this
      linear message should be sent as a message to the verifier.
    """

    agent_level_in_keys = ("graph_level_repr", "node_level_repr")

    @property
    def agent_level_out_keys(self) -> tuple[str, ...]:
        """The agent-level output keys.

        These are the keys that are output by the agent head.
        """

        agent_level_out_keys = ("node_selected_logits", "decision_logits")

        if self.hyper_params.include_linear_message_space:
            agent_level_out_keys = (
                *agent_level_out_keys,
                "linear_message_selected_logits",
            )

        return agent_level_out_keys

    def forward(
        self,
        body_output: TensorDict,
        hooks: Optional[GraphIsomorphismAgentHooks] = None,
    ) -> TensorDict:
        """Output a uniform distribution.

        Parameters
        ----------
        body_output : TensorDict
            The output of the body module. A tensor dict with keys:

            - "graph_level_repr" (... 2 d_representation): The output graph-level
              representations.
            - "node_level_repr" (... 2 max_nodes d_representation): The output
              node-level representations.

        hooks : GraphIsomorphismAgentHooks, optional
            Hooks to run at various points in the agent forward pass (ignored).

        Returns
        -------
        out : TensorDict
            A tensor dict with keys (all zeros):

            - "node_selected_logits" (... channel position 2*max_nodes): A logit for
              each node, indicating the probability that this node should be sent as a
              message to the verifier.
            - "decision_logits" (... 3): A logit for each of the three options: continue
              exchanging messages, guess that the graphs are isomorphic, or guess that
              the graphs are not isomorphic. Set to zeros when the decider is not
              present.
            - "linear_message_selected_logits" (... channel position
              d_linear_message_space) (optional): A logit for each linear message,
              indicating the probability that this linear message should be sent as a
              message to the verifier.
        """

        max_num_nodes = body_output["node_level_repr"].shape[-2]

        update_dict = {}

        update_dict["node_selected_logits"] = torch.zeros(
            *body_output.batch_size,
            self.num_visible_message_channels,
            self.hyper_params.message_size,
            2 * max_num_nodes,
            device=self.device,
            dtype=torch.float32,
        )
        update_dict["decision_logits"] = torch.zeros(
            *body_output.batch_size,
            3,
            device=self.device,
            dtype=torch.float32,
        )

        # Multiply the outputs by the dummy parameter, so that the gradients PyTorch
        # doesn't complain about not having any gradients
        update_dict["node_selected_logits"] = (
            update_dict["node_selected_logits"] * self.dummy_parameter
        )
        update_dict["decision_logits"] = (
            update_dict["decision_logits"] * self.dummy_parameter
        )

        if self.hyper_params.include_linear_message_space:
            update_dict["linear_message_selected_logits"] = torch.zeros(
                *body_output.batch_size,
                self.num_visible_message_channels,
                self.hyper_params.message_size,
                self.hyper_params.d_linear_message_space,
                device=self.device,
                dtype=torch.float32,
            )
            update_dict["linear_message_selected_logits"] = (
                update_dict["linear_message_selected_logits"] * self.dummy_parameter
            )

        return body_output.update(update_dict)


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
    hyper_params : HyperParameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The settings of the experiment.
    agent_name : str
        The name of the agent.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    device : TorchDevice, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    agent_level_in_keys = ("graph_level_repr",)

    @property
    def env_level_in_keys(self) -> tuple[str, ...]:
        """The environment-level input keys."""
        if self.agent_params.include_round_in_value:
            return ("round",)
        else:
            return ()

    agent_level_out_keys = ("value",)

    def __init__(
        self,
        hyper_params: HyperParameters,
        settings: ExperimentSettings,
        agent_name: str,
        protocol_handler: ProtocolHandler,
    ):
        super().__init__(
            hyper_params=hyper_params,
            settings=settings,
            agent_name=agent_name,
            protocol_handler=protocol_handler,
        )

        self.value_mlp = self._build_mlp()

        self._init_weights()

    def _build_mlp(self) -> TensorDictModule:
        """Build the module which computes the value function.

        Returns
        -------
        value_mlp : TensorDictModule
            The value module.
        """
        return self._build_graph_level_mlp(
            d_in=self.hyper_params.d_representation,
            d_hidden=self.agent_params.d_value,
            d_out=1,
            num_layers=self.agent_params.num_value_layers,
            include_round=self.agent_params.include_round_in_value,
            out_key="value",
            squeeze=True,
        )

    def forward(
        self,
        body_output: TensorDict,
        hooks: Optional[GraphIsomorphismAgentHooks] = None,
    ) -> TensorDict:
        """Run the value head on the given body output.

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
        """Move the agent to the given device.

        Parameters
        ----------
        device : TorchDevice, optional
            The device to use. If not given, the CPU is used.
        """
        super().to(device)
        self.device = device
        self.value_mlp.to(device)


@register_scenario_class(GI_SCENARIO, ConstantAgentValueHead)
class GraphIsomorphismConstantAgentValueHead(
    GraphIsomorphismDummyAgentPart, ConstantAgentValueHead
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

    agent_level_in_keys = ("graph_level_repr", "node_level_repr")
    agent_level_out_keys = ("value",)

    def forward(
        self,
        body_output: TensorDict,
        hooks: Optional[GraphIsomorphismAgentHooks] = None,
    ) -> TensorDict:
        """Return a constant value.

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
    hyper_params : HyperParameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The settings of the experiment.
    agent_name : str
        The name of the agent.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    device : TorchDevice, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    agent_level_in_keys = ("graph_level_repr",)
    agent_level_out_keys = ("decision_logits",)

    def __init__(
        self,
        hyper_params: HyperParameters,
        settings: ExperimentSettings,
        agent_name: str,
        protocol_handler: ProtocolHandler,
    ):
        super().__init__(
            hyper_params=hyper_params,
            settings=settings,
            agent_name=agent_name,
            protocol_handler=protocol_handler,
        )

        self.decider = self._build_decider(d_out=2, include_round=False)

        self._init_weights()

    def forward(
        self,
        body_output: TensorDict,
        hooks: Optional[GraphIsomorphismAgentHooks] = None,
    ) -> TensorDict:
        """Run the solo agent head on the given body output.

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
        """Move the agent to the given device.

        Parameters
        ----------
        device : TorchDevice, optional
            The device to use. If not given, the CPU is used.
        """
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
    - "x" (... round channel position pair node): The graph node features (message
      history)
    - "adjacency" (... pair node node): The adjacency matrices.
    - "message" (... channel position pair node), optional: The most recent message.
    - "node_mask" (... pair node): Which nodes actually exist.
    - "linear_message_history" : (... round channel position linear_message), optional:
      The linear message history, if using.

    Output:

    - ("agents", "node_level_repr") (... agent pair max_nodes d_representation): The
      output node-level representations.
    - ("agents", "graph_level_repr") (... agent pair d_representation): The output
      graph-level representations.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The settings of the experiment.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    bodies : dict[str, GraphIsomorphismAgentBody]
        The agent bodies to combine.
    """

    additional_in_keys = ("round",)

    def forward(
        self,
        data: TensorDictBase,
        hooks: Optional[GraphIsomorphismAgentHooks] = None,
    ) -> TensorDict:
        """Run the agent bodies and combine their outputs.

        Parameters
        ----------
        data : TensorDictBase
            The input data. A tensor dict with keys:

            - "round" (...): The current round number.
            - "x" (... round channel position pair node): The graph node features
              (message history)
            - "adjacency" (... pair node node): The adjacency matrices.
            - "message" (... channel position pair node), optional: The most recent
              message.
            - "node_mask" (... pair node): Which nodes actually exist.
            - "linear_message_history" : (... round channel position linear_message),
              optional: The linear message history, if using.

        hooks : GraphIsomorphismAgentHooks, optional
            Hooks to run at various points in the agent forward pass.

        Returns
        -------
        data : TensorDict
            The tensordict updated in place with the output of the agent bodies.
        """

        round_id: Int[Tensor, "..."] = data["round"]

        # Run the agent bodies
        body_outputs: dict[str, TensorDict] = {}
        for agent_name in self.agent_names:
            # Build the input dict for the agent body. For the message and history keys
            # we need to mask out the invisible channels
            input_dict = {}
            for key in self.bodies[agent_name].in_keys:
                if key == "ignore_message":
                    input_dict[key] = round_id == 0
                elif key == "message":
                    if "message" not in data.keys():
                        continue
                    input_dict[key] = self._restrict_input_to_visible_channels(
                        agent_name, data[key], "... channel position pair node"
                    )
                elif key == "linear_message_history":
                    input_dict[key] = self._restrict_input_to_visible_channels(
                        agent_name,
                        data[key],
                        "... round channel position linear_message",
                    )
                elif key == "x":
                    input_dict[key] = self._restrict_input_to_visible_channels(
                        agent_name, data[key], "... round channel position pair node"
                    )
                else:
                    input_dict[key] = data[key]
            input_td = TensorDict(
                input_dict,
                batch_size=data.batch_size,
            )

            # Run the agent body
            body_outputs[agent_name] = self.bodies[agent_name](input_td, hooks=hooks)

        # Stack the outputs
        node_level_repr = rearrange(
            [body_outputs[name]["node_level_repr"] for name in self.agent_names],
            "agent ... pair node feature -> ... agent pair node feature",
        )
        graph_level_repr = rearrange(
            [body_outputs[name]["graph_level_repr"] for name in self.agent_names],
            "agent ... pair feature -> ... agent pair feature",
        )

        return data.update(
            dict(
                agents=TensorDict(
                    dict(
                        node_level_repr=node_level_repr,
                        graph_level_repr=graph_level_repr,
                    ),
                    batch_size=(*data.batch_size, len(self.agent_names)),
                )
            )
        )


@register_scenario_class(GI_SCENARIO, CombinedPolicyHead)
class GraphIsomorphismCombinedPolicyHead(CombinedPolicyHead):
    """A module which combines the agent policy heads for the graph isomorphism task.

    Shapes
    ------
    Input:

    - ("agents", "node_level_repr") (... agent pair node feature): The output node-level
      representations.
    - ("agents", "graph_level_repr") (... agent pair feature): The output graph-level
      representations.
    - "round" (...): The current round number.
    - "node_mask" (... pair node): Which nodes actually exist.
    - "message" (... channel position pair node): The most recent message.
    - "ignore_message" (...): Whether to ignore the message
    - "decision_restriction" (...): The restriction on what decisions are allowed.

    Output:

    - ("agents", "node_selected_logits") (... agent channel position 2*max_nodes): A
      logit for each node, indicating the probability that this node should be sent as a
      message to the verifier.
    - ("agents", "main_message_logits") (... agents channel position logit): The same as
      "node_selected_logits".
    - ("agents", "decision_logits") (... agent 3): A logit for each of the three
      options: guess that the graphs are isomorphic, guess that the graphs are not
      isomorphic, or continue exchanging messages. d_linear_message_space) (optional): A
      logit for each linear message, indicating the probability that this linear message
      should be sent as a message to the verifier.
    - ("agents", "linear_message_selected_logits") (... agent channel position
      d_linear_message_space) (optional): A logit for each linear message, indicating
      the probability that this linear message should be sent as a message to the
      verifier.


    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The settings of the experiment.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    policy_heads : dict[str, GraphIsomorphismAgentPolicyHead]
        The agent policy heads to combine.
    """

    additional_in_keys = ("ignore_message", "decision_restriction")

    def forward(
        self,
        body_output: TensorDictBase,
        hooks: Optional[GraphIsomorphismAgentHooks] = None,
    ) -> TensorDict:
        """Run the agent policy heads and combine their outputs.

        Parameters
        ----------
        body_output : TensorDictBase
            The combined output of the agent bodies.
        hooks : GraphIsomorphismAgentHooks, optional
            Hooks to run at various points in the agent forward pass.

        Returns
        -------
        body_output: TensorDict
            The tensordict updated in place with the output of the policy heads.
        """

        # Run the policy heads to obtain the probability distributions
        policy_outputs: dict[str, TensorDict] = {}
        for i, agent_name in enumerate(self.agent_names):

            # Build the input dict by selecting the correct agent's body output
            input_td = TensorDict(
                dict(
                    node_level_repr=body_output["agents", "node_level_repr"][
                        ..., i, :, :, :
                    ],
                    graph_level_repr=body_output["agents", "graph_level_repr"][
                        ..., i, :, :
                    ],
                    message=body_output["message"],
                ),
                batch_size=body_output.batch_size,
            )
            if "round" in body_output.keys():
                input_td["round"] = body_output["round"]

            # Run the policy head for the agent
            policy_outputs[agent_name] = self.policy_heads[agent_name](
                input_td, hooks=hooks
            )

            # Expand the logits to all channels (not just the ones visible to the agent)
            policy_outputs[agent_name]["node_selected_logits"] = (
                self._expand_logits_to_all_channels(
                    agent_name,
                    policy_outputs[agent_name]["node_selected_logits"],
                    "... channel position node",
                )
            )
            if "linear_message_selected_logits" in policy_outputs[agent_name].keys():
                policy_outputs[agent_name]["linear_message_selected_logits"] = (
                    self._expand_logits_to_all_channels(
                        agent_name,
                        policy_outputs[agent_name]["linear_message_selected_logits"],
                        "... channel position logit",
                    )
                )

            # Make sure the provers only select nodes in the opposite graph to the most
            # recent message
            if agent_name in self.protocol_handler.prover_names:
                message: Tensor = body_output["message"]
                max_num_nodes = body_output["agents", "node_level_repr"].shape[-2]
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

        agents_update = {}

        # Stack the outputs
        agents_update["node_selected_logits"] = rearrange(
            [policy_outputs[name]["node_selected_logits"] for name in self.agent_names],
            "agent ... channel position node -> ... agent channel position node",
        )
        agents_update["decision_logits"] = rearrange(
            [policy_outputs[name]["decision_logits"] for name in self.agent_names],
            "agent ... logit-> ... agent logit",
        )
        if self.hyper_params.include_linear_message_space:
            agents_update["linear_message_selected_logits"] = rearrange(
                [
                    policy_outputs[name]["linear_message_selected_logits"]
                    for name in self.agent_names
                ],
                "agent ... channel position logit -> ... agent channel position logit",
            )

        # Make sure the agents only select nodes which exist
        node_mask_flatter = rearrange(
            body_output["node_mask"], "... pair node -> ... 1 1 1 (pair node)"
        )
        agents_update["node_selected_logits"] = torch.where(
            node_mask_flatter,
            agents_update["node_selected_logits"],
            -1e9,
        )

        # Make sure the verifier only selects decisions which are allowed
        agents_update["decision_logits"] = self._restrict_decisions(
            body_output["decision_restriction"], agents_update["decision_logits"]
        )

        # Copy the main message logits to the main message logits key
        agents_update["main_message_logits"] = agents_update["node_selected_logits"]

        return body_output.update(
            dict(
                agents=TensorDict(
                    agents_update,
                    batch_size=(*body_output.batch_size, len(self.agent_names)),
                )
            )
        )


@register_scenario_class(GI_SCENARIO, CombinedValueHead)
class GraphIsomorphismCombinedValueHead(CombinedValueHead):
    """A module which combines the agent value heads for the graph isomorphism task.

    Shapes
    ------
    Input:

    - ("agents", "graph_level_repr") (... agent d_representation): The output
      graph-level representations.
    - "round" (...): The current round number.

    Output:

    - ("agents", "value") (... agent): The estimated value for each batch item

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The settings of the experiment.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    value_heads : dict[str, GraphIsomorphismAgentValueHead]
        The agent value heads to combine.
    """

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

        # Run the value heads to obtain the value estimates
        value_outputs: dict[str, TensorDict] = {}
        for i, agent_name in enumerate(self.agent_names):
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
            if "round" in head_output.keys():
                input_td["round"] = head_output["round"]
            value_outputs[agent_name] = self.value_heads[agent_name](
                input_td, hooks=hooks
            )

        # Stack the outputs
        value = rearrange(
            [value_outputs[name]["value"] for name in self.agent_names],
            "agent ... -> ... agent",
        )

        return head_output.update(
            dict(
                agents=TensorDict(
                    dict(value=value),
                    batch_size=(*head_output.batch_size, len(self.agent_names)),
                )
            ),
        )


@register_scenario_class(GI_SCENARIO, Agent)
@dataclass
class GraphIsomorphismAgent(Agent):
    """An agent for the graph isomorphism task.

    This is a dataclass which contains all the parts of the agent.
    """

    agent_params: ClassVar[GraphIsomorphismAgentParameters | RandomAgentParameters]

    message_logits_key: ClassVar[str] = "node_selected_logits"

    def get_model_parameter_dicts(
        self,
        base_lr: float,
        named_parameters: Optional[Iterable[tuple[str, TorchParameter]]] = None,
        body_lr_factor_override: bool = False,
    ) -> Iterable[dict[str, Any]]:
        """Get the Torch parameters of the agent, and their learning rates.

        Parameters
        ----------
        base_lr : float
            The base learning rate for the trainer.
        named_parameters : Iterable[tuple[str, TorchParameter]], optional
            The named parameters of the loss module, usually obtained by
            ``loss_module.named_parameters()``. If not given, the parameters of all the
            agent parts are used.
        body_lr_factor_override : bool
            If true, this overrides the learning rate factor for the body (for both the
            actor and critic), effectively setting it to 1.

        Returns
        -------
        param_dict : Iterable[dict[str, Any]]
            The Torch parameters of the agent, and their learning rates. This is an
            iterable of dictionaries with the keys ``hyper_params`` and ``lr``.
        """

        # Check for mistakes
        if (
            self.hyper_params.rl.use_shared_body
            and self.agent_params.agent_lr_factor.actor
            != self.agent_params.agent_lr_factor.critic
        ):
            raise ValueError(
                "The agent learning rate factor for the actor and critic must be the "
                "same if the body is shared."
            )
        if (
            self.hyper_params.rl.use_shared_body
            and self.agent_params.body_lr_factor.actor
            != self.agent_params.body_lr_factor.critic
        ):
            raise ValueError(
                "The body learning rate factor for the actor and critic must be the "
                "same if the body is shared."
            )
        if hasattr(self.agent_params, "gnn_lr_factor"):
            if (
                self.hyper_params.rl.use_shared_body
                and self.agent_params.gnn_lr_factor.actor
                != self.agent_params.gnn_lr_factor.critic
            ):
                raise ValueError(
                    "The GNN learning rate factor for the actor and critic must be the "
                    "same if the body is shared."
                )

        # The learning rate of the whole agent
        agent_lr = {
            "actor": self.agent_params.agent_lr_factor.actor * base_lr,
            "critic": self.agent_params.agent_lr_factor.critic * base_lr,
        }

        # Determine the learning rate of the body
        body_lr = {
            "actor": (
                agent_lr["actor"] * self.agent_params.body_lr_factor.actor
                if not body_lr_factor_override
                else agent_lr["actor"]
            ),
            "critic": (
                agent_lr["critic"] * self.agent_params.body_lr_factor.critic
                if not body_lr_factor_override
                else agent_lr["critic"]
            ),
        }

        # Determine the learning rate for the GNN encoder
        gnn_lr = {
            "actor": (
                body_lr["actor"] * self.agent_params.gnn_lr_factor.actor
                if isinstance(self.agent_params, GraphIsomorphismAgentParameters)
                else 0
            ),
            "critic": (
                body_lr["critic"] * self.agent_params.gnn_lr_factor.critic
                if isinstance(self.agent_params, GraphIsomorphismAgentParameters)
                else 0
            ),
        }

        model_param_dict = []

        # If named_parameters is not given, use the parameters of all the agent parts
        if named_parameters is None:
            for part in ["actor", "critic"]:
                self._append_filtered_params(
                    model_param_dict,
                    self._body_named_parameters,
                    lambda x: part in x and x.startswith("gnn"),
                    gnn_lr[part],
                )
                self._append_filtered_params(
                    model_param_dict,
                    self._body_named_parameters,
                    lambda x: part in x and not x.startswith("gnn"),
                    body_lr[part],
                )
            if self.policy_head is not None:
                model_param_dict.append(
                    dict(params=self.policy_head.parameters(), lr=agent_lr["actor"])
                )
            if self.value_head is not None:
                model_param_dict.append(
                    dict(params=self.value_head.parameters(), lr=agent_lr["critic"])
                )
            if self.solo_head is not None:
                model_param_dict.append(
                    dict(params=self.solo_head.parameters(), lr=agent_lr["actor"])
                )
            return model_param_dict

        # Convert the named parameters to a list, so that we can iterate over it
        # multiple times
        named_parameters = list(named_parameters)

        def is_gnn_param(name: str, part: str):
            return re.match(f"{self._body_param_regex(part)}.gnn", name)

        def is_body_param(name: str, part: str):
            return re.match(self._body_param_regex(part), name) and not is_gnn_param(
                name, part
            )

        for part in ["actor", "critic"]:
            # Set the learning rate for the GNN parameters
            self._append_filtered_params(
                model_param_dict,
                named_parameters,
                partial(is_gnn_param, part=part),
                gnn_lr[part],
            )

            # Set the learning rate for the body parameters other than the GNN
            # parameters
            self._append_filtered_params(
                model_param_dict,
                named_parameters,
                partial(is_body_param, part=part),
                body_lr[part],
            )

            # Set the learning rate for the non-body parameters
            self._append_filtered_params(
                model_param_dict,
                named_parameters,
                lambda name: re.match(self._non_body_param_regex(part), name),
                agent_lr[part],
            )

        return model_param_dict
