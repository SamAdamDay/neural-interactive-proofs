"""Image classification agents components.

Contains classes for building agent bodies and heads for the image classification task.

The structure of all agent bodies is the same:

- An encoder layer, which takes as input the image and the message history and outputs
  the initial pixel-level encodings.
- A sequence of `num_conv_groups` groups of convolutional layers. 
    + Each layer is followed by a non-linearity and each group by a max pooling layer. 
    + For each group we halve the output size and double the number of channels. 
    + The number of convolutional layers in each group is given by the
      `num_convs_per_group` parameter. 
    + Before the last group we add the most recent message as a new channel.
    + The output of the last group is the 'latent pixel-level' representations, which
      provides a representation for each latent pixel.
- A global pooling layer, which pools the latent pixel-level representations to obtain
  the image-level representations.
- A representation encoder which takes as input the image-level and latent pixel-level
  representations and outputs the final representations.
"""

from abc import ABC
from typing import Optional

import torch
from torch.nn import Sequential, Linear, Conv2d, MaxPool2d, Upsample
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Categorical

from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from jaxtyping import Float, Bool, Int

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
from pvg.parameters import (
    Parameters,
    ImageClassificationParameters,
    ImageClassificationAgentParameters,
)
from pvg.utils.torch_modules import (
    ACTIVATION_CLASSES,
    Squeeze,
    BatchNorm1dBatchDims,
    TensorDictCat,
    ParallelTensorDictModule,
    Print,
)
from pvg.utils.types import TorchDevice
from pvg.image_classification.data import IMAGE_DATASETS


class ImageClassificationAgentPart(AgentPart, ABC):
    """Base class for all image classification agent parts.

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

        self._agent_params: ImageClassificationAgentParameters = params.agents[
            agent_name
        ]
        for i, _agent_name in enumerate(params.agents):
            if _agent_name == agent_name:
                self.agent_index = i
                break

        # Get some dataset parameters
        self._dataset_num_channels = IMAGE_DATASETS[params.dataset].num_channels
        self._image_width = IMAGE_DATASETS[params.dataset].width
        self._image_height = IMAGE_DATASETS[params.dataset].height
        self._latent_width = (
            self._image_width // 2**self._agent_params.num_conv_groups
        )
        self._latent_height = (
            self._image_height // 2**self._agent_params.num_conv_groups
        )
        self._latent_num_channels = (
            2**self._agent_params.num_conv_groups
            * self._agent_params.initial_num_channels
        )

        self.activation_function = ACTIVATION_CLASSES[
            self._agent_params.activation_function
        ]


class ImageClassificationAgentBody(ImageClassificationAgentPart, AgentBody):
    """The body of an image classification agent.

    Takes as input the image, message history and the most recent message and outputs
    the image-level and latent pixel-level representations.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    agent_name : str
        The name of the agent.
    device : TorchDevice, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    in_keys = ("x", "image", "message", "ignore_message")
    out_keys = ("image_level_repr", "latent_pixel_level_repr")

    def __init__(
        self,
        params: Parameters,
        agent_name: str,
        device: Optional[TorchDevice] = None,
    ):
        super().__init__(params, agent_name, device)

        self.module_sequence = TensorDictSequential(
            [
                self.build_message_history_upsampler(),
                self.build_initial_encoder(),
                self.build_cnn_encoder(),
                self.build_global_pooling(),
                self.build_final_encoder(),
            ]
        ).to(self.device)

    def build_message_history_upsampler(self) -> TensorDictModule:
        """Build the module which upsamples the history to the size of the image.

        Shapes
        ------
        Input:
            - message : (... max_message_rounds latent_width latent_height)

        Output:
            - message_upsampled : (... max_message_rounds width height)

        Returns
        -------
        message_history_upsampler : TensorDictModule
            The module which upsamples the message history to the size of the image.
        """
        return TensorDictModule(
            Upsample(
                size=(self._dataset_num_channels, self._dataset_num_channels),
                mode="nearest",
            ),
            in_keys="x",
            out_keys="x_upsampled",
        )

    def build_initial_encoder(self) -> TensorDictSequential:
        """Build the initial encoding layer.

        Concatenates the upsampled message history with the image, then applies a linear
        layer to obtain the initial pixel-level representations.

        Shapes
        ------
        Input:
            - x_upsampled : (... max_message_rounds width height) image : (...
            num_channels width height)
        Output:
            - latent_pixel_level_repr : (... initial_num_channels width height)

        Returns
        -------
        TensorDictSequential
            The initial encoding layer.
        """
        return TensorDictSequential(
            [
                TensorDictCat(
                    in_keys=("x_upsampled", "image"),
                    out_keys="latent_pixel_level_repr",
                    dim=-3,
                ),
                TensorDictModule(
                    Linear(
                        self.params.max_message_rounds + self._dataset_num_channels,
                        self._agent_params.initial_num_channels,
                    ),
                    in_keys="latent_pixel_level_repr",
                    out_keys="latent_pixel_level_repr",
                ),
            ]
        )

    def build_cnn_encoder(self) -> TensorDictSequential:
        """Build the the sequence of groups of convolutional layers.

        Shapes
        ------
        Input:
            - latent_pixel_level_repr : (... initial_num_channels width height)
        Output:
            - latent_pixel_level_repr : (... latent_num_channels latent_width
            latent_height)
        where `latent_num_channels = initial_num_channels * 2**num_conv_groups`

        Returns
        -------
        cnn_encoder : TensorDictSequential
            The sequence of groups of convolutional layers.
        """
        cnn_encoder = []
        for i in range(self._agent_params.num_conv_groups):
            # Add the message as a new channel before the last group
            if i == self._agent_params.num_conv_groups - 1:
                cnn_encoder.append(
                    TensorDictCat(
                        in_keys=("latent_pixel_level_repr", "message"),
                        out_keys="latent_pixel_level_repr",
                    )
                )

            # Add the convolutional layers
            for j in range(self._agent_params.num_convs_per_group):
                # Determine the number of input channels. In all groups except the first
                # the number of input channels is the same as the number of output
                # channels of the previous group. In the last group we add the message
                # as a new channel.
                if j == 0 and i == self._agent_params.num_conv_groups - 1:
                    in_channels = (
                        2 ** (i - 1) * self._agent_params.initial_num_channels + 1
                    )
                elif j == 0 and i > 0:
                    in_channels = 2 ** (i - 1) * self._agent_params.initial_num_channels
                else:
                    in_channels = 2**i * self._agent_params.initial_num_channels

                # Add the convolutional layer and non-linearity
                cnn_encoder.append(
                    TensorDictModule(
                        Conv2d(
                            in_channels=in_channels,
                            out_channels=2**i
                            * self._agent_params.initial_num_channels,
                            kernel_size=self._agent_params.kernel_size,
                            stride=self._agent_params.stride,
                            padding="same",
                        ),
                        in_keys="latent_pixel_level_repr",
                        out_keys="latent_pixel_level_repr",
                    )
                )
                cnn_encoder.append(
                    TensorDictModule(
                        self.activation_function(),
                        in_keys="latent_pixel_level_repr",
                        out_keys="latent_pixel_level_repr",
                    )
                )

            # Add the max pooling layer after each group
            cnn_encoder.append(
                TensorDictModule(
                    MaxPool2d(kernel_size=2, stride=2),
                    in_keys="latent_pixel_level_repr",
                    out_keys="latent_pixel_level_repr",
                )
            )

        return TensorDictSequential(*cnn_encoder)

    def build_global_pooling(self) -> TensorDictSequential:
        """Build the global pooling layer.

        Shapes
        ------
        Input:
            - latent_pixel_level_repr : (... latent_num_channels latent_width
            - latent_height)
        Output:
            - image_level_repr : (... latent_num_channels)

        Returns
        -------
        TensorDictSequential
            The global pooling layer.
        """
        return TensorDictSequential(
            Reduce("... channel height width -> ... channel", reduction="mean"),
            in_keys="latent_pixel_level_repr",
            out_keys="image_level_repr",
        )

    def build_final_encoder(self) -> ParallelTensorDictModule:
        """Build the final encoder.

        This rearranges the latent pixel-level representations to put the channel
        dimension last, then applies a linear layer to obtain the final representations.

        Shapes
        ------
        Input:
            - image_level_repr : (... latent_num_channels)
            - latent_pixel_level_repr : (... latent_num_channels latent_width
              latent_height)
        Output:
            - image_level_repr : (... d_representation)
            - latent_pixel_level_repr : (... latent_width latent_height
              d_representation)

        Returns
        -------
        ParallelTensorDictModule
            The final encoder.
        """
        return TensorDictSequential(
            [
                TensorDict(
                    Rearrange(
                        "... latent_num_channels latent_width latent_height -> "
                        "... latent_width latent_height latent_num_channels"
                    ),
                ),
                ParallelTensorDictModule(
                    Linear(
                        in_features=2**self._agent_params.num_conv_groups
                        * self._agent_params.initial_num_channels,
                        out_features=self.params.d_representation,
                    ),
                ),
            ]
        )

    def forward(self, data: TensorDictBase) -> TensorDict:
        """Run the image classification body

        Parameters
        ----------
        data : TensorDictBase
            The data to run the body on. A TensorDictBase with keys:

            - "x" (... pair node feature): The graph node features (message history)
            - "image" (... num_channels width height): The image
            - "message" (... max_message_rounds latent_width latent_height): The message
              history

        Returns
        -------
        out : TensorDict
            A tensor dict with keys:

            - "image_level_repr" (... d_representation): The output image-level
              representations.
            - "latent_pixel_level_repr" (... latent_width latent_height
              d_representation): The output latent-pixel-level representations.
        """

        return self.module_sequence(data)

    def to(self, device: Optional[TorchDevice] = None):
        super().to(device)
        self.device = device
        self.module_sequence.to(self.device)
        return self


class ImageClassificationDummyAgentBody(ImageClassificationAgentPart, DummyAgentBody):
    """Dummy agent body for the image classification task."""

    in_keys = ("x", "image", "message", "ignore_message")
    out_keys = ("image_level_repr", "latent_pixel_level_repr")

    def forward(self, data: TensorDictBase) -> TensorDict:
        """Returns dummy outputs.

        Parameters
        ----------
        data : TensorDictBase
            TensorDictBase with keys:

            - "x" (... pair node feature): The graph node features (message history)
            - "image" (... num_channels width height): The image
            - "message" (... max_message_rounds latent_width latent_height): The message
              history

        Returns
        -------
        out : TensorDict
            A tensor dict with keys:

            - "image_level_repr" (... d_representation): The output image-level
              representations.
            - "latent_pixel_level_repr" (... latent_width latent_height
              d_representation): The output latent-pixel-level representations.
        """

        # The dummy image-level representations
        image_level_repr = torch.zeros(
            *data.batch_size,
            self.params.d_representation,
            device=self.device,
            dtype=torch.float32,
        )

        # The dummy node-level representations
        latent_pixel_level_repr = torch.zeros(
            *data.batch_size,
            self._latent_width,
            self._latent_height,
            self.params.d_representation,
            device=self.device,
            dtype=torch.float32,
        )

        # Multiply the outputs by the dummy parameter, so that the gradients PyTorch
        # doesn't complain about not having any gradients
        image_level_repr = image_level_repr * self.dummy_parameter
        latent_pixel_level_repr = latent_pixel_level_repr * self.dummy_parameter

        return data.update(
            dict(
                image_level_repr=image_level_repr,
                latent_pixel_level_repr=latent_pixel_level_repr,
            )
        )


class ImageClassificationAgentHead(ImageClassificationAgentPart, AgentHead, ABC):
    """Base class for all image classification agent heads.

    This class provides some utility methods for constructing and running the various
    modules.
    """

    def _build_latent_pixel_mlp(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        num_layers: int,
        out_key: str = "latent_pixel_mlp_output",
    ) -> TensorDictModule:
        """Builds an MLP which acts on the node-level representations.

        Shapes
        ------
        Input:
            - latent_pixel_repr : (... latent_width latent_height d_in)
        Output:
            - latent_pixel_mlp_output : (... latent_width latent_height d_out)

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
        out_key : str, default="latent_pixel_mlp_output"
            The tensordict key to use for the output of the MLP.

        Returns
        -------
        latent_pixel_mlp : TensorDictModule
            The node-level MLP.
        """
        layers = []

        # The layers of the MLP
        layers.append(Linear(d_in, d_hidden))
        layers.append(self.activation_function(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(Linear(d_hidden, d_hidden))
            layers.append(self.activation_function(inplace=True))
        layers.append(Linear(d_hidden, d_out))

        # Make the layers into a sequential module and wrap it in a TensorDictModule
        sequential = Sequential(*layers)
        tensor_dict_sequential = TensorDictModule(
            sequential, in_keys=("latent_pixel_repr",), out_keys=(out_key,)
        )

        tensor_dict_sequential = tensor_dict_sequential.to(self.device)

        return tensor_dict_sequential

    def _build_image_level_mlp(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        num_layers: int,
        out_key: str = "image_level_mlp_output",
        squeeze: bool = False,
    ) -> TensorDictModule:
        """Builds an MLP which acts on the node-level representations.

        Shapes
        ------
        Input:
            - image_level_repr : (... d_in)
        Output:
            - image_level_mlp_output : (... d_out)

        Parameters
        ----------
        d_in : int
            The dimensionality of the image-level representations.
        d_hidden : int
            The dimensionality of the hidden layers.
        d_out : int
            The dimensionality of the output.
        num_layers : int
            The number of hidden layers in the MLP.
        out_key : str, default="image_level_mlp_output"
            The tensordict key to use for the output of the MLP.
        squeeze : bool, default=False
            Whether to squeeze the output dimension. Only use this if the output
            dimension is 1.

        Returns
        -------
        latent_pixel_mlp : TensorDictModule
            The node-level MLP.
        """
        layers = []

        # The layers of the MLP
        layers.append(Linear(d_in, d_hidden))
        layers.append(self.activation_function(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(Linear(d_hidden, d_hidden))
            layers.append(self.activation_function(inplace=True))
        layers.append(Linear(d_hidden, d_out))

        # Squeeze the output dimension if necessary
        if squeeze:
            layers.append(Squeeze())

        # Make the layers into a sequential module, and wrap it in a TensorDictModule
        sequential = Sequential(*layers)
        tensor_dict_sequential = TensorDictModule(
            sequential, in_keys=("image_level_repr",), out_keys=(out_key,)
        )

        tensor_dict_sequential = tensor_dict_sequential.to(self.device)

        return tensor_dict_sequential

    def _build_decider(self, d_out: int = 3) -> TensorDictModule:
        """Builds the module which produces a image-level output.

        By default it is used to decide whether to continue exchanging messages. In this
        case it outputs a single triple of logits for the three options: guess a
        classification for the image or continue exchanging messages.

        Parameters
        ----------
        d_out : int, default=3
            The dimensionality of the output.

        Returns
        -------
        decider : TensorDictModule
            The decider module.
        """
        return self._build_image_level_mlp(
            d_in=self.params.d_representation,
            d_hidden=self._agent_params.d_decider,
            d_out=d_out,
            num_layers=self._agent_params.num_decider_layers,
            out_key="decision_logits",
        )
