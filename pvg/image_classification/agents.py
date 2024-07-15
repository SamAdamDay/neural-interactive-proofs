"""Image classification agents components.

Contains classes for building agent bodies and heads for the image classification task.

The structure of all agent bodies is the same:

- An encoder layer, which takes as input the image and the message history and outputs
  the initial pixel-level encodings.
- A sequence of `num_block_groups` groups of building blocks (e.g. convolutional
  layers). 
    + Each layer is followed by a non-linearity and each group by a max pooling layer. 
    + For each group we halve the output size and double the number of channels. 
    + The number of building blocks in each group is given by the `num_blocks_per_group`
      parameter. 
    + The output of the last group is the 'latent pixel-level' representations, which
      provides a representation for each latent pixel.
- We add a channel to the latent pixel-level representations to represent the most
  recent message.
- A global pooling layer, which pools the latent pixel-level representations to obtain
  the image-level representations.
- A representation encoder which takes as input the image-level and latent pixel-level
  representations and outputs the final representations.
"""

from abc import ABC
from typing import Optional, ClassVar
from dataclasses import dataclass
from functools import partial

import torch
from torch.nn import Sequential, Linear, Conv2d, BatchNorm2d
from torch import Tensor

from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from jaxtyping import Int

from pvg.scenario_base.agents import (
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
    Agent,
)
from pvg.scenario_base.pretrained_models import get_pretrained_model_class
from pvg.scenario_instance import register_scenario_class
from pvg.protocols import ProtocolHandler
from pvg.parameters import (
    Parameters,
    ImageClassificationAgentParameters,
    RandomAgentParameters,
    ScenarioType,
    ImageBuildingBlockType,
)
from pvg.utils.torch import (
    ACTIVATION_CLASSES,
    Squeeze,
    UpsampleSimulateBatchDims,
    MaxPool2dSimulateBatchDims,
    Conv2dSimulateBatchDims,
    ResNetBasicBlockSimulateBatchDims,
    ResNetBottleneckBlockSimulateBatchDims,
    TensorDictCat,
    ParallelTensorDictModule,
    TensorDictCloneKeys,
    OneHot,
    NormalizeOneHotMessageHistory,
    Print,
    TensorDictPrint,
)
from pvg.utils.types import TorchDevice
from pvg.image_classification.data import DATASET_WRAPPER_CLASSES
from pvg.image_classification.pretrained_models import PretrainedImageModel


IC_SCENARIO = ScenarioType.IMAGE_CLASSIFICATION


class ImageClassificationAgentPart(AgentPart, ABC):
    """Base class for all image classification agent parts.

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

    _agent_params: ImageClassificationAgentParameters | RandomAgentParameters
    _pretrained_model_class: Optional[PretrainedImageModel] = None

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
        self.agent_index = protocol_handler.agent_names.index(agent_name)

        # Get some parameters
        self.num_block_groups = self.params.image_classification.num_block_groups
        self.initial_num_channels = (
            self.params.image_classification.initial_num_channels
        )
        self.dataset_num_channels = DATASET_WRAPPER_CLASSES[params.dataset].num_channels
        self.image_width = DATASET_WRAPPER_CLASSES[params.dataset].width
        self.image_height = DATASET_WRAPPER_CLASSES[params.dataset].height
        self.latent_width = self.image_width // 2**self.num_block_groups
        self.latent_height = self.image_height // 2**self.num_block_groups
        self.latent_num_channels = 2**self.num_block_groups * self.initial_num_channels

        if isinstance(self._agent_params, ImageClassificationAgentParameters):
            self.activation_function = ACTIVATION_CLASSES[
                self._agent_params.activation_function
            ]


@register_scenario_class(IC_SCENARIO, AgentBody)
class ImageClassificationAgentBody(ImageClassificationAgentPart, AgentBody):
    """The body of an image classification agent.

    Takes as input the image, message history and the most recent message and outputs
    the image-level and latent pixel-level representations.

    Shapes
    ------
    Input:
        - "x" (... round latent_height latent_width): The message history
        - "image" (... num_channels height width): The image
        - "message" (... latent_height latent_width), optional: The most recent message
        - "ignore_message" (...), optional: Whether to ignore the message
        - ("pretrained_embeddings", model_name) (... embedding_width embedding_height),
          optional: The embeddings of a pretrained model, if using.
        - "linear_message_history" : (... round linear_message), optional: The linear
          message history, if using

    Output:
        - "image_level_repr" (... d_representation): The output image-level
          representations.
        - "latent_pixel_level_repr" (... latent_height latent_width d_representation):
          The output latent-pixel-level representations.

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

    agent_level_in_keys = ("ignore_message",)

    @property
    def env_level_in_keys(self) -> tuple[str, ...]:

        env_level_in_keys = ("x", "image", "message")

        if self.include_pretrained_embeddings:
            env_level_in_keys = (
                *env_level_in_keys,
                self.pretrained_model_key,
            )

        if self.params.include_linear_message_space:
            env_level_in_keys = (*env_level_in_keys, "linear_message_history")

        return env_level_in_keys

    agent_level_out_keys = ("image_level_repr", "latent_pixel_level_repr")

    @property
    def required_pretrained_models(self) -> list[tuple[str, int]]:
        if self.include_pretrained_embeddings:
            return [(self.pretrained_model_name, self.pretrained_model_num_channels)]
        else:
            return []

    def __init__(
        self,
        params: Parameters,
        agent_name: str,
        protocol_handler: ProtocolHandler,
        *,
        device: Optional[TorchDevice] = None,
    ):
        super().__init__(params, agent_name, protocol_handler, device=device)

        if self._agent_params.use_manual_architecture:
            raise NotImplementedError(
                "Manual architecture is not implemented for the image classification "
                "task."
            )

        # Build the message history normalizer if necessary
        if self._agent_params.normalize_message_history:
            self.message_history_normalizer = NormalizeOneHotMessageHistory(
                max_message_rounds=self.protocol_handler.max_message_rounds,
                message_out_key="x_normalized",
                num_structure_dims=2,
                round_dim_last=False,
            )

        # Build the pretrained encoding scaler if necessary
        if self.include_pretrained_embeddings:
            self.pretrained_embedding_scaler = self.build_pretrained_embedding_scaler()
            self.pretrained_embedding_scaler = self.pretrained_embedding_scaler.to(
                device
            )

        self.message_history_upsampler = self.build_message_history_upsampler().to(
            device
        )
        self.initial_encoder = self.build_initial_encoder().to(device)
        self.cnn_encoder = self.build_cnn_encoder().to(device)
        self.global_pooling = self.build_global_pooling().to(device)
        self.final_encoder = self.build_final_encoder().to(device)

    def build_message_history_upsampler(self) -> TensorDictModule:
        """Build the module which upsamples the history and message.

        The message history is upsampled to the size of the image.

        Shapes
        ------
        Input:
            - "x" : (... max_message_rounds latent_height latent_width)

        Output:
            - "x_upsampled" : (... max_message_rounds height width)

        Returns
        -------
        message_history_upsampler : TensorDictModule
            The module which upsamples the message history to the size of the image.
        """
        return TensorDictModule(
            UpsampleSimulateBatchDims(
                size=(self.image_height, self.image_width),
                mode="nearest",
            ),
            in_keys="x_normalized",
            out_keys="x_upsampled",
        )

    def build_pretrained_embedding_scaler(self) -> TensorDictModule:
        """Build the module which scales the pretrained embeddings.

        The pretrained embeddings scaled to the image size. This can be by upsampling or
        mean pooling, depending on  whether the image size is larger or smaller than the
        embedding size.

        Shapes
        ------
        Input:
            - "pretrained_embeddings" : (... embedding_dim embedding_height
              embedding_width) : The embeddings of the pretrained model

        Output:
            - "pretrained_embeddings_scaled" : (... embedding_dim height width) :
              The scaled embeddings

        Returns
        -------
        pretrained_embedding_scaler : TensorDictModule
            The module which scales the pretrained embeddings.
        """

        embedding_width = self.pretrained_model_class.embedding_width
        embedding_height = self.pretrained_model_class.embedding_height

        if (
            embedding_width == self.image_width
            and embedding_height == self.image_height
        ):
            return TensorDictModule(
                TensorDictCloneKeys(
                    in_keys=self.pretrained_model_key,
                    out_keys="pretrained_embeddings_scaled",
                )
            )
        elif (
            embedding_width > self.image_width and embedding_height > self.image_height
        ):
            return TensorDictModule(
                Reduce(
                    "... channel scale_height*image_height scale_width*image_width "
                    "-> ... channel image_height image_width",
                    reduction="mean",
                    image_height=self.image_height,
                    image_width=self.image_width,
                ),
                in_keys=self.pretrained_model_key,
                out_keys="pretrained_embeddings_scaled",
            )
        elif (
            embedding_width < self.image_width and embedding_height < self.image_height
        ):
            return TensorDictModule(
                UpsampleSimulateBatchDims(
                    size=(self.image_height, self.image_width),
                    mode="nearest",
                ),
                in_keys=self.pretrained_model_key,
                out_keys="pretrained_embeddings_scaled",
            )
        else:
            raise ValueError(
                f"The pretrained embeddings must either be the same size as the image, "
                f"smaller in both dimensions or larger in both dimensions. Got "
                f"image size ({self.image_height}, {self.image_width}) and embedding "
                f"size ({embedding_height}, {embedding_width})."
            )

    def build_initial_encoder(self) -> TensorDictSequential:
        """Build the initial encoding layer.

        Concatenates the upsampled message history with the image and pretrained
        embeddings if using, then applies a two-layer MLP to obtain the initial
        pixel-level representations.

        Shapes
        ------
        Input:
            - "x_upsampled" : (... round height width)
            - "image" : (... num_channels height width)
            - "pretrained_embeddings_scaled" : (... embedding_channels height width),
              optional

        Output:
            - "latent_pixel_level_repr" : (... initial_num_channels height width)

        Returns
        -------
        TensorDictSequential
            The initial encoding layer.
        """

        in_channels = (
            self.protocol_handler.max_message_rounds + self.dataset_num_channels
        )

        if not self.include_pretrained_embeddings:
            cat_keys = ("x_upsampled", "image")
        else:
            cat_keys = ("x_upsampled", "image", "pretrained_embeddings_scaled")
            in_channels += self.pretrained_model_num_channels

        return TensorDictSequential(
            TensorDictCat(
                in_keys=cat_keys,
                out_key="latent_pixel_level_repr",
                dim=-3,
            ),
            TensorDictModule(
                Rearrange("... channel height width -> ... height width channel"),
                in_keys="latent_pixel_level_repr",
                out_keys="latent_pixel_level_repr",
            ),
            TensorDictModule(
                Linear(
                    in_channels,
                    in_channels,
                ),
                in_keys="latent_pixel_level_repr",
                out_keys="latent_pixel_level_repr",
            ),
            TensorDictModule(
                self.activation_function(),
                in_keys="latent_pixel_level_repr",
                out_keys="latent_pixel_level_repr",
            ),
            TensorDictModule(
                Linear(
                    in_channels,
                    self.initial_num_channels,
                ),
                in_keys="latent_pixel_level_repr",
                out_keys="latent_pixel_level_repr",
            ),
            TensorDictModule(
                Rearrange("... height width channel -> ... channel height width"),
                in_keys="latent_pixel_level_repr",
                out_keys="latent_pixel_level_repr",
            ),
        )

    def build_cnn_encoder(self) -> TensorDictSequential:
        """Build the the sequence of groups of building blocks.

        Shapes
        ------
        Input:
            - "latent_pixel_level_repr" : (... initial_channels height width)

        Output:
            - "latent_pixel_level_repr" : (... latent_channels latent_height
            latent_width)

        where `latent_channels = initial_channels * 2**num_block_groups`

        Returns
        -------
        cnn_encoder : TensorDictSequential
            The sequence of groups of building blocks.
        """

        stride = self._agent_params.stride
        track_running_stats = self.params.functionalize_modules

        cnn_encoder = []

        for group_index in range(self.num_block_groups):
            # Add the building blocks
            for conv_index in range(self._agent_params.num_blocks_per_group):
                # Determine the number of input and output channels.
                if conv_index == 0:
                    in_channels = 2**group_index * self.initial_num_channels
                else:
                    in_channels = 2 ** (group_index + 1) * self.initial_num_channels
                out_channels = 2 ** (group_index + 1) * self.initial_num_channels

                # Create the appropriate building block
                match self._agent_params.building_block_type:
                    case ImageBuildingBlockType.CONV2D:
                        building_block = Conv2dSimulateBatchDims(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=self._agent_params.kernel_size,
                            stride=stride,
                            padding="same",
                        )
                    case ImageBuildingBlockType.RESIDUAL_BASIC:
                        if stride != 1 or in_channels != out_channels:
                            downsample = Sequential(
                                Conv2d(
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    stride=stride,
                                    kernel_size=1,
                                    bias=False,
                                ),
                                BatchNorm2d(
                                    out_channels,
                                    track_running_stats=track_running_stats,
                                ),
                            )
                        else:
                            downsample = None
                        building_block = ResNetBasicBlockSimulateBatchDims(
                            inplanes=in_channels,
                            planes=out_channels,
                            stride=stride,
                            downsample=downsample,
                            norm_layer=partial(
                                BatchNorm2d,
                                track_running_stats=track_running_stats,
                            ),
                        )

                # Add the building block and non-linearity
                cnn_encoder.append(
                    TensorDictModule(
                        building_block,
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
                    MaxPool2dSimulateBatchDims(kernel_size=2, stride=2),
                    in_keys="latent_pixel_level_repr",
                    out_keys="latent_pixel_level_repr",
                )
            )

        return TensorDictSequential(*cnn_encoder)

    def build_global_pooling(self) -> TensorDictModule:
        """Build the global pooling layer.

        Shapes
        ------
        Input:
            - "latent_pixel_level_repr" : (... latent_channels+1 latent_height
            latent_width)

        Output:
            - "image_level_repr" : (... latent_channels+1)

        Returns
        -------
        global_pooling : TensorDictModule
            The global pooling layer.
        """
        return TensorDictModule(
            Reduce("... channel height width -> ... channel", reduction="mean"),
            in_keys="latent_pixel_level_repr",
            out_keys="image_level_repr",
        )

    def build_final_encoder(self) -> TensorDictSequential:
        """Build the final encoder.

        This rearranges the latent pixel-level representations to put the channel
        dimension last, then applies a linear layer to obtain the final latent
        pixel-level representations. It also concatenates the image-level
        representations with the linear message history if using, then applies a linear
        layer to obtain the final image-level representations.

        Shapes
        ------
        Input:
            - "image_level_repr" : (... latent_channels+1)
            - "latent_pixel_level_repr" : (... latent_channels+1 latent_height
              latent_width)
            - "linear_message_history" : (... round linear_message), optional

        Output:
            - "image_level_repr" : (... d_representation)
            - "latent_pixel_level_repr" : (... latent_height latent_width
              d_representation)

        Returns
        -------
        final_encoder : TensorDictSequential
            The final encoder.
        """

        image_level_num_channels = latent_pixel_level_num_channels = (
            self.latent_num_channels + 1
        )

        final_encoder = []

        # Rearrange the latent pixel-level representations to put the channel dimension
        # last
        final_encoder.append(
            TensorDictModule(
                Rearrange(
                    "... channel latent_height latent_width -> "
                    "... latent_height latent_width channel"
                ),
                in_keys="latent_pixel_level_repr",
                out_keys="latent_pixel_level_repr",
            )
        )

        # Obtain the final latent pixel-level representations
        final_encoder.append(
            TensorDictModule(
                Linear(
                    in_features=latent_pixel_level_num_channels,
                    out_features=self.params.d_representation,
                ),
                in_keys="latent_pixel_level_repr",
                out_keys="latent_pixel_level_repr",
            )
        )

        if self.params.include_linear_message_space:
            # Flatten the linear message history
            final_encoder.append(
                TensorDictModule(
                    Rearrange("... round linear_message -> ... (round linear_message)"),
                    in_keys="linear_message_history",
                    out_keys="linear_message_history_flat",
                )
            )

            # Concatenate the image-level and linear message history representations
            final_encoder.append(
                TensorDictCat(
                    in_keys=("image_level_repr", "linear_message_history_flat"),
                    out_key="image_level_repr",
                    dim=-1,
                )
            )

            image_level_num_channels += (
                self.params.d_linear_message_space
                * self.protocol_handler.max_message_rounds
            )

        # Obtain the final image-level representations
        final_encoder.append(
            TensorDictModule(
                Linear(
                    in_features=image_level_num_channels,
                    out_features=self.params.d_representation,
                ),
                in_keys="image_level_repr",
                out_keys="image_level_repr",
            )
        )

        return TensorDictSequential(*final_encoder)

    def forward(self, data: TensorDictBase) -> TensorDict:
        """Run the image classification body

        Parameters
        ----------
        data : TensorDictBase
            The data to run the body on. A TensorDictBase with keys:

            - "x" (... round latent_height latent_width): The message history
            - "image" (... channel height width): The image
            - "message" (... latent_height latent_width), optional: The most recent
              message
            - "ignore_message" (...), optional: Whether to ignore the message. For
              example, in the first round the there is no message, and the "message"
              field is set to a dummy value.
            - ("pretrained_embeddings", model_name) (... embedding_width
              embedding_height), optional: The embeddings of a pretrained model, if
              using.
            - "linear_message_history" : (... round linear_message), optional: The
              linear message history, if using.

        Returns
        -------
        out : TensorDict
            A tensor dict with keys:

            - "image_level_repr" (... d_representation): The output image-level
              representations.
            - "latent_pixel_level_repr" (... latent_height latent_width
              d_representation): The output latent-pixel-level representations.
        """

        if "message" in data.keys():
            # Add a channel dimension to the message
            message = rearrange(
                data["message"],
                "... latent_height latent_width -> ... 1 latent_height latent_width",
            )
            # If the message is to be ignored, set it to zero
            if "ignore_message" in data.keys():
                message = torch.where(
                    data["ignore_message"][..., None, None, None],
                    0,
                    message,
                )
        else:
            message = torch.zeros(
                (*data.batch_size, 1, self.latent_height, self.latent_width),
                device=self.device,
                dtype=torch.float,
            )

        # Normalize the message history if necessary
        if self._agent_params.normalize_message_history:
            data = self.message_history_normalizer(data)
        else:
            data = data.update(dict(x_normalized=data["x"]))

        # Upsample the message history
        data = self.message_history_upsampler(data)

        # Scale the pretrained embeddings if necessary
        if self.include_pretrained_embeddings:
            data = self.pretrained_embedding_scaler(data)

        # Encode the image and message history
        data = self.initial_encoder(data)

        # Encode the image and message history using the CNN encoder
        data = self.cnn_encoder(data)

        # Add the message to the latent pixel-level representations as a new channel
        data["latent_pixel_level_repr"] = torch.cat(
            (data["latent_pixel_level_repr"], message), dim=-3
        )

        # Pool the latent pixel-level representations to obtain the image-level
        # representations
        data = self.global_pooling(data)

        # Encode the image-level and latent pixel-level representations to obtain the
        # final representations
        data = self.final_encoder(data)

        return data

    def to(self, device: Optional[TorchDevice] = None):
        super().to(device)
        self.device = device
        self.message_history_upsampler = self.message_history_upsampler.to(device)
        self.initial_encoder = self.initial_encoder.to(device)
        self.cnn_encoder = self.cnn_encoder.to(device)
        self.global_pooling = self.global_pooling.to(device)
        self.final_encoder = self.final_encoder.to(device)
        return self

    @property
    def include_pretrained_embeddings(self) -> bool:
        """Whether to include pretrained embeddings."""
        return self._agent_params.pretrained_embeddings_model is not None

    @property
    def pretrained_model_class(self) -> PretrainedImageModel:
        """The pretrained model class to use.

        Raises
        ------
        ValueError
            If no pretrained model is being used.
        """

        if self.include_pretrained_embeddings:
            if self._pretrained_model_class is None:
                self._pretrained_model_class = get_pretrained_model_class(
                    self._agent_params.pretrained_embeddings_model, self.params
                )
            return self._pretrained_model_class

        raise ValueError("No pretrained model is being used.")

    @property
    def pretrained_model_name(self) -> str:
        """The full name of the pretrained model to use, if any.

        This may be different from the model name in the parameters, if the latter is
        a shorthand.

        Raises
        ------
        ValueError
            If no pretrained model is being used.
        """
        return self.pretrained_model_class.name

    @property
    def pretrained_model_num_channels(self) -> int:
        """The number of channels used in the embeddings of the pretrained model.

        This can be different from the base number of channels in the pretrained model,
        if the embeddings are downsampled.

        Raises
        ------
        ValueError
            If no pretrained model is being used.
        """

        if self.include_pretrained_embeddings:
            if self._agent_params.pretrained_embedding_num_channels is not None:
                return self._agent_params.pretrained_embedding_num_channels
            else:
                return self.pretrained_model_class.embedding_channels

        raise ValueError("No pretrained model is being used.")

    @property
    def pretrained_model_key(self) -> tuple[str, str]:
        """The key to use for the pretrained embeddings in the data dict."""
        return (
            "pretrained_embeddings",
            f"{self.pretrained_model_name}_d{self.pretrained_model_num_channels}",
        )


@register_scenario_class(IC_SCENARIO, DummyAgentBody)
class ImageClassificationDummyAgentBody(ImageClassificationAgentPart, DummyAgentBody):
    """Dummy agent body for the image classification task.

    Shapes
    ------
    Input:
        - "x" (... max_message_rounds latent_height latent_width): The message history
        - "image" (... num_channels height width): The image
        - "message" (... latent_height latent_width), optional: The most recent message
        - "ignore_message" (...), optional: Whether to ignore the message

    Output:
        - "image_level_repr" (... d_representation): The output image-level
          representations.
        - "latent_pixel_level_repr" (... latent_height latent_width d_representation):
          The output latent-pixel-level representations.

    """

    env_level_in_keys = ("x", "image")
    agent_level_out_keys = ("image_level_repr", "latent_pixel_level_repr")

    def forward(self, data: TensorDictBase) -> TensorDict:
        """Returns dummy outputs.

        Parameters
        ----------
        data : TensorDictBase
            The data to run the body on.

        Returns
        -------
        out : TensorDict
            The dummy outputs.
        """

        # The dummy image-level representations
        image_level_repr = torch.zeros(
            *data.batch_size,
            self.params.d_representation,
            device=self.device,
            dtype=torch.float32,
        )

        # The dummy latent-pixel-level representations
        latent_pixel_level_repr = torch.zeros(
            *data.batch_size,
            self.latent_width,
            self.latent_height,
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
        flatten_output: bool = True,
        out_key: str = "latent_pixel_mlp_output",
    ) -> TensorDictModule:
        """Builds an MLP which acts on the latent-pixel-level representations.

        Shapes
        ------
        Input:
            - "latent_pixel_level_repr" : (... latent_height latent_width d_in)

        Output:
            - latent_pixel_mlp_output : (... latent_height*latent_width d_out)

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
        flatten_output : bool, default=True
            Whether to flatten the output dimension to `latent_height * latent_width`.
        out_key : str, default="latent_pixel_mlp_output"
            The tensordict key to use for the output of the MLP.

        Returns
        -------
        latent_pixel_mlp : TensorDictModule
            The latent-pixel-level MLP.
        """
        layers = []

        # The layers of the MLP
        layers.append(Linear(d_in, d_hidden))
        layers.append(self.activation_function())
        for _ in range(num_layers - 2):
            layers.append(Linear(d_hidden, d_hidden))
            layers.append(self.activation_function())
        layers.append(Linear(d_hidden, d_out))

        # Flatten the output dimension if necessary
        if flatten_output:
            layers.append(
                Rearrange(
                    "... latent_height latent_width d_out "
                    "-> ... (latent_height latent_width) d_out"
                )
            )

        # Make the layers into a sequential module and wrap it in a TensorDictModule
        sequential = Sequential(*layers)
        tensor_dict_sequential = TensorDictModule(
            sequential, in_keys=("latent_pixel_level_repr",), out_keys=(out_key,)
        )

        tensor_dict_sequential = tensor_dict_sequential.to(self.device)

        return tensor_dict_sequential

    def _build_image_level_mlp(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        num_layers: int,
        include_round: bool = False,
        out_key: str = "image_level_mlp_output",
        squeeze: bool = False,
    ) -> TensorDictSequential:
        """Builds an MLP which acts on the image-level representations.

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
        include_round : bool, default=False
            Whether to include the round number as a (one-hot encoded) input to the MLP.
        out_key : str, default="image_level_mlp_output"
            The tensordict key to use for the output of the MLP.
        squeeze : bool, default=False
            Whether to squeeze the output dimension. Only use this if the output
            dimension is 1.

        Returns
        -------
        image_level_mlp : TensorDictSequential
            The image-level MLP.
        """

        # The final module includes one or two more things then the MLP
        td_sequential_layers = []

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
                    in_keys=("image_level_repr", "round_one_hot"),
                    out_key="image_level_mlp_input",
                    dim=-1,
                ),
            )
        else:
            td_sequential_layers.append(
                TensorDictCloneKeys(
                    in_keys="image_level_repr", out_keys="image_level_mlp_input"
                )
            )

        # The layers of the MLP
        mlp_layers = []

        # The layers of the MLP
        updated_d_in = d_in
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
            mlp, in_keys=("image_level_mlp_input",), out_keys=(out_key,)
        )

        td_sequential_layers.append(mlp)

        return TensorDictSequential(*td_sequential_layers).to(self.device)

    def _build_decider(
        self, d_out: int = 3, include_round: Optional[bool] = None
    ) -> TensorDictModule:
        """Builds the module which produces a image-level output.

        By default it is used to decide whether to continue exchanging messages. In this
        case it outputs a single triple of logits for the three options: guess a
        classification for the image or continue exchanging messages.

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

        return self._build_image_level_mlp(
            d_in=self.params.d_representation,
            d_hidden=self._agent_params.d_decider,
            d_out=d_out,
            num_layers=self._agent_params.num_decider_layers,
            include_round=include_round,
            out_key="decision_logits",
        )


@register_scenario_class(IC_SCENARIO, AgentPolicyHead)
class ImageClassificationAgentPolicyHead(ImageClassificationAgentHead, AgentPolicyHead):
    """Agent policy head for the image classification task.

    Takes as input the output of the agent body and outputs a policy distribution over
    the actions. Both agents select a node to send as a message, and the verifier also
    decides whether to continue exchanging messages or to make a guess.

    Shapes
    ------
    Input:
        - "image_level_repr" (... d_representation): The output image-level
          representations.
        - "latent_pixel_level_repr" (... latent_height latent_width
          d_representation): The output latent-pixel-level representations.
        - "round" (optional) (...): The round number.

    Output:
        - "latent_pixel_selected_logits" (... latent_height*latent_width): A logit for
          each latent pixel, indicating the probability that this latent pixel should be
          sent as a message to the verifier.
        - "decision_logits" (... 3): A logit for each of the three options: guess a
          classification one way or the other, or continue exchanging messages. Set to
          zeros when the decider is not present.
        - "linear_message_selected_logits" (... d_linear_message_space) (optional):
          A logit for each linear message, indicating the probability that this linear
          message should be sent as a message to the verifier.

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

    agent_level_in_keys = ("image_level_repr", "latent_pixel_level_repr")

    @property
    def env_level_in_keys(self) -> tuple[str, ...]:
        if self.decider is not None and self._agent_params.include_round_in_decider:
            return ("round",)
        else:
            return ()

    @property
    def agent_level_out_keys(self) -> tuple[str, ...]:

        agent_level_out_keys = ("latent_pixel_selected_logits", "decision_logits")

        if self.params.include_linear_message_space:
            agent_level_out_keys = (
                *agent_level_out_keys,
                "linear_message_selected_logits",
            )

        return agent_level_out_keys

    def __init__(
        self,
        params: Parameters,
        agent_name: str,
        protocol_handler: ProtocolHandler,
        *,
        device: Optional[TorchDevice] = None,
    ):
        super().__init__(params, agent_name, protocol_handler, device=device)

        # Build the latent pixel selector module
        self.latent_pixel_selector = self._build_latent_pixel_selector()

        # Build the decider module if necessary
        if agent_name == "verifier":
            self.decider = self._build_decider()
        else:
            self.decider = None

        # Build the linear message selector if necessary
        if self.params.include_linear_message_space:
            self.linear_message_selector = self._build_linear_message_selector()
        else:
            self.linear_message_selector = None

    def _build_latent_pixel_selector(self) -> TensorDictModule:
        """Builds the module which selects which latent pixel to send as a message.

        Returns
        -------
        latent_pixel_selector : TensorDictModule
            The latent pixel selector module.
        """
        return self._build_latent_pixel_mlp(
            d_in=self.params.d_representation,
            d_hidden=self._agent_params.d_latent_pixel_selector,
            d_out=1,
            flatten_output=True,
            num_layers=self._agent_params.num_latent_pixel_selector_layers,
            out_key="latent_pixel_selected_logits",
        )

    def _build_linear_message_selector(self) -> TensorDictModule:
        """Builds the module which selects which linear message to send.

        Returns
        -------
        linear_message_selector : TensorDictModule
            The linear message selector module.
        """
        return self._build_image_level_mlp(
            d_in=self.params.d_representation,
            d_hidden=self._agent_params.d_linear_message_selector,
            d_out=self.params.d_linear_message_space,
            num_layers=self._agent_params.num_linear_message_selector_layers,
            include_round=False,
            out_key="linear_message_selected_logits",
        )

    def forward(self, body_output: TensorDict) -> TensorDict:
        """Runs the policy head on the given body output.

        Runs the latent pixel selector module and the decider module if present.

        Parameters
        ----------
        body_output : TensorDict
            The output of the body module. A tensor dict with keys:

            - "image_level_repr" (... d_representation): The output image-level
              representations.
            - "latent_pixel_level_repr" (... latent_height latent_width
              d_representation): The output latent-pixel-level representations.

        Returns
        -------
        out : TensorDict
            A tensor dict with keys:

            - "latent_pixel_selected_logits" (... latent_height*latent_width): A logit
              for each latent pixel, indicating the probability that this latent pixel
              should be sent as a message to the verifier.
            - "decision_logits" (... 3): A logit for each of the three options: guess a
              classification one way or the other, or continue exchanging messages. Set
              to zeros when the decider is not present.
            - "linear_message_selected_logits" (... d_linear_message_space) (optional):
              A logit for each linear message, indicating the probability that this
              linear message should be sent as a message to the verifier.
        """

        out_dict = {}

        out_dict["latent_pixel_selected_logits"] = self.latent_pixel_selector(
            body_output
        )["latent_pixel_selected_logits"].squeeze(-1)

        if self.decider is not None:
            out_dict["decision_logits"] = self.decider(body_output)["decision_logits"]
        else:
            out_dict["decision_logits"] = torch.zeros(
                (*body_output.batch_size, 3),
                device=self.device,
                dtype=torch.float32,
            )

        if self.params.include_linear_message_space:
            out_dict["linear_message_selected_logits"] = self.linear_message_selector(
                body_output
            )["linear_message_selected_logits"]
        else:
            out_dict["linear_message_selected_logits"] = torch.zeros(
                (*body_output.batch_size, self.params.d_linear_message_space),
                device=self.device,
                dtype=torch.float32,
            )

        return TensorDict(out_dict, batch_size=body_output.batch_size)

    def to(self, device: Optional[TorchDevice] = None):
        super().to(device)
        self.device = device
        self.latent_pixel_selector.to(device)
        if self.decider is not None:
            self.decider.to(device)


@register_scenario_class(IC_SCENARIO, RandomAgentPolicyHead)
class ImageClassificationRandomAgentPolicyHead(
    ImageClassificationAgentPart, RandomAgentPolicyHead
):
    """Policy head for the image classification task yielding a uniform distribution.

    Shapes
    ------
    Input:
        - "image_level_repr" (... d_representation): The output image-level
          representations.
        - "latent_pixel_level_repr" (... latent_height latent_width d_representation):
          The output latent-pixel-level representations.

    Output:
        - "latent_pixel_selected_logits" (... latent_height*latent_width): A logit for
          each latent pixel, indicating the probability that this latent pixel should be
          sent as a message to the verifier.
        - "decision_logits" (... 3): A logit for each of the three options: guess a
          classification one way or the other, or continue exchanging messages. Set to
          zeros when the decider is not present.
        - "linear_message_selected_logits" (... d_linear_message_space) (optional):
          A logit for each linear message, indicating the probability that this linear
          message should be sent as a message to the verifier.
    """

    agent_level_in_keys = ("image_level_repr", "latent_pixel_level_repr")

    @property
    def agent_level_out_keys(self) -> tuple[str, ...]:

        agent_level_out_keys = ("latent_pixel_selected_logits", "decision_logits")

        if self.params.include_linear_message_space:
            agent_level_out_keys = (
                *agent_level_out_keys,
                "linear_message_selected_logits",
            )

        return agent_level_out_keys

    def forward(self, body_output: TensorDict) -> TensorDict:
        """Outputs a uniform distribution.

        Parameters
        ----------
        body_output : TensorDict
            The output of the body module.

        Returns
        -------
        out : TensorDict
            A tensor dict with all zero outputs.
        """

        latent_pixel_selected_logits = torch.zeros(
            *body_output.batch_size,
            self.latent_width * self.latent_height,
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
        latent_pixel_selected_logits = (
            latent_pixel_selected_logits * self.dummy_parameter
        )
        decision_logits = decision_logits * self.dummy_parameter

        return body_output.update(
            dict(
                latent_pixel_selected_logits=latent_pixel_selected_logits,
                decision_logits=decision_logits,
            )
        )


@register_scenario_class(IC_SCENARIO, AgentValueHead)
class ImageClassificationAgentValueHead(ImageClassificationAgentHead, AgentValueHead):
    """Value head for the image classification task.

    Takes as input the output of the agent body and outputs a value function.

    Shapes
    ------
    Input:
        - "image_level_repr" (... d_representation): The output image-level
          representations.
        - "round" (optional) (...): The round number.

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

    agent_level_in_keys = ("image_level_repr",)

    @property
    def env_level_in_keys(self):
        if self._agent_params.include_round_in_value:
            return ("round",)
        else:
            return ()

    agent_level_out_keys = ("value",)

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
        return self._build_image_level_mlp(
            d_in=self.params.d_representation,
            d_hidden=self._agent_params.d_value,
            d_out=1,
            num_layers=self._agent_params.num_value_layers,
            include_round=self._agent_params.include_round_in_value,
            out_key="value",
            squeeze=True,
        )

    def forward(self, body_output: TensorDict) -> TensorDict:
        """Runs the value head on the given body output.

        Parameters
        ----------
        body_output : TensorDict
            The output of the body module. A tensor dict with keys:

            - "image_level_repr" (... d_representation): The output graph-level
              representations.

        Returns
        -------
        value_out : TensorDict
            A tensor dict with keys:

            - "value" (...): The estimated value for each batch item
        """

        return self.value_mlp(body_output)

    def to(self, device: Optional[TorchDevice] = None):
        super().to(device)
        self.device = device
        self.value_mlp.to(device)


@register_scenario_class(IC_SCENARIO, ConstantAgentValueHead)
class ImageClassificationConstantAgentValueHead(
    ImageClassificationAgentHead, ConstantAgentValueHead
):
    """A constant value head for the image classification task.

    Shapes
    ------
    Input:
        - "image_level_repr" (... d_representation): The output image-level
          representations.
        - "latent_pixel_level_repr" (... latent_height latent_width d_representation):
          The output latent-pixel-level representations.

    Output:
        - "value" (...): The 'value' for each batch item, which is a constant zero.
    """

    agent_level_in_keys = ("image_level_repr", "latent_pixel_level_repr")
    agent_level_out_keys = ("value",)

    def forward(self, body_output: TensorDict) -> TensorDict:
        """Returns a constant value.

        Parameters
        ----------
        body_output : TensorDict
            The output of the body module.

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


@register_scenario_class(IC_SCENARIO, SoloAgentHead)
class ImageClassificationSoloAgentHead(ImageClassificationAgentHead, SoloAgentHead):
    """Solo agent head for the image classification task.

    Solo agents try to solve the task on their own, without interacting with another
    agents.

    Shapes
    ------
    Input:
        - "image_level_repr" (... d_representation): The output image-level
          representations.

    Output:
        - "decision_logits" (... 2): A logit for each of the two options: guess that the
          graphs are isomorphic, or guess that the graphs are not isomorphic.
    """

    agent_level_in_keys = ("image_level_repr",)
    agent_level_out_keys = ("decision_logits",)

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

    def forward(self, body_output: TensorDict) -> TensorDict:
        """Runs the solo agent head on the given body output.

        Parameters
        ----------
        body_output : TensorDict
            The output of the body module. A tensor dict with keys:

            - "image_level_repr" (... d_representation): The output graph-level
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


@register_scenario_class(IC_SCENARIO, CombinedBody)
class ImageClassificationCombinedBody(CombinedBody):
    """A module which combines the agent bodies for the image classification task.

    Shapes
    ------
    Input:
        - "round" (...): The round number.
        - "x" (... max_message_rounds latent_height latent_width): The message history
        - "image" (... num_channels height width): The image
        - "message" (... latent_height latent_width), optional: The most recent message.
        - "linear_message_history" : (... round linear_message), optional: The
          linear message history, if using.

    Output:
        - ("agents", "latent_pixel_level_repr") (... agents latent_height latent_width
          d_representation): The output latent-pixel-level representations.
        - ("agents", "image_level_repr") (... agents d_representation): The output
          image-level representations.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    bodies : dict[str, ImageClassificationAgentBody]
        The agent bodies to combine.
    """

    additional_in_keys = ("round",)
    excluded_in_keys = (("agents", "ignore_message"),)
    additional_out_keys = ("round",)

    def __init__(
        self,
        params: Parameters,
        protocol_handler: ProtocolHandler,
        bodies: dict[str, ImageClassificationAgentBody],
    ) -> None:
        super().__init__(params, protocol_handler, bodies)

    def forward(self, data: TensorDictBase) -> TensorDict:
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
            body_outputs[agent_name] = self.bodies[agent_name](input_td)

        # Stack the outputs
        latent_pixel_level_repr = torch.stack(
            [
                body_outputs[name]["latent_pixel_level_repr"]
                for name in self._agent_names
            ],
            dim=-4,
        )
        image_level_repr = torch.stack(
            [body_outputs[name]["image_level_repr"] for name in self._agent_names],
            dim=-2,
        )

        return data.update(
            dict(
                agents=dict(
                    latent_pixel_level_repr=latent_pixel_level_repr,
                    image_level_repr=image_level_repr,
                )
            )
        )


@register_scenario_class(IC_SCENARIO, CombinedPolicyHead)
class ImageClassificationCombinedPolicyHead(CombinedPolicyHead):
    """A module which combines the agent policy heads for the image classification task.

    Shapes
    ------
    Input:
        - "round" (...): The round number.
        - ("agents", "latent_pixel_level_repr") (... agents latent_height latent_width
          d_representation): The output latent-pixel-level representations.
        - ("agents", "image_level_repr") (... agents d_representation): The output
          image-level representations.
        - "decision_restriction" (...): The restriction on what decisions are allowed.

    Output:
        - ("agents", "latent_pixel_selected_logits") (... agents
          latent_height*latent_width): A logit for each latent pixel, indicating the
          probability that this latent pixel should be sent as a message to the
          verifier.
        - ("agents", "decision_logits") (... agents 3): A logit for each of the three
          options: guess a classification one way or the other, or continue exchanging
          messages. Set to zeros when the decider is not present.
        - ("agents", "linear_message_selected_logits") (... agents
          d_linear_message_space) (optional): A logit for each linear message,
          indicating the probability that this linear message should be sent as a
          message to the verifier.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    policy_heads : dict[str, ImageClassificationAgentPolicyHead]
        The agent policy heads to combine.
    """

    additional_in_keys = ("decision_restriction",)

    def __init__(
        self,
        params: Parameters,
        protocol_handler: ProtocolHandler,
        policy_heads: dict[str, ImageClassificationAgentPolicyHead],
    ):
        super().__init__(params, protocol_handler, policy_heads)

    def forward(self, head_output: TensorDictBase) -> TensorDict:
        """Run the agent policy heads and combine their outputs.

        Parameters
        ----------
        tensordict : TensorDictBase
            The input to the value heads.

        Returns
        -------
        tensordict: TensorDict
            The tensordict update in place with the output of the value heads.
        """

        # Run the policy heads to obtain the probability distributions
        policy_outputs: dict[str, TensorDict] = {}
        for i, agent_name in enumerate(self._agent_names):
            latent_pixel_level_repr = head_output["agents", "latent_pixel_level_repr"]
            image_level_repr = head_output["agents", "image_level_repr"]
            input_td = TensorDict(
                dict(
                    latent_pixel_level_repr=latent_pixel_level_repr[..., i, :, :, :],
                    image_level_repr=image_level_repr[..., i, :],
                ),
                batch_size=head_output.batch_size,
            )
            if "round" in head_output.keys():
                input_td["round"] = head_output["round"]
            policy_outputs[agent_name] = self.policy_heads[agent_name](input_td)

        agents_update = {}

        # Stack the outputs
        agents_update["latent_pixel_selected_logits"] = torch.stack(
            [
                policy_outputs[name]["latent_pixel_selected_logits"]
                for name in self._agent_names
            ],
            dim=-2,
        )
        agents_update["decision_logits"] = torch.stack(
            [policy_outputs[name]["decision_logits"] for name in self._agent_names],
            dim=-2,
        )
        if self.params.include_linear_message_space:
            agents_update["linear_message_selected_logits"] = torch.stack(
                [
                    policy_outputs[name]["linear_message_selected_logits"]
                    for name in self._agent_names
                ],
                dim=-2,
            )

        # Make sure the verifier only selects decisions which are allowed
        agents_update["decision_logits"] = self._restrict_decisions(
            head_output["decision_restriction"], agents_update["decision_logits"]
        )

        return head_output.update(
            dict(
                agents=TensorDict(
                    agents_update,
                    batch_size=head_output.batch_size,
                )
            )
        )


@register_scenario_class(IC_SCENARIO, CombinedValueHead)
class ImageClassificationCombinedValueHead(CombinedValueHead):
    """A module which combines the agent value heads for the image classification task.

    Shapes
    ------
    Input:
        - "round" (...): The round number.
        - ("agents", "latent_pixel_level_repr") (... agents latent_height latent_width
          d_representation): The output latent-pixel-level representations.
        - ("agents", "image_level_repr") (... agents d_representation): The output
          image-level representations.

    Output:
        - ("agents", "value") (... agents): The estimated value for each batch item

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    value_heads : dict[str, ImageClassificationAgentValueHead]
        The agent value heads to combine.
    """

    def __init__(
        self,
        params: Parameters,
        protocol_handler: ProtocolHandler,
        value_heads: dict[str, ImageClassificationAgentValueHead],
    ):
        super().__init__(params, protocol_handler, value_heads)

    def forward(self, head_output: TensorDictBase) -> TensorDict:
        """Run the agent value heads and combine their values.

        Parameters
        ----------
        tensordict : TensorDictBase
            The input to the value heads. Should contain the keys:

            - ("agents", "image_level_repr"): The node-level representation from the
              body.

        Returns
        -------
        tensordict: TensorDict
            The tensordict update in place with the output of the value heads.
        """

        # Run the policy heads to obtain the value estimates
        value_outputs: dict[str, TensorDict] = {}
        for i, agent_name in enumerate(self._agent_names):
            latent_pixel_level_repr = head_output["agents", "latent_pixel_level_repr"]
            image_level_repr = head_output["agents", "image_level_repr"]
            input_td = TensorDict(
                dict(
                    latent_pixel_level_repr=latent_pixel_level_repr[..., i, :, :, :],
                    image_level_repr=image_level_repr[..., i, :],
                ),
                batch_size=head_output.batch_size,
            )
            if "round" in head_output.keys():
                input_td["round"] = head_output["round"]
            value_outputs[agent_name] = self.value_heads[agent_name](input_td)

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


@register_scenario_class(IC_SCENARIO, Agent)
@dataclass
class ImageClassificationAgent(Agent):
    message_logits_key: ClassVar[str] = "latent_pixel_selected_logits"
