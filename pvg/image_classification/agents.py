"""Image classification agents components.

Contains classes for building agent bodies and heads for the image classification task.

The structure of all agent bodies is the same:

- An encoder layer, which takes as input the image and the message history and outputs
  the initial pixel-level encodings.
- A sequence of `num_conv_groups` groups of convolutional layers. 
    + Each layer is followed by a ReLU and each group by a max pooling layer. 
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
from torch.nn import Sequential, ReLU, Linear, Conv2d, MaxPool2d, Upsample
from torch import Tensor

from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

from einops.layers.torch import Rearrange, Reduce

from jaxtyping import Int

from pvg.scenario_base import (
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
from pvg.parameters import (
    Parameters,
    ImageClassificationAgentParameters,
)
from pvg.utils.torch_modules import (
    Squeeze,
    TensorDictCat,
    ParallelTensorDictModule,
    OneHot,
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


class ImageClassificationAgentBody(ImageClassificationAgentPart, AgentBody):
    """The body of an image classification agent.

    Takes as input the image, message history and the most recent message and outputs
    the image-level and latent pixel-level representations.

    Shapes
    ------
    Input:
        - "x" (... max_message_rounds latent_width latent_height): The message history
        - "image" (... num_channels width height): The image
        - "message" (... latent_width latent_height): The most recent message
        - "ignore_message" (...): Whether to ignore the message

    Output:
        - "image_level_repr" (... d_representation): The output image-level
          representations.
        - "latent_pixel_level_repr" (... latent_width latent_height d_representation):
          The output latent-pixel-level representations.

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
            - "message" : (... max_message_rounds latent_width latent_height)

        Output:
            - "message_upsampled" : (... max_message_rounds width height)

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
            - "x_upsampled" : (... max_message_rounds width height) image : (...
            num_channels width height)

        Output:
            - "latent_pixel_level_repr" : (... initial_num_channels width height)

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
            - "latent_pixel_level_repr" : (... initial_num_channels width height)

        Output:
            - "latent_pixel_level_repr" : (... latent_num_channels latent_width
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

                # Add the convolutional layer and ReLU
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
                        ReLU(),
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
            - "latent_pixel_level_repr" : (... latent_num_channels latent_width
            latent_height)

        Output:
            - "image_level_repr" : (... latent_num_channels)

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
            - "image_level_repr" : (... latent_num_channels)
            - "latent_pixel_level_repr" : (... latent_num_channels latent_width
              latent_height)

        Output:
            - "image_level_repr" : (... d_representation)
            - "latent_pixel_level_repr" : (... latent_width latent_height
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

            - "x" (... max_message_rounds latent_width latent_height): The message
              history
            - "image" (... num_channels width height): The image
            - "message" (... latent_width latent_height): The message most recent
              message

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
    """Dummy agent body for the image classification task.

    Shapes
    ------
    Input:
        - "x" (... max_message_rounds latent_width latent_height): The message history
        - "image" (... num_channels width height): The image
        - "message" (... latent_width latent_height): The most recent message
        - "ignore_message" (...): Whether to ignore the message

    Output:
        - "image_level_repr" (... d_representation): The output image-level
          representations.
        - "latent_pixel_level_repr" (... latent_width latent_height d_representation):
          The output latent-pixel-level representations.

    """

    in_keys = ("x", "image", "message", "ignore_message")
    out_keys = ("image_level_repr", "latent_pixel_level_repr")

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
        """Builds an MLP which acts on the latent-pixel-level representations.

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
            The latent-pixel-level MLP.
        """
        layers = []

        # The layers of the MLP
        layers.append(Linear(d_in, d_hidden))
        layers.append(ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(Linear(d_hidden, d_hidden))
            layers.append(ReLU(inplace=True))
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
        include_round: bool = False,
        out_key: str = "image_level_mlp_output",
        squeeze: bool = False,
    ) -> TensorDictSequential:
        """Builds an MLP which acts on the latent-pixel-level representations.

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
        latent_pixel_mlp : TensorDictSequential
            The latent-pixel-level MLP.
        """
        mlp_layers = []

        # The layers of the MLP
        updated_d_in = d_in
        if include_round:
            updated_d_in += self.params.max_message_rounds + 1
        mlp_layers.append(Linear(updated_d_in, d_hidden))
        mlp_layers.append(ReLU(inplace=True))
        for _ in range(num_layers - 2):
            mlp_layers.append(Linear(d_hidden, d_hidden))
            mlp_layers.append(ReLU(inplace=True))
        mlp_layers.append(Linear(d_hidden, d_out))

        # Squeeze the output dimension if necessary
        if squeeze:
            mlp_layers.append(Squeeze())

        # Make the layers into a sequential module, and wrap it in a TensorDictModule
        mlp = Sequential(*mlp_layers)
        mlp = TensorDictModule(
            mlp, in_keys=("image_level_mlp_input",), out_keys=(out_key,)
        )

        # The final module includes one or two more things before the MLP
        td_sequential_layers = []

        if include_round:
            # Add the round number as an input to the MLP
            td_sequential_layers.append(
                TensorDictModule(
                    OneHot(num_classes=self.params.max_message_rounds + 1),
                    in_keys=("round"),
                    out_keys=("round_one_hot",),
                )
            )
            td_sequential_layers.append(
                TensorDictCat(
                    in_keys=("image_level_mlp_input", "round_one_hot"),
                    out_key="image_level_mlp_input",
                    dim=-1,
                ),
            )

        td_sequential_layers.append(mlp)

        return TensorDictSequential(*td_sequential_layers).to(self.device)

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
            include_round=self._agent_params.include_round_in_decider,
            out_key="decision_logits",
        )


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
        - "latent_pixel_level_repr" (... latent_width latent_height
          d_representation): The output latent-pixel-level representations.
        - "round" (optional) (...): The round number.

    Output:
        - "latent_pixel_selected_logits" (... latent_width latent_height): A logit for
          each latent pixel, indicating the probability that this latent pixel should be
          sent as a message to the verifier.
        - "decision_logits" (... 3): A logit for each of the three options: guess a
          classification one way or the other, or continue exchanging messages. Set to
          zeros when the decider is not present.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    agent_name : str
        The name of the agent.
    device : TorchDevice, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    @property
    def in_keys(self):
        if self.decider is not None and self._agent_params.include_round_in_decider:
            return ("image_level_repr", "latent_pixel_level_repr", "round")
        else:
            return ("image_level_repr", "latent_pixel_level_repr")

    @property
    def out_keys(self):
        if self.decider is None:
            return ("latent_pixel_selected_logits",)
        else:
            return ("latent_pixel_selected_logits", "decision_logits")

    def __init__(
        self,
        params: Parameters,
        agent_name: str,
        device: Optional[TorchDevice] = None,
    ):
        super().__init__(params, agent_name, device)

        # Build the latent pixel selector module
        self.latent_pixel_selector = self._build_latent_pixel_selector()

        # Build the decider module if necessary
        if agent_name == "verifier":
            self.decider = self._build_decider()
        else:
            self.decider = None

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
            num_layers=self._agent_params.num_latent_pixel_selector_layers,
            out_key="latent_pixel_selected_logits",
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
            - "latent_pixel_level_repr" (... latent_width latent_height
              d_representation): The output latent-pixel-level representations.

        Returns
        -------
        out : TensorDict
            A tensor dict with keys:

            - "latent_pixel_selected_logits" (... latent_width latent_height): A logit
              for each latent pixel, indicating the probability that this latent pixel
              should be sent as a message to the verifier.
            - "decision_logits" (... 3): A logit for each of the three options: guess a
              classification one way or the other, or continue exchanging messages. Set
              to zeros when the decider is not present.
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

        return TensorDict(out_dict, batch_size=body_output.batch_size)

    def to(self, device: Optional[TorchDevice] = None):
        super().to(device)
        self.device = device
        self.latent_pixel_selector.to(device)
        if self.decider is not None:
            self.decider.to(device)


class ImageClassificationRandomAgentPolicyHead(
    ImageClassificationAgentPart, RandomAgentPolicyHead
):
    """Policy head for the image classification task yielding a uniform distribution.

    Shapes
    ------
    Input:
        - "image_level_repr" (... d_representation): The output image-level
          representations.
        - "latent_pixel_level_repr" (... latent_width latent_height d_representation):
          The output latent-pixel-level representations.

    Output:
        - "latent_pixel_selected_logits" (... latent_width latent_height): A logit for
          each latent pixel, indicating the probability that this latent pixel should be
          sent as a message to the verifier.
        - "decision_logits" (... 3): A logit for each of the three options: guess a
          classification one way or the other, or continue exchanging messages. Set to
          zeros when the decider is not present.
    """

    in_keys = ("image_level_repr", "latent_pixel_level_repr")

    @property
    def out_keys(self):
        if self.decider:
            return ("latent_pixel_selected_logits",)
        else:
            return ("latent_pixel_selected_logits", "decision_logits")

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
            The output of the body module.

        Returns
        -------
        out : TensorDict
            A tensor dict with all zero outputs.
        """

        latent_pixel_selected_logits = torch.zeros(
            *body_output.batch_size,
            self._latent_width,
            self._latent_height,
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
    device : TorchDevice, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    out_keys = ("value",)

    @property
    def in_keys(self):
        if self._agent_params.include_round_in_value:
            return ("image_level_repr", "round")
        else:
            return "image_level_repr"

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


class ImageClassificationConstantAgentValueHead(
    ImageClassificationAgentHead, ConstantAgentValueHead
):
    """A constant value head for the image classification task.

    Shapes
    ------
    Input:
        - "image_level_repr" (... d_representation): The output image-level
          representations.
        - "latent_pixel_level_repr" (... latent_width latent_height d_representation):
          The output latent-pixel-level representations.

    Output:
        - "value" (...): The 'value' for each batch item, which is a constant zero.
    """

    in_keys = ("image_level_repr", "latent_pixel_level_repr")
    out_keys = ("value",)

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

    in_keys = ("image_level_repr",)
    out_keys = ("decision_logits",)

    def __init__(
        self,
        params: Parameters,
        agent_name: str,
        device: Optional[TorchDevice] = None,
    ):
        super().__init__(params, agent_name, device)

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


class ImageClassificationCombinedBody(CombinedBody):
    """A module which combines the agent bodies for the image classification task.

    Shapes
    ------
    Input:
        - "round" (...): The round number.
        - "x" (... max_message_rounds latent_width latent_height): The message history
        - "image" (... num_channels width height): The image

    Output:
        - ("agents", "latent_pixel_level_repr") (... num_agents latent_width
          latent_height d_representation): The output latent-pixel-level
          representations.
        - ("agents", "image_level_repr") (... num_agents d_representation): The output
          image-level representations.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    bodies : dict[str, ImageClassificationAgentBody]
        The agent bodies to combine.
    """

    in_keys = ("round", "x", "image", "message")
    out_keys = (
        "round",
        ("agents", "latent_pixel_level_repr"),
        ("agents", "image_level_repr"),
    )

    def __init__(
        self,
        params: Parameters,
        bodies: dict[str, ImageClassificationAgentBody],
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
        latent_pixel_level_repr = torch.stack(
            [
                body_outputs[name]["latent_pixel_level_repr"]
                for name in self.params.agents
            ],
            dim=-4,
        )
        image_level_repr = torch.stack(
            [body_outputs[name]["image_level_repr"] for name in self.params.agents],
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


class ImageClassificationCombinedPolicyHead(CombinedPolicyHead):
    """A module which combines the agent policy heads for the image classification task.

    Shapes
    ------
    Input:
        - "round" (...): The round number.
        - ("agents", "latent_pixel_level_repr") (... num_agents latent_width
          latent_height d_representation): The output latent-pixel-level
          representations.
        - ("agents", "image_level_repr") (... num_agents d_representation): The output
          image-level representations.

    Output:
        - ("agents", "latent_pixel_selected_logits") (... num_agents latent_width
          latent_height): A logit for each latent pixel, indicating the probability that
          this latent pixel should be sent as a message to the verifier.
        - ("agents", "decision_logits") (... num_agents 3): A logit for each of the
          three options: guess a classification one way or the other, or continue
          exchanging messages. Set to zeros when the decider is not present.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    policy_heads : dict[str, ImageClassificationAgentPolicyHead]
        The agent policy heads to combine.
    """

    in_keys = (
        ("agents", "latent_pixel_level_repr"),
        ("agents", "image_level_repr"),
        "round",
    )
    out_keys = (
        ("agents", "latent_pixel_selected_logits"),
        ("agents", "decision_logits"),
    )

    def __init__(
        self,
        params: Parameters,
        policy_heads: dict[str, ImageClassificationAgentPolicyHead],
    ):
        super().__init__(params, policy_heads)

    def forward(self, head_output: TensorDictBase) -> TensorDict:
        """Run the agent policy heads and combine their outputs.

        Parameters
        ----------
        tensordict : TensorDictBase
            The input to the value heads. Should contain the keys:

            - ("agents", "latent_pixel_level_repr"): The node-level representation from
              the body.
            - ("agents", "image_level_repr"): The node-level representation from the
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
                    latent_pixel_level_repr=head_output[
                        "agents", "latent_pixel_level_repr"
                    ][..., i, :, :, :],
                    image_level_repr=head_output["agents", "image_level_repr"][
                        ..., i, :, :
                    ],
                    round=head_output["round"],
                ),
                batch_size=head_output.batch_size,
            )
            policy_outputs[agent_name] = self.policy_heads[agent_name](input_td)

        # Stack the outputs
        latent_pixel_selected_logits = torch.stack(
            [
                policy_outputs[name]["latent_pixel_selected_logits"]
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
                        latent_pixel_selected_logits=latent_pixel_selected_logits,
                        decision_logits=decision_logits,
                    ),
                    batch_size=head_output.batch_size,
                )
            )
        )


class ImageClassificationCombinedValueHead(CombinedValueHead):
    """A module which combines the agent value heads for the image classification task.

    Shapes
    ------
    Input:
        - "round" (...): The round number.
        - ("agents", "latent_pixel_level_repr") (... num_agents latent_width
          latent_height d_representation): The output latent-pixel-level
          representations.
        - ("agents", "image_level_repr") (... num_agents d_representation): The output
          image-level representations.

    Output:
        - ("agents", "value") (... num_agents): The estimated value for each batch item

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    value_heads : dict[str, ImageClassificationAgentValueHead]
        The agent value heads to combine.
    """

    in_keys = (("agents", "image_level_repr"), "round")
    out_keys = (("agents", "value"),)

    def __init__(
        self,
        params: Parameters,
        value_heads: dict[str, ImageClassificationAgentValueHead],
    ):
        super().__init__(params, value_heads)

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
        for i, agent_name in enumerate(self.params.agents):
            input_td = TensorDict(
                dict(
                    latent_pixel_level_repr=head_output[
                        "agents", "latent_pixel_level_repr"
                    ][..., i, :, :, :],
                    image_level_repr=head_output["agents", "image_level_repr"][
                        ..., i, :, :
                    ],
                    round=head_output["round"],
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
