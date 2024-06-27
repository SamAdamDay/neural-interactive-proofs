"""The image classification RL environment."""

from typing import Optional
from math import floor

import torch
from torch import Tensor
import torch.nn.functional as F

from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.data.tensor_specs import (
    CompositeSpec,
    DiscreteTensorSpec,
    BinaryDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)

from jaxtyping import Float, Int

from pvg.parameters import ScenarioType
from pvg.scenario_base import Environment
from pvg.scenario_instance import register_scenario_class
from pvg.image_classification.data import DATASET_WRAPPER_CLASSES


@register_scenario_class(ScenarioType.IMAGE_CLASSIFICATION, Environment)
class ImageClassificationEnvironment(Environment):
    """The image classification RL environment.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The settings of the experiment.
    dataset : Dataset
        The dataset for the environment.
    protocol_handler : ProtocolHandler
        The protocol handler for the environment.
    train : bool, optional
        Whether the environment is used for training or evaluation.
    """

    _int_dtype: torch.dtype = torch.int32
    _max_num_nodes: Optional[int] = None

    @property
    def num_block_groups(self):
        return self.params.image_classification.num_block_groups

    @property
    def initial_num_channels(self):
        return self.params.image_classification.initial_num_channels

    @property
    def dataset_num_channels(self):
        return DATASET_WRAPPER_CLASSES[self.params.dataset].num_channels

    @property
    def image_width(self):
        return DATASET_WRAPPER_CLASSES[self.params.dataset].width

    @property
    def image_height(self):
        return DATASET_WRAPPER_CLASSES[self.params.dataset].height

    @property
    def latent_width(self):
        latent_width = self.image_width
        for _ in range(self.num_block_groups):
            latent_width = floor(latent_width / 2)
        return latent_width

    @property
    def latent_height(self):
        latent_height = self.image_height
        for _ in range(self.num_block_groups):
            latent_height = floor(latent_height / 2)
        return latent_height

    @property
    def latent_num_channels(self):
        return 2**self.num_block_groups * self.initial_num_channels

    @property
    def _message_history_shape(self) -> tuple:
        return (
            self.num_envs,
            self.protocol_handler.max_message_rounds,
            self.latent_height,
            self.latent_width,
        )

    def _get_observation_spec(self) -> CompositeSpec:
        """Get the specification of the agent observations.

        Agents see the image and the messages sent so far. The "message" field contains
        the most recent message.

        Returns
        -------
        observation_spec : CompositeSpec
            The observation specification.
        """

        observation_spec = super()._get_observation_spec()

        observation_spec["image"] = UnboundedContinuousTensorSpec(
            shape=(
                self.num_envs,
                self.dataset_num_channels,
                self.image_height,
                self.image_width,
            ),
            dtype=torch.float,
            device=self.device,
        )

        observation_spec["message"] = DiscreteTensorSpec(
            self.latent_width,
            shape=(
                self.num_envs,
                self.latent_height,
                self.latent_width,
            ),
            dtype=torch.float,
            device=self.device,
        )

        return observation_spec

    def _get_action_spec(self) -> CompositeSpec:
        """Get the specification of the agent actions.

        Each action space has shape (batch_size, num_agents). Each agent chooses both a
        latent pixel and a decision: reject, accept or continue (represented as 0, 1 or
        2).

        Returns
        -------
        action_spec : CompositeSpec
            The action specification.
        """
        base_action_spec = super()._get_action_spec()
        base_action_spec["agents"]["latent_pixel_selected"] = DiscreteTensorSpec(
            self.latent_height * self.latent_width,
            shape=(self.num_envs, self.num_agents),
            dtype=torch.long,
            device=self.device,
        )
        return base_action_spec

    def _compute_message_history(
        self,
        env_td: TensorDictBase,
        next_td: TensorDictBase,
    ) -> TensorDictBase:
        """Compute the message history and next message.

        Used in the `_step` method of the environment.

        Parameters
        ----------
        env_td : TensorDictBase
            The current observation and state.
        next_td : TensorDictBase
            The 'next' tensordict, to be updated with the message history and next
            message.

        Returns
        -------
        next_td : TensorDictBase
            The updated 'next' tensordict.
        """

        # Extract the tensors from the dict
        message_history: Float[Tensor, "... round latent_height latent_width"] = env_td[
            "message_history"
        ]
        round: Int[Tensor, "..."] = env_td["round"]
        latent_pixel_selected: Int[Tensor, "... agent"] = env_td[
            "agents", "latent_pixel_selected"
        ]

        # Compute index of the agent whose turn it is.
        # (... agent)
        active_agents_mask = self.protocol_handler.get_active_agents_mask(round)

        # Sum up the messages from the agents whose turn it is. If two agents select the
        # same latent pixel, the message will be 2.
        # (... latent_height latent_width)
        message = F.one_hot(
            latent_pixel_selected, self.latent_height * self.latent_width
        ).float()
        message = torch.where(
            active_agents_mask[..., None], message, torch.zeros_like(message)
        )
        message = message.sum(dim=-2)
        message = message.view(
            *message.shape[:-1], self.latent_height, self.latent_width
        )

        # Insert the message into the message history at the current round
        round_mask = F.one_hot(round, self.protocol_handler.max_message_rounds).bool()
        message_history = message_history.masked_scatter(
            round_mask[..., :, None, None], message
        )

        # Add the message history and next message to the next tensordict
        next_td["message_history"] = message_history
        next_td["message"] = message

        return next_td

    def _masked_reset(
        self, env_td: TensorDictBase, mask: Tensor, data_batch: TensorDict
    ) -> TensorDictBase:
        """Reset the environment for a subset of the episodes.

        Takes a new sample from the dataset and inserts it into the given episodes. Also
        resets the other elements of the episodes.

        Parameters
        ----------
        env_td : TensorDictBase
            The current observation, state and done signal.
        mask : torch.Tensor
            A boolean mask of the episodes to reset.
        data_batch : TensorDict
            The data batch to insert into the episodes.

        Returns
        -------
        env_td : TensorDictBase
            The reset environment tensordict.
        """

        env_td["image"][mask] = data_batch["image"]
        env_td["y"][mask] = data_batch["y"].unsqueeze(-1)
        env_td["message_history"][mask] = torch.zeros_like(
            env_td["message_history"][mask]
        )
        env_td["x"][mask] = torch.zeros_like(env_td["x"][mask])
        env_td["message"][mask] = 0
        env_td["round"][mask] = 0
        env_td["done"][mask] = False
        env_td["terminated"][mask] = False
        env_td["decision_restriction"][mask] = 0

        pretrained_model_names = self.dataset.pretrained_model_names
        for model_name in pretrained_model_names:
            env_td["pretrained_embeddings", model_name][mask] = data_batch[
                "pretrained_embeddings", model_name
            ]

        return env_td
