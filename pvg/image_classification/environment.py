"""The image classification RL environment."""

from typing import Optional
from math import floor

import torch
from torch import Tensor

from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.data.tensor_specs import (
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)

from pvg.parameters import ScenarioType
from pvg.scenario_base import Environment, TensorDictEnvironment
from pvg.factory import register_scenario_class
from pvg.image_classification.data import DATASET_WRAPPER_CLASSES


@register_scenario_class(ScenarioType.IMAGE_CLASSIFICATION, Environment)
class ImageClassificationEnvironment(TensorDictEnvironment):
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

    main_message_out_key = "latent_pixel_selected"

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
    def main_message_space_shape(self) -> tuple:
        return (self.latent_height, self.latent_width)

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
                self.protocol_handler.num_message_channels,
                self.params.message_size,
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
        action_spec = super()._get_action_spec()
        action_spec["agents"]["latent_pixel_selected"] = DiscreteTensorSpec(
            self.latent_height * self.latent_width,
            shape=(
                self.num_envs,
                self.num_agents,
                self.protocol_handler.num_message_channels,
                self.params.message_size,
            ),
            dtype=torch.long,
            device=self.device,
        )
        return action_spec

    def _masked_reset(
        self, env_td: TensorDictBase, mask: Tensor, data_batch: TensorDict
    ) -> TensorDictBase:

        env_td = super()._masked_reset(env_td, mask, data_batch)

        env_td["image"][mask] = data_batch["image"]

        return env_td
