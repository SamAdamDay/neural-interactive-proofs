"""The image classification RL environment."""

from typing import Optional

import torch
from torch import Tensor

from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.data.tensor_specs import (
    CompositeSpec,
    DiscreteTensorSpec,
    BinaryDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)

from jaxtyping import Float, Int

from pvg.scenario_base import Environment
from pvg.image_classification.data import IMAGE_DATASETS


class ImageClassificationEnvironment(Environment):
    """The image classification RL environment.

    Parameters
    ----------
    params : Parameters
        The parameters of the environment.
    device : torch.device, optional
        The device on which the environment should be stored.
    """

    _int_dtype: torch.dtype = torch.int32
    _max_num_nodes: Optional[int] = None

    @property
    def num_conv_groups(self):
        return self.params.image_classification.num_conv_groups

    @property
    def initial_num_channels(self):
        return self.params.image_classification.initial_num_channels

    @property
    def dataset_num_channels(self):
        return IMAGE_DATASETS[self.params.dataset].num_channels

    @property
    def image_width(self):
        return IMAGE_DATASETS[self.params.dataset].width

    @property
    def image_height(self):
        return IMAGE_DATASETS[self.params.dataset].height

    @property
    def latent_width(self):
        return self.image_width // 2**self.num_conv_groups

    @property
    def latent_height(self):
        return self.image_height // 2**self.num_conv_groups

    @property
    def latent_num_channels(self):
        return 2**self.num_conv_groups * self.initial_num_channels

    def _get_observation_spec(self) -> CompositeSpec:
        """Get the specification of the agent observations.

        Agents see the adjacency matrix and the messages sent so far. The "message"
        field contains the most recent message.

        Returns
        -------
        observation_spec : CompositeSpec
            The observation specification.
        """
        return CompositeSpec(
            image=UnboundedContinuousTensorSpec(
                shape=(
                    self.num_envs,
                    self.dataset_num_channels,
                    self.image_width,
                    self.image_height,
                ),
                dtype=torch.float,
            ),
            x=BinaryDiscreteTensorSpec(
                self.latent_height,
                shape=(
                    self.num_envs,
                    self.params.max_message_rounds,
                    self.latent_width,
                    self.latent_height,
                ),
                dtype=torch.float,
            ),
            message=DiscreteTensorSpec(
                self.latent_height * self.latent_width,
                shape=(self.num_envs,),
                dtype=torch.long,
            ),
            round=DiscreteTensorSpec(
                self.params.max_message_rounds,
                shape=(self.num_envs,),
                dtype=torch.long,
            ),
            shape=(self.num_envs,),
        )

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
        return CompositeSpec(
            agents=CompositeSpec(
                latent_pixel_selected=DiscreteTensorSpec(
                    self.latent_height * self.latent_width,
                    shape=(self.num_envs, 2),
                    dtype=torch.long,
                ),
                decision=DiscreteTensorSpec(
                    3,
                    shape=(self.num_envs, 2),
                    dtype=self._int_dtype,
                ),
                shape=(self.num_envs,),
            ),
            shape=(self.num_envs,),
        )

    def _compute_x_and_message(
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
        x: Float[Tensor, "batch round latent_height latent_width"] = env_td["x"]
        round: Int[Tensor, "batch"] = env_td["round"]
        latent_pixel_selected: Int[Tensor, "batch agent"] = env_td[
            "agents", "latent_pixel_selected"
        ]

        batch_size = x.shape[0]

        # Compute index of the agent whose turn it is.
        agent_index: Int[Tensor, "batch"] = round % len(self.agent_names)

        # Compute the latent pixel selected by the agents
        # (batch agent)
        latent_pixel_selected_x = latent_pixel_selected // self.latent_height
        latent_pixel_selected_y = latent_pixel_selected % self.latent_height

        # Write the latent pixel selected by the agent whose turn it is as a (one-hot)
        # message
        x[
            torch.arange(batch_size),
            round,
            latent_pixel_selected_y[torch.arange(batch_size), agent_index],
            latent_pixel_selected_x[torch.arange(batch_size), agent_index],
        ] = 1

        # Set the latent pixel selected by the agent whose turn it is as the message
        message = latent_pixel_selected[torch.arange(batch_size), agent_index].long()

        # Add the message history and next message to the next tensordict
        next_td["x"] = x
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
        env_td["x"][mask] = torch.zeros_like(env_td["x"][mask])
        env_td["message"][mask] = 0
        env_td["round"][mask] = 0
        env_td["done"][mask] = False

        return env_td
