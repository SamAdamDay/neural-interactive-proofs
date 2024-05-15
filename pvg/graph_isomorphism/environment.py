"""The graph isomorphism RL environment."""

from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.data.tensor_specs import (
    CompositeSpec,
    DiscreteTensorSpec,
    BinaryDiscreteTensorSpec,
    TensorSpec,
    Box,
)

from jaxtyping import Float, Int

from pvg.parameters import ScenarioType
from pvg.scenario_base import Environment
from pvg.scenario_instance import register_scenario_class
from pvg.utils.types import TorchDevice


class AdjacencyMatrixBox(Box):
    """An abstract representation of the space of adjacency matrices.

    Parameters
    ----------
    max_num_nodes : int
        The maximum number of nodes in the graph.
    """

    def __init__(self, max_num_nodes: int):
        self.max_num_nodes = max_num_nodes

    def clone(self) -> "AdjacencyMatrixBox":
        return AdjacencyMatrixBox(self.max_num_nodes)


class AdjacencyMatrixSpec(TensorSpec):
    """A specification of the space of adjacency matrices.

    This represents a space of adjacency matrices with a fixed number of nodes, for use
    in an RL environment.

    Parameters
    ----------
    max_num_nodes : int
        The maximum number of nodes in the graph.
    shape : torch.Size, optional
        The shape of the adjacency matrix. If None, the shape will be
        (max_num_nodes, max_num_nodes).
    device : torch.device, optional
        The device on which the adjacency matrix spec should be stored.
    dtype : torch.dtype, optional
        The dtype of the adjacency matrix spec.
    """

    def __init__(
        self,
        max_num_nodes: int,
        shape: torch.Size | None = None,
        device: Optional[TorchDevice] = None,
        dtype: str | torch.dtype = torch.int32,
    ):
        self.max_num_nodes = max_num_nodes
        self.device = device
        self.dtype = dtype

        if shape is None:
            self.shape = torch.Size([max_num_nodes, max_num_nodes])
        else:
            if shape[-2:] != (max_num_nodes, max_num_nodes):
                raise ValueError(
                    f"The last two dimensions of the shape must be {max_num_nodes}. "
                    f"Got {shape[-2:]}."
                )
            self.shape = torch.Size(shape)

        self.space = AdjacencyMatrixBox(max_num_nodes)

    def is_in(self, val: torch.Tensor) -> bool:
        """Check if a value is a valid adjacency matrix.

        Parameters
        ----------
        val : torch.Tensor
            The value to check.

        Returns
        -------
        is_in : bool
            Whether the value is a valid adjacency matrix.
        """

        # Basic type checks
        if not isinstance(val, torch.Tensor):
            return False
        if val.shape[-2:] != (self.max_num_nodes, self.max_num_nodes):
            return False
        if val.dtype != self.dtype:
            return False

        # Make sure the values are either 0 or 1
        if not torch.all(torch.isin(val, torch.tensor([0, 1], device=self.device))):
            return False

        # Make sure the matrix is symmetric
        if not torch.all(val.transpose(-1, -2) == val):
            return False

        # Make sure the diagonal is all zeros
        if not torch.all(torch.isin(torch.diagonal(val, dim1=-2, dim2=-1), 0)):
            return False

        return True

    def rand(self, shape: Optional[list[int] | torch.Size] = None) -> torch.Tensor:
        """Generate a random 1/2 Erdos-Renyi adjacency matrix.

        Parameters
        ----------
        shape : list[int] | torch.Size, optional
            The batch shape of the adjacency matrix. If None, the batch shape will be
            ().

        Returns
        -------
        adjacency : torch.Tensor
            A random adjacency matrix.
        """

        if shape is None:
            shape = shape = torch.Size([])

        adjacency_values = torch.rand(*shape, *self.shape, device=self.device)
        adjacency = (adjacency_values < 0.5).to(self.dtype)
        adjacency = adjacency.triu(diagonal=1)
        adjacency += adjacency.transpose(1, 2).clone()

        return adjacency

    def _project(self, val: torch.Tensor) -> torch.Tensor:
        """Project a value to the space of valid adjacency matrices.

        Symmetrizes the matrix, sets the diagonal to zero and rounds the values to 0 or
        1.

        Parameters
        ----------
        val : torch.Tensor
            The value to project.

        Returns
        -------
        projected_val : torch.Tensor
            The projected value.
        """

        # Symmetrize the matrix
        val = (val + val.transpose(1, 2)) / 2

        # Make sure the diagonal is all zeros
        val[..., torch.arange(self.max_num_nodes), torch.arange(self.max_num_nodes)] = 0

        # Make sure the values are either 0 or 1
        return torch.clamp(torch.round(val), min=0, max=1).to(self.dtype)

    def to(self, dest: torch.dtype | torch.device | str | int) -> TensorSpec:
        if isinstance(dest, torch.dtype):
            self.dtype = dest
        elif isinstance(dest, (torch.device, str, int)):
            self.device = dest
        else:
            raise ValueError(f"Invalid destination {dest}")
        return self

    def clone(self) -> "AdjacencyMatrixSpec":
        return AdjacencyMatrixSpec(
            self.max_num_nodes,
            self.shape,
            self.device,
            self.dtype,
        )


@register_scenario_class(ScenarioType.GRAPH_ISOMORPHISM, Environment)
class GraphIsomorphismEnvironment(Environment):
    """The graph isomorphism RL environment.

    Agents see the adjacency matrix and the messages sent so far.

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
    def max_num_nodes(self) -> int:
        if self._max_num_nodes is None:
            self._max_num_nodes = self.dataset["x"].shape[-2]
        return self._max_num_nodes

    @property
    def _message_history_shape(self) -> tuple:
        return (
            self.num_envs,
            2,
            self.max_num_nodes,
            self.protocol_handler.max_message_rounds,
        )

    def _get_observation_spec(self) -> CompositeSpec:
        """Get the specification of the agent observations.

        Agents see the adjacency matrix and the messages sent so far. The "message"
        field contains the most recent message.

        Returns
        -------
        observation_spec : CompositeSpec
            The observation specification.
        """
        base_observation_spec = super()._get_observation_spec()
        base_observation_spec["adjacency"] = AdjacencyMatrixSpec(
            self.max_num_nodes,
            shape=(
                self.num_envs,
                2,
                self.max_num_nodes,
                self.max_num_nodes,
            ),
            dtype=self._int_dtype,
            device=self.device,
        )
        base_observation_spec["node_mask"] = BinaryDiscreteTensorSpec(
            self.max_num_nodes,
            shape=(
                self.num_envs,
                2,
                self.max_num_nodes,
            ),
            dtype=torch.bool,
            device=self.device,
        )
        base_observation_spec["message"] = DiscreteTensorSpec(
            self.max_num_nodes,
            shape=(
                self.num_envs,
                2,
                self.max_num_nodes,
            ),
            dtype=torch.float,
            device=self.device,
        )

        # full_observation_spec = CompositeSpec()
        # for c in self.protocol_handler.conversations:
        #     k = "<->".join(c)
        #     full_observation_spec[k] = base_observation_spec

        return base_observation_spec

    def _get_action_spec(self) -> CompositeSpec:
        """Get the specification of the agent actions.

        Each action space has shape (batch_size, num_agents). Each agent chooses both a
        node and a decision: reject, accept or continue (represented as 0, 1 or 2). The
        node is is a number between 0 and 2 * max_num_nodes - 1. If it is less than
        max_num_nodes, it is a node in the first graph, otherwise it is a node in the
        second graph.

        Returns
        -------
        action_spec : CompositeSpec
            The action specification.
        """
        base_action_spec = super()._get_action_spec()
        base_action_spec["agents"]["node_selected"] = DiscreteTensorSpec(
            2 * self.max_num_nodes,
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
        message_history: Float[Tensor, "... graph node message_round"] = env_td[
            "message_history"
        ]
        round: Int[Tensor, "..."] = env_td["round"]
        node_selected: Int[Tensor, "... agent"] = env_td["agents", "node_selected"]

        # Compute index of the agent whose turn it is.
        # (... agent)
        active_agents_mask = self.protocol_handler.get_active_agents_mask(round)

        # Sum up the messages from the agents whose turn it is. If two agents select the
        # same node, the message will be 2.
        # (... 2 max_num_nodes)
        message = F.one_hot(node_selected, 2 * self.max_num_nodes).float()
        message = torch.where(
            active_agents_mask[..., None], message, torch.zeros_like(message)
        )
        message = message.sum(dim=-2)
        message = message.view(*message.shape[:-1], 2, self.max_num_nodes)

        # Insert the message into the message history at the current round
        round_mask = F.one_hot(round, self.protocol_handler.max_message_rounds).bool()
        message_history = message_history.masked_scatter(
            round_mask[..., None, None, :], message
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

        env_td["adjacency"][mask] = data_batch["adjacency"]
        env_td["node_mask"][mask] = data_batch["node_mask"]
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

        return env_td
