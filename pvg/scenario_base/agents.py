"""Base classes for building agents.

An agent is composed of a body and one or more heads. The body computes a representation
of the environment state, and the heads use this representation to compute the agent's
policy, value function, etc.

All modules are TensorDictModules, which means they take and return TensorDicts. Input
and output keys are specified in the module's `input_keys` and `output_keys` attributes.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Iterable, Callable, ClassVar, Literal
from dataclasses import dataclass, fields, InitVar
from functools import partial, cached_property
import re
import itertools

import torch
from torch import Tensor
from torch.nn.parameter import Parameter as TorchParameter

import numpy as np
from numpy.typing import NDArray

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey

from einops import repeat, rearrange

from jaxtyping import Float, Int, Bool

from pvg.parameters import HyperParameters, PureTextAgentParameters
from pvg.experiment_settings import ExperimentSettings
from pvg.protocols import ProtocolHandler
from pvg.scenario_base.environment import PureTextEnvironment
from pvg.utils.types import TorchDevice
from pvg.utils.hyper_params import get_agent_part_flags
from pvg.utils.torch import apply_orthogonal_initialisation
from pvg.utils.nested_array_dict import NestedArrayDict


@dataclass
class AgentHooks:
    """Holder for hooks to run at various points in the agent forward pass."""

    @classmethod
    def create_recorder_hooks(
        cls, storage: dict | TensorDict, per_agent: bool = True
    ) -> "AgentHooks":
        """Create hooks to record the agent's output.

        Parameters
        ----------
        storage : dict | TensorDict
            The dictionary to store the agent's output in.
        per_agent : bool, default=True
            Whether to store the output of each agent separately.

        Returns
        -------
        hooks : AgentHooks
            The hooks to record the agent's output.
        """

        def recorder_hook(
            hook_name: str,
            storage: dict | TensorDict,
            output: Tensor,
            *,
            agent_name: Optional[str] = None,
        ):
            if agent_name is not None and per_agent:
                if agent_name not in storage:
                    storage[agent_name] = {}
                storage[agent_name][hook_name] = output.clone()
            else:
                storage[hook_name] = output.clone()

        cls_args = {
            field.name: partial(recorder_hook, field.name, storage)
            for field in fields(cls)
        }

        return cls(**cls_args)


@dataclass
class AgentState(ABC):
    """Base class for storing all the data needed to restore an agent."""


class AgentPart(ABC):
    """Base class for all agent parts: bodies and heads.

    The in and out keys are split into agent-level and environment-level keys.
    Agent-level keys are nested under "agents" in the environment's state dict, while
    environment-level keys are at the top level.

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

    Class attributes
    ----------------
    agent_level_in_keys : Iterable[NestedKey]
        The keys required by the agent part whose values are per-agent (so in the
        environment's state dict will be nested under "agents").
    env_level_in_keys : Iterable[NestedKey]
        The keys required by the agent part whose values are per-environment (so in the
        environment's state dict will be at the top level).
    agent_level_out_keys : Iterable[NestedKey]
        The keys produced by the agent part whose values are per-agent (so in the
        environment's state dict will be nested under "agents").
    env_level_out_keys : Iterable[NestedKey]
        The keys produced by the agent part whose values are per-environment (so in the
        environment's state dict will be at the top level).
    """

    agent_level_in_keys: Iterable[NestedKey] = []
    env_level_in_keys: Iterable[NestedKey] = []
    agent_level_out_keys: Iterable[NestedKey] = []
    env_level_out_keys: Iterable[NestedKey] = []

    @property
    def in_keys(self) -> set[NestedKey]:
        """The keys required by the module.

        Computed by taking the union of `agent_level_in_keys` and `env_level_in_keys`.

        Returns
        -------
        in_keys : set[str]
            The keys required by the module.
        """

        in_keys = set()
        in_keys.update(self.agent_level_in_keys)
        in_keys.update(self.env_level_in_keys)
        return in_keys

    @property
    def out_keys(self) -> set[NestedKey]:
        """The keys produced by the module.

        Computed by taking the union of `agent_level_out_keys` and `env_level_out_keys`.

        Returns
        -------
        out_keys : set[str]
            The keys produced by the module.
        """

        out_keys = set()
        out_keys.update(self.agent_level_out_keys)
        out_keys.update(self.env_level_out_keys)
        return out_keys

    @property
    def is_prover(self) -> bool:
        """Whether the agent is a prover."""
        return self.agent_name in self.protocol_handler.prover_names

    @property
    def is_verifier(self) -> bool:
        """Whether the agent is a verifier."""
        return self.agent_name in self.protocol_handler.verifier_names

    @property
    def max_message_rounds(self) -> int:
        return self.protocol_handler.max_message_rounds

    @cached_property
    def visible_message_channel_names(self) -> list[str]:
        """The names of the message channels visible to the agent."""

        return self.protocol_handler.get_agent_visible_channels(self.agent_name)

    @cached_property
    def visible_message_channel_indices(self) -> list[int]:
        """The indices of the message channels visible to the agent."""

        visible_channels = self.visible_message_channel_names
        all_channels = self.protocol_handler.message_channel_names
        return [all_channels.index(channel) for channel in visible_channels]

    _visible_message_channel_mask: Optional[Bool[Tensor, "channel"]] = None

    @property
    def visible_message_channel_mask(self) -> Bool[Tensor, "channel"]:
        """The mask for the message channels visible to the agent."""

        if self._visible_message_channel_mask is None:
            num_message_channels = len(self.protocol_handler.message_channel_names)
            self._visible_message_channel_mask = torch.zeros(
                num_message_channels, dtype=torch.bool, device=self.device
            )
            self._visible_message_channel_mask[self.visible_message_channel_indices] = (
                True
            )

        return self._visible_message_channel_mask.to(self.device)

    @cached_property
    def num_visible_message_channels(self) -> int:
        """The number of message channels visible to the agent."""
        return len(self.visible_message_channel_names)

    @property
    def required_pretrained_models(self) -> Iterable[str]:
        """The pretrained models used by the agent.

        The embeddings of these models will be added to the dataset.
        """
        return []

    def set_state(self, checkpoint: AgentState):
        """Set the state of the agent from a checkpoint.

        This method should be overridden by subclasses to restore the state of the agent
        from a checkpoint.

        Parameters
        ----------
        checkpoint : AgentCheckpoint
            The checkpoint to restore the state from.
        """

    def get_state_dict(self) -> dict:
        """Get the state of the agent part as a dict.

        This method should be implemented by subclasses capable of saving their state.

        Returns
        -------
        state_dict : dict
            The state of the agent part.
        """
        raise NotImplementedError(
            f"Getting the agent state is not implemented for "
            f"{self.__class__.__name__}"
        )

    def __init__(
        self,
        hyper_params: HyperParameters,
        settings: ExperimentSettings,
        agent_name: str,
        protocol_handler: ProtocolHandler,
    ):
        super().__init__()
        self.hyper_params = hyper_params
        self.settings = settings
        self.agent_name = agent_name
        self.protocol_handler = protocol_handler

        self.agent_params = hyper_params.agents[agent_name]
        self.agent_index = self.protocol_handler.agent_names.index(agent_name)

    @abstractmethod
    def forward(self, data: Any) -> Any:
        """Forward pass through the agent part.

        Parameters
        ----------
        data : Any
            The input to the agent part.

        Returns
        -------
        output : Any
            The output of the forward pass on the input.
        """


class TensorDictAgentPartMixin(AgentPart, TensorDictModuleBase, ABC):
    """Mixin for agent parts which use TensorDicts as input and output."""

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
        self.device = settings.device

    def _init_weights(self):
        """Initialise the module weights

        Should be called at the end of `__init__`
        """
        if self.agent_params.use_orthogonal_initialisation:
            apply_orthogonal_initialisation(
                self, self.agent_params.orthogonal_initialisation_gain
            )

    def _run_recorder_hook(
        self,
        hooks: Optional[AgentHooks],
        hook_name: str,
        output: Optional[Tensor],
    ):
        if hooks is not None and output is not None:
            hooks.__getattribute__(hook_name)(output, agent_name=self.agent_name)

    @abstractmethod
    def to(self, device: TorchDevice):
        """Move the agent to the given device."""
        pass


class TensorDictDummyAgentPartMixin(TensorDictAgentPartMixin, ABC):
    """A tensordict mixin for agent parts which are dummy (e.g. random or constant).

    Adds a dummy parameter to the agent part, so that PyTorch can calculate gradients
    and so that tensordict can determine the device.
    """

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
        self.dummy_parameter = TorchParameter(torch.tensor(0.0, device=self.device))

    def to(self, device: TorchDevice):
        """Move the agent to the given device."""
        self.device = device
        self.dummy_parameter = self.dummy_parameter.to(device)


class WholeAgent(AgentPart, ABC):
    """Base class for agents which are not split into parts."""


class PureTextWholeAgent(WholeAgent, ABC):
    """Base class for whole agents which process text input and call APIs."""

    shared_model_group: Optional["PureTextSharedModelGroup"] = None
    agent_params: PureTextAgentParameters
    _visible_message_channel_mask: Optional[Bool[np.ndarray, "channel"]] = None

    @cached_property
    def visible_message_channel_mask(self) -> Bool[np.ndarray, "channel"]:
        """The mask for the message channels visible to the agent."""
        return super().visible_message_channel_mask.cpu().detach().numpy()

    @abstractmethod
    def forward(
        self, data: NestedArrayDict, environment: PureTextEnvironment
    ) -> NestedArrayDict:
        """Forward pass through the agent

        Parameters
        ----------
        data : NestedArrayDict
            The input to the agent.
        environment : PureTextEnvironment
            The environment the agent is interacting with.

        Returns
        -------
        output : NestedArrayDict
            The output of the forward pass on the input.
        """

    @abstractmethod
    def build_fine_tune_dataset(self, rollouts: NestedArrayDict) -> list:
        """Build a dataset for fine-tuning the agent from sampled rollouts.

        This method generates a dataset of examples ready to pass to the fine-tune API.

        Parameters
        ----------
        rollouts : NestedArrayDict
            The sampled rollouts.

        Returns
        -------
        fine_tune_dataset : list
            The dataset for fine-tuning the agent.
        """

    def __call__(
        self, data: NestedArrayDict, environment: PureTextEnvironment
    ) -> NestedArrayDict:
        return self.forward(data, environment)


@dataclass
class PureTextSharedModelGroupState(ABC):
    """Base class for storing all the data needed to restore a shared model group."""


class PureTextSharedModelGroup(ABC):
    """A class representing a group of pure text agents which share the same model

    The shared model is fine-tuned on the data from all agents in the group.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The settings of the experiment.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    agent_wholes : Iterable[PureTextWholeAgent]
        The agents in the shared model group.
    group_name : str
        The name of the shared model group.
    """

    state_class: ClassVar[type[PureTextSharedModelGroupState]] = (
        PureTextSharedModelGroupState
    )

    @dataclass
    class SharedAgentParams:
        """The parameters shared by all agents in the group."""

        model_name: str
        freeze_agent: bool
        use_dummy_api: bool
        fine_tune_from_scratch: bool

    @property
    def model_name(self) -> str:
        """The current model name, which may be the base model or a fine-tuned model."""
        if self.fine_tuned_model_name is not None:
            return self.fine_tuned_model_name
        else:
            return self.base_model_name

    def __init__(
        self,
        hyper_params: HyperParameters,
        settings: ExperimentSettings,
        protocol_handler: ProtocolHandler,
        agent_wholes: Iterable[PureTextWholeAgent],
        group_name: str,
    ):
        self.hyper_params = hyper_params
        self.settings = settings
        self.protocol_handler = protocol_handler
        self.group_name = group_name

        self.agent_wholes = {agent.agent_name: agent for agent in agent_wholes}
        self.agent_names = list(self.agent_wholes.keys())

        shared_agent_params = {}
        for agent in agent_wholes:

            # Check that the agent's shared_model_group attribute is set correctly
            if (
                agent.agent_params.shared_model_group is not None
                and agent.agent_params.shared_model_group != self.group_name
            ):
                raise ValueError(
                    f"Tried to create a shared model group named {group_name!r} "
                    f"containing agent {agent.agent_name!r}, but its "
                    f"`shared_model_group` attribute is set to "
                    f"{agent.agent_params.shared_model_group!r}."
                )
            elif (
                agent.agent_params.shared_model_group is None
                and group_name != agent.agent_name
            ):
                raise ValueError(
                    f"Tried to create a shared model group named {group_name!r} "
                    f"containing agent {agent.agent_name!r}, but its "
                    f"`shared_model_group` attribute is not set."
                )

            # Get the value of the shared agent parameters and check that they are the
            # same for all agents in the group
            for shared_agent_param_field in fields(self.SharedAgentParams):
                param_name = shared_agent_param_field.name
                if param_name not in shared_agent_params:
                    shared_agent_params[param_name] = getattr(
                        agent.agent_params, param_name
                    )
                elif (
                    getattr(agent.agent_params, param_name)
                    != shared_agent_params[param_name]
                ):
                    raise ValueError(
                        f"All agents in a shared model group must have the same "
                        f"{param_name!r} parameter, but got "
                        f"{shared_agent_params[param_name]!r} for agent "
                        f"{self.agent_names[0]!r} and "
                        f"{getattr(agent.agent_params, param_name)!r} for "
                        f"agent {agent.agent_name!r}."
                    )

            # Set the agent's shared_model_group attribute
            if agent.shared_model_group is not None:
                raise ValueError(
                    f"Agent {agent.agent_name} is already in a shared model group."
                )
            agent.shared_model_group = self

        self.shared_agent_params = self.SharedAgentParams(**shared_agent_params)
        self.base_model_name = self.shared_agent_params.model_name

        self.fine_tune_job_id: Optional[str] = None
        self.fine_tuned_model_name: Optional[str] = None

    @abstractmethod
    def create_fine_tune_job(self, data_per_agent: dict[str, NestedArrayDict]):
        """Create a fine-tune job for the agent group given sampled rollouts

        Parameters
        ----------
        data_per_agent : dict[str, NestedArrayDict]
            The data for each agent in the group, sampled from the environment.
        """

    @abstractmethod
    def get_fine_tune_job_status(
        self,
    ) -> Literal["pending", "running", "succeeded", "failed", "cancelled"]:
        """Get the status of the fine-tune job"""

    @abstractmethod
    def get_fine_tune_job_error_repr(self) -> str:
        """Get a string representation of the error for the fine-tune job"""

    @abstractmethod
    def switch_to_next_model(self):
        """Switch to the next model after fine-tuning"""

    def set_state(self, checkpoint: PureTextSharedModelGroupState):
        """Set the state of the shared model group from a checkpoint.

        This method should be overridden by subclasses to restore the state of the
        shared model group from a checkpoint.

        Parameters
        ----------
        checkpoint : AgentCheckpoint
            The checkpoint to restore the state from.
        """

    def get_state_dict(self) -> dict:
        """Get the state of the shared model group as a dict.

        This method should be implemented by subclasses capable of saving their state.

        Returns
        -------
        state_dict : dict
            The state of the shared model group.
        """
        raise NotImplementedError(
            f"Getting the agent state is not implemented for "
            f"{self.__class__.__name__}"
        )

    def get_state(self) -> PureTextSharedModelGroupState:
        """Get the state of the shared model group."""
        return self.state_class(**self.get_state_dict())


class RandomWholeAgent(WholeAgent, ABC):
    """Base class for whole random agents."""


class AgentBody(AgentPart, ABC):
    """Base class for all agent bodies, which compute representations for heads.

    Representations should have dimension `hyper_params.d_representation`.
    """


class DummyAgentBody(AgentBody, ABC):
    """A dummy agent body which does nothing."""


class AgentHead(AgentPart, ABC):
    """Base class for all agent heads."""


class AgentPolicyHead(AgentHead, ABC):
    """Base class for all agent policy heads."""

    @property
    def has_decider(self) -> bool:
        """Whether the policy head has an output yielding a decision.

        By default a decider is used to decide whether to continue exchanging messages.
        In this case it outputs a single triple of logits for the three options: guess
        that the graphs are not isomorphic, guess that the graphs are isomorphic, or
        continue exchanging messages.
        """
        return self.is_verifier


class RandomAgentPolicyHead(AgentPolicyHead, ABC):
    """A policy head which samples actions randomly."""


class AgentValueHead(AgentHead, ABC):
    """Base class for all agent value heads, to the value of a state."""


class ConstantAgentValueHead(AgentValueHead, ABC):
    """A value head which returns a constant value."""


class SoloAgentHead(AgentHead, ABC):
    """Base class for all solo agent heads, which attempt the task on their own."""


class CombinedAgentPart(ABC):
    """Base class for modules which combine agent parts together.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The settings of the experiment.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    parts : dict[str, AgentPart]
        The agent parts to combine.

    Class attributes
    ----------------
    additional_in_keys : list[NestedKey]
        Input keys required by the module, in addition to the keys required by the agent
        parts.
    excluded_in_keys : list[NestedKey]
        Input keys required by the agent parts, which are not required as inputs to this
        module (i.e. these keys are populated by this module when called). Agent-level
        keys should be specified as nested keys, with the first element being "agents".
    additional_out_keys : list[NestedKey]
        Output keys produced by the module, in addition to the keys produced by the
        agent parts.
    excluded_out_keys : list[NestedKey]
        Output keys produced by the agent parts, which are not output by this module.
        Agent-level keys should be specified as nested keys, with the first element
        being "agents".
    """

    additional_in_keys: list[NestedKey] = []
    excluded_in_keys: list[NestedKey] = []
    additional_out_keys: list[NestedKey] = []
    excluded_out_keys: list[NestedKey] = []

    @property
    def in_keys(self) -> set[NestedKey]:
        """The keys required by the module.

        Computed by taking the union of the `agent_level_in_keys` and
        `env_level_in_keys` of all the parts, and then removing the keys in
        `excluded_in_keys` and adding the keys in `additional_in_keys`.

        Returns
        -------
        in_keys : set[str]
            The keys required by the module.
        """

        in_keys = set()
        for part in self.parts.values():
            for in_key in part.agent_level_in_keys:
                if ("agents", in_key) in self.excluded_in_keys:
                    continue
                in_keys.add(("agents", in_key))
            for in_key in part.env_level_in_keys:
                if in_key in self.excluded_in_keys:
                    continue
                in_keys.add(in_key)

        in_keys.update(self.additional_in_keys)

        return in_keys

    @property
    def out_keys(self) -> set[NestedKey]:
        """The keys produced by the module.

        Computed by taking the union of the `agent_level_out_keys` and
        `env_level_out_keys` of all the parts, and then removing the keys in
        `excluded_out_keys` and adding the keys in `additional_out_keys`.

        Returns
        -------
        out_keys : set[str]
            The keys produced by the module.
        """

        out_keys = set()
        for part in self.parts.values():
            for out_key in part.agent_level_out_keys:
                if ("agents", out_key) in self.excluded_out_keys:
                    continue
                out_keys.add(("agents", out_key))
            for out_key in part.env_level_out_keys:
                if out_key in self.excluded_out_keys:
                    continue
                out_keys.add(out_key)

        out_keys.update(self.additional_out_keys)

        return out_keys

    def __init__(
        self,
        hyper_params: HyperParameters,
        settings: ExperimentSettings,
        protocol_handler: ProtocolHandler,
        parts: dict[str, AgentPart],
    ):
        super().__init__()
        self.hyper_params = hyper_params
        self.settings = settings
        self.protocol_handler = protocol_handler
        self.parts = parts

        self.agent_names = protocol_handler.agent_names

        if set(parts.keys()) != set(self.agent_names):
            raise ValueError(
                f"The agent names in {type(self).__name__} must match the agent names "
                f"in the protocol handler. Expected {self.agent_names}, got "
                f"{parts.keys()}."
            )

    def _restrict_input_to_visible_channels(
        self, agent_name: str, input_array: Tensor | NDArray, shape_spec: str
    ) -> Tensor:
        """Restrict an agent's input to its visible message channels.

        Agents only receive messages from the channels they can see. This function
        restricts the input to the agent to only the visible message channels.

        Parameters
        ----------
        agent_name : str
            The name of the agent.
        input_array : Tensor | NDArray
            The input array to the agent.
        shape_spec : str
            The shape of the input. This is a space-separated string of the dimensions
            of the input. One of these must be "channel".

        Returns
        -------
        restricted_input : Tensor | NDArray
            The input restricted to the visible message channels.
        """

        agent_index = self.agent_names.index(agent_name)

        dim_names = shape_spec.split(" ")

        if dim_names.count("channel") != 1:
            raise ValueError(
                f"The input shape must contain exactly one 'channel' dimension. Got "
                f"{shape_spec!r}."
            )

        channel_dim = dim_names.index("channel")

        if "..." in dim_names[channel_dim + 1 :]:
            raise ValueError(
                f"An ellipsis (...) is not allowed after the 'channel' dimension. Got "
                f"{shape_spec!r}."
            )

        channel_dim = channel_dim - len(dim_names)

        # If the input already has the correct number of channels, return it
        if input_array.shape[channel_dim] == len(
            self.protocol_handler.get_agent_visible_channels(agent_name)
        ):
            return input_array

        # Create an index for the tensor, which selects the visible channels using a
        # mask along the channel dimension
        visible_mask = self.protocol_handler.agent_channel_visibility_mask[agent_index]
        if isinstance(input_array, np.ndarray):
            visible_mask = visible_mask.cpu().numpy()
        index = (Ellipsis, visible_mask) + (slice(None),) * (-1 - channel_dim)

        # Apply the mask to the input
        return input_array[index]


class CombinedWhole(CombinedAgentPart, ABC):
    """Base class for modules which combine whole agents together."""

    @abstractmethod
    def forward(self, data: Any) -> Any:
        """Run a forward pass through all the agents and combine the output."""

    def __init__(
        self,
        hyper_params: HyperParameters,
        settings: ExperimentSettings,
        protocol_handler: ProtocolHandler,
        wholes: dict[str, WholeAgent],
    ):
        super().__init__(
            hyper_params=hyper_params,
            settings=settings,
            protocol_handler=protocol_handler,
            parts=wholes,
        )
        self.wholes = wholes


class PureTextCombinedWhole(CombinedWhole, ABC):
    """Base class for modules which combine whole pure-text agents together."""

    @abstractmethod
    def forward(
        self, data: NestedArrayDict, environment: PureTextEnvironment
    ) -> NestedArrayDict:
        """Run a forward pass through all the agents and combine the output."""


class CombinedTensorDictAgentPart(CombinedAgentPart, TensorDictModuleBase, ABC):
    """Base class for modules which combine agent parts together and use TensorDicts."""

    @property
    def device(self) -> TorchDevice:
        device = None
        for part in self.parts.values():
            if device is None:
                device = part.device
            elif device != part.device:
                raise RuntimeError(
                    f"The device of all {type(self).__name__} parts must be the same,"
                    f" but got {device} and {part.device}."
                )
        return device

    def __init__(
        self,
        hyper_params: HyperParameters,
        settings: ExperimentSettings,
        protocol_handler: ProtocolHandler,
        parts: dict[str, AgentPart],
    ):
        super().__init__(
            hyper_params=hyper_params,
            settings=settings,
            protocol_handler=protocol_handler,
            parts=parts,
        )

        # Add the parts as submodules, so that PyTorch knows about them
        for agent_name in self.agent_names:
            self.add_module(agent_name, parts[agent_name])


class CombinedBody(CombinedTensorDictAgentPart, ABC):
    """A module which combines all the agent bodies together.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The settings of the experiment.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    bodies : dict[str, AgentBody]
        The agent bodies to combine.
    """

    def __init__(
        self,
        hyper_params: HyperParameters,
        settings: ExperimentSettings,
        protocol_handler: ProtocolHandler,
        bodies: dict[str, AgentBody],
    ):
        super().__init__(
            hyper_params=hyper_params,
            settings=settings,
            protocol_handler=protocol_handler,
            parts=bodies,
        )
        self.bodies = bodies

    @abstractmethod
    def forward(self, data: TensorDictBase) -> TensorDict:
        """Forward pass through the combined body.

        Parameters
        ----------
        data : TensorDict
            The input to the combined body.

        Returns
        -------
        body_output : TensorDict
            The output of the combined body.
        """
        pass


class CombinedPolicyHead(CombinedTensorDictAgentPart, ABC):
    """A module which combines all the agent policy heads together.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The settings of the experiment.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    policy_heads : dict[str, AgentPolicyHead]
        The agent policy heads to combine.
    """

    def __init__(
        self,
        hyper_params: HyperParameters,
        settings: ExperimentSettings,
        protocol_handler: ProtocolHandler,
        policy_heads: dict[str, AgentPolicyHead],
    ):
        super().__init__(
            hyper_params=hyper_params,
            settings=settings,
            protocol_handler=protocol_handler,
            parts=policy_heads,
        )
        self.policy_heads = policy_heads

    @abstractmethod
    def forward(self, data: TensorDictBase) -> TensorDict:
        """Forward pass through the combined policy head.

        Parameters
        ----------
        data : TensorDict
            The input to the combined policy head.

        Returns
        -------
        policy_output : TensorDict
            The output of the combined policy head. This must contain the key ("agents",
            "main_message_logits"), which has shape "... agents channel position logit"
            and contains the logits for the agents' messages in the main message space.
        """
        pass

    def _expand_logits_to_all_channels(
        self, agent_name: str, logits: Tensor, shape_spec: str, fill_value: float = -1e9
    ) -> Tensor:
        """Expand an agent's logits from its visible message channels to all.

        Agents only output messages for the channels they can see. This function expands
        the output to all channels, by filling in `fill_value` for the logits in the
        channels the agent cannot see.

        Parameters
        ----------
        agent_name : str
            The name of the agent.
        logits : Tensor
            A tensor of output logits. This is a single key in the output of the agent's
            forward pass.
        shape_spec : str
            The shape of the output. This is a space-separated string of the dimensions
            of the output. One of these must be "channel".
        fill_value : float, default=-1e9
            The value to fill in for the channels the agent cannot see.

        Returns
        -------
        expanded_logits : Tensor
            The output expanded to all channels. This has the same shape as `logits`,
            except that the channel dimension is the full set of message channels.
        """

        agent_index = self.agent_names.index(agent_name)

        dim_names = shape_spec.split(" ")

        if dim_names.count("channel") != 1:
            raise ValueError(
                f"The output shape must contain exactly one 'channel' dimension. Got "
                f"{shape_spec!r}."
            )

        channel_dim = dim_names.index("channel")

        if "..." in dim_names[channel_dim + 1 :]:
            raise ValueError(
                f"An ellipsis (...) is not allowed after the 'channel' dimension. Got "
                f"{shape_spec!r}."
            )

        channel_dim = channel_dim - len(dim_names)

        # If the output is already expanded, return it
        if logits.shape[channel_dim] == self.protocol_handler.num_message_channels:
            return logits

        # Create a tensor filled with `fill_value` of the correct shape
        full_shape = list(logits.shape)
        full_shape[channel_dim] = self.protocol_handler.num_message_channels
        expanded_logits = torch.full(
            full_shape, fill_value, device=self.device, dtype=logits.dtype
        )

        # Create an index for the tensor, which selects the visible channels using a
        # mask along the channel dimension
        visible_mask = self.protocol_handler.agent_channel_visibility_mask[agent_index]
        index = (Ellipsis, visible_mask) + (slice(None),) * (-1 - channel_dim)

        # Fill in the visible channels
        expanded_logits[index] = logits

        return expanded_logits

    def _restrict_decisions(
        self,
        decision_restriction: Int[Tensor, "..."],
        decision_logits: Float[Tensor, "... agents 3"],
    ) -> TensorDictBase:
        """Make sure the agent's decisions comply with the restrictions

        Parameters
        ----------
        decision_restriction : Int[Tensor, "..."]
            The restrictions on the agents' decisions. The possible values are:#

                - 0: The verifier can decide anything.
                - 1: The verifier can only decide to continue interacting.
                - 2: The verifier can only make a guess.

        decision_logits : Float[Tensor, "... agents 3"]
            The logits for the agents' decisions.

        Returns
        -------
        decision_logits : Float[Tensor, "... agents 3"]
            The logits for the agents' decisions, with the restricted decisions set to
            -1e9.
        """

        num_agents = len(self.agent_names)

        no_guess_mask = decision_restriction == 1
        no_guess_mask = repeat(no_guess_mask, f"... -> ... {num_agents} 3").clone()
        no_guess_mask[..., :, 2] = False
        decision_logits[no_guess_mask] = -1e9

        no_continue_mask = decision_restriction == 2
        no_continue_mask = repeat(
            no_continue_mask, f"... -> ... {num_agents} 3"
        ).clone()
        no_continue_mask[..., :, :2] = False
        decision_logits[no_continue_mask] = -1e9

        return decision_logits


class CombinedValueHead(CombinedTensorDictAgentPart, ABC):
    """A module which combines all the agent value heads together.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The settings of the experiment.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    value_heads : dict[str, AgentValueHead]
        The agent value heads to combine.
    """

    def __init__(
        self,
        hyper_params: HyperParameters,
        settings: ExperimentSettings,
        protocol_handler: ProtocolHandler,
        value_heads: dict[str, AgentValueHead],
    ):
        super().__init__(
            hyper_params=hyper_params,
            settings=settings,
            protocol_handler=protocol_handler,
            parts=value_heads,
        )
        self.value_heads = value_heads

    @abstractmethod
    def forward(self, data: TensorDictBase) -> TensorDict:
        """Forward pass through the combined value head.

        Parameters
        ----------
        data : TensorDict
            The input to the combined value head.

        Returns
        -------
        value_output : TensorDict
            The output of the combined value head.
        """


@dataclass
class Agent(ABC):
    """A base class for holding all the parts of an agent for an experiment.

    Subclasses should define the `message_logits_key` class variable, which is the key
    in the output of the policy head which contains the logits for the message.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    agent_name : str
        The name of the agent.
    whole : WholeAgent, optional
        The whole agent, if the agent is not split into parts.
    body : AgentBody, optional
        The (shared) body of the agent.
    policy_body : AgentBody, optional
        The body of the agent's policy head, if not using a shared body.
    value_body : AgentBody, optional
        The body of the agent's value head, if not using a shared body.
    policy_head : AgentPolicyHead, optional
        The policy head of the agent.
    value_head : AgentValueHead, optional
        The value head of the agent.
    solo_head : SoloAgentHead, optional
        The solo head of the agent.
    """

    hyper_params: InitVar[HyperParameters]
    agent_name: InitVar[str]
    whole: Optional[WholeAgent] = None
    body: Optional[AgentBody] = None
    policy_body: Optional[AgentBody] = None
    value_body: Optional[AgentBody] = None
    policy_head: Optional[AgentPolicyHead] = None
    value_head: Optional[AgentValueHead] = None
    solo_head: Optional[SoloAgentHead] = None

    message_logits_key: ClassVar[str]

    agent_state_class: ClassVar[type[AgentState]] = AgentState

    def __post_init__(
        self,
        hyper_params: HyperParameters,
        agent_name: str,
    ):
        if self.body is None and self.policy_body is None and self.whole is None:
            raise ValueError(
                "An agent must have either a body or a policy body, or be a whole agent"
            )

        if self.body is not None and self.policy_body is not None:
            raise ValueError("An agent cannot have both a body and a policy body")

        if self.value_body is not None and self.policy_body is None:
            raise ValueError("An agent with a value body must have a policy body")

        if self.policy_head is None and self.solo_head is None and self.whole is None:
            raise ValueError(
                "An agent which is not whole must have either a policy head or a solo "
                "head, or both."
            )

        if self.value_head is not None and self.policy_head is None:
            raise ValueError("An agent with a value head must have a policy head")

        if (
            self.policy_head is not None
            and self.body is None
            and self.policy_body is None
        ):
            raise ValueError(
                "An agent with a policy head must have a body or a policy body"
            )

        if (
            self.value_head is not None
            and self.body is None
            and self.value_body is None
        ):
            raise ValueError(
                "An agent with a value head must have a body or a value body"
            )

        if (
            self.solo_head is not None
            and self.body is None
            and self.policy_body is None
        ):
            raise ValueError(
                "An agent with a solo head must have a body or a policy body"
            )

        if (
            self.policy_head is not None
            and self.value_head is None
            and self.body is None
        ):
            raise ValueError(
                "An agent with a policy head but no value head must have a 'body', and"
                " not a 'policy_body'"
            )

        self.hyper_params = hyper_params
        self.agent_name = agent_name

        self.agent_params = hyper_params.agents[agent_name]

    def set_state(self, checkpoint: AgentState):
        """Set the state of the agent from a checkpoint.

        This method restores the state of all the agent parts from the checkpoint.

        Parameters
        ----------
        checkpoint : AgentCheckpoint
            The checkpoint to restore the state from.
        """

        for part_field in fields(self):
            part: AgentPart = getattr(self, part_field.name)
            if part is not None:
                part.set_state(checkpoint)

    def get_state(self) -> AgentState:
        """Get a checkpoint of the agent's state.

        This method gets a checkpoint of the state of all the agent parts.

        Returns
        -------
        checkpoint : AgentCheckpoint
            The checkpoint of the agent's state.
        """

        state_dict = {}
        for part_field in fields(self):
            part: AgentPart = getattr(self, part_field.name)
            if part is not None:
                for key, value in part.get_state_dict().items():
                    if key in state_dict:
                        raise ValueError(
                            f"Duplicate key {key!r} in agent state checkpoint."
                        )
                    state_dict[key] = value

        return self.agent_state_class(**state_dict)

    @staticmethod
    def _append_filtered_params(
        model_param_dict: list[dict[str, Any]],
        named_parameters: list[tuple[str, TorchParameter]],
        filter: Callable[[str], bool],
        lr: float,
    ):
        """Filter the parameters and set their learning rate, and append them to a list.

        Normally appends a dictionary with the keys `hyper_params` and `lr`, consisting of the
        filtered parameters and their learning rate. If the learning rate is 0, the
        parameters are frozen instead.

        Parameters
        ----------
        model_param_dict : list[dict[str, Any]]
            The list of parameter dictionaries to append to.
        named_parameters : list[tuple[str, TorchParameter]]
            A list of the named parameters.
        filter : Callable[[str], bool]
            A function which returns True for the parameters to include.
        lr : float
            The learning rate for the parameters.
        """

        filtered_params = [
            param for param_name, param in named_parameters if filter(param_name)
        ]

        if lr == 0:
            for param in filtered_params:
                param.requires_grad = False
        else:
            model_param_dict.append(dict(params=filtered_params, lr=lr))

    def _body_param_regex(self, part: str) -> str:
        use_critic, use_single_body, _ = get_agent_part_flags(self.hyper_params)
        network_suffix = "network"
        if self.hyper_params.functionalize_modules:
            network_suffix += "_params"
        if use_single_body and use_critic and part == "actor":
            return f"actor_{network_suffix}.module.0.{self.agent_name}"
        else:
            if part == "actor":
                return f"actor_{network_suffix}.module.0.module.0.{self.agent_name}"
            elif part == "critic":
                return f"critic_{network_suffix}.module.0.{self.agent_name}"
            else:
                raise ValueError(f"Unknown part: {part}")

    def _non_body_param_regex(self, part: str) -> str:
        use_critic, use_single_body, _ = get_agent_part_flags(self.hyper_params)
        nums = {"actor": "1-9", "critic": "0-9"}
        network_suffix = "network"
        if self.hyper_params.functionalize_modules:
            network_suffix += "_params"
        if use_single_body and use_critic:
            return f"{part}_{network_suffix}.module.[{nums[part]}].{self.agent_name}"
        else:
            if part == "actor":
                return f"actor_{network_suffix}.module.0.module.[1-9].{self.agent_name}"
            elif part == "critic":
                return f"critic_{network_suffix}.module.[1-9].{self.agent_name}"
            else:
                raise ValueError(f"Unknown part: {part}")

    @property
    def _body_named_parameters(self) -> Iterable[tuple[str, TorchParameter]]:
        use_critic, use_single_body, _ = get_agent_part_flags(self.hyper_params)
        if use_critic and not use_single_body:
            return itertools.chain(
                self.policy_body.named_parameters(), self.value_body.named_parameters()
            )
        return self.body.named_parameters()

    @property
    def _body_parameters(self) -> Iterable[TorchParameter]:
        use_critic, use_single_body, _ = get_agent_part_flags(self.hyper_params)
        if use_critic and not use_single_body:
            return itertools.chain(
                self.policy_body.parameters(), self.value_body.parameters()
            )
        return self.body.parameters()

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
            `loss_module.named_parameters()`. If not given, the parameters of all the
            agent parts are used.
        body_lr_factor_override : bool
            If true, this overrides the learning rate factor for the body (for both the actor and critic), effectively setting it to 1.

        Returns
        -------
        param_dict : Iterable[dict[str, Any]]
            The Torch parameters of the agent, and their learning rates. This is an
            iterable of dictionaries with the keys `hyper_params` and `lr`.
        """

        # Check for mistakes
        if (
            self.hyper_params.rl.use_shared_body
            and self.agent_params.agent_lr_factor.actor
            != self.agent_params.agent_lr_factor.critic
        ):
            raise ValueError(
                "The agent learning rate factor for the actor and critic must be the same if the body is shared."
            )
        if (
            self.hyper_params.rl.use_shared_body
            and self.agent_params.body_lr_factor.actor
            != self.agent_params.body_lr_factor.critic
        ):
            raise ValueError(
                "The body learning rate factor for the actor and critic must be the same if the body is shared."
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

        model_param_dict = []

        # If named_parameters is not given, use the parameters of all the agent parts.
        if named_parameters is None:
            for part in ["actor", "critic"]:
                self._append_filtered_params(
                    model_param_dict,
                    self._body_named_parameters,
                    lambda name: re.match(self._body_param_regex(part), name),
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

        # Set the learning rate for the body parameters
        for part in ["actor", "critic"]:
            self._append_filtered_params(
                model_param_dict,
                named_parameters,
                lambda name: re.match(self._body_param_regex(part), name),
                body_lr[part],
            )

        # Set the learning rate for the non-body parameters
        for part in ["actor", "critic"]:
            self._append_filtered_params(
                model_param_dict,
                named_parameters,
                lambda name: re.match(self._non_body_param_regex(part), name),
                agent_lr[part],
            )

        return model_param_dict

    def train(self):
        """Set the agent to training mode."""
        if self.body is not None:
            self.body.train()
        if self.policy_body is not None:
            self.policy_body.train()
        if self.value_body is not None:
            self.value_body.train()
        if self.policy_head is not None:
            self.policy_head.train()
        if self.value_head is not None:
            self.value_head.train()
        if self.solo_head is not None:
            self.solo_head.train()

    def eval(self):
        """Set the agent to evaluation mode."""
        if self.body is not None:
            self.body.eval()
        if self.policy_body is not None:
            self.policy_body.eval()
        if self.value_body is not None:
            self.value_body.eval()
        if self.policy_head is not None:
            self.policy_head.eval()
        if self.value_head is not None:
            self.value_head.eval()
        if self.solo_head is not None:
            self.solo_head.eval()
