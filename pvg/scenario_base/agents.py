"""Base classes for building agents.

An agent is composed of a body and one or more heads. The body computes a representation
of the environment state, and the heads use this representation to compute the agent's
policy, value function, etc.

All modules are TensorDictModules, which means they take and return TensorDicts. Input
and output keys are specified in the module's `input_keys` and `output_keys` attributes.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Iterable, Callable
from dataclasses import dataclass, fields
from functools import partial
import re

import torch
from torch import Tensor
from torch.nn.parameter import Parameter as TorchParameter

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase

from einops import repeat

from jaxtyping import Float, Int

from pvg.parameters import Parameters, TrainerType, ScenarioType
from pvg.protocols import ProtocolHandler
from pvg.utils.types import TorchDevice


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


class AgentPart(TensorDictModuleBase, ABC):
    """Base class for all agent parts: bodies and heads.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    device : TorchDevice, optional
        The device to use for this agent part. If not given, the CPU is used.
    """

    def __init__(
        self,
        params: Parameters,
        agent_name: str,
        protocol_handler: ProtocolHandler,
        *,
        device: Optional[TorchDevice] = None,
    ):
        super().__init__()
        self.params = params
        self.agent_name = agent_name
        self.protocol_handler = protocol_handler
        if device is None:
            device = "cpu"
        self.device = device

    def _run_recorder_hook(
        self,
        hooks: Optional[AgentHooks],
        hook_name: str,
        output: Optional[Tensor],
    ):
        if hooks is not None and output is not None:
            hooks.__getattribute__(hook_name)(output, agent_name=self.agent_name)

    @abstractmethod
    def to(device: TorchDevice):
        """Move the agent to the given device."""
        pass


class DummyAgentPartMixin(AgentPart, ABC):
    """A mixin for agent parts which are dummy (e.g. random or constant).

    Adds a dummy parameter to the agent part, so that PyTorch can calculate gradients
    and so that tensordict can determine the device.
    """

    def __init__(
        self,
        params: Parameters,
        agent_name: str,
        protocol_handler: ProtocolHandler,
        *,
        device: TorchDevice | None = None,
    ):
        super().__init__(params, agent_name, protocol_handler, device=device)
        self.dummy_parameter = TorchParameter(torch.tensor(0.0, device=self.device))

    def to(self, device: TorchDevice):
        """Move the agent to the given device."""
        self.device = device
        self.dummy_parameter = self.dummy_parameter.to(device)


class AgentBody(AgentPart, ABC):
    """Base class for all agent bodies, which compute representations for heads.

    Representations should have dimension `params.d_representation`.
    """


class DummyAgentBody(DummyAgentPartMixin, AgentBody, ABC):
    """A dummy agent body which does nothing."""


class AgentHead(AgentPart, ABC):
    """Base class for all agent heads."""


class AgentPolicyHead(AgentHead, ABC):
    """Base class for all agent policy heads."""


class RandomAgentPolicyHead(DummyAgentPartMixin, AgentPolicyHead, ABC):
    """A policy head which samples actions randomly."""


class AgentValueHead(AgentHead, ABC):
    """Base class for all agent value heads, to the value of a state."""


class ConstantAgentValueHead(DummyAgentPartMixin, AgentValueHead, ABC):
    """A value head which returns a constant value."""


class SoloAgentHead(AgentHead, ABC):
    """Base class for all solo agent heads, which attempt the task on their own."""


class CombinedAgentPart(TensorDictModuleBase, ABC):
    """Base class for modules which combine agent parts together.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    parts : dict[str, AgentPart]
        The agent parts to combine.
    """

    def __init__(
        self,
        params: Parameters,
        protocol_handler: ProtocolHandler,
        parts: dict[str, AgentPart],
    ):
        super().__init__()
        self.params = params
        self.protocol_handler = protocol_handler

        self._agent_names = protocol_handler.agent_names

        if set(parts.keys()) != set(self._agent_names):
            raise ValueError(
                f"The agent names in {type(self).__name__} must match the agent names "
                f"in the protocol handler. Expected {self._agent_names}, got "
                f"{parts.keys()}."
            )

        # Add the parts as submodules, so that PyTorch knows about them
        for agent_name in self._agent_names:
            self.add_module(agent_name, parts[agent_name])


class CombinedBody(CombinedAgentPart, ABC):
    """A module which combines all the agent bodies together.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    bodies : dict[str, AgentBody]
        The agent bodies to combine.
    """

    def __init__(
        self,
        params: Parameters,
        protocol_handler: ProtocolHandler,
        bodies: dict[str, AgentBody],
    ):
        super().__init__(params, protocol_handler, bodies)
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


class CombinedPolicyHead(CombinedAgentPart, ABC):
    """A module which combines all the agent policy heads together.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    policy_heads : dict[str, AgentPolicyHead]
        The agent policy heads to combine.
    """

    def __init__(
        self,
        params: Parameters,
        protocol_handler: ProtocolHandler,
        policy_heads: dict[str, AgentPolicyHead],
    ):
        super().__init__(params, protocol_handler, policy_heads)
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
            The output of the combined policy head.
        """
        pass

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

        num_agents = len(self._agent_names)

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


class CombinedValueHead(CombinedAgentPart, ABC):
    """A module which combines all the agent value heads together.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    value_heads : dict[str, AgentValueHead]
        The agent value heads to combine.
    """

    def __init__(
        self,
        params: Parameters,
        protocol_handler: ProtocolHandler,
        value_heads: dict[str, AgentValueHead],
    ):
        super().__init__(params, protocol_handler, value_heads)
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


class Agent:
    """A class which holds all the parts of an agent for an experiment.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    agent_name : str
        The name of the agent.
    body : AgentBody
        The body of the agent.
    policy_head : AgentPolicyHead, optional
        The policy head of the agent.
    value_head : AgentValueHead, optional
        The value head of the agent.
    solo_head : SoloAgentHead, optional
        The solo head of the agent.
    """

    def __init__(
        self,
        params: Parameters,
        agent_name: str,
        body: AgentBody,
        policy_head: Optional[AgentPolicyHead] = None,
        value_head: Optional[AgentValueHead] = None,
        solo_head: Optional[SoloAgentHead] = None,
    ):
        if policy_head is None and solo_head is None:
            raise ValueError(
                "An agent must have either a policy head or a solo head, or both."
            )

        if policy_head is not None and value_head is None:
            raise ValueError("An agent with a policy head must have a value head")

        self.params = params
        self.agent_name = agent_name
        self.body = body
        self.policy_head = policy_head
        self.value_head = value_head
        self.solo_head = solo_head

        self._agent_params = params.agents[agent_name]

    @staticmethod
    def _append_filtered_params(
        model_param_dict: list[dict[str, Any]],
        named_parameters: list[tuple[str, TorchParameter]],
        filter: Callable[[str], bool],
        lr: float,
    ):
        """Filter the parameters and set their learning rate, and append them to a list.

        Normally appends a dictionary with the keys `params` and `lr`, consisting of the
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

    def get_param_dicts(
        self,
        base_lr: float,
        named_parameters: Optional[Iterable[tuple[str, TorchParameter]]] = None,
        body_lr_factor_override: Optional[float] = None,
        body_param_regex_prefixes: Iterable[str] = ("actor_network_params.module_0_",),
        non_body_param_regex_prefixes: Iterable[str] = (
            "actor_network_params.module_[1-9]_",
            "critic_network_params.module_[0-9]_",
        ),
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
        body_lr_factor_override : float, optional
            The learning rate factor for the body, which overrides the one set in the
            agent params.
        body_param_regex_prefixes : Iterable[str], optional
            The regular expression prefixes for the names of the body parameters. If not
            given, the default ones are used.
        non_body_param_regex_prefixes : Iterable[str], optional
            The regular expression prefixes for the names of the non-body parameters. If
            not given, the default ones are used.

        Returns
        -------
        param_dict : Iterable[dict[str, Any]]
            The Torch parameters of the agent, and their learning rates. This is an
            iterable of dictionaries with the keys `params` and `lr`.
        """

        # The learning rate of the whole agent
        agent_lr = self._agent_params.agent_lr_factor * base_lr

        # Determine the learning rate of the body.
        if body_lr_factor_override is None:
            body_lr = agent_lr * self._agent_params.body_lr_factor
        else:
            body_lr = agent_lr * body_lr_factor_override

        model_param_dict = []

        # If named_parameters is not given, use the parameters of all the agent parts.
        # In this case, we don't need to filter by name
        if named_parameters is None:
            model_param_dict.append(dict(params=self.body.parameters(), lr=body_lr))
            if self.policy_head is not None:
                model_param_dict.append(
                    dict(params=self.policy_head.parameters(), lr=agent_lr)
                )
            if self.value_head is not None:
                model_param_dict.append(
                    dict(params=self.value_head.parameters(), lr=agent_lr)
                )
            if self.solo_head is not None:
                model_param_dict.append(
                    dict(params=self.solo_head.parameters(), lr=agent_lr)
                )
            return model_param_dict

        # Convert the named parameters to a list, so that we can iterate over it
        # multiple times
        named_parameters = list(named_parameters)

        def is_body_param(param_name: str):
            return any(
                re.match(f"{prefix}{self.agent_name}", param_name)
                for prefix in body_param_regex_prefixes
            )

        # Set the learning rate for the body parameters
        self._append_filtered_params(
            model_param_dict, named_parameters, is_body_param, body_lr
        )

        # Set the learning rate for the non-body parameters
        def is_non_body_param(param_name: str):
            return any(
                re.match(f"{prefix}{self.agent_name}", param_name)
                for prefix in non_body_param_regex_prefixes
            )

        # Set the learning rate for the non-body parameters
        self._append_filtered_params(
            model_param_dict, named_parameters, is_non_body_param, agent_lr
        )

        return model_param_dict
