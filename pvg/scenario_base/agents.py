"""Base classes for building agents.

An agent is composed of a body and one or more heads. The body computes a representation
of the environment state, and the heads use this representation to compute the agent's
policy, value function, etc.

All modules are TensorDictModules, which means they take and return TensorDicts. Input
and output keys are specified in the module's `input_keys` and `output_keys` attributes.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Iterable, Callable, ClassVar
from dataclasses import dataclass, fields, InitVar
from functools import partial
import re
import itertools

import torch
from torch import Tensor
from torch.nn.parameter import Parameter as TorchParameter

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey

from einops import repeat

from jaxtyping import Float, Int

from pvg.parameters import Parameters
from pvg.protocols import ProtocolHandler
from pvg.utils.types import TorchDevice
from pvg.utils.params import check_if_critic_and_single_body


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

    The in and out keys are split into agent-level and environment-level keys.
    Agent-level keys are nested under "agents" in the environment's TensorDict, while
    environment-level keys are at the top level.

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

    Class attributes
    ----------------
    agent_level_in_keys : Iterable[NestedKey]
        The keys required by the agent part whose values are per-agent (so in the
        environment's TensorDict will be nested under "agents").
    env_level_in_keys : Iterable[NestedKey]
        The keys required by the agent part whose values are per-environment (so in the
        environment's TensorDict will be at the top level).
    agent_level_out_keys : Iterable[NestedKey]
        The keys produced by the agent part whose values are per-agent (so in the
        environment's TensorDict will be nested under "agents").
    env_level_out_keys : Iterable[NestedKey]
        The keys produced by the agent part whose values are per-environment (so in the
        environment's TensorDict will be at the top level).
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

    @property
    def required_pretrained_models(self) -> Iterable[str]:
        """The pretrained models used by the agent.

        The embeddings of these models will be added to the dataset.
        """
        return []


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

    @property
    def has_decider(self) -> bool:
        """Whether the policy head has an output yielding a decision.

        By default a decider is used to decide whether to continue exchanging messages.
        In this case it outputs a single triple of logits for the three options: guess
        that the graphs are not isomorphic, guess that the graphs are isomorphic, or
        continue exchanging messages.
        """
        return "verifier" in self.agent_name


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
        params: Parameters,
        protocol_handler: ProtocolHandler,
        parts: dict[str, AgentPart],
    ):
        super().__init__()
        self.params = params
        self.protocol_handler = protocol_handler
        self.parts = parts

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


@dataclass
class Agent(ABC):
    """A base class for holding all the parts of an agent for an experiment.

    Subclasses should define the `message_logits_key` class variable, which is the key
    in the output of the policy head which contains the logits for the message.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    agent_name : str
        The name of the agent.
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

    params: InitVar[Parameters]
    agent_name: InitVar[str]
    body: Optional[AgentBody] = None
    policy_body: Optional[AgentBody] = None
    value_body: Optional[AgentBody] = None
    policy_head: Optional[AgentPolicyHead] = None
    value_head: Optional[AgentValueHead] = None
    solo_head: Optional[SoloAgentHead] = None

    message_logits_key: ClassVar[str]

    def __post_init__(
        self,
        params: Parameters,
        agent_name: str,
    ):
        if self.body is None and self.policy_body is None:
            raise ValueError("An agent must have either a body or a policy body")

        if self.body is not None and self.policy_body is not None:
            raise ValueError("An agent cannot have both a body and a policy body")

        if self.value_body is not None and self.policy_body is None:
            raise ValueError("An agent with a value body must have a policy body")

        if self.policy_head is None and self.solo_head is None:
            raise ValueError(
                "An agent must have either a policy head or a solo head, or both."
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

        self.params = params
        self.agent_name = agent_name

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

    def _body_param_regex(self, part: str) -> str:
        use_critic, use_single_body = check_if_critic_and_single_body(self.params)
        network_suffix = "network"
        if self.params.functionalize_modules:
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
        use_critic, use_single_body = check_if_critic_and_single_body(self.params)
        nums = {"actor": "1-9", "critic": "0-9"}
        network_suffix = "network"
        if self.params.functionalize_modules:
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
        use_critic, use_single_body = check_if_critic_and_single_body(self.params)
        if use_critic and not use_single_body:
            return itertools.chain(
                self.policy_body.named_parameters(), self.value_body.named_parameters()
            )
        return self.body.named_parameters()

    @property
    def _body_parameters(self) -> Iterable[TorchParameter]:
        use_critic, use_single_body = check_if_critic_and_single_body(self.params)
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
            iterable of dictionaries with the keys `params` and `lr`.
        """

        # Check for mistakes
        if (
            self.params.rl.use_shared_body
            and self._agent_params.agent_lr_factor.actor
            != self._agent_params.agent_lr_factor.critic
        ):
            raise ValueError(
                "The agent learning rate factor for the actor and critic must be the same if the body is shared."
            )
        if (
            self.params.rl.use_shared_body
            and self._agent_params.body_lr_factor.actor
            != self._agent_params.body_lr_factor.critic
        ):
            raise ValueError(
                "The body learning rate factor for the actor and critic must be the same if the body is shared."
            )

        # The learning rate of the whole agent
        agent_lr = {
            "actor": self._agent_params.agent_lr_factor.actor * base_lr,
            "critic": self._agent_params.agent_lr_factor.critic * base_lr,
        }

        # Determine the learning rate of the body
        body_lr = {
            "actor": (
                agent_lr["actor"] * self._agent_params.body_lr_factor.actor
                if not body_lr_factor_override
                else agent_lr["actor"]
            ),
            "critic": (
                agent_lr["critic"] * self._agent_params.body_lr_factor.critic
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
