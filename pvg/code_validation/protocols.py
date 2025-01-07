"""Implementations of the parts of interaction protocols specific to code validation.

This module controls how prompts are created and how messages are interpreted for each
protocol.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Iterator
import importlib.resources
from string import Template
from functools import cache
from collections import OrderedDict
from random import Random

from torch import Tensor, as_tensor

from jaxtyping import Bool, Int, Float

from pvg.parameters import InteractionProtocolType, ScenarioType
from pvg.protocols.protocol_base import ProtocolHandler
from pvg.protocols.registry import register_protocol_handler
from pvg.protocols.main_protocols import (
    PvgProtocol,
    DebateProtocol,
    AdpProtocol,
    MerlinArthurProtocol,
    MnipProtocol,
    SoloVerifierProtocol,
)
from pvg.utils.api import InvalidDecisionError, NotAllActiveChannelsInResponseError
from pvg.utils.nested_array_dict import NestedArrayDict


@dataclass
class CodeValidationAgentSpec:
    """A specification for an agent in a code validation protocol.

    This dataclass specifies how the model representing each agent is interfaced with.

    Parameters
    ----------
    human_name : str
        The human-friendly name of the agent, used in prompts.
    response_channel_headers : dict[str, str], optional
        In multi-channel protocols, the completion from the model should contain
        messages for all channels in which the agent is active. Each message is prefaced
        by a header that specifies the channel. This dictionary maps channel names to
        headers. This can be `None` if the agent is active in only one channel.
    channel_order : list[str | set[str]], optional
        When making a request to the model, in each round the channels are ordered
        according to this list. It is a list of either channel names or sets of channel
        names. If a set of channel names is given, the channels are ordered randomly
        within that set. It is recommended to put the channels in which the agent is
        active last. If `None`, the order is determined by the protocol handler.
    anonymous : bool, optional
        Whether the agent is anonymous. If True, the agent's name will not be used in
        prompts. Default is False.
    last_round_system_message : str, optional
        If set, this message will be sent as a system message at the beginning of the
        last round of the interaction to the agent. This can be used to tell the agent
        to make a decision.
    use_raw_message_for_self_prompt : bool, default=True
        When prompting the agent for a message, whether messages sent from this agent
        should be included in the chat history as raw messages (rather than being split
        by channel). This agent's messages will still be split by channel in the chat
        histories used to prompt other agents. When prompting a model with its own past
        messages, it makes sense to use the raw messages, since the model will do some
        in-context learning. If the model always sees the messages split by channel, it
        will eventually learn to generate messages that are split by channel, which is
        not what we want.
    default_stance : int, default=1
        The default verdict that the agent will argue for, where 0 is reject and 1 is
        accept. This is only relevant for provers.
    """

    human_name: str
    response_channel_headers: Optional[dict[str, str]] = None
    channel_order: Optional[list[str | set[str]]] = None
    anonymous: bool = False
    last_round_system_message: Optional[str] = None
    use_raw_message_for_self_prompt: bool = True
    default_stance: int = 1


class CodeValidationProtocolHandler(ProtocolHandler, ABC):
    """Mixin for code validation protocol handlers."""

    prover_stance_can_be_randomized: bool = False

    @property
    @abstractmethod
    def agent_specs(self) -> dict[str, CodeValidationAgentSpec]:
        """A dictionary mapping agent names to specifications."""

    @property
    def system_prompt_directory(self) -> str:
        """The dot-separated path to the directory containing the system prompts."""

        return (
            f"pvg.code_validation.prompt_templates.system_prompts"
            f".{self.hyper_params.interaction_protocol}"
        )

    @cache
    def get_agent_system_prompt_template(self, agent_name: str) -> Template:
        """Get the system prompt template for a given agent.

        This template is used to generate system prompts at the beginning of the
        chat history for the agent.

        Parameters
        ----------
        agent_name : str
            The name of the agent.

        Returns
        -------
        Template
            The system prompt template for the agent.
        """

        try:
            prompt_template_traversable = importlib.resources.files(
                self.system_prompt_directory
            )
        except ModuleNotFoundError:
            raise NotImplementedError(
                f"System prompt directory for protocol "
                f"{self.hyper_params.interaction_protocol!s} not found."
            )

        template_filename = f"{agent_name}.txt"
        return Template(
            prompt_template_traversable.joinpath(template_filename).read_text()
        )

    def get_agent_system_prompt(self, agent_name: str, **kwargs) -> str:
        """Get the system prompt for a given agent.

        This prompt is used to generate system prompts at the beginning of the chat
        history for the agent.

        Parameters
        ----------
        agent_name : str
            The name of the agent.
        kwargs
            Additional keyword arguments to pass to the template.

        Returns
        -------
        str
            The system prompt for the agent.
        """

        # Determine the stance of the agent as a string, if it can be randomized
        if (
            self.prover_stance_can_be_randomized
            and self.hyper_params.protocol_common.randomize_prover_stance
        ):
            agent_stance: int = kwargs.pop(
                "agent_stance", self.agent_specs[agent_name].default_stance
            )
        else:
            agent_stance = self.agent_specs[agent_name].default_stance
        agent_stance_string = "accept" if agent_stance == 1 else "reject"

        return self.get_agent_system_prompt_template(agent_name).substitute(
            **kwargs, agent_stance_string=agent_stance_string
        )

    def get_agent_ordered_channels(self, agent_name: str, seed: int) -> Iterator[str]:
        """Get an iterator over the channels ordered for the agent.

        Channels can be ordered differently for each agent, when creating the request to
        the model.

        Parameters
        ----------
        agent_name : str
            The name of the agent.
        seed : int
            The random seed to use for shuffling the channels, if necessary.

        Yields
        ------
        channel_name : str
            The name of the channel.
        """

        if self.agent_specs[agent_name].channel_order is not None:

            random = Random(seed)

            for list_item in self.agent_specs[agent_name].channel_order:
                if isinstance(list_item, set):
                    yield from random.sample(list(list_item), len(list_item))
                else:
                    yield list_item

        else:
            yield from self.message_channel_names

    def parse_chat_completion(
        self, completion_text: str, agent_name: str, round_id: int
    ) -> tuple[OrderedDict[str, str] | None, int]:
        """Parse a chat completion into a message to each channel and a decision.

        Parameters
        ----------
        completion_text : str
            The completion to parse.
        agent_name : str
            The name of the agent that generated the completion.
        round_id : int
            The current round of the interaction.

        Returns
        -------
        channel_messages : OrderedDict[str, str] | None
            A dictionary mapping channel names to messages, ordered by channel order. If
            the model has made a decision, this will be None.
        decision : int
            The decision made by the model. This is either 0 (reject), 1 (accept) or 2
            (no decision).

        Raises
        ------
        InvalidDecisionError
            If the completion text does not contain a valid decision.
        NotAllActiveChannelsInResponseError
            If the response does not contain messages for all active channels
        """

        # First check if the completion contains a decision
        decision = self._parse_decision(completion_text)
        if decision != 2:
            return None, decision

        # Get the channels where the agent can be active
        active_channels = [
            channel_name
            for channel_name in self.message_channel_names
            if self.can_agent_be_active(agent_name, round_id, channel_name)
        ]

        if len(active_channels) == 0:
            raise ValueError(
                f"Tried to parse response for {agent_name!r} in round {round_id}, but it "
                f"is not active in any channel."
            )

        # If the agent is active in only one channel, parse the message for that channel
        if len(active_channels) == 1:
            return (
                self._parse_single_channel_message(completion_text, active_channels[0]),
                2,
            )

        # Otherwise, parse the message for all active channels
        return (
            self._parse_multi_channel_message(
                completion_text, agent_name, active_channels
            ),
            2,
        )

    @property
    def empty_channel_message(self) -> OrderedDict[str, str]:
        """An empty message for each channel.

        This is used as a placeholder when the model fails to generate a valid response.
        """

        return OrderedDict(
            [(channel_name, "") for channel_name in self.message_channel_names]
        )

    def _parse_decision(self, completion_text: str) -> int:
        """Parse a completion text to extract the decision.

        Parameters
        ----------
        completion_text : str
            The completion text to parse.

        Returns
        -------
        decision : int
            The decision extracted from the completion text. This is either 0 (reject),
            1 (accept) or 2 (no decision).

        Raises
        ------
        InvalidDecisionError
            If the completion text does not contain a valid decision.
        """

        if "decision:" in completion_text.lower():
            if "decision: accept" in completion_text.lower():
                return 1
            elif "decision: reject" in completion_text.lower():
                return 0
            else:
                raise InvalidDecisionError(response_text=completion_text)
        else:
            return 2

    def _parse_single_channel_message(
        self, completion_text: str, active_channel_name: str
    ) -> OrderedDict[str, str]:
        """Parse a completion text to extract the message when it is to one channel.

        Parameters
        ----------
        completion_text : str
            The completion text to parse.
        active_channel_name : str
            The name of the message channel where the agent is active.

        Returns
        -------
        channel_messages : OrderedDict[str, str]
            A dictionary mapping the channel name to the message.
        """

        if active_channel_name not in self.message_channel_names:
            raise ValueError(
                f"Tried to parse response for {active_channel_name!r}, but it is not a "
                f"valid channel name."
            )

        channel_messages = OrderedDict(
            [(channel_name, None) for channel_name in self.message_channel_names]
        )
        channel_messages[active_channel_name] = completion_text
        return channel_messages

    def _parse_multi_channel_message(
        self,
        completion_text: str,
        agent_name: str,
        active_channels: list[str],
    ) -> OrderedDict[str, str]:
        """Parse a completion text to extract messages when it is to multiple channels.

        Parameters
        ----------
        completion_text : str
            The completion text to parse.
        agent_name : str
            The name of the agent that generated the completion.
        active_channels : list[str]
            The names of the message channels where the agent is active.

        Returns
        -------
        channel_messages : OrderedDict[str, str]
            A dictionary mapping the channel name to the message.

        Raises
        ------
        NotAllActiveChannelsInResponseError
            If the response does not contain messages for all active channels
        """

        channel_messages = OrderedDict(
            [(channel_name, None) for channel_name in self.message_channel_names]
        )

        # Get the location of each channel header in the completion text
        header_locations = {}
        for channel_name in active_channels:
            header = self.agent_specs[agent_name].response_channel_headers[channel_name]
            if header.lower() not in completion_text.lower():
                raise NotAllActiveChannelsInResponseError(response_text=completion_text)
            header_locations[channel_name] = completion_text.lower().index(
                header.lower()
            )

        # Sort the active channels by the location of their headers in the completion
        # text
        active_channels_sorted = sorted(
            active_channels, key=lambda channel_name: header_locations[channel_name]
        )

        # Extract the message for each channel
        for i, channel_name in enumerate(active_channels_sorted):

            if channel_name not in self.message_channel_names:
                raise ValueError(
                    f"Tried to parse response for {channel_name!r}, but it is not a "
                    f"valid channel name."
                )

            header = self.agent_specs[agent_name].response_channel_headers[channel_name]

            start_index = header_locations[channel_name]
            if i < len(active_channels_sorted) - 1:
                end_index = header_locations[active_channels_sorted[i + 1]]
            else:
                end_index = None

            channel_messages[channel_name] = completion_text[
                start_index:end_index
            ].strip()

        return channel_messages

    def _include_prover_rewards(
        self,
        verifier_decision_made: Bool[Tensor, "..."],
        verifier_decision: Int[Tensor, "..."],
        reward: Float[Tensor, "... agent"],
        env_td: NestedArrayDict,
    ):
        """Compute the rewards for the other agents and add them to the current reward.

        This modifies the default implementation to allow following the prover's stance
        when the stance can be randomized.

        The `reward` tensor is updated in place, adding in the rewards for the agents
        at the appropriate indices.

        Parameters
        ----------
        verifier_decision_made : Bool[Tensor, "..."]
            A boolean mask indicating whether the verifier has made a decision.
        verifier_decision : Int[Tensor, "..."]
            The verifier's decision.
        reward : Float[Tensor, "... agent"]
            The currently computed reward, which should include the reward for the
            verifier.
        env_td : NestedArrayDict
            The current observation and state.
        """

        if self.hyper_params.protocol_common.shared_reward:
            for prover_index in self.prover_indices:
                reward[..., prover_index] = reward[..., self.verifier_index]
        else:
            if len(self.prover_names) == 1:
                if (
                    self.prover_stance_can_be_randomized
                    and self.hyper_params.protocol_common.randomize_prover_stance
                ):
                    prover_stance = as_tensor(env_td["prover_stance"])
                else:
                    prover_stance = self.agent_specs[
                        self.prover_names[0]
                    ].default_stance
                reward[..., self.prover_indices[0]] = (
                    verifier_decision_made & (verifier_decision == prover_stance)
                ).float() * self.hyper_params.protocol_common.prover_reward
            else:
                reward[..., self.prover_indices[0]] = (
                    verifier_decision_made & (verifier_decision == 0)
                ).float() * self.hyper_params.protocol_common.prover_reward
                reward[..., self.prover_indices[1]] = (
                    verifier_decision_made & (verifier_decision == 1)
                ).float() * self.hyper_params.protocol_common.prover_reward


@register_protocol_handler("pvg", "code_validation")
class PvgCodeValidationProtocol(CodeValidationProtocolHandler, PvgProtocol):
    """The PVG protocol for code validation."""

    prover_stance_can_be_randomized = True

    agent_specs = {
        "verifier": CodeValidationAgentSpec(
            "Verifier",
            last_round_system_message="You cannot ask any more questions. You must now "
            "make a decision.",
        ),
        "prover": CodeValidationAgentSpec("Expert"),
    }


@register_protocol_handler("abstract_decision_problem", "code_validation")
class AdpCodeValidationProtocol(CodeValidationProtocolHandler, AdpProtocol):
    """The abstract decision problem protocol for code validation."""

    prover_stance_can_be_randomized = True

    agent_specs = {
        "verifier": CodeValidationAgentSpec("Verifier"),
        "prover": CodeValidationAgentSpec("Expert"),
    }


@register_protocol_handler("debate", "code_validation")
class DebateCodeValidationProtocol(CodeValidationProtocolHandler, DebateProtocol):
    """The debate protocol for code validation."""

    @property
    def agent_specs(self) -> dict[str, CodeValidationAgentSpec]:
        """A dictionary mapping agent names to specifications."""

        if self.hyper_params.debate_protocol.randomize_channel_order:
            verifier_channel_order = [{"prover0_channel", "prover1_channel"}]
        else:
            verifier_channel_order = None

        return {
            "verifier": CodeValidationAgentSpec(
                "Verifier",
                response_channel_headers={
                    "prover0_channel": "Question for Expert_1:",
                    "prover1_channel": "Question for Expert_2:",
                },
                last_round_system_message="You cannot ask any more questions. You must "
                "now make a decision.",
                channel_order=verifier_channel_order,
            ),
            "prover0": CodeValidationAgentSpec(
                "Expert_1",
                channel_order=["prover1_channel", "prover0_channel"],
                default_stance=0,
            ),
            "prover1": CodeValidationAgentSpec(
                "Expert_2",
                channel_order=["prover0_channel", "prover1_channel"],
                default_stance=1,
            ),
        }


@register_protocol_handler("merlin_arthur", "code_validation")
class MerlinArthurCodeValidationProtocol(
    CodeValidationProtocolHandler, MerlinArthurProtocol
):
    """The Merlin-Arthur protocol for code validation."""

    agent_specs = {
        "verifier": CodeValidationAgentSpec("Verifier"),
        "prover0": CodeValidationAgentSpec(
            "Expert_1", default_stance=0, anonymous=True
        ),
        "prover1": CodeValidationAgentSpec(
            "Expert_2", default_stance=1, anonymous=True
        ),
    }


@register_protocol_handler("mnip", "code_validation")
class MnipCodeValidationProtocol(CodeValidationProtocolHandler, MnipProtocol):
    """The MNIP protocol for code validation."""

    prover_stance_can_be_randomized = True

    @property
    def agent_specs(self) -> dict[str, CodeValidationAgentSpec]:
        """A dictionary mapping agent names to specifications."""

        if self.hyper_params.mnip_protocol.randomize_channel_order:
            verifier_channel_order = [{"prover0_channel", "prover1_channel"}]
        else:
            verifier_channel_order = None

        return {
            "verifier": CodeValidationAgentSpec(
                "Verifier",
                response_channel_headers={
                    "prover0_channel": "Question for Expert_1:",
                    "prover1_channel": "Question for Expert_2:",
                },
                last_round_system_message="You cannot ask any more questions. You must "
                "now make a decision.",
                channel_order=verifier_channel_order,
            ),
            "prover0": CodeValidationAgentSpec(
                "Expert_1", channel_order=["prover1_channel", "prover0_channel"]
            ),
            "prover1": CodeValidationAgentSpec(
                "Expert_2", channel_order=["prover0_channel", "prover1_channel"]
            ),
        }


@register_protocol_handler("solo_verifier", "code_validation")
class SoloVerifierCodeValidationProtocol(
    CodeValidationProtocolHandler, SoloVerifierProtocol
):
    """A protocol where the verifier acts alone."""

    agent_specs = {"verifier": CodeValidationAgentSpec("Verifier")}

    def _include_prover_rewards(
        self,
        verifier_decision_made: Bool[Tensor, "..."],
        verifier_decision: Int[Tensor, "..."],
        reward: Float[Tensor, "... agent"],
        env_td: NestedArrayDict,
    ):
        pass
