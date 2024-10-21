"""Implementations of the parts of interaction protocols specific to code validation.

This module controls how prompts are created and how messages are interpreted for each
protocol.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import importlib.resources
from string import Template
from functools import cache
from collections import OrderedDict

from pvg.parameters import InteractionProtocolType
from pvg.protocols.base import ProtocolHandler
from pvg.protocols.registry import register_protocol_handler
from pvg.protocols.main_protocols import (
    PvgProtocol,
    DebateProtocol,
    AdpProtocol,
    MerlinArthurProtocol,
    MnipProtocol,
)
from pvg.utils.api import InvalidDecisionError, NotAllActiveChannelsInResponseError


@dataclass
class CodeValidationAgentSpec:
    """A specification for an agent in a code validation protocol.

    This dataclass specifies how the model representing each agent is interfaced with.

    Parameters
    ----------
    human_name : str
        The human-friendly name of the agent, used in prompts.
    response_channel_headers : Optional[dict[str, str]], optional
        In multi-channel protocols, the completion from the model should contain
        messages for all channels in which the agent is active. Each message is prefaced
        by a header that specifies the channel. This dictionary maps channel names to
        headers. This can be `None` if the agent is active in only one channel.
    anonymous : bool, optional
        Whether the agent is anonymous. If True, the agent's name will not be used in
        prompts. Default is False.
    """

    human_name: str
    response_channel_headers: Optional[dict[str, str]] = None
    anonymous: bool = False


class CodeValidationProtocolHandler(ProtocolHandler, ABC):
    """Mixin for code validation protocol handlers"""

    @property
    @abstractmethod
    def agent_specs(self) -> dict[str, CodeValidationAgentSpec]:
        """A dictionary mapping agent names to specifications"""

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
                f"{self.params.interaction_protocol!s} not found."
            )

        template_filename = f"{agent_name}.txt"
        return Template(
            prompt_template_traversable.joinpath(template_filename).read_text()
        )

    @property
    def system_prompt_directory(self) -> str:
        """The dot-separated path to the directory containing the system prompts."""

        return (
            f"pvg.code_validation.prompt_templates.system_prompts"
            f".{self.params.interaction_protocol!s}"
        )

    def parse_chat_completion(
        self, completion_text: str, agent_name: str, round: int
    ) -> tuple[OrderedDict[str, str] | None, int]:
        """Parse a chat completion into a message to each channel and a decision

        Parameters
        ----------
        completion_text : str
            The completion to parse.
        agent_name : str
            The name of the agent that generated the completion.
        round : int
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
            if self.can_agent_be_active(agent_name, round, channel_name)
        ]

        if len(active_channels) == 0:
            raise ValueError(
                f"Tried to parse response for {agent_name!r} in round {round}, but it "
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


@register_protocol_handler(InteractionProtocolType.PVG)
class PvgCodeValidationProtocol(CodeValidationProtocolHandler, PvgProtocol):

    agent_specs = {
        "verifier": CodeValidationAgentSpec("Verifier"),
        "prover": CodeValidationAgentSpec("Expert"),
    }


@register_protocol_handler(InteractionProtocolType.ABSTRACT_DECISION_PROBLEM)
class AdpCodeValidationProtocol(CodeValidationProtocolHandler, AdpProtocol):

    agent_specs = {
        "verifier": CodeValidationAgentSpec("Verifier"),
        "prover": CodeValidationAgentSpec("Expert"),
    }


@register_protocol_handler(InteractionProtocolType.DEBATE)
class DebateCodeValidationProtocol(CodeValidationProtocolHandler, DebateProtocol):

    agent_specs = {
        "verifier": CodeValidationAgentSpec(
            "Verifier",
            response_channel_headers={
                "prover0_channel": "Question for Expert_1:",
                "prover1_channel": "Question for Expert_2:",
            },
        ),
        "prover0": CodeValidationAgentSpec("Expert_1"),
        "prover1": CodeValidationAgentSpec("Expert_2"),
    }


@register_protocol_handler(InteractionProtocolType.MERLIN_ARTHUR)
class MerlinArthurCodeValidationProtocol(
    CodeValidationProtocolHandler, MerlinArthurProtocol
):

    agent_specs = {
        "verifier": CodeValidationAgentSpec("Verifier"),
        "prover0": CodeValidationAgentSpec("Expert_1", anonymous=True),
        "prover1": CodeValidationAgentSpec("Expert_2", anonymous=True),
    }


@register_protocol_handler(InteractionProtocolType.MNIP)
class MnipCodeValidationProtocol(CodeValidationProtocolHandler, MnipProtocol):

    agent_specs = {
        "verifier": CodeValidationAgentSpec(
            "Verifier",
            response_channel_headers={
                "prover0_channel": "Question for Expert_1:",
                "prover1_channel": "Question for Expert_2:",
            },
        ),
        "prover0": CodeValidationAgentSpec("Expert_1"),
        "prover1": CodeValidationAgentSpec("Expert_2"),
    }
