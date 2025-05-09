"""Implementations of the parts of interaction protocols specific to code validation.

This module controls how prompts are created and how messages are interpreted for each
protocol.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Iterator, Literal
from functools import cache, cached_property
from collections import OrderedDict
from random import Random

from torch import Tensor, as_tensor

from jaxtyping import Bool, Int, Float

from jinja2 import (
    Environment as JinjaEnvironment,
    PackageLoader,
    Template,
    StrictUndefined,
)

from nip.parameters.agents import CodeValidationAgentParameters
from nip.protocols.protocol_base import ProtocolHandler
from nip.protocols.registry import register_protocol_handler
from nip.protocols.main_protocols import (
    NipProtocol,
    DebateProtocol,
    AdpProtocol,
    MerlinArthurProtocol,
    MnipProtocol,
    SoloVerifierProtocol,
)
from nip.protocols.verifier_decision_scale import VerifierDecisionParseError
from nip.utils.api import InvalidDecisionError, NotAllActiveChannelsInResponseError
from nip.utils.nested_array_dict import NestedArrayDict
from nip.utils.jinja_filters import capitalise_first_letter, add_s_plural
from nip.constants import PACKAGE_ROOT


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
        headers. This can be ``None`` if the agent is active in only one channel.
    channel_order : list[str | set[str]], optional
        When making a request to the model, in each round the channels are ordered
        according to this list. It is a list of either channel names or sets of channel
        names. If a set of channel names is given, the channels are ordered randomly
        within that set. It is recommended to put the channels in which the agent is
        active last. If ``None``, the order is determined by the protocol handler.
    anonymous : bool, optional
        Whether the agent is anonymous. If True, the agent's name will not be used in
        prompts. Default is False.
    last_round_system_message : str, optional
        If set, this message will be sent as a system message at the beginning of the
        last round of the interaction to the agent. This can be used to tell the agent
        to make a decision. NOTE: This functionality overlaps with the 'supervisor
        message'. If the supervisor message would be sent in the last round, this
        setting is ignored.
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
    def agent_params(self) -> dict[str, CodeValidationAgentParameters]:
        """A dictionary mapping agent names to parameters."""
        return self.hyper_params.agents

    def modify_system_prompt_context(self, agent_name: str, context: dict) -> dict:
        """Modify the template context for a system prompt for a given agent.

        This method can be overridden by any protocol handler which needs to include
        additional variables in the system prompts.

        Parameters
        ----------
        agent_name : str
            The name of the agent.
        context : dict
            The current variables to include in the system prompts.

        Returns
        -------
        system_prompt_variables : dict
            A dictionary mapping variable names to values. To add new variables, return
            ``context`` with the new variables added.
        """

        return context

    @cached_property
    def jinja_filters(self) -> dict[str, callable]:
        """The custom Jinja2 filters for this protocol handler."""
        return {
            "capitalise_first_letter": capitalise_first_letter,
            "add_s_plural": add_s_plural,
        }

    @cached_property
    def jinja_environment(self) -> JinjaEnvironment:
        """The Jinja2 environment for loading templates."""

        environment = JinjaEnvironment(
            loader=PackageLoader(
                "nip",
                f"code_validation/templates/rollout_generation"
                f"/{self.hyper_params.code_validation.system_prompt_version}",
            ),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
        )

        # Add custom filters to the Jinja2 environment
        for filter_name, filter_func in self.jinja_filters.items():
            environment.filters[filter_name] = filter_func

        return environment

    @cached_property
    def verifier_decision_instructions_prompt_template(self) -> Template:
        """The template containing the instructions for the verifier decision."""

        if self.hyper_params.code_validation.system_prompt_version == "v1":
            return self.jinja_environment.from_string("")

        return self.jinja_environment.get_template(
            f"verifier_decision_instructions/"
            f"{self.hyper_params.protocol_common.verifier_decision_scale}.txt"
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
        system_prompt_template : jinja2.Template
            The system prompt template for the agent.
        """

        if self.agent_params[agent_name].system_prompt_template_path is not None:
            with open(
                self.agent_params[agent_name].system_prompt_template_path, "r"
            ) as f:
                template_string = f.read()
            return self.jinja_environment.from_string(template_string)
        else:
            return self.jinja_environment.get_template(
                f"main_system_prompt/{self.hyper_params.interaction_protocol}"
                f"/{agent_name}.txt"
            )

    def get_agent_system_prompt(self, agent_name: str, **context) -> str:
        """Get the system prompt for a given agent.

        This prompt is used to generate system prompts at the beginning of the chat
        history for the agent.

        Parameters
        ----------
        agent_name : str
            The name of the agent.
        kwargs
            Additional keyword arguments to pass to the template as context.

        Returns
        -------
        system_prompt : str
            The system prompt for the agent.
        """

        context = self.modify_system_prompt_context(agent_name, context)

        verifier_decision_instructions = (
            self.verifier_decision_instructions_prompt_template.render(**context)
        )

        return self.get_agent_system_prompt_template(agent_name).render(
            **context,
            agent_stance_string=self._get_agent_stance_string(agent_name, context),
            verifier_decision_instructions=verifier_decision_instructions,
        )

    @cache
    def get_agent_supervisor_message_template(self, agent_name: str) -> Template:
        """Get the supervisor message template for a given agent.

        This template is used to generate a message appended to the chat history before
        it is passed to the agent model.

        Parameters
        ----------
        agent_name : str
            The name of the agent.

        Returns
        -------
        supervisor_message_template : jinja2.Template
            The supervisor message template for the agent.
        """
        return self.jinja_environment.get_template(
            f"supervisor_message/{self.hyper_params.interaction_protocol}"
            f"/{agent_name}.txt"
        )

    def get_agent_supervisor_message(
        self,
        agent_name: str,
        round_id: int,
        num_questions_left: int,
        **context,
    ) -> str:
        """Get the supervisor message for a given agent.

        This message is appended to the chat history before it is passed to the agent
        model.

        Parameters
        ----------
        agent_name : str
            The name of the agent.
        kwargs
            Additional keyword arguments to pass to the template as context.

        Returns
        -------
        supervisor_message : str
            The system prompt for the agent.
        """

        verifier_decision_instructions = (
            self.verifier_decision_instructions_prompt_template.render(**context)
        )

        return self.get_agent_supervisor_message_template(agent_name).render(
            round_id=round_id,
            num_questions_left=num_questions_left,
            agent_stance_string=self._get_agent_stance_string(agent_name, context),
            verifier_decision_instructions=verifier_decision_instructions,
            **context,
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
    ) -> tuple[OrderedDict[str, str] | None, Literal[0, 1, 2, 3], float, str]:
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
        discrete_decision : Literal[0, 1, 2, 3]
            The discrete decision from the verifier model, with the following meanings:

            - 0: reject
            - 1: accept
            - 2: no decision
            - 3: end with neither accept nor reject

        continuous_decision : float
            The continuous decision from the verifier model. This is a number between -1
            and 1, where -1 is "reject" and 1 is "accept". This is a more fine-grained
            version of ``discrete_decision``.
        raw_decision : str
            The raw decision text from the verifier model. This is the text which
            appears after "Decision: " in the completion text.

        Raises
        ------
        InvalidDecisionError
            If the completion text does not contain a valid decision.
        NotAllActiveChannelsInResponseError
            If the response does not contain messages for all active channels
        """

        # First check if the completion contains a decision
        discrete_decision, continuous_decision, raw_decision = self._parse_decision(
            completion_text
        )
        if discrete_decision != 2:
            return None, discrete_decision, continuous_decision, raw_decision

        # Get the channels where the agent can be active
        active_channels = [
            channel_name
            for channel_name in self.message_channel_names
            if self.can_agent_be_active(agent_name, round_id, channel_name)
        ]

        if len(active_channels) == 0:
            raise ValueError(
                f"Tried to parse response for {agent_name!r} in round {round_id}, but "
                f"it is not active in any channel."
            )

        # If the agent is active in only one channel, parse the message for that channel
        if len(active_channels) == 1:
            return (
                self._parse_single_channel_message(completion_text, active_channels[0]),
                discrete_decision,
                continuous_decision,
                raw_decision,
            )

        # Otherwise, parse the message for all active channels
        return (
            self._parse_multi_channel_message(
                completion_text, agent_name, active_channels
            ),
            discrete_decision,
            continuous_decision,
            raw_decision,
        )

    @property
    def empty_channel_message(self) -> OrderedDict[str, str]:
        """An empty message for each channel.

        This is used as a placeholder when the model fails to generate a valid response.
        """

        return OrderedDict(
            [(channel_name, "") for channel_name in self.message_channel_names]
        )

    def _get_agent_stance_string(self, agent_name: str, context: dict) -> str:
        """Get the stance of the agent as a string, when substituting a template.

        The stance is either "accept" or "reject", and tells the agent what to argue
        for. This is only relevant for provers.

        Parameters
        ----------
        agent_name : str
            The name of the agent.
        context : dict
            The template variables to use for rendering the template.

        Returns
        -------
        agent_stance_string : str
            The stance of the agent as a string.
        """

        if (
            self.prover_stance_can_be_randomized
            and self.hyper_params.protocol_common.randomize_prover_stance
        ):
            agent_stance: int = context.pop(
                "agent_stance", self.agent_specs[agent_name].default_stance
            )
        else:
            agent_stance = self.agent_specs[agent_name].default_stance

        if agent_stance == 0:
            return "reject"
        else:
            return "accept"

    def _parse_decision(
        self, completion_text: str
    ) -> tuple[Literal[0, 1, 2, 3], float, str]:
        """Parse a completion text to extract the decision.

        Parameters
        ----------
        completion_text : str
            The completion text to parse.

        Returns
        -------
        discrete_decision : Literal[0, 1, 2, 3]
            The discrete decision from the verifier model, with the following meanings:

            - 0: reject
            - 1: accept
            - 2: no decision
            - 3: end with neither accept nor reject

        continuous_decision : float
            The continuous decision from the verifier model. This is a number between -1
            and 1, where -1 is "reject" and 1 is "accept". This is a more fine-grained
            version of ``discrete_decision``.
        raw_decision : str
            The raw decision text from the verifier model. This is the text which
            appears after "Decision: " in the completion text.

        Raises
        ------
        InvalidDecisionError
            If the completion text does not contain a valid decision.
        """

        if "decision:" in completion_text.lower():
            first_decision_index = completion_text.lower().index("decision:")
            decision_text = completion_text[first_decision_index + len("decision:") :]
            try:
                return self.verifier_decision_scale_handler.extract_decision(
                    decision_text
                )
            except VerifierDecisionParseError as e:
                raise InvalidDecisionError(response_text=completion_text) from e
        else:
            return 2, 0.0, ""

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
        verifier_float_decision: Float[Tensor, "..."] | None,
        reward: Float[Tensor, "... agent"],
        env_td: NestedArrayDict,
    ):
        """Compute the rewards for the other agents and add them to the current reward.

        This modifies the default implementation to allow following the prover's stance
        when the stance can be randomized.

        The ``reward`` tensor is updated in place, adding in the rewards for the agents
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

                if verifier_float_decision is not None:
                    reward[..., self.prover_indices[0]][~verifier_decision_made] = 0.0
                    reward[..., self.prover_indices[0]][verifier_decision_made] = (
                        (verifier_float_decision / 2 + 0.5)[verifier_decision_made]
                        * self.hyper_params.protocol_common.prover_reward
                        * (2 * prover_stance - 1)
                    )

                else:
                    reward[..., self.prover_indices[0]] = (
                        verifier_decision_made & (verifier_decision == prover_stance)
                    ).float() * self.hyper_params.protocol_common.prover_reward

            else:

                if verifier_float_decision is not None:

                    reward[..., self.prover_indices[0]][~verifier_decision_made] = 0.0
                    reward[..., self.prover_indices[0]][verifier_decision_made] = (
                        verifier_float_decision / 2 + 0.5
                    )[
                        verifier_decision_made
                    ] * self.hyper_params.protocol_common.prover_reward

                    reward[..., self.prover_indices[1]][~verifier_decision_made] = 0.0
                    reward[..., self.prover_indices[1]][verifier_decision_made] = (
                        -verifier_float_decision / 2 + 0.5
                    )[
                        verifier_decision_made
                    ] * self.hyper_params.protocol_common.prover_reward

                else:
                    reward[..., self.prover_indices[0]] = (
                        verifier_decision_made & (verifier_decision == 0)
                    ).float() * self.hyper_params.protocol_common.prover_reward
                    reward[..., self.prover_indices[1]] = (
                        verifier_decision_made & (verifier_decision == 1)
                    ).float() * self.hyper_params.protocol_common.prover_reward


@register_protocol_handler("nip", "code_validation")
class NipCodeValidationProtocol(CodeValidationProtocolHandler, NipProtocol):
    """The NIP protocol for code validation."""

    prover_stance_can_be_randomized = True

    agent_specs = {
        "verifier": CodeValidationAgentSpec(
            "Verifier",
            last_round_system_message="You cannot ask any more questions. You must now "
            "make a decision.",
        ),
        "prover": CodeValidationAgentSpec("Expert"),
    }


@register_protocol_handler("adp", "code_validation")
class AdpCodeValidationProtocol(CodeValidationProtocolHandler, AdpProtocol):
    """The abstract decision problem (ADP) protocol for code validation."""

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
    """The Merlin-Arthur Classifier (MAC) protocol for code validation."""

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

    def _include_prover_rewards(
        self,
        verifier_decision_made: Bool[Tensor, "..."],
        verifier_decision: Int[Tensor, "..."],
        verifier_float_decision: Float[Tensor, "..."] | None,
        reward: Float[Tensor, "... agent"],
        env_td: NestedArrayDict,
    ):
        super(CodeValidationProtocolHandler, self)._include_prover_rewards(
            verifier_decision_made=verifier_decision_made,
            verifier_decision=verifier_decision,
            verifier_float_decision=verifier_float_decision,
            reward=reward,
            env_td=env_td,
        )


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
        verifier_float_decision: Float[Tensor, "..."] | None,
        reward: Float[Tensor, "... agent"],
        env_td: NestedArrayDict,
    ):
        pass
