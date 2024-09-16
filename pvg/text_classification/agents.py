"""Text classification agent components.

To integrate easily into the codebase, the text classification agents a split into
three parts: the body, the policy head, and the value head. However, for text
classification, the split is not necessary, so only the policy head actually does
anything. The body and value head return the input data unchanged.
"""

from abc import ABC
from typing import Optional, Literal
from functools import cached_property
import importlib.resources
from string import Template
from itertools import product

from torch import from_numpy

import numpy as np
from numpy.typing import NDArray

from einops import rearrange

from jaxtyping import Float, Int, Bool

from openai import OpenAI

from pvg.scenario_base import WholeAgent, RandomWholeAgent, CombinedWhole
from pvg.parameters import (
    Parameters,
    TextClassificationAgentParameters,
    RandomAgentParameters,
    ScenarioType,
    InteractionProtocolType,
)
from pvg.experiment_settings import ExperimentSettings
from pvg.factory import register_scenario_class
from pvg.protocols import ProtocolHandler
from pvg.utils.nested_array_dict import NestedArrayDict
from pvg.utils.types import NumpyStringDtype

TC_SCENARIO = ScenarioType.TEXT_CLASSIFICATION


class GenerationError(Exception, ABC):
    """Base class for exceptions raised during generation of the next message"""

    def __init__(self, num_retries: int):
        self.num_retries = num_retries
        super().__init__(f"Generation failed after {num_retries} retries")


class NotGuessedError(GenerationError):
    """Raised when the agent has not made a decision within the max number of turns"""


class ContentFilterError(GenerationError):
    """Raised when the agent's response is blocked by a content filter"""


class UnknownFinishReasonError(GenerationError):
    """Raised when the agent's finishes generating for an unknown reason"""

    def __init__(self, num_retries: int, reason: str):
        self.reason = reason
        self.num_retries = num_retries
        super(GenerationError, self).__init__(
            f"Generation failed after {num_retries} retries with reason {reason!r}"
        )


class InvalidResponseError(GenerationError):
    """Raised when the agent's response is invalid"""


class InvalidDecisionError(InvalidResponseError):
    """Raised when the agent's decision is invalid (i.e. not accept or reject)"""


class TextClassificationWholeAgent(WholeAgent, ABC):
    """Base class for text classification agent parts."""

    _visible_message_channel_mask: Optional[Bool[NDArray, "channel"]] = None

    @cached_property
    def visible_message_channel_mask(self) -> Bool[NDArray, "channel"]:
        """The mask for the message channels visible to the agent."""
        return super().visible_message_channel_mask.cpu().detach().numpy()


@register_scenario_class(TC_SCENARIO, WholeAgent, {"model_provider": "OpenAI"})
class OpenAiWholeAgent(TextClassificationWholeAgent):
    """The whole agent for text classification, using OpenAI's API."""

    agent_params: TextClassificationAgentParameters

    agent_level_in_keys = ["active_mask"]
    env_level_in_keys = ["message_history", "problem", "solution"]
    agent_level_out_keys = ["message", "decision"]
    env_level_out_keys = ["retry_count", "token_limit"]

    @property
    def client(self) -> OpenAI:
        """The OpenAI client to use for interacting with the OpenAI API."""
        return self.settings.openai_client

    def __init__(
        self,
        params: Parameters,
        settings: ExperimentSettings,
        agent_name: str,
        protocol_handler: ProtocolHandler,
    ):
        super().__init__(
            params=params,
            settings=settings,
            agent_name=agent_name,
            protocol_handler=protocol_handler,
        )

        self.model_name = self.agent_params.model_name

        # Load the system prompt template
        prompt_template_traversable = importlib.resources.files(
            "pvg.text_classification.prompt_templates"
        )
        if self.is_verifier:
            template_filename = "verifier_system_prompt.txt"
        elif self.is_prover:
            template_filename = "prover_system_prompt.txt"
        else:
            raise ValueError(f"Unable to get system prompt for agent {agent_name!r}")
        self.system_template = Template(
            prompt_template_traversable.joinpath(template_filename).read_text()
        )

    def forward(self, data: NestedArrayDict) -> NestedArrayDict:
        """Forward pass through the agent policy head.

        Parameters
        ----------
        data : NestedArrayDict
            The input data. A nested dictionary of arrays with keys:

            - "round" (...): The current round number.
            - "message_history" (... round channel): The history of messages exchanged
                between the agents in each channel.
            - "problem" (...): The problem text.
            - "solution" (...): The proposed solution text.
            - "active_mask" (... channel): A boolean array indicating whether the agent
                is active in each channel.

        Returns
        -------
        output_data : NestedArrayDict
            The output of running the agent policy head. This is a nested dictionary of
            arrays with keys:

            - "message" (... channel): The message generated by the agent.
            - "decision" (...): The decision made by the agent. This is either 0
                (reject), 1 (accept) or 2 (no decision).
            - "retry_count" (... channel): The number of retries before the generation
                was successful.
            - "token_limit" (... channel): A boolean array indicating whether the
                generation was stopped because the maximum number of tokens was reached.
        """

        batch_size = data.batch_size

        output_data = NestedArrayDict(
            dict(
                message=np.full(
                    (*batch_size, self.num_visible_message_channels),
                    None,
                    dtype=NumpyStringDtype(),
                ),
                decision=np.full(batch_size, 2, dtype=np.int32),
                retry_count=np.zeros(
                    (*batch_size, self.num_visible_message_channels), dtype=np.int32
                ),
                token_limit=np.zeros(
                    (*batch_size, self.num_visible_message_channels), dtype=np.bool
                ),
            ),
            batch_size=batch_size,
        )

        for batch_id in product(range(dim) for dim in batch_size):
            for channel_id, channel_name in enumerate(
                self.protocol_handler.message_channel_names
            ):
                if not data["active_mask"][*batch_id, channel_id]:
                    continue

                # Generate and store the next message and decision
                next_message, decision, retry_count, warning = (
                    self._generate_next_message_and_decision(
                        message_history=data["message_history"][*batch_id, channel_id],
                        round=data["round"][batch_id],
                        channel_name=channel_name,
                        problem=data["problem"][batch_id],
                        solution=data["solution"][batch_id],
                    )
                )
                output_data["message"][*batch_id, channel_id] = next_message
                output_data["decision"][batch_id] = decision
                output_data["retry_count"][*batch_id, channel_id] = retry_count
                output_data["token_limit"][*batch_id, channel_id] = (
                    warning == "max_tokens"
                )

        return output_data

    def _generate_next_message_and_decision(
        self,
        message_history: list[str],
        round: int,
        channel_name: str,
        problem: str,
        solution: str,
    ) -> tuple[str | None, int, int, Literal["max_tokens"] | None]:
        """Generate the next message and decision for the agent, with retries.

        This method is called when the agent is active in the channel. It builds and
        runs the API request to generate the next action, which can be a message or a
        decision.

        If the there is an error in the generation, this method will retry a number of
        times before raising an exception (detailed below).

        Parameters
        ----------
        message_history : list[str]
            The list of messages in the chat history.
        round : int
            The current round number.
        channel_name : str
            The name of the message channel.
        problem : str
            The problem text.
        solution : str
            The proposed solution text.

        Returns
        -------
        next_message : str | None
            The next message generated by the agent. If the agent makes a decision
            instead of a message, this will be None.
        decision : int
            The decision made by the agent. This is either 0 (reject), 1 (accept) or 2
            (no decision).
        retry_count : int
            The number of retries before the generation was successful.
        warning : Literal["max_tokens"] | None
            If the generation was stopped because the maximum number of tokens was
            reached, this will be "max_tokens". Otherwise, this will be None.

        Raises
        ------
        ContentFilterError
            If the agent's response is blocked by a content filter.
        UnknownFinishReasonError
            If the agent finishes generating for an unknown reason.
        InvalidResponseError
            If the agent generates a response in an invalid format.
        InvalidDecisionError
            If the agent generates an invalid decision (i.e. not accept or reject).
        """

        chat_messages_prompt = self._build_chat_messages_prompt(
            message_history=message_history,
            round_id=round,
            channel_name=channel_name,
            problem=problem,
            solution=solution,
        )

        def try_generation(
            retry: int,
        ) -> tuple[str | None, int, Literal["max_tokens"] | None]:

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=chat_messages_prompt,
                max_tokens=self.agent_params.max_tokens_per_message,
            )

            warning = None

            # Validate the reason for finishing the generation
            finish_reason = completion.choices[0].finish_reason
            if finish_reason == "content_filter":
                raise ContentFilterError(retry)
            elif finish_reason == "length":
                warning = "max_tokens"
            elif finish_reason != "stop":
                raise UnknownFinishReasonError(retry, finish_reason)

            completion_text = completion.choices[0].message.content
            completion_text = completion_text.strip()

            # Match based on the completion text
            if completion_text.startswith("Question: "):
                return completion_text, 2, retry, warning
            elif completion_text.startswith("Decision: "):
                if completion_text == "Decision: accept":
                    return None, 1, retry, warning
                elif completion_text == "Decision: reject":
                    return None, 0, retry, warning
                else:
                    raise InvalidDecisionError(retry)
            else:
                raise InvalidResponseError(retry)

        # Try the generation a number of times
        for retry in range(self.agent_params.num_invalid_generation_retries):
            try:
                return try_generation(retry)
            except GenerationError as e:
                if retry == self.agent_params.num_invalid_generation_retries - 1:
                    raise e

    def _build_chat_messages_prompt(
        self,
        message_history: list[str],
        round_id: int,
        channel_name: str,
        problem: str,
        solution: str,
    ) -> list[dict[str, str]]:
        """Construct the chat history ready to feed to the API.

        Parameters
        ----------
        message_history : list[str]
            The list of messages in the chat history.
        round : int
            The current round number.
        channel_name : str
            The name of the message channel.
        problem : str
            The problem text.
        solution : str
            The proposed solution text.

        Returns
        -------
        chat_messages : list[dict[str, str]]
            The chat messages ready to feed to the API.
        """

        # First add the system prompt
        system_prompt = self.system_template.substitute(
            problem=problem,
            solution=solution,
            max_questions=self.protocol_handler.max_verifier_turns,
        )
        chat_messages = [dict(role="system", content=system_prompt)]

        # Then add the chat history, with messages from the current agent
        for round_id in range(message_history):
            if self.protocol_handler.can_agent_be_active(
                self.agent_name, round_id, channel_name
            ):
                role = "assistant"
            else:
                role = "user"
            chat_messages.append(dict(role=role, content=message_history[round_id]))

        return chat_messages


@register_scenario_class(TC_SCENARIO, RandomWholeAgent)
class TextClassificationRandomAgentPolicyHead(
    TextClassificationWholeAgent, RandomWholeAgent
):
    """Random agent for text classification, yielding random strings."""


@register_scenario_class(TC_SCENARIO, CombinedWhole)
class TextClassificationCombinedWholeAgent(CombinedWhole):
    """Module which combines all agents for text classification."""

    def forward(self, data: NestedArrayDict) -> NestedArrayDict:
        """Run the forward pass through all agent parts and combine the outputs.

        Parameters
        ----------
        data : NestedArrayDict
            The input data. A nested dictionary of arrays with keys:

            - "round" (...): The current round number.
            - "message_history" (... round channel): The history of messages exchanged
                between the agents in each channel.
            - "problem" (...): The problem text.
            - "solution" (...): The proposed solution text.

        Returns
        -------
        data : NestedArrayDict
            The input data updated with the output of running the agent. This has the
            added keys:

            - ("agents", "message") (... agent channel): The message generated by the
              agent.
            - ("agents", "decision") (... agent): The decision made by the agent. This
              is either 0 (reject), 1 (accept) or 2 (no decision).
        """

        # Get the active agent mask for the batch
        active_agent_mask = self.protocol_handler.get_active_agents_mask_from_rounds(
            from_numpy(data["round"])
        )
        active_agent_mask: Bool[NDArray, "... agent channel"] = (
            active_agent_mask.cpu().detach().numpy()
        )

        whole_outputs: dict[str, NestedArrayDict] = {}
        for agent_id, agent_name in enumerate(self.agent_names):

            # Build the input dictionary for the agent
            input_dict = {}
            for key in self.wholes[agent_name].in_keys:

                # For the message history, restrict to the visible channels
                if key == "message_history":
                    input_dict[key] = self._restrict_input_to_visible_channels(
                        agent_name, data[key], "... round channel"
                    )

                # For the active mask, restrict to the agent
                elif key == "active_mask":
                    input_dict[key] = active_agent_mask[..., agent_id, :]

                # Everything else is passed through unchanged
                else:
                    input_dict[key] = data[key]

            input_nad = NestedArrayDict(input_dict, batch_size=data.batch_size)

            # Run the agent
            whole_outputs[agent_name] = self.wholes[agent_name](input_nad)

        agents_update = {}

        # Stack the outputs
        agents_update["message"] = rearrange(
            [whole_outputs[agent_name]["message"] for agent_name in self.agent_names],
            "agent ... channel -> ... agent channel",
        )
        agents_update["decision"] = rearrange(
            [whole_outputs[agent_name]["decision"] for agent_name in self.agent_names],
            "agent ... -> ... agent",
        )

        return data.update(dict(agents=agents_update))
