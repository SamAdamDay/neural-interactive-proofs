"""Code validation agent components.

To integrate easily into the codebase, the code validation agents a split into
three parts: the body, the policy head, and the value head. However, for text
classification, the split is not necessary, so only the policy head actually does
anything. The body and value head return the input data unchanged.
"""

from abc import ABC, abstractmethod
from typing import Optional, Literal, ClassVar, Any
from functools import cached_property
import importlib.resources
from string import Template
from itertools import product
from dataclasses import dataclass
from random import randrange
from tempfile import TemporaryDirectory
from pathlib import Path
import json
from time import sleep

from torch import from_numpy

import numpy as np
from numpy.typing import NDArray

from einops import rearrange

from jaxtyping import Float, Int, Bool

from openai import OpenAI, APITimeoutError, APIStatusError, RateLimitError
from openai.types.fine_tuning import FineTuningJob as OpenAIFineTuningJob

from pvg.scenario_base import (
    WholeAgent,
    PureTextWholeAgent,
    RandomWholeAgent,
    CombinedWhole,
    Agent,
    AgentState,
)
from pvg.parameters import (
    Parameters,
    CodeValidationAgentParameters,
    RandomAgentParameters,
    ScenarioType,
    InteractionProtocolType,
)
from pvg.experiment_settings import ExperimentSettings
from pvg.factory import register_scenario_class
from pvg.protocols import ProtocolHandler
from pvg.utils.nested_array_dict import NestedArrayDict
from pvg.utils.types import NumpyStringDtype
from pvg.utils.env import load_env_once
from pvg.utils.string import random_string
from pvg.constants import WANDB_OPENAI_FINETUNE_PROJECT

CV_SCENARIO = ScenarioType.CODE_VALIDATION


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

    def __init__(self, num_retries: int, response_text: str):
        self.response_text = response_text
        self.num_retries = num_retries
        super(GenerationError, self).__init__(
            f"Invalid generation after {num_retries} retries. Response: "
            "{response_text!r}"
        )


class InvalidDecisionError(InvalidResponseError):
    """Raised when the agent's decision is invalid (i.e. not accept or reject)"""


@dataclass
class OpenAiAgentState(AgentState):
    """The state of an OpenAI agent.

    Attributes
    ----------
    fine_tune_job_id : Optional[str]
        The ID of the OpenAI API fine-tune job.
    fine_tuned_model_name : Optional[str]
        The name of the most recently fine-tuned model.
    """

    fine_tune_job_id: Optional[str] = None
    fine_tuned_model_name: Optional[str] = None


@register_scenario_class(CV_SCENARIO, WholeAgent, {"model_provider": "OpenAI"})
class OpenAiWholeAgent(PureTextWholeAgent):
    """The whole agent for code validation, using OpenAI's API."""

    agent_params: CodeValidationAgentParameters

    agent_level_in_keys = ["active_mask"]
    env_level_in_keys = ["message_history", "question", "solution", "round"]
    agent_level_out_keys = ["message", "decision", "retry_count", "token_limit"]

    @property
    def client(self) -> OpenAI:
        """The OpenAI client to use for interacting with the OpenAI API."""
        if self._openai_client is None:
            self._openai_client = OpenAI()
        return self._openai_client

    @property
    def base_model_name(self) -> str:
        """The base OpenAI model name, before any fine-tuning."""
        return self.agent_params.model_name

    @property
    def model_name(self) -> str:
        """The OpenAI model name, including any fine-tuning."""
        if self.fine_tuned_model_name is not None:
            return self.fine_tuned_model_name
        else:
            return self.base_model_name

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

        # Make sure the environment variables are loaded, so that we can access the
        # OpenAI API key
        load_env_once()

        # Load the system prompt template
        prompt_template_traversable = importlib.resources.files(
            "pvg.code_validation.prompt_templates"
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

        self._openai_client: Optional[OpenAI] = None
        self.fine_tune_job_id: Optional[str] = None
        self.fine_tuned_model_name: Optional[str] = None

    def forward(self, data: NestedArrayDict) -> NestedArrayDict:
        """Forward pass through the agent policy head.

        Parameters
        ----------
        data : NestedArrayDict
            The input data. A nested dictionary of arrays with keys:

            - "round" (...): The current round number.
            - "message_history" (... round channel): The history of messages exchanged
                between the agents in each channel.
            - "question" (...): The problem text.
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
                    dtype=NumpyStringDtype,
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

        for batch_id in product(*(range(dim) for dim in batch_size)):
            for channel_id, channel_name in enumerate(
                self.protocol_handler.message_channel_names
            ):
                if not data["active_mask"][(*batch_id, channel_id)]:
                    continue

                # Generate and store the next message and decision
                next_message, decision, retry_count, warning = (
                    self._generate_next_message_and_decision(
                        message_history=data["message_history"][batch_id][
                            :, channel_id
                        ],
                        round=data["round"][batch_id],
                        channel_name=channel_name,
                        question=data["question"][batch_id],
                        solution=data["solution"][batch_id],
                    )
                )
                output_data["message"][(*batch_id, channel_id)] = next_message
                output_data["decision"][batch_id] = decision
                output_data["retry_count"][(*batch_id, channel_id)] = retry_count
                output_data["token_limit"][(*batch_id, channel_id)] = (
                    warning == "max_tokens"
                )

        return output_data

    def get_state_dict(self) -> dict:
        """Get the state dictionary of the agent.

        Returns
        -------
        state_dict : dict
            The state dictionary of the agent.
        """

        return dict(
            fine_tune_job_id=self.fine_tune_job_id,
            fine_tuned_model_name=self.fine_tuned_model_name,
        )

    def set_state(self, checkpoint: OpenAiAgentState | dict[str, Any]):

        if isinstance(checkpoint, dict):
            checkpoint = OpenAiAgentState(**checkpoint)

        self.fine_tune_job_id = checkpoint.fine_tune_job_id
        self.fine_tuned_model_name = checkpoint.fine_tuned_model_name

    def create_fine_tune_job(self, rollouts: NestedArrayDict):
        """Create a fine-tune job for the agent.

        This method generates a dataset of examples ready to pass to the fine-tune API.

        Parameters
        ----------
        rollouts : NestedArrayDict
            The sampled rollouts. A nested dictionary of arrays with keys:

            - "round" (...): The current round number.
            - "message_history" (... round channel): The history of messages exchanged
                between the agents in each channel.
            - "question" (...): The problem text.
            - "solution" (...): The proposed solution text.

        """

        fine_tune_dataset = self._build_fine_tune_dataset(rollouts)

        if self.agent_params.use_dummy_api:
            self.fine_tune_job_id = "dummy_job_id"
            return

        with TemporaryDirectory() as temp_dir:

            # Write the dataset to a temporary file
            file_path = Path(temp_dir, "fine_tune_dataset.jsonl")
            with open(file_path, "w") as f:
                for example in fine_tune_dataset:
                    f.write(json.dumps(example) + "\n")

            # Upload the file to OpenAI
            uploaded_file = self.client.files.create(
                file=open(file_path, "rb"), purpose="fine-tune"
            )

        file_id = uploaded_file.id

        if self.agent_params.fine_tune_from_scratch:
            model_name = self.base_model_name
        else:
            model_name = self.model_name

        # Create the fine-tune job
        while True:
            try:
                job = self.client.fine_tuning.jobs.create(
                    model=model_name,
                    training_file=file_id,
                    integrations=[
                        {
                            "type": "wandb",
                            "wandb": {"project": WANDB_OPENAI_FINETUNE_PROJECT},
                        }
                    ],
                )

            # If we are day rate limited, sleep for an hour and try again
            except RateLimitError as e:
                if e.code == "daily_rate_limit_exceeded":
                    sleep(60 * 60)
                    continue
                else:
                    raise e

            break

        self.fine_tune_job_id = job.id

    def get_fine_tune_job_status(
        self,
    ) -> Literal["pending", "running", "succeeded", "failed", "cancelled"]:
        """Get the status of the fine-tune job"""

        if self.agent_params.use_dummy_api:
            return "succeeded"

        status = self._get_fine_tune_job().status

        if status in ["validating_files", "queued"]:
            return "pending"
        elif status in ["running", "succeeded", "failed", "cancelled"]:
            return status
        else:
            raise ValueError(f"Unknown OpenAI fine-tune job status {status!r}")

    def get_fine_tune_job_error_repr(self) -> str:
        """Get a string representation of the error for the fine-tune job"""

        if self.agent_params.use_dummy_api:
            raise ValueError("Cannot get error for dummy API")

        error = self._get_fine_tune_job().error

        output = f"Code: {error.code}. Message: {error.message}."
        if error.param is not None:
            output += f" Parameter: {error.param}."

        if isinstance(error, APIStatusError):
            output += f" Headers: {error.response.headers}."

        return output

    def switch_to_next_model(self):
        """Switch to the next model after fine-tuning"""

        if self.fine_tune_job_id is None:
            raise ValueError("Fine-tune job ID not set")

        if self.agent_params.use_dummy_api:
            self.fine_tuned_model_name = "dummy_model_name"
            return

        job = self._get_fine_tune_job()

        if job.status != "succeeded":
            raise ValueError(
                f"Cannot switch to next model: fine-tune job status is {job.status!r}"
            )

        if job.fine_tuned_model is None:
            raise ValueError("Fine-tuned model name not set in fine-tune job")

        self.fine_tuned_model_name = job.fine_tuned_model

    def _build_fine_tune_dataset(
        self, rollouts: NestedArrayDict
    ) -> list[dict[Literal["messages"], list[dict[Literal["role", "content"], str]]]]:
        """Build the dataset for fine-tuning the agent given sampled rollouts

        This method generates a dataset of examples ready to pass to the fine-tune API.

        Parameters
        ----------
        rollouts : NestedArrayDict
            The sampled rollouts. A nested dictionary of arrays with keys:

            - "round" (...): The current round number.
            - "message_history" (... round channel): The history of messages exchanged
                between the agents in each channel.
            - "question" (...): The problem text.
            - "solution" (...): The proposed solution text.

        Returns
        -------
        fine_tune_dataset : list[dict[str, list[dict[str, str]]]]
            The dataset for fine-tuning the agent. This is a list of examples, where
            each example is a dictionary with key "messages", whose value is a list of
            dictionaries with keys "role" and "content".
        """

        batch_size = rollouts.batch_size[:-1]

        fine_tune_dataset: list[
            dict[Literal["messages"], list[dict[Literal["role", "content"], str]]]
        ] = []

        for batch_id in product(*(range(dim) for dim in batch_size)):

            # Get the environment state for the final round
            rollout_final_state = rollouts[batch_id][-1]

            for channel_id, channel_name in enumerate(
                self.protocol_handler.message_channel_names
            ):

                if not self.protocol_handler.can_agent_see_channel(
                    self.agent_name, channel_name
                ):
                    continue

                chat_messages_prompt = self._build_chat_messages_prompt(
                    message_history=rollout_final_state["message_history"][
                        :, channel_id
                    ],
                    round_id=rollout_final_state["round"],
                    channel_name=channel_name,
                    question=rollout_final_state["question"],
                    solution=rollout_final_state["solution"],
                    ensure_last_message_is_assistant=True,
                )

                fine_tune_dataset.append(dict(messages=chat_messages_prompt))

        return fine_tune_dataset

    def _get_fine_tune_job(self) -> OpenAIFineTuningJob:
        """Get the fine-tune job from the OpenAI API"""

        if self.fine_tune_job_id is None:
            raise ValueError("Fine-tune job ID not set")

        return self.client.fine_tuning.jobs.retrieve(self.fine_tune_job_id)

    def _generate_next_message_and_decision(
        self,
        message_history: NDArray,
        round: int,
        channel_name: str,
        question: str,
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
        message_history : NDArray
            The array of messages in the chat history.
        round : int
            The current round number.
        channel_name : str
            The name of the message channel.
        question : str
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
            question=question,
            solution=solution,
        )

        def try_generation(
            retry: int,
        ) -> tuple[str | None, int, Literal["max_tokens"] | None]:

            completion_text, finish_reason = self._make_generation_api_call(
                chat_messages_prompt
            )

            warning = None

            # Validate the reason for finishing the generation
            if finish_reason == "content_filter":
                raise ContentFilterError(num_retries=retry)
            elif finish_reason == "length":
                warning = "max_tokens"
            elif finish_reason != "stop":
                raise UnknownFinishReasonError(num_retries=retry, reason=finish_reason)

            completion_text = completion_text.strip()

            # Match based on the completion text
            if completion_text.startswith("Question:") or completion_text.startswith(
                "Answer:"
            ):
                return completion_text, 2, retry, warning
            elif completion_text.startswith("Decision:"):
                if completion_text == "Decision: accept":
                    return completion_text, 1, retry, warning
                elif completion_text == "Decision: reject":
                    return completion_text, 0, retry, warning
                else:
                    raise InvalidDecisionError(
                        num_retries=retry, response_text=completion_text
                    )
            else:
                raise InvalidResponseError(
                    num_retries=retry, response_text=completion_text
                )

        # Try the generation a number of times
        num_generation_errors = 0
        num_timeouts = 0
        while True:
            try:
                return try_generation(num_generation_errors)

            # Retry if there is a generation error
            except GenerationError as e:
                num_generation_errors += 1
                if (
                    num_generation_errors
                    == self.agent_params.num_invalid_generation_retries
                ):
                    raise e

            # Retry if there is a timeout, but wait a bit first
            except APITimeoutError as e:
                num_timeouts += 1
                if num_timeouts == self.settings.num_api_generation_timeouts:
                    raise e
                sleep(10)

    def _make_generation_api_call(
        self,
        chat_messages_prompt: list[dict[Literal["role", "content"], str]],
    ) -> tuple[str, Literal["stop", "content_filter", "length"]]:
        """Call the OpenAI API to generate the next message.

        Parameters
        ----------
        chat_messages_prompt : list[dict[str, str]]
            The message history to feed to the API. A list of dicts with keys "role"
            and "content".

        Returns
        -------
        completion_text : str
            The text of the completion generated by the API.
        finish_reason : Literal["stop", "content_filter", "length"]
            The reason for finishing the generation.
        """

        if self.agent_params.use_dummy_api:
            if chat_messages_prompt[-1]["content"].startswith("Question: "):
                return f"Answer: {random_string(20)}", "stop"
            else:
                output_type = randrange(3)
                if output_type == 0:
                    return "Decision: reject", "stop"
                elif output_type == 1:
                    return "Decision: accept", "stop"
                else:
                    return f"Question: {random_string(20)}", "stop"
        else:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=chat_messages_prompt,
                max_tokens=self.agent_params.max_tokens_per_message,
                temperature=self.agent_params.temperature,
                top_p=self.agent_params.top_p,
            )
            choice = completion.choices[0]
            return choice.message.content, choice.finish_reason

    def _build_chat_messages_prompt(
        self,
        message_history: NDArray,
        round_id: int,
        channel_name: str,
        question: str,
        solution: str,
        ensure_last_message_is_assistant: bool = False,
    ) -> list[dict[str, str]]:
        """Construct the chat history ready to feed to the API.

        Parameters
        ----------
        message_history : NDArray
            The list of messages in the chat history.
        round : int
            The current round number.
        channel_name : str
            The name of the message channel.
        question : str
            The problem text.
        solution : str
            The proposed solution text.
        ensure_last_message_is_assistant : bool, default=False
            Whether to ensure the last message is from the assistant, by removing
            messages from the user.

        Returns
        -------
        chat_messages : list[dict[str, str]]
            The chat messages ready to feed to the API.
        """

        # First add the system prompt
        system_prompt = self.system_template.substitute(
            question=question,
            solution=solution,
            max_questions=self.protocol_handler.max_verifier_turns - 1,
        )
        chat_messages = [dict(role="system", content=system_prompt)]

        # Then add the chat history, with messages from the current agent
        for round_id, message in enumerate(message_history):
            if message is None:
                break
            if self.protocol_handler.can_agent_be_active(
                self.agent_name, round_id, channel_name
            ):
                role = "assistant"
            else:
                role = "user"
            chat_messages.append(dict(role=role, content=str(message)))

        if ensure_last_message_is_assistant:
            while len(chat_messages) > 0 and chat_messages[-1]["role"] != "assistant":
                chat_messages.pop()

        return chat_messages

    def __getstate__(self) -> dict[str, any]:
        """Get the state of the object for pickling.

        We don't pickle the OpenAI client, as it is not picklable.

        Returns
        -------
        state : dict[str, any]
            The state of the object.
        """

        state = self.__dict__
        state["_openai_client"] = None

        return state


@register_scenario_class(CV_SCENARIO, RandomWholeAgent)
class CodeValidationRandomAgentPolicyHead(PureTextWholeAgent, RandomWholeAgent):
    """Random agent for code validation, yielding random strings."""


@register_scenario_class(CV_SCENARIO, CombinedWhole)
class CodeValidationCombinedWholeAgent(CombinedWhole):
    """Module which combines all agents for code validation."""

    def forward(self, data: NestedArrayDict) -> NestedArrayDict:
        """Run the forward pass through all agent parts and combine the outputs.

        Parameters
        ----------
        data : NestedArrayDict
            The input data. A nested dictionary of arrays with keys:

            - "round" (...): The current round number.
            - "message_history" (... round channel): The history of messages exchanged
                between the agents in each channel.
            - "question" (...): The problem text.
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
            - ("agents", "retry_count") (... agent channel): The number of retries
              before the generation was successful for each agent.
            - ("agents", "token_limit") (... agent channel): A boolean array indicating
              whether the generation was stopped because the maximum number of tokens
              was reached for each agent.
        """

        # Get the active agent mask for the batch
        active_agent_mask = self.protocol_handler.get_active_agents_mask_from_rounds(
            from_numpy(data["round"])
        )
        active_agent_mask: Bool[np.ndarray, "... agent channel"] = (
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
        agents_update["retry_count"] = rearrange(
            [
                whole_outputs[agent_name]["retry_count"]
                for agent_name in self.agent_names
            ],
            "agent ... channel -> ... agent channel",
        )
        agents_update["token_limit"] = rearrange(
            [
                whole_outputs[agent_name]["token_limit"]
                for agent_name in self.agent_names
            ],
            "agent ... channel -> ... agent channel",
        )

        data = data.update(dict(agents=agents_update))

        return data


@register_scenario_class(CV_SCENARIO, Agent)
@dataclass
class CodeValidationAgent(Agent):

    agent_params: ClassVar[CodeValidationAgentParameters | RandomAgentParameters]

    agent_state_class: ClassVar[type[OpenAiAgentState]] = OpenAiAgentState
