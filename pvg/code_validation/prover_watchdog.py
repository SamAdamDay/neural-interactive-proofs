"""Use a language model to evaluate how well the prover(s) are conforming to their roles
"""

import importlib.resources
from string import Template
from itertools import product
from typing import Literal, Optional
from random import randrange

from numpy.typing import NDArray
import numpy as np
from numpy import ma

from openai import OpenAI

from tqdm import tqdm

from pvg.parameters import Parameters, InteractionProtocolType
from pvg.protocols import ProtocolHandler
from pvg.experiment_settings import ExperimentSettings
from pvg.utils.string import random_string
from pvg.utils.nested_array_dict import NestedArrayDict
from pvg.utils.env import load_env_once
from pvg.utils.api import (
    GenerationError,
    ContentFilterError,
    UnknownFinishReasonError,
    InvalidResponseError,
)


class CodeValidationProverWatchdog:
    """A watchdog to evaluate how well the prover(s) are conforming to their roles.

    The watchdog uses a language model to evaluate the message histories.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The instance-specific settings of the experiment, like device, logging, etc.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    """

    @property
    def client(self) -> OpenAI:
        """The OpenAI client to use for interacting with the OpenAI API."""
        if self._openai_client is None:
            self._openai_client = OpenAI()
        return self._openai_client

    @property
    def model_name(self) -> str:
        """The name of the language model to use as the watchdog."""
        return self.params.ei.prover_watchdog_model_name

    def __init__(
        self,
        params: Parameters,
        settings: ExperimentSettings,
        protocol_handler: ProtocolHandler,
    ):
        self.params = params
        self.settings = settings
        self.protocol_handler = protocol_handler

        if params.interaction_protocol == InteractionProtocolType.MERLIN_ARTHUR:
            raise NotImplementedError(
                "Prover watchdog is not implemented for the Merlin-Arthur protocol."
            )

        # Make sure the environment variables are loaded, so that we can access the
        # OpenAI API key
        load_env_once()

        # Load the system prompt template
        prompt_template_traversable = importlib.resources.files(
            "pvg.code_validation.prompt_templates"
        )
        self.system_template = Template(
            prompt_template_traversable.joinpath(
                "prover_watchdog_system_prompt.txt"
            ).read_text()
        )

        self._openai_client: Optional[OpenAI] = None

    def forward(
        self, rollouts: NestedArrayDict, use_tqdm: bool = False
    ) -> dict[tuple[str, str], ma.MaskedArray]:
        """Evaluate how well the prover(s) are conforming to their roles.

        Evaluations are either 0 or 1, where 1 indicates that the prover is conforming
        to its role.

        Parameters
        ----------
        rollouts : NestedArrayDict
            The sampled rollouts. A nested dictionary of arrays with keys:

            - "round" (... round): The current round number.
            - "message_history" (... round round channel): The history of messages
              exchanged between the agents in each channel.

        Returns
        -------
        watchdog_evaluations : dict[tuple[str, str], NDArray]
            The evaluations of the watchdog. A dictionary indexed by agent name and
            channel name, where `watchdog_evaluations[agent_name, channel_name]` is a
            0-1 array of evaluations of shape (...)
        """

        batch_size = rollouts.batch_size[:-1]

        watchdog_evaluations: dict[tuple[str, str], ma.MaskedArray] = {}

        for agent_name, channel_name in self.protocol_handler.agent_channel_visibility:
            if agent_name not in self.protocol_handler.prover_names:
                continue

            channel_id = self.protocol_handler.message_channel_names.index(channel_name)

            watchdog_evaluation = ma.array(
                np.empty(batch_size, dtype=np.int8),
                mask=np.zeros(batch_size, dtype=bool),
            )

            iterator = product(*(range(dim) for dim in batch_size))
            if use_tqdm:
                total = 1
                for dim in batch_size:
                    total *= dim
                iterator = tqdm(
                    iterator, desc=f"Watchdog evaluating {agent_name!r}", total=total
                )
            for batch_id in iterator:

                last_message_history = rollouts["message_history"][batch_id][
                    -1, :, channel_id
                ]

                evaluation = self._generate_evaluation(
                    last_message_history, channel_name
                )

                # Mask the evaluation if it could not be generated, otherwise store it
                if evaluation is None:
                    watchdog_evaluation.mask[batch_id] = True
                else:
                    watchdog_evaluation[batch_id] = evaluation

            watchdog_evaluations[agent_name, channel_name] = watchdog_evaluation

        return watchdog_evaluations

    def _generate_evaluation(
        self, message_history: NDArray, channel_name: str
    ) -> int | None:
        """Generate an evaluation of how well the prover is conforming to its role.

        Parameters
        ----------
        message_history : NDArray
            The history of messages exchanged between the agents in the channel.
        channel_name : str
            The name of the message channel.

        Returns
        -------
        evaluation : int | None
            The evaluation of the prover. 1 indicates that the prover is conforming to
            its role. 0 indicates that the prover is not conforming to its role. None
            indicates that the evaluation could not be generated.
        """

        chat_messages_prompt = self._build_chat_messages_prompt(
            message_history, channel_name
        )

        def try_generation(
            retry: int,
        ) -> int:

            completion_text, finish_reason = self._make_generation_api_call(
                chat_messages_prompt
            )

            # Validate the reason for finishing the generation
            if finish_reason == "content_filter":
                raise ContentFilterError(num_retries=retry)
            elif finish_reason not in ["stop", "length"]:
                raise UnknownFinishReasonError(num_retries=retry, reason=finish_reason)

            completion_text = completion_text.strip()

            # Match based on the completion text
            if completion_text.lower().startswith("Yes"):
                return 0
            elif completion_text.lower().startswith("No"):
                return 1
            else:
                raise InvalidResponseError(
                    num_retries=retry, response_text=completion_text
                )

        # Try the generation a number of times
        num_generation_errors = 0
        while True:
            try:
                return try_generation(num_generation_errors)

            # Retry if there is a generation error
            except GenerationError:
                num_generation_errors += 1
                if (
                    num_generation_errors
                    == self.params.ei.prover_watchdog_num_invalid_generation_retries
                ):
                    return None

    def _make_generation_api_call(
        self,
        chat_messages_prompt: list[dict[Literal["role", "content", "user"], str]],
    ) -> tuple[str, Literal["stop", "content_filter", "length"]]:
        """Call the OpenAI API to generate the evaluation.

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

        if self.params.ei.prover_watchdog_use_dummy_api:
            output_type = randrange(2)
            if output_type == 0:
                return "Yes", "stop"
            else:
                return "No", "stop"
        else:
            completion = self.client.chat.completions.create(
                model=self.model_name, messages=chat_messages_prompt
            )
            choice = completion.choices[0]
            return choice.message.content, choice.finish_reason

    def _build_chat_messages_prompt(
        self,
        message_history: NDArray,
        channel_name: str,
    ) -> list[dict[str, str]]:
        """Construct the chat history ready to feed to the API.

        Parameters
        ----------
        message_history : NDArray
            The list of messages in the chat history.
        channel_name : str
            The name of the message channel.

        Returns
        -------
        chat_messages : list[dict[str, str]]
            The chat messages ready to feed to the API.
        """

        # First add the system prompt
        system_prompt = self.system_template.substitute()
        chat_messages = [dict(role="system", content=system_prompt)]

        # Then add the chat history, with messages from the current agent
        for round_id, message in enumerate(message_history):
            if message is None:
                break

            # Add the message with the name of the agent that can be active
            # TODO: This will need to be modified for the Merlin-Arthur protocol
            for agent_name in self.protocol_handler.agent_names:
                if self.protocol_handler.can_agent_be_active(
                    agent_name, round_id, channel_name
                ):
                    chat_messages.append(
                        dict(role="user", name=agent_name, content=str(message))
                    )
                    break

        chat_messages.append(
            dict(
                role="user",
                name="supervisor",
                content="Does the prover consistently try to convince the verifier "
                "that the solution is valid? Answer 'Yes' or 'No'.",
            )
        )

        return chat_messages
