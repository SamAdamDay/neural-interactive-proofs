"""Use a language model to do analysis on rollouts."""

import importlib.resources
from string import Template
from itertools import product
from typing import Literal, Optional, Iterator, ClassVar
from random import randrange
from abc import ABC, abstractmethod
from numbers import Real

from numpy.typing import NDArray
import numpy as np
from numpy import ma

from openai import OpenAI

from tqdm import tqdm

from nip.parameters import HyperParameters
from nip.protocols import ProtocolHandler
from nip.experiment_settings import ExperimentSettings
from nip.scenario_base.rollout_analysis import (
    PureTextRolloutAnalyser,
    register_rollout_analyser,
)
from nip.utils.nested_array_dict import NestedArrayDict
from nip.utils.env import load_env_once
from nip.utils.api import (
    GenerationError,
    ContentFilterError,
    UnknownFinishReasonError,
    InvalidResponseError,
)


class CodeValidationRolloutAnalyser(PureTextRolloutAnalyser, ABC):
    """Base class for analysing code validation rollouts."""

    score_dtype: ClassVar[np.dtype] = np.int8
    """The data type of the scores returned by the rollout analyser."""

    max_generation_retries: ClassVar[int] = 3
    """The number of times to retry if the model generates an invalid response."""

    @property
    @abstractmethod
    def system_prompt_template_filename(self) -> str:
        """The filename of the system prompt template."""

    @property
    @abstractmethod
    def supervisor_question(self) -> Template:
        """The question asked by the supervisor agent."""

    @property
    def client(self) -> OpenAI:
        """The OpenAI client to use for interacting with the OpenAI API."""
        if self._openai_client is None:
            self._openai_client = OpenAI()
        return self._openai_client

    def __init__(
        self,
        hyper_params: HyperParameters,
        settings: ExperimentSettings,
        protocol_handler: ProtocolHandler,
        model_name: str,
        *,
        use_dummy_api: bool = False,
    ):
        super().__init__(
            hyper_params=hyper_params,
            settings=settings,
            protocol_handler=protocol_handler,
            model_name=model_name,
            use_dummy_api=use_dummy_api,
        )

        if hyper_params.interaction_protocol == "merlin_arthur":
            raise NotImplementedError(
                "Rollout analysers are not implemented for the Merlin-Arthur Classifier (MAC) protocol."
            )

        # Make sure the environment variables are loaded, so that we can access the
        # OpenAI API key
        load_env_once()

        # Load the system prompt template
        prompt_template_traversable = importlib.resources.files(
            "nip.code_validation.prompt_templates.rollout_analysis"
        )
        self.system_template = Template(
            prompt_template_traversable.joinpath(
                self.system_prompt_template_filename
            ).read_text()
        )

        self._openai_client: Optional[OpenAI] = None

    def forward(
        self, rollouts: NestedArrayDict, use_tqdm: bool = False
    ) -> dict[tuple[str, str], ma.MaskedArray]:
        """Score the rollouts by running a language model on the message history.

        Parameters
        ----------
        rollouts : NestedArrayDict
            The sampled rollouts. A nested dictionary of arrays with keys:

            - "round" (... round): The current round number.
            - "message_history" (... round round channel): The history of messages
              exchanged between the agents in each channel.
            - "question" (... round): The coding question.
            - "solution" (... round): The proposed solution to the coding question.
            - "prover_stance" (...): When randomizing the prover stance, the verdict
              that the prover is arguing for, where 0 means "reject" and 1 means
              "accept".


        Returns
        -------
        evaluations : dict[tuple[str, str], ma.MaskedArray]
            The evaluations. A dictionary indexed by agent name and channel name, where
            ``evaluations[agent_name, channel_name]`` is an array of scores of shape
            (...)
        """

        batch_size = rollouts.batch_size[:-1]

        evaluations: dict[tuple[str, str], ma.MaskedArray] = {}

        for agent_name, channel_name in self.relevant_agents_and_channels():

            channel_id = self.protocol_handler.message_channel_names.index(channel_name)

            evaluation_array = ma.array(
                np.empty(batch_size, dtype=self.score_dtype),
                mask=np.zeros(batch_size, dtype=bool),
            )

            iterator = product(*(range(dim) for dim in batch_size))
            if use_tqdm:
                total = 1
                for dim in batch_size:
                    total *= dim
                iterator = tqdm(
                    iterator,
                    desc=f"Analysis({self.name}, {agent_name}, {channel_name})",
                    total=total,
                )
            for batch_id in iterator:

                last_message_history = rollouts["message_history"][batch_id][
                    -1, :, channel_id
                ]
                last_message_agent_id = rollouts["message_agent_id"][batch_id][
                    -1, :, channel_id
                ]

                evaluation = self._generate_evaluation(
                    last_message_history,
                    last_message_agent_id,
                    agent_name=agent_name,
                    channel_name=channel_name,
                    question=rollouts["question"][batch_id][-1],
                    solution=rollouts["solution"][batch_id][-1],
                )

                # Mask the evaluation if it could not be generated, otherwise store it
                if evaluation is None:
                    evaluation_array.mask[batch_id] = True
                else:
                    evaluation_array[batch_id] = evaluation

            evaluations[agent_name, channel_name] = evaluation_array

        return evaluations

    @abstractmethod
    def _generate_dummy_response(self) -> str:
        """Generate a dummy response for the rollout analyser.

        This is used when the use_dummy_api flag is set to True, to generate a dummy
        response to the API call.

        Returns
        -------
        response : str
            The dummy response.
        """

    @abstractmethod
    def _get_score_from_response(self, response: str) -> Real:
        """Convert the language model response to a score.

        Parameters
        ----------
        response : str
            The response from the language model.

        Returns
        -------
        score : Real
            The response converted to a score.

        Raises
        ------
        InvalidResponseError
            If the response is not a valid response.
        """

    def _generate_evaluation(
        self,
        message_history: NDArray,
        message_agent_id: NDArray,
        agent_name: str,
        channel_name: str,
        question: str,
        solution: str,
    ) -> Real | None:
        """Generate an evaluation for a rollout.

        Parameters
        ----------
        message_history : NDArray
            The history of messages exchanged between the agents in the channel.
        message_agent_id : NDArray
            The agent ID of the agent which sent each message in the message history.
        agent_name : str
            The name of the agent being evaluated.
        channel_name : str
            The name of the message channel.
        question : str
            The coding question.
        solution : str
            The proposed solution to the coding question.

        Returns
        -------
        evaluation : Real | None
            The evaluation. None indicates that the evaluation could not be generated.
        """

        chat_messages_prompt = self._build_chat_messages_prompt(
            message_history,
            message_agent_id,
            agent_name=agent_name,
            channel_name=channel_name,
            question=question,
            solution=solution,
        )

        def try_generation(
            retry: int,
        ) -> Real:

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
            return self._get_score_from_response(completion_text)

        # Try the generation a number of times
        num_generation_errors = 0
        while True:
            try:
                return try_generation(num_generation_errors)

            # Retry if there is a generation error
            except GenerationError:
                num_generation_errors += 1
                if num_generation_errors == self.max_generation_retries:
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

        if self.use_dummy_api:
            return self._generate_dummy_response(), "stop"
        else:
            completion = self.client.chat.completions.create(
                model=self.model_name, messages=chat_messages_prompt
            )
            choice = completion.choices[0]
            return choice.message.content, choice.finish_reason

    def _build_chat_messages_prompt(
        self,
        message_history: NDArray,
        message_agent_id: NDArray,
        agent_name: str,
        channel_name: str,
        question: str,
        solution: str,
    ) -> list[dict[str, str]]:
        """Construct the chat history ready to feed to the API.

        Parameters
        ----------
        message_history : NDArray
            The list of messages in the chat history.
        message_agent_id : NDArray
            The agent ID of the agent which sent each message in the message history.
        agent_name : str
            The name of the agent being evaluated.
        channel_name : str
            The name of the message channel.
        question : str
            The coding question.
        solution : str
            The proposed solution to the coding question.

        Returns
        -------
        chat_messages : list[dict[str, str]]
            The chat messages ready to feed to the API.
        """

        template_mapping = dict(
            agent_name=agent_name,
            channel_name=channel_name,
            question=question,
            solution=solution,
        )

        # First add the system prompt
        system_prompt = self.system_template.substitute(template_mapping)
        chat_messages = [dict(role="system", content=system_prompt)]

        # Then add the chat history, with messages from the current agent
        for round_id, message in enumerate(message_history):
            if message is None:
                break

            message_agent_name = self.protocol_handler.agent_names[
                message_agent_id[round_id]
            ]

            chat_messages.append(
                dict(role="user", name=message_agent_name, content=str(message))
            )

        chat_messages.append(
            dict(
                role="user",
                name="supervisor",
                content=self.supervisor_question.substitute(template_mapping),
            )
        )

        return chat_messages


class BinaryRolloutAnalyser(CodeValidationRolloutAnalyser, ABC):
    """Base class for rollout analyser which yield a binary classification.

    Each rollout is analysed by a language model to generate a binary classification.
    This is done by first giving the system prompt, then the message history, and
    finally asking a question, which is done by the "supervisor" agent.
    """

    def _generate_dummy_response(self):
        """Generate a dummy response for the rollout analyser.

        This is used when the use_dummy_api flag is set to True, to generate a dummy
        response to the API call.

        Returns
        -------
        response : str
            The dummy response.
        """
        return "Yes" if randrange(2) == 0 else "No"

    def _get_score_from_response(self, response: str) -> Literal[0, 1]:
        """Get the binary classification from language model response.

        Parameters
        ----------
        response : str
            The response from the language model.

        Returns
        -------
        classification : Literal[0, 1]
            The binary classification.

        Raises
        ------
        InvalidResponseError
            If the response is not a valid response.
        """
        response = response.strip().lower()

        if response.startswith("yes"):
            return 1
        elif response.startswith("no"):
            return 0
        else:
            raise InvalidResponseError(response_text=response)


class OutOfTenRolloutAnalyser(CodeValidationRolloutAnalyser, ABC):
    """Base class for rollout analyser which score from 0 to 10.

    Each rollout is analysed by a language model to generate a score out of 10.
    This is done by first giving the system prompt, then the message history, and
    finally asking a question, which is done by the "supervisor" agent.
    """

    text_to_int = {
        "zero": 0,
        "naught": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    def _generate_dummy_response(self):
        """Generate a dummy response for the rollout analyser.

        This is used when the use_dummy_api flag is set to True, to generate a dummy
        response to the API call.

        Returns
        -------
        response : str
            The dummy response.
        """
        return str(randrange(11))

    def _get_score_from_response(self, response: str) -> int:
        """Get the score out of ten from language model response.

        Parameters
        ----------
        response : str
            The response from the language model.

        Returns
        -------
        classification : int
            The score out of ten. This is an integer between 0 and 10.

        Raises
        ------
        InvalidResponseError
            If the response is not a valid response.
        """
        response = response.strip().lower()

        try:
            as_int = int(response)
        except ValueError:
            if response in self.text_to_int:
                as_int = self.text_to_int[response]
            else:
                raise InvalidResponseError(response_text=response)
        if as_int < 0 or as_int > 10:
            raise InvalidResponseError(response_text=response)
        return as_int


class ProverAnalyserMixin:
    """Mixin class for analysing provers."""

    protocol_handler: ProtocolHandler

    def relevant_agents_and_channels(self) -> Iterator[tuple[str, str]]:
        """Get the relevant agents and channels for the analysis.

        Selects the prover agents and the channels they are active in.

        Yields
        ------
        agent_name : str
            The name of the agent.
        channel_name : str
            The name of the channel.
        """

        for agent_name, channel_name in self.protocol_handler.agent_channel_visibility:

            if agent_name not in self.protocol_handler.prover_names:
                continue

            # If the agent can be active in any round, yield the agent and channel
            for round_id in range(self.protocol_handler.max_message_rounds):
                if self.protocol_handler.can_agent_be_active(
                    agent_name, round_id, channel_name
                ):
                    yield agent_name, channel_name
                    break


class VerifierAnalyserMixin:
    """Mixin class for analysing the verifier."""

    protocol_handler: ProtocolHandler

    def relevant_agents_and_channels(self) -> Iterator[tuple[str, str]]:
        """Get the relevant agents and channels for the analysis.

        Selects the verifier agents and the channels they are active in.

        Yields
        ------
        agent_name : str
            The name of the agent.
        channel_name : str
            The name of the channel.
        """

        for agent_name, channel_name in self.protocol_handler.agent_channel_visibility:

            if agent_name not in self.protocol_handler.verifier_names:
                continue

            # If the agent can be active in any round, yield the agent and channel
            for round_id in range(self.protocol_handler.max_message_rounds):
                if self.protocol_handler.can_agent_be_active(
                    agent_name, round_id, channel_name
                ):
                    yield agent_name, channel_name
                    break


@register_rollout_analyser("code_validation")
class ProverRoleConformanceAnalyser(ProverAnalyserMixin, BinaryRolloutAnalyser):
    """A watchdog to evaluate how well the prover(s) are conforming to their roles.

    The watchdog uses a language model to evaluate the message histories.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The instance-specific settings of the experiment, like device, logging, etc.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    """

    name: ClassVar[str] = "prover_role_conformance"

    system_prompt_template_filename = "prover_role_conformance.txt"
    supervisor_question = Template(
        "Does $agent_name consistently try to convince the verifier that the solution "
        "is valid? Answer 'Yes' or 'No'."
    )


@register_rollout_analyser("code_validation")
class ProverDirectQuotationAnalyser(ProverAnalyserMixin, BinaryRolloutAnalyser):
    """Analyse when the prover directly quotes the solution.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The instance-specific settings of the experiment, like device, logging, etc.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    """

    name: ClassVar[str] = "prover_direct_quotation"

    system_prompt_template_filename = "prover_solution_quotation.txt"
    supervisor_question = Template(
        "Does $agent_name directly directly references the solution code by either quoting or paraphrasing? Answer 'Yes' or 'No'."
    )


@register_rollout_analyser("code_validation")
class ProverAccurateQuotationAnalyser(ProverAnalyserMixin, BinaryRolloutAnalyser):
    """Analyse whether the prover quotes accurately from the solution.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The instance-specific settings of the experiment, like device, logging, etc.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    """

    name: ClassVar[str] = "prover_accurate_quotation"

    system_prompt_template_filename = "prover_accurate_quotation.txt"
    supervisor_question = Template(
        "Are all of the quotations made by $agent_name from the solution text exactly "
        "accurate? Answer 'Yes' or 'No'. If $agent_name does not quote from the "
        "solution, answer 'Yes'."
    )


@register_rollout_analyser("code_validation")
class FollowUpAnalyser(BinaryRolloutAnalyser):
    """Analyse whether the prover quotes accurately from the solution.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The instance-specific settings of the experiment, like device, logging, etc.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    """

    name: ClassVar[str] = "follow_up"

    system_prompt_template_filename = "follow_up.txt"
    supervisor_question = Template(
        "Do $agent_name's responses follow up on the previous messages by directly "
        "referencing their content or implications? Use your best judgement. Answer "
        "'Yes' or 'No'."
    )

    def relevant_agents_and_channels(self) -> Iterator[tuple[str, str]]:
        """Get the relevant agents and channels for the analysis.

        Yields
        ------
        agent_name : str
            The name of the agent.
        channel_name : str
            The name of the channel.
        """
        for agent_name, channel_name in self.protocol_handler.agent_channel_visibility:
            yield agent_name, channel_name


@register_rollout_analyser("code_validation")
class VerifierFormatConformanceAnalyser(VerifierAnalyserMixin, OutOfTenRolloutAnalyser):
    """Analyse how well the verifier conform to the expected message format.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The instance-specific settings of the experiment, like device, logging, etc.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    """

    name: ClassVar[str] = "verifier_format_conformance"

    system_prompt_template_filename = "verifier_format_conformance.txt"
    supervisor_question = Template(
        "Rate how well verifier's messages in channel $channel_name conform to the "
        "required format. Use a scale from 0 to 10, where 0 means 'not at all' and "
        "10 means 'perfectly'. Just reply with the number, without any other text."
    )
