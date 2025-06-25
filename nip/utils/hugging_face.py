"""Utility functions for Hugging Face libraries."""

from typing import Optional
from dataclasses import dataclass

import numpy as np

from jaxtyping import Int

import einops

from peft.utils.constants import CONFIG_NAME as PEFT_CONFIG_NAME

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nip.utils.data import prompt_array_to_list
from nip.utils.nested_array_dict import NestedArrayDict
from nip.utils.types import String, PromptMessage


def is_model_peft(model_name: str) -> bool:
    """Check if a model is a PEFT model by looking for the PEFT config file.

    Parameters
    ----------
    model_name : str
        The name of the model to check, typically a Hugging Face identifier.

    Returns
    -------
    is_peft : bool
        True if the model is a PEFT model (e.g., LoRA-adapted), False otherwise.
    """

    try:
        hf_hub_download(model_name, PEFT_CONFIG_NAME)
    except EntryNotFoundError:
        return False
    return True


@dataclass
class TokenCounts:
    """Dataclass to hold token count information."""

    prompt: Int[np.ndarray, "rollout round"]
    """The number of tokens in the prompt for each rollout, round, and agent."""

    completion: Int[np.ndarray, "rollout round"]
    """The number of tokens in the completion for each rollout, round, and agent."""

    total: Int[np.ndarray, "rollout round"]
    """The total number of tokens (prompt + completion)"""


def count_tokens(
    rollouts: NestedArrayDict, agent_id: int, model_name: str
) -> TokenCounts:
    """Count the number of tokens in the rollouts.

    This function counts both the prompt and completion tokens for each rollout and
    round. It uses the Hugging Face tokenizer for the specified model.

    For the prompt, it first takes the chat history and puts it into the chat template
    for the model.

    Parameters
    ----------
    rollouts : NestedArrayDict
        The rollouts nested array dictionary. Has keys:

        - ("agents", "prompt") (rollout round agent message field): The prompt messages
          passed to each model, as a chat history.
        - ("agents", "raw_message") (rollout round agent): The completion messages
          returned by each model.

    agent_id : int
        The ID of the agent for which to count tokens. This is used to index into the
        rollouts dictionary.
    model_name : str
        The name of the model to use for tokenization, typically a Hugging Face
        identifier.

    Returns
    -------
    token_count : TokenCount
        A dataclass containing the token counts for prompts, completions, and total
        tokens. The shape of all elements is (rollout round).
    """

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)

    prompt: String[np.ndarray, "rollout round agent message field"] = rollouts[
        "agents", "prompt"
    ]
    prompt: String[np.ndarray, "rollout round message field"] = prompt[
        ..., agent_id, :, :
    ]
    raw_message: String[np.ndarray, "rollout round agent"] = rollouts[
        "agents", "raw_message"
    ]
    raw_message: String[np.ndarray, "rollout round"] = raw_message[..., agent_id]

    prompt_flattened = einops.rearrange(
        prompt,
        "rollout round message field -> (rollout round) message field",
    )
    raw_message_flattened = einops.rearrange(
        raw_message,
        "rollout round -> (rollout round)",
    )

    prompt_list: list[list[PromptMessage]] = [
        prompt_array_to_list(prompt_flattened[i])
        for i in range(prompt_flattened.shape[0])
    ]
    prompt_nonempty_mask = np.array(
        [len(prompt) != 0 for prompt in prompt_list], dtype=bool
    )
    prompt_nonempty_list = list(filter(lambda x: len(x) > 0, prompt_list))

    # Apply the chat template and tokenizer to the non-empty prompts
    prompt_nonempty_tokenized: list[list[int]] = tokenizer.apply_chat_template(
        prompt_nonempty_list
    )
    prompt_nonempty_lengths = [len(tokens) for tokens in prompt_nonempty_tokenized]

    # Fill in the lengths for empty prompts
    prompt_lengths = np.zeros_like(prompt_nonempty_mask, dtype=int)
    prompt_lengths[prompt_nonempty_mask] = prompt_nonempty_lengths

    prompt_lengths = einops.rearrange(
        prompt_lengths,
        "(rollout round) -> rollout round",
        rollout=prompt.shape[0],
        round=prompt.shape[1],
    )

    completion_list: list[str | None] = raw_message_flattened.tolist()
    completion_list: list[str] = [
        message if message is not None else "" for message in completion_list
    ]

    completion_tokenized: list[list[int]] = tokenizer(completion_list)["input_ids"]
    completion_lengths: Int[np.ndarray, "(rollout round)"] = np.array(
        [len(tokens) for tokens in completion_tokenized]
    )
    completion_lengths = einops.rearrange(
        completion_lengths,
        "(rollout round) -> rollout round",
        rollout=prompt.shape[0],
        round=prompt.shape[1],
    )

    return TokenCounts(
        prompt=prompt_lengths,
        completion=completion_lengths,
        total=prompt_lengths + completion_lengths,
    )
