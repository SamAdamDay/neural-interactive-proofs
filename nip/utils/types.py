"""Type definitions for the NIP package."""

import logging
from typing import (
    Annotated as String,
    Any,
    get_origin,
    get_args,
    Union,
    TypeAlias,
    Literal,
    NotRequired,
)
from types import UnionType

from typing_extensions import TypedDict

import torch

from numpy.dtypes import StringDType

TorchDevice: TypeAlias = torch.device | str | int

LoggingType: TypeAlias = logging.Logger | logging.LoggerAdapter

NumpyStringDtype = StringDType(na_object=None)

NOT_GIVEN = object()

NestedKey: TypeAlias = str | tuple["NestedKey", ...]
"""A nested key type used for tensordict keys."""


def get_union_elements(tp: Any) -> list:
    """Get the elements of a union type.

    If the type is not a union, returns a singleton list containing the type.

    Parameters
    ----------
    tp : Any
        The type, which may be a union type

    Returns
    -------
    type_list : list
        A list of types that are part of the union, or a singleton list if the type is
        not a union.
    """

    if get_origin(tp) is UnionType or get_origin(tp) is Union:
        return sum([get_union_elements(sub_tp) for sub_tp in get_args(tp)], [])
    else:
        return [tp]


class PromptMessage(TypedDict):
    """A message in the prompt for a language model API.

    The prompt is a list of messages, where each message is a dictionary with keys as
    follows.

    Attributes
    ----------
    role : Literal["system", "assistant", "user"]
        The role of the message sender.
    content : str
        The content of the message.
    name : str, optional
        The name of the message sender.
    """

    role: Literal["system", "assistant", "user"]
    content: str
    name: NotRequired[str]


class SupervisedDatasetItem(TypedDict):
    """A single item in a supervised dataset.

    This is used for training language models with supervised fine-tuning.
    """

    messages: list[PromptMessage]
    """The input chat history for the supervised dataset item.

    The messages are a list of :class:`PromptMessage` objects representing the chat
    history, where each message has a role (system, assistant, or user) and content.
    """


class DpoDatasetItem(TypedDict):
    """A single item in a DPO dataset.

    This is used for training language models with Direct Preference Optimization (DPO).
    """

    input: dict[Literal["messages"], list[PromptMessage]]
    """The input chat history for the DPO dataset item.

    The input is a dictionary with a single key "messages", which maps to a list of
    :class:`PromptMessage` objects representing the chat history.
    """

    preferred_output: list[PromptMessage]
    """The preferred part of the preference pair."""

    non_preferred_output: list[PromptMessage]
    """The non-preferred part of the preference pair."""


class HuggingFaceDpoDatasetItem(TypedDict):
    """A single item in a DPO dataset using the Hugging Face format.

    This is similar to :class:`DpoDatasetItem`, but uses the Hugging Face format for
    the input chat history.
    """

    prompt: list[PromptMessage]
    """The input chat history for the DPO dataset item.

    The input is a list of :class:`PromptMessage` objects representing the chat history.
    """

    chosen: list[PromptMessage]
    """The preferred part of the preference pair."""

    rejected: list[PromptMessage]
    """The non-preferred part of the preference pair."""


class ExperimentConfig(TypedDict):
    """Configuration for an experiment, which specifies the hyperparameters."""

    kind: Literal["single_experiment", "grid"]
    """The type of experiment configuration.

    - "single_experiment": A single experiment with specific hyperparameters.
    - "grid": A grid search over hyperparameters.
    """

    parameters: dict[str, Any]
    """A dictionary of parameters for the experiment.

    They keys are parameter names, which may differ from the names of the
    hyperparameters in :class:`nip.parameters.Hyperparameters`, but which are
    interpreted by the experiment script.

    The values are either the values of the hyperparameters for a single
    experiment or lists of values for grid search.

    It is possible to use a nested dictionary here. Nested keys will be flattened
    using the dot notation, e.g. ``{"a": {"b": 1, "c": 2}}`` will be flattened to
    ``{"a.b": 1, "a.c": 2}``.
    """
