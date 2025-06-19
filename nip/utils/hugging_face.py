"""Utility functions for Hugging Face libraries."""

from peft.utils.constants import CONFIG_NAME as PEFT_CONFIG_NAME

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError


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
