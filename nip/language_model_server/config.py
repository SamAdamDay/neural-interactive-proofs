"""Configuration settings for the language model server."""

from typing import Literal
from functools import lru_cache

from pydantic_settings import BaseSettings, CliSuppress

from nip.language_model_server.types import SubprocessOutputDestination
from nip.utils.env import get_env_var


class Settings(
    BaseSettings,
    cli_kebab_case=True,
    cli_ignore_unknown_args=True,
    cli_implicit_flags=True,
    use_attribute_docstrings=True,
):
    """Configuration settings for the language model server.

    This class uses `pydantic_settings` to load settings from environment variables or
    a `.env` file.
    """

    lm_server_port: int = get_env_var("DEFAULT_LM_SERVER_PORT")
    """The port on which the main language model server will run."""

    vllm_port: int = get_env_var("DEFAULT_VLLM_SERVER_PORT")
    """The port on which the vLLM server will run."""

    subprocess_output_destination: CliSuppress[SubprocessOutputDestination] = (
        "stdout_std_err"
    )
    """Where to send the output of the vLLM server subprocess."""

    max_training_jobs: int = 1
    """The maximum number of concurrent training jobs allowed."""

    vllm_num_gpus: int | Literal["auto"] = "auto"
    """The maximum number of GPUs to use for the vLLM server. 

    If set to 'auto', it will use all available GPUs.
    
    The actual number of GPUs used may be less than this value, because it must divide
    the number of attention heads in the model.
    """

    vllm_clear_cache: bool = False
    """Whether to clear the Hugging Face model cache before starting the server.

    If ``True``, all cached models other than the one being loaded will be cleared
    before starting the vLLM server.
    """

    vllm_max_lora_rank: int | Literal["auto"] = "auto"
    """The maximum rank for LoRA layers permitted in the vLLM server.
    
    This should be set to the maximum rank of the LoRA layers in the model being
    trained.

    If set to 'auto', it will use the rank of the LoRA model to be served, if it is
    available. If no LoRA model is available, it will use the vLLM default value.
    """

    accelerate_config_path: str = "accelerate_config.yaml.jinja2"
    """Path to the configuration file for the accelerate library.
    
    If the filename ends with `.jinja2`, it will be treated as a Jinja2 template and
    rendered. If empty, no configuration file will be passed to the ``accelerate``
    command. In this case, the ``accelerate`` command will use the default configuration
    file, which is usually located at
    `~/.cache/huggingface/accelerate/default_config.yaml`.

    Relative paths are resolved against the current working directory, or if that fails
    against the template directory: ``nip/language_model_server/templates/``.
    """

    offload_optimizer: bool = False
    """Whether to offload the optimizer state to CPU memory when training.

    This can reduce GPU memory usage, but may slow down training.
    """

    offload_parameters: bool = False
    """Whether to offload the model parameters to CPU memory when training.

    This can reduce GPU memory usage, but may slow down training.
    """

    parent_script_cwd: CliSuppress[str | None] = None
    """Path to the working directory of the script which called this process.
    
    The script may run the FastAPI process with a different working directory, but this
    would mess up any relative paths. So this setting records the original working
    directory for path resolution.
    """

    debug: bool = False
    """Whether to enable debug mode."""

    external: bool = False
    """Whether to run the server in external mode, with host set to '0.0.0.0'.

    This allows the server to be accessed from outside the local machine.
    """

    reload: bool = False
    """Whether to enable auto-reload for the uvicorn server.
    
    This auto-reloads the server when any of the source files change, at the cost of
    some performance.
    """


@lru_cache
def get_settings():
    """Get the settings for the language model server.

    This function uses `lru_cache` to cache the settings, so that they are only loaded
    once per process.

    Returns
    -------
    settings: Settings
        The settings for the language model server.
    """
    return Settings()
