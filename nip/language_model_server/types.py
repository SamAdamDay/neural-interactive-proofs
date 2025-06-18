"""Types for the language model server, including request and response structures."""

from typing import Literal, TypeAlias, Optional

from pydantic import BaseModel, Field

from nip.utils.types import DpoDatasetItem

VllmServerStatus: TypeAlias = Literal[
    "online",
    "not_started",
    "crashed",
    "not_accepting_connections",
    "timeout",
    "server_error",
    "other_error",
]
"""The status of the vLLM server. One of:

- "online": The server is running and accepting connections.
- "not_started": The server has not been started.
- "crashed": The server has exited unexpectedly.
- "not_accepting_connections": The server is running but not accepting connections. This
  can happen if the server is still starting up or if it has crashed.
- "timeout": A timeout occurred when trying to connect to the server. Retrying may help.
- "server_error": A 5xx error occurred when trying to connect to the server.
- "other_error": Any other error occurred when trying to connect to the server.
"""

TrainingJobStatus: TypeAlias = Literal[
    "pending",
    "starting",
    "running",
    "succeeded",
    "crashed",
    "interrupted",
    "cancelled",
    "unknown",
]
"""The status of a training job. One of:

- "pending": The job has not been started yet.
- "starting": The job is being started.
- "running": The job is currently running.
- "succeeded": The job has completed successfully.
- "crashed": The job process has crashed unexpectedly.
- "interrupted": The job was interrupted by the user.
- "cancelled": The job has been canceled by the user.
- "unknown": The status of the job is unknown, possibly due to a failure in checking the
  status.
"""

SubprocessOutputDestination: TypeAlias = Literal["stdout_std_err", "log_file"]
"""The destination for subprocess output.

One of:

- "stdout_std_err": Output is printed to standard output and standard error.
- "log_file": Output is written to a log file.
"""


class ServerVersionResponse(BaseModel):
    """A response containing the version of the language model server."""

    version: str
    """The version of the language model server, as a string."""


class VllmStartRequest(BaseModel):
    """A request to start the vLLM server with a specific model."""

    model_name: str
    """The name of the model to be served by the vLLM server."""


class VllmStartResponse(BaseModel):
    """A response obtained when starting the vLLM server."""

    message: str
    """A message indicating the result of the start operation."""

    model_name: str
    """The name of the model that the vLLM server is serving."""

    port: int
    """The port on which the vLLM server is running."""


class VllmStopRequest(BaseModel):
    """A request to stop the vLLM server."""

    ignore_not_running: bool = False
    """If True, the server will not raise an error if it is not running.
    
    Instead, it will log a warning and return a success message indicating that the
    server was not running and is being ignored.
    """

    terminate_timeout: float = 10.0
    """The timeout in seconds to wait for the server to terminate gracefully.

    If the server does not terminate within this time, it will be forcefully killed.
    """


class VllmStatusResponse(BaseModel):
    """A response obtained when checking the vLLM server status."""

    status: VllmServerStatus
    """The status of the vLLM server, as defined in ServerStatus."""

    error: str | None
    """An error message if the server is not online, otherwise None."""


class LmDpoTrainingConfig(BaseModel):
    """Configuration for Direct Preference Optimization (DPO) training."""

    beta: float
    """The beta parameter controlling trade-off between exploration and exploitation."""

    learning_rate: float
    """The learning rate for the DPO training."""

    max_prompt_length: int | None = None
    """The maximum length of the prompt sequence."""

    max_completion_length: int | None = None
    """The maximum length of the completion sequence."""

    max_length: int | None = None
    """The maximum length full sequence (prompt + completion)."""


class LmLoraAdapterConfig(BaseModel):
    """Configuration for a LoRA adapter to be applied on top of a base model.

    See :cite:t:`Yu2023` for the original LoRA paper.
    """

    r: int
    """The rank of the LoRA adapter, controlling the number of trainable parameters."""

    lora_alpha: int
    """The scaling factor for the LoRA adapter, for the strength of the adapter."""

    lora_dropout: float
    """The dropout rate for the LoRA layers."""


class LmTrainingConfig(BaseModel):
    """Configuration for training a language model with the language model server."""

    model_name: str
    """The name of the model to be trained, typically a Hugging Face identifier."""

    method: Literal["dpo"]
    """The method to be used for training.

    Currently, only "dpo" (Direct Preference Optimization) is supported
    :cite:p:`Rafailov2023`.
    """

    dpo_config: LmDpoTrainingConfig = Field(default_factory=LmDpoTrainingConfig)
    """Configuration specific to DPO training."""

    training_lora_config: LmLoraAdapterConfig | None = None
    """Configuration for the LoRA adapter to use when training.
    
    If ``None``, no LoRA adapter will be applied during training.
    """

    model_already_lora_strategy: Literal["reuse", "stack"] = "reuse"
    """Strategy for handling models that are already LoRA-adapted.

    If ``training_lora_adapter_config`` is not ``None``, and the model specified by
    ``model_name`` is already LoRA-adapted, this strategy determines how to handle it.

    - "reuse": Reuse the existing LoRA adapter without modification. If
      ``training_lora_adapter_config`` is not compatible with the existing adapter, an
      error will be raised.
    - "stack": Stack the new LoRA adapter on top of the existing one, allowing for
      multiple LoRA adapters to be applied sequentially.
    """

    mixed_precision: Literal["fp16", "bf16", "no"] = "fp16"
    """The mixed precision to use during training.

    - "fp16": Use 16-bit floating point precision.
    - "bf16": Use bfloat16 precision.
    - "no": Use full 32-bit floating point precision.
    """

    gradient_checkpointing: bool = True
    """Whether to use gradient checkpointing to save memory during training."""


class CreateTrainingJobRequest(BaseModel):
    """A request to create a new training job."""

    config: LmTrainingConfig
    """The configuration for the training job."""

    dataset: list[DpoDatasetItem]
    """The dataset to be used for training.
    
    Consists of a list of `DpoDatasetItem` objects, each containing the necessary
    information for training, such as prompts and completions.
    """

    job_name: Optional[str] = None
    """An optional name for the job, to make it more recognizable."""


class TrainingJobInfo(BaseModel):
    """A data structure representing information about a training job."""

    job_id: str
    """The unique identifier for the training job."""

    status: TrainingJobStatus
    """The current status of the training job."""

    config: LmTrainingConfig
    """The configuration for the training job."""

    new_model_name: str
    """The name of the model that will be created after training is complete."""

    error_message: str = ""
    """An error message if the job has failed, otherwise an empty string."""
