"""A server which allows for controlling vLLM and doing language model training.

The server is a FastAPI application with the following endpoints:

- ``version``: Returns the ``nip`` package version.
- ``/vllm/start``: Starts a vLLM server with the specified model.
- ``/vllm/stop``: Stops the vLLM server.
- ``/vllm/status``: Returns the status of the vLLM server.
- ``/training/jobs``: Create or list fine-tuning jobs.
- ``/training/jobs/<job_id>``: Get info about a fine-tuning job or cancel it.

Example
-------

>>> from nip.language_model_server.server import LanguageModelServer
>>> from quart import Quart
>>> app = Quart(__name__)
>>> async with LanguageModelServer(app, vllm_port=8000):
...     app.run(port=8080)
"""

from typing import Optional
import logging
from asyncio import TaskGroup
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from nip.language_model_server.types import (
    VllmStartRequest,
    VllmStartResponse,
    VllmStopRequest,
    VllmStatusResponse,
    ServerVersionResponse,
    CreateTrainingJobRequest,
    TrainingJobInfo,
)
from nip.language_model_server.exceptions import (
    LanguageModelServerError,
    VllmServerNotRunningError,
)
from nip.language_model_server.trainer_handler import TrainerHandler
from nip.language_model_server.vllm_server_handler import VllmServerHandler
from nip.language_model_server.config import get_settings
from nip.utils.version import get_version


logger = logging.getLogger(__name__)

vllm_server_handler: Optional[VllmServerHandler] = None
trainer_handler: Optional[TrainerHandler] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for the FastAPI application.

    This context manager is used to initialize and clean up resources when the FastAPI
    application starts and stops.
    """

    global vllm_server_handler, trainer_handler

    settings = get_settings()

    vllm_server_handler = VllmServerHandler(settings)
    trainer_handler = TrainerHandler(settings)

    try:
        yield
    finally:
        async with TaskGroup() as task_group:
            task_group.create_task(vllm_server_handler.close())
            task_group.create_task(trainer_handler.close())


app = FastAPI(lifespan=lifespan)


def _raise_server_error_as_http_exception(
    error: LanguageModelServerError,
    description: str = "Language model server error",
    log_level: int = logging.WARNING,
):
    """Raise an HTTPException based on a LanguageModelServerError.

    Parameters
    ----------
    error : LanguageModelServerError
        The error to convert into an HTTPException.
    description : str, default="Language model server error"
        A prefix for the error message
    log_level : int, default=logging.WARNING
        The logging level to use when logging the error.

    Raises
    ------
    HTTPException
        An HTTPException with the status code and detail from the error.
    """
    detail = f"{description}: {error!s}"
    logger.log(log_level, detail)
    raise HTTPException(status_code=error.status_code, detail=detail)


@app.get("/version")
def get_package_version() -> ServerVersionResponse:
    """Get the version of the language model server.

    Returns
    -------
    response : ServerVersionResponse
        A data structure containing the version of the language model server.
    """

    return ServerVersionResponse(version=get_version(as_tuple=False))


@app.post("/vllm/start", status_code=201)
async def start_vllm_server(request: VllmStartRequest) -> VllmStartResponse:
    """Start the vLLM server with the specified model.

    Parameters
    ----------
    request : VllmStartRequest
        A request containing the model name to be served by the vLLM server.

    Returns
    -------
    response : VllmStartResponse
        A data structure containing the success message, model name, and port on which the
        vLLM server is running.

    Raises
    ------
    HTTPException
        If the vLLM server fails to start.
    """

    try:
        success_message = await vllm_server_handler.start_server(
            model_name=request.model_name
        )
    except LanguageModelServerError as e:
        _raise_server_error_as_http_exception(
            e,
            description="Failed to start vLLM server",
            log_level=logging.ERROR,
        )

    return VllmStartResponse(
        message=success_message,
        model_name=vllm_server_handler.model_name,
        port=vllm_server_handler.port,
    )


@app.post("/vllm/stop", status_code=204)
async def stop_vllm_server(request: VllmStopRequest):
    """Stop the vLLM server.

    Parameters
    ----------
    request : VllmStopRequest
        A request containing the ignore_not_running flag.
    """

    try:
        await vllm_server_handler.stop_server(timeout=request.terminate_timeout)
    except VllmServerNotRunningError as e:
        if request.ignore_not_running:
            logger.warning("vLLM server was not running, not stopping.")
        else:
            _raise_server_error_as_http_exception(
                e, description="Failed to stop vLLM server"
            )
    except LanguageModelServerError as e:
        _raise_server_error_as_http_exception(
            e, description="Failed to stop vLLM server"
        )


@app.get("/vllm/status")
async def get_vllm_server_status() -> VllmStatusResponse:
    """Get the status of the vLLM server.

    Returns
    -------
    response : VllmStatusResponse
        A data structure containing the status of the vLLM server and any error message
        if the server is not online.
    """

    status, error_message = await vllm_server_handler.get_status()
    return VllmStatusResponse(status=status, error=error_message)


@app.get("/training/jobs")
async def get_training_jobs() -> list[TrainingJobInfo]:
    """List all training jobs managed by the server.

    Returns
    -------
    response : list[TrainingJobInfo]
        A list of ``TrainingJobInfo`` data structures representing the training jobs,
        each containing job ID, status, and configuration.
    """

    return await trainer_handler.get_training_job_infos()


@app.get("/training/jobs/{job_id}")
async def get_training_job(job_id: str) -> TrainingJobInfo:
    """Get info about a training job.

    Parameters
    ----------
    job_id : str
        The ID of the training job to retrieve.

    Returns
    -------
    response : TrainingJobInfo
        A data structure containing information about the training job, including its
        ID, status, and configuration.
    """

    try:
        return await trainer_handler.get_job_info(job_id)

    except LanguageModelServerError as e:
        _raise_server_error_as_http_exception(
            e, description=f"Failed to get training job {job_id}"
        )


@app.post("/training/jobs", status_code=201)
async def create_training_job(request: CreateTrainingJobRequest) -> TrainingJobInfo:
    """Create a new training job.

    Parameters
    ----------
    request : CreateTrainingJobRequest
        A request containing the configuration for the training job, the dataset to be
        used, and an optional job ID suffix.

    Returns
    -------
    response : TrainingJobInfo | LanguageModelErrorResponse
        A dictionary containing either the created training job or an error response
        if an error occurs.
    """

    try:
        job = await trainer_handler.add_job(request)
        return await job.get_info()

    except LanguageModelServerError as e:
        _raise_server_error_as_http_exception(
            e, description="Failed to create training job"
        )


@app.delete("/training/jobs/{job_id}", status_code=204)
async def cancel_training_job(job_id: str):
    """Cancel a training job.

    Parameters
    ----------
    job_id : str
        The ID of the training job to cancel.
    """

    try:
        await trainer_handler.cancel_job(job_id)

    except LanguageModelServerError as e:
        _raise_server_error_as_http_exception(
            e, description=f"Failed to cancel training job {job_id}"
        )
