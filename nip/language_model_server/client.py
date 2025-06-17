"""A client for interacting with the self-hosting language model server.

This client provides a simple interface to interact with the language model server,
allowing for controlling the vLLM server and performing language model training
tasks.
"""

from typing import Optional
import asyncio
from warnings import warn

from httpx import AsyncClient

from pydantic import ValidationError

from nip.utils.types import DpoDatasetItem
from nip.utils.asyncio import run_coroutine_sync
from nip.utils.version import get_version, compare_versions
from nip.language_model_server.types import (
    ServerVersionResponse,
    VllmServerStatus,
    VllmStartRequest,
    VllmStartResponse,
    VllmStopRequest,
    VllmStatusResponse,
    CreateTrainingJobRequest,
    TrainingJobInfo,
    LmTrainingConfig,
)
from nip.language_model_server.exceptions import (
    BadResponseError,
    ClientTimeoutError,
    VllmServerError,
)


class LanguageModelClient:
    """A client for interacting with the language model server.

    This client provides methods to interact with the language model server, allowing
    for controlling the vLLM server and performing language model training tasks.

    Parameters
    ----------
    server_url : str, default="http://localhost:5000"
        The URL of the language model server. This should include the protocol (http or
        https) and the port number if applicable.
    """

    def __init__(self, server_url: str = "http://localhost:5000"):
        self.server_url = server_url

        server_version = run_coroutine_sync(self.get_server_version())
        _, difference = compare_versions(server_version, get_version())
        if difference == "major":
            raise RuntimeError(
                f"Language model server version {server_version!r} differs from "
                f"client version {get_version()!r} by a major version. For "
                f"compatibility reasons, the client and server must have the same "
                f"major version."
            )
        elif difference == "minor":
            warn(
                f"Language model server version {server_version!r} differs from "
                f"client version {get_version()!r} by a minor version. This may be ok "
                f"but is not guaranteed to be compatible. If you encounter issues, "
                f"please ensure that the client and server versions match.",
                UserWarning,
            )

    async def get_server_version(self) -> str:
        """Get the version of the language model server.

        Returns
        -------
        version : str
            The version of the language model server, as a string.

        Raises
        ------
        BadResponseError
            If the server returns an invalid response or if the response does not
            contain the expected 'version' field.
        """

        async with AsyncClient() as httpx_client:
            response = await httpx_client.get(f"{self.server_url}/version")
        response.raise_for_status()

        data = response.json()

        try:
            data = ServerVersionResponse(**data)
        except ValidationError as e:
            raise BadResponseError(
                "The response from the server does not match the expected format.",
                response=response,
            ) from e

        return data.version

    async def start_vllm_server(self, model_name: str) -> str:
        """Start the vLLM language model server with the specified model.

        Parameters
        ----------
        model_name : str
            The name of the model to be served by vLLM. This should match a model that
            is available in the vLLM installation.

        Returns
        -------
        success_message : str
            A message indicating that the vLLM server has been started successfully, or
            was already running.

        Raises
        ------
        BadResponseError
            If the server returns an invalid response or if the response does not
            contain the expected data.
        """

        async with AsyncClient() as httpx_client:
            response = await httpx_client.post(
                f"{self.server_url}/vllm/start",
                json=VllmStartRequest(model_name=model_name).model_dump(),
            )
        response.raise_for_status()

        data = response.json()

        try:
            data = VllmStartResponse(**data)
        except ValidationError as e:
            raise BadResponseError(
                "The response from the server does not match the expected format.",
                response=response,
            ) from e

        return data.message

    async def stop_vllm_server(
        self, ignore_not_running: bool = False, timeout: float = 15.0
    ):
        """Stop the vLLM language model server.

        Parameters
        ----------
        ignore_not_running : bool, default=False
            If True, the server will not raise an error if it is not running. Instead,
            it will log a warning and return a success message indicating that the
            server was not running and is being ignored.
        timeout : float, default=15.0
            The maximum time to wait for the vLLM server to stop, in seconds. If the
            server does not stop within this time, a timeout error will be raised. The
            server will attempt to terminate gracefully for `max(timeout - 5.0, 1.0)`
            seconds, after which it will be forcefully killed if it is still running.

        Raises
        ------
        HTTPStatusError
            If the server returns an error status code while stopping the vLLM server.
        """

        async with AsyncClient() as httpx_client:
            response = await httpx_client.post(
                f"{self.server_url}/vllm/stop",
                json=VllmStopRequest(
                    ignore_not_running=ignore_not_running,
                    terminate_timeout=max(1.0, timeout - 5.0),
                ).model_dump(),
                timeout=timeout,
            )
        response.raise_for_status()

    async def get_vllm_server_status(self) -> VllmServerStatus:
        """Get the current status of the vLLM language model server.

        Returns
        -------
        vllm_server_status : ServerStatus
            The current status of the vLLM server. See the documentation for
            :const:`ServerStatus <nip.language_model_server.types.ServerStatus>` for
            possible values.

        Raises
        ------
        BadResponseError
            If the server returns an invalid response or if the response does not
            contain the expected 'status' field, or if the status is not a valid
            `ServerStatus`.
        """

        async with AsyncClient() as httpx_client:
            response = await httpx_client.get(f"{self.server_url}/vllm/status")
        response.raise_for_status()

        data = response.json()

        try:
            data = VllmStatusResponse(**data)
        except ValidationError as e:
            raise BadResponseError(
                "The response from the server does not match the expected format.",
                response=response,
            ) from e

        return data.status

    async def wait_for_vllm_server(self, timeout: float = 300):
        """Wait for the vLLM server to be online.

        Parameters
        ----------
        timeout : float, default=300
            The maximum time to wait for the vLLM server to be online, in seconds.

        Raises
        ------
        ClientTimeoutError
            If the vLLM server does not become online within the specified timeout.
        """

        start_time = asyncio.get_event_loop().time()

        while True:

            status = await self.get_vllm_server_status()

            if status == "online":
                return

            elif status in ["crashed", "server_error", "other_error"]:
                raise VllmServerError(status)

            if asyncio.get_event_loop().time() - start_time > timeout:
                raise ClientTimeoutError(
                    f"Timed out waiting for vLLM server to be online after {timeout}s."
                )

            await asyncio.sleep(5)

    async def get_training_jobs(self) -> list[TrainingJobInfo]:
        """Get the list of training jobs currently managed by the server.

        Returns
        -------
        training_jobs : list[TrainingJobInfo]
            A list of :class:`TrainingJobInfo` objects, each containing information
            about a training job, including its ID, status, and configuration.

        Raises
        ------
        HTTPStatusError
            If the server returns an error status code while creating the training job.
        BadResponseError
            If the server returns an invalid response or if the response does not
            contain the expected data.
        """

        async with AsyncClient() as httpx_client:
            response = await httpx_client.get(f"{self.server_url}/training/jobs")
        response.raise_for_status()

        data = response.json()

        try:
            data = [TrainingJobInfo(**job) for job in data]
        except ValidationError as e:
            raise BadResponseError(
                "The response from the server does not match the expected format.",
                response=response,
            ) from e

        return data

    async def get_training_job(self, job_id: str) -> TrainingJobInfo:
        """Get the details of a specific training job by its ID.

        Parameters
        ----------
        job_id : str
            The ID of the training job to retrieve.

        Returns
        -------
        training_job : TrainingJobInfo
            An object containing the details of the training job, including its ID,
            status, and configuration.

        Raises
        ------
        HTTPStatusError
            If the server returns an error status code while creating the training job.
        BadResponseError
            If the server returns an invalid response or if the response does not
            contain the expected data.
        """

        async with AsyncClient() as httpx_client:
            response = await httpx_client.get(
                f"{self.server_url}/training/jobs/{job_id}"
            )
        response.raise_for_status()

        data = response.json()

        try:
            data = TrainingJobInfo(**data)
        except ValidationError as e:
            raise BadResponseError(
                "The response from the server does not match the expected format.",
                response=response,
            ) from e

        return data

    async def create_training_job(
        self,
        training_config: LmTrainingConfig,
        dataset: list[DpoDatasetItem],
        job_id_suffix: Optional[str] = None,
    ) -> TrainingJobInfo:
        """Create a new training job with the specified configuration.

        Parameters
        ----------
        training_config : LmTrainingConfig
            The configuration for the training job, including model name and training
            parameters.
        dataset : list[DpoDatasetItem]
            The dataset to be used for training. This should be a list of dictionaries
            where each dictionary represents a single data point in the dataset.
        job_id_suffix : Optional[str], default=None
            An optional suffix to append to the job ID, to make it more recognizable.

        Returns
        -------
        training_job : TrainingJobInfo
            An object containing the details of the created training job, including its
            ID, status, and configuration.

        Raises
        ------
        HTTPStatusError
            If the server returns an error status code while creating the training job.
        BadResponseError
            If the server returns an invalid response or if the response does not
            contain the expected data.
        """

        request = CreateTrainingJobRequest(
            config=training_config,
            dataset=dataset,
            job_id_suffix=job_id_suffix,
        )

        async with AsyncClient() as httpx_client:
            response = await httpx_client.post(
                f"{self.server_url}/training/jobs", json=request.model_dump()
            )
        response.raise_for_status()

        data = response.json()
        try:
            data = TrainingJobInfo(**data)
        except ValidationError as e:
            raise BadResponseError(
                "The response from the server does not match the expected format.",
                response=response,
            ) from e

        return data

    async def cancel_training_job(self, job_id: str):
        """Cancel a training job by its ID.

        Parameters
        ----------
        job_id : str
            The ID of the training job to cancel.

        Raises
        ------
        HTTPStatusError
            If the server returns an error status code while cancelling the training
            job.
        """

        async with AsyncClient() as httpx_client:
            response = await httpx_client.delete(
                f"{self.server_url}/training/jobs/{job_id}"
            )
        response.raise_for_status()
