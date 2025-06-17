"""A handler the vLLM server, which provides inference access for language models.

This module provides the `VllmServerHandler` class, which allows starting, stopping,
and checking the status of a vLLM server.
"""

from typing import Optional, ClassVar
import shutil
from datetime import datetime
import logging
from asyncio import wait_for, TimeoutError, Lock
from asyncio.subprocess import create_subprocess_exec, STDOUT, Process
from contextlib import nullcontext

from httpx import HTTPStatusError, ConnectError, AsyncClient, ConnectTimeout

import torch

from nip.constants import VLLM_LOG_DIR
from nip.language_model_server.types import (
    VllmServerStatus,
    SubprocessOutputDestination,
)
from nip.language_model_server.exceptions import (
    VllmNotInstalledError,
    VllmNoGpusError,
    VllmServerNotRunningError,
)
from nip.language_model_server.config import Settings


logger = logging.getLogger(__name__)


class VllmServerHandler:
    """A class to handle vLLM server operations.

    This class provides methods to start, stop, and check the status of a vLLM server.

    Note
    ----
    This class is designed to be used in an asynchronous context, as it uses
    asynchronous subprocess management and HTTP requests to interact with the vLLM
    server.

    Parameters
    ----------
    settings : Settings
        The settings for the vLLM server, including the port and output destination.
    """

    server_process_lock: ClassVar[Lock] = Lock()
    """Lock to ensure that only one server process operation is performed at a time."""

    @property
    def port(self) -> int:
        """The port on which the vLLM server is running."""
        return self.settings.vllm_port

    @property
    def vllm_server_url(self) -> str:
        """The URL of the vLLM server."""
        return f"http://localhost:{self.port}"

    @property
    def subprocess_output_destination(self) -> SubprocessOutputDestination:
        """The destination for subprocess output."""
        return self.settings.subprocess_output_destination

    def __init__(self, settings: Settings):

        self.settings = settings

        self.server_process: Optional[Process] = None
        self.model_name: Optional[str] = None

        if self.subprocess_output_destination == "log_file":
            VLLM_LOG_DIR.mkdir(parents=True, exist_ok=True)
            time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            vllm_log_filepath = VLLM_LOG_DIR.joinpath(f"vllm_{time_string}.log")
            self.log_file = open(vllm_log_filepath, "w")
            logger.info(f"Logging vLLM server to {vllm_log_filepath}")

    async def close(self):
        """Perform cleanup when the server handler is no longer needed."""

        try:
            await self.stop_server(ignore_lock=True)
        except VllmServerNotRunningError:
            pass

        if self.subprocess_output_destination == "log_file":
            self.log_file.close()

    async def start_server(self, model_name: str) -> str:
        """Start the vLLM server with the specified model.

        If the server is already running with the current model, this method will do
        nothing. If the server is running with a different model, it will stop the
        current server and start a new one with the specified model.

        Parameters
        ----------
        model_name : str
            The name of the model to serve with vLLM.

        Raises
        ------
        RuntimeError
            If the vLLM log file is not initialized.
        VllmNotInstalledError
            If vLLM is not installed.

        Returns
        -------
        success_message : str
            A message indicating success (either that the server was started or that it
            was already running with the specified model).
        """

        async with self.server_process_lock:

            if self.model_name == model_name and self.server_process is not None:
                logger.info(
                    f"vLLM server is already running with model '{model_name}'."
                )
                return "vLLM server is already running with the specified model."

            try:
                await self.stop_server(ignore_lock=True)
            except VllmServerNotRunningError:
                pass

            if shutil.which("vllm") is None:
                raise VllmNotInstalledError

            num_available_gpus = torch.cuda.device_count()
            if num_available_gpus == 0:
                raise VllmNoGpusError

            if self.subprocess_output_destination == "log_file":
                output_kwargs = {
                    "stdout": self.log_file,
                    "stderr": STDOUT,
                }
            else:
                output_kwargs = {}

            if self.settings.vllm_num_gpus == "auto":
                num_gpus = num_available_gpus
            else:
                num_gpus = self.settings.vllm_num_gpus

            self.server_process = await create_subprocess_exec(
                "vllm",
                "serve",
                model_name,
                "--port",
                str(self.port),
                "--tensor-parallel-size",
                "2",
                **output_kwargs,
            )

            self.model_name = model_name

            logger.info(
                f"Started vLLM server with model '{model_name}' on port {self.port}."
            )
            return "vLLM server started successfully with the specified model."

    async def stop_server(self, timeout: float = 5.0, ignore_lock: bool = False):
        """Stop the vLLM server if it is running.

        Parameters
        ----------
        timeout : float, default=5.0
            The maximum time to wait for the server to stop, by default 5.0 seconds. If
            it takes longer than this, the process will be killed.
        ignore_lock : bool, default=False
            If True, the server process lock will be ignored.

        Raises
        ------
        VllmServerNotRunningError
            If the vLLM server is not running.
        """

        if ignore_lock:
            context_manager = nullcontext()
        else:
            context_manager = self.server_process_lock

        async with context_manager:

            if (
                self.server_process is None
                or self.server_process.returncode is not None
            ):
                raise VllmServerNotRunningError

            logger.info("Stopping vLLM server...")

            self.server_process.terminate()

            try:
                await wait_for(self.server_process.wait(), timeout)
            except TimeoutError:
                logger.warning(
                    f"vLLM server not terminated after {timeout}s. Sending kill signal"
                )
                self.server_process.kill()

            self.server_process = None

            logger.info("vLLM server stopped.")

    async def get_status(
        self, timeout: float = 0.5
    ) -> tuple[VllmServerStatus, str | None]:
        """Get the status of the vLLM server by trying to list available models.

        Parameters
        ----------
        timeout : float, default=0.5
            The timeout to use when trying to connect to the server.

        Returns
        -------
        vllm_server_status : ServerStatus
            The status of the server. See the documentation for :const:`ServerStatus
            <nip.language_model_server.types.ServerStatus>` for possible values.
        error_message : str | None
            An error message if the server is not running or if an error occurs while
            trying to connect to it. If there is no error, this will be ``None``.
        """

        if self.server_process is None:
            return "not_started", None

        returncode = self.server_process.returncode
        if returncode is not None:
            if returncode == 0:
                return "not_started", None
            else:
                return "crashed", None

        try:
            async with AsyncClient() as httpx_client:
                response = await httpx_client.get(
                    f"{self.vllm_server_url}/v1/models", timeout=timeout
                )
            response.raise_for_status()
            return "online", None
        except ConnectTimeout:
            return "timeout", "Connection to vLLM server timed out."
        except ConnectError as e:
            return "not_accepting_connections", str(e)
        except HTTPStatusError as e:
            if 500 <= e.response.status_code < 600:
                return "server_error", str(e)
            else:
                return "other_error", str(e)
