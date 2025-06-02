"""A client for interacting with the self-hosting language model server.

This client provides a simple interface to interact with the language model server,
allowing for controlling the vLLM server and performing language model training
tasks.
"""

import typing
import asyncio

from httpx import AsyncClient

from nip.language_model_server.types import (
    ServerStatus,
    VllmStartResponse,
    VllmStopResponse,
    VllmStatusResponse,
)
from nip.language_model_server.exceptions import BadResponseError, TimeoutError


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
                f"{self.server_url}/vllm/start", data={"model_name": model_name}
            )
        response.raise_for_status()

        data: VllmStartResponse = response.json()

        if "message" not in data:
            raise BadResponseError(
                "The response from the server does not contain a 'message' field.",
                response=response,
            )

        return data["message"]

    async def stop_vllm_server(self) -> str:
        """Stop the vLLM language model server.

        Returns
        -------
        success_message : str
            A message indicating that the vLLM server has been stopped successfully.

        Raises
        ------
        BadResponseError
            If the server returns an invalid response or if the response does not
            contain the expected data.
        """

        async with AsyncClient() as httpx_client:
            response = await httpx_client.post(f"{self.server_url}/vllm/stop")
        response.raise_for_status()

        data: VllmStopResponse = response.json()

        if "message" not in data:
            raise BadResponseError(
                "The response from the server does not contain a 'message' field.",
                response=response,
            )

        return data["message"]

    async def get_vllm_server_status(self) -> ServerStatus:
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

        data: VllmStatusResponse = response.json()

        if "status" not in data:
            raise BadResponseError(
                "The response from the server does not contain a 'status' field.",
                response=response,
            )

        if data["status"] not in typing.get_args(ServerStatus):
            raise BadResponseError(
                f"Invalid status '{data['status']}' received from the server.",
                response=response,
            )

        return data["status"]

    async def wait_for_vllm_server(self, timeout: float = 300):
        """Wait for the vLLM server to be online.

        Parameters
        ----------
        timeout : float, default=300
            The maximum time to wait for the vLLM server to be online, in seconds.

        Raises
        ------
        TimeoutError
            If the vLLM server does not become online within the specified timeout.
        """

        start_time = asyncio.get_event_loop().time()

        while True:

            status = await self.get_vllm_server_status()

            if status == "online":
                return

            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(
                    f"Timed out waiting for vLLM server to be online after {timeout} "
                    f"seconds."
                )

            await asyncio.sleep(5)
