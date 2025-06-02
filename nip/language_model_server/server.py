"""A server which allows for controlling vLLM and doing language model training.

The server is a Flask application using JSON as the data format for requests and
responses. Is has the following endpoints:

- `/vllm/start`: Starts a vLLM server with the specified model.
- `/vllm/stop`: Stops the vLLM server.
- `/vllm/status`: Returns the status of the vLLM server.
- `/training/v1/files`: Upload or list files for training.
- `/training/v1/fine_tuning/jobs`: Create or list fine-tuning jobs.
- `/training/v1/fine_tuning/jobs/<job_id>`: Get info about a fine-tuning job.

Example
-------

>>> from nip.language_model_server.server import LanguageModelServer
>>> from flask import Flask
>>> app = Flask(__name__)
>>> with LanguageModelServer(app, vllm_port=8000):
...     app.run(port=8080)
"""

from subprocess import Popen, TimeoutExpired, STDOUT
from typing import (
    Optional,
    Callable,
    TypeVar,
    ClassVar,
    Literal,
    ParamSpec,
    Concatenate,
)
from types import TracebackType
import shutil
from datetime import datetime
from io import TextIOWrapper
import logging
from dataclasses import dataclass
from functools import partial, wraps

from flask import Flask, request

import httpx
from httpx import HTTPError, ConnectError

from nip.constants import VLLM_LOG_DIR
from nip.language_model_server.types import (
    LanguageModelErrorResponse,
    VllmStartResponse,
    VllmStopResponse,
    VllmStatusResponse,
    ServerStatus,
)
from nip.language_model_server.exceptions import (
    LanguageModelServerError,
    VllmNotInstalledError,
    VllmServerNotRunningError,
)


M = TypeVar("M", bound=Callable)
P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)


@dataclass
class _Route:
    """A data class representing a route in the language model server."""

    rule: str
    """The URL rule for the route, e.g., `/vllm/start`."""
    endpoint: str
    """The endpoint name for the route, e.g., `vllm_start`."""
    view_func: callable
    """The view function that handles requests to this route."""
    methods: list[str]
    """The HTTP methods allowed for this route."""


class _RouteHolder:
    """A class to hold the routes for the language model server.

    This class is used to store the routes that will be added to the Flask application,
    and provides a decorator method to register new routes.
    """

    def __init__(self):
        self.routes: list[_Route] = []

    def register_route(
        self,
        rule: str,
        endpoint: str,
        methods: list[str] = ["GET"],
    ) -> Callable[[M], M]:
        """Register a new route for the language model server.

        Using this decorator allows the route specification to appear directly above the
        function definition, making it clear which function handles which route.

        Parameters
        ----------
        rule : str
            The URL rule for the route, e.g., `/vllm/start`.
        endpoint : str
            The endpoint name for the route, e.g., `vllm_start`.
        methods : list[str], default=["GET"]
            The HTTP methods allowed for this route, e.g., `["POST"]`.

        Returns
        -------
        decorator : Callable[[M], M]
            A decorator that can be applied to a function to register it as a route
            handler.
        """

        def decorator(method: M) -> M:
            """Register a method as a route handler."""
            self.routes.append(
                _Route(
                    rule=rule,
                    endpoint=endpoint,
                    view_func=method,
                    methods=methods,
                )
            )
            return method

        return decorator

    def add_routes_to_app(
        self, language_model_server: "LanguageModelServer", app: Flask
    ):
        """Add the registered routes to the given Flask application.

        This method should be called after all routes have been registered using the
        `register_route` decorator.

        Parameters
        ----------
        language_model_server : LanguageModelServer
            The language model server instance to which the routes belong. This is
            needed to pass the ``self`` reference to the route handlers.
        app : Flask
            The Flask application to which the routes will be added.
        """

        for route in self.routes:
            app.add_url_rule(
                rule=route.rule,
                endpoint=route.endpoint,
                view_func=partial(route.view_func, self=language_model_server),
                methods=route.methods,
            )


def _check_context_manager_entered(
    method: Callable[Concatenate["LanguageModelServer", P], R],
) -> Callable[Concatenate["LanguageModelServer", P], R]:
    """Check if the context manager has been entered.

    This function is used as a decorator to ensure that the language model server is
    being used as a context manager. It raises an error if the context manager has not
    been entered before calling the decorated method.

    Parameters
    ----------
    method : Callable[Concatenate[LanguageModelServer, P], R]
        The method to be decorated. It should take the `LanguageModelServer` instance
        as its first argument (the ``self`` parameter).

    Returns
    -------
    Callable[Concatenate[LanguageModelServer, P], R]
        A wrapped method that checks if the context manager has been entered before
        executing the original method.

    Raises
    ------
    RuntimeError
        If the context manager has not been entered.
    """

    @wraps(method)
    def wrapped_method(
        self: "LanguageModelServer",
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        if not self.context_entered:
            raise RuntimeError("LanguageModelServer must be used as a context manager.")
        return method(self, *args, **kwargs)

    return wrapped_method


class LanguageModelServer:
    """A class to manage the language model server.

    Must be used as a context manager to ensure proper logging and cleanup.

    Parameters
    ----------
    flask_app : Flask
        The Flask application to use for the server.
    vllm_port : int, default=8000
        The port on which the vLLM server will run.
    subprocess_output_destination : str
        Where to send the output of the vLLM server subprocess. One of:

        - "stdout_std_err": Output will be sent to standard output and error.
        - "log_file": Output will be written to a log file in the `VLLM_LOG_DIR`
          directory, named with the current date and time.
    """

    route_handler: ClassVar[_RouteHolder] = _RouteHolder()

    @property
    def vllm_server_url(self) -> str:
        """The URL of the vLLM server."""
        return f"http://localhost:{self.vllm_port}"

    def __init__(
        self,
        flask_app: Flask,
        vllm_port: int = 8000,
        subprocess_output_destination: Literal[
            "stdout_std_err", "log_file"
        ] = "stdout_std_err",
    ):
        self.flask_app = flask_app
        self.vllm_port = vllm_port
        self.subprocess_output_destination = subprocess_output_destination

        self.vllm_server_process: Optional[Popen] = None
        self.vllm_log_file: Optional[TextIOWrapper] = None
        self.vllm_model_name: Optional[str] = None

        self.context_entered = False

        self.route_handler.add_routes_to_app(self, flask_app)

    def __enter__(self):
        time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if self.subprocess_output_destination == "log_file":
            VLLM_LOG_DIR.mkdir(parents=True, exist_ok=True)
            vllm_log_filepath = VLLM_LOG_DIR.joinpath(f"vllm_{time_string}.log")
            self.vllm_log_file = open(vllm_log_filepath, "w")
            logger.info(f"Logging vLLM server to {vllm_log_filepath}")

        self.context_entered = True

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ):
        if self.subprocess_output_destination == "log_file":
            self.vllm_log_file.close()

        try:
            self._stop_vllm_server()
        except VllmServerNotRunningError:
            pass

        self.context_entered = False

    @route_handler.register_route("/vllm/start", "vllm_start", methods=["POST"])
    def vllm_start_command(
        self,
    ) -> tuple[LanguageModelErrorResponse | VllmStartResponse, int]:
        """Handle the ``/vllm/start`` command to start the vLLM server.

        This method is intended to be used as a Flask route handler for the
        ``/vllm/start`` endpoint. It starts the vLLM server with the specified model
        name from the request data.

        Expects the request data to be a JSON object with keys:

        - "model_name" (str): The name of the model to serve with vLLM.

        Returns
        -------
        response : LanguageModelErrorResponse | VllmStartResponse
            A dictionary containing either an error response or the status of the vLLM
            server after starting it.
        status_code : int
            The HTTP status code for the response. If the server starts successfully,
            this will be 200. If an error occurs, it will be a 4xx or 5xx code depending
            on the error type.
        """

        try:
            success_message = self._start_vllm_server(
                model_name=request.form["model_name"]
            )
        except LanguageModelServerError as e:
            logger.error(f"Failed to start vLLM server: {e}")
            return e.to_dict(), e.status_code

        response = {
            "message": success_message,
            "model_name": self.vllm_model_name,
            "port": self.vllm_port,
        }
        return response, 200

    @route_handler.register_route("/vllm/stop", "vllm_stop", methods=["POST"])
    def vllm_stop_command(
        self,
    ) -> tuple[VllmStopResponse | LanguageModelErrorResponse, int]:
        """Handle the ``/vllm/stop`` command to stop the vLLM server.

        This method is intended to be used as a Flask route handler for the
        ``/vllm/stop`` endpoint. It stops the vLLM server if it is running.

        Expects the request data to be a JSON object with keys:

        - "ignore_not_running" (bool): If True, will not raise an error if the server is
          not running.

        Returns
        -------
        response : VllmStopResponse | LanguageModelErrorResponse
            A dictionary containing either a success message or an error response if the
            server is not running or another error occurs.
        status_code : int
            The HTTP status code for the response. If the server stops successfully,
            this will be 200. If an error occurs, it will be a 4xx or 5xx code depending
            on the error type.
        """

        try:
            self._stop_vllm_server()
        except VllmServerNotRunningError as e:
            if request.form["ignore_not_running"]:
                logger.warning("vLLM server was not running, not stopping.")
                return {"message": "vLLM server was not running, ignoring."}, 200
            else:
                logger.error(f"Failed to stop vLLM server: {e}")
                return e.to_dict(), e.status_code
        except LanguageModelServerError as e:
            logger.error(f"Failed to stop vLLM server: {e}")
            return e.to_dict(), e.status_code

        return {"message": "vLLM server stopped successfully."}, 200

    @route_handler.register_route("/vllm/status", "vllm_status", methods=["GET"])
    def vllm_status_command(
        self,
    ) -> tuple[VllmStatusResponse, int]:
        """Handle the ``/vllm/status`` command to get the status of the vLLM server.

        This method is intended to be used as a Flask route handler for the
        ``/vllm/status`` endpoint. It checks the status of the vLLM server and returns
        the status as a JSON response.

        Returns
        -------
        response : VllmStatusResponse
            A dictionary containing the status of the vLLM server.
        status_code : int
            The HTTP status code for the response, which for now will always be 200,
            indicating that the request was successful.
        """

        status, error_message = self._get_vllm_server_status()
        return {"status": status, "error": error_message}, 200

    @_check_context_manager_entered
    def _start_vllm_server(self, model_name: str) -> str:
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

        if not self.context_entered:
            raise RuntimeError("LanguageModelServer must be used as a context manager.")

        if self.vllm_model_name == model_name and self.vllm_server_process is not None:
            logger.info(f"vLLM server is already running with model '{model_name}'.")
            return "vLLM server is already running with the specified model."

        try:
            self._stop_vllm_server()
        except VllmServerNotRunningError:
            pass

        if shutil.which("vllm") is None:
            raise VllmNotInstalledError

        if self.subprocess_output_destination == "log_file":
            output_kwargs = {
                "stdout": self.vllm_log_file,
                "stderr": STDOUT,
            }
        else:
            output_kwargs = {}

        self.vllm_server_process = Popen(
            [
                "vllm",
                "serve",
                model_name,
                "--port",
                str(self.vllm_port),
            ],
            **output_kwargs,
        )

        self.vllm_model_name = model_name

        logger.info(
            f"Started vLLM server with model '{model_name}' on port {self.vllm_port}."
        )
        return "vLLM server started successfully with the specified model."

    @_check_context_manager_entered
    def _stop_vllm_server(self, timeout: float = 5.0):
        """Stop the vLLM server if it is running.

        Parameters
        ----------
        timeout : float, default=5.0
            The maximum time to wait for the server to stop, by default 5.0 seconds. If
            it takes longer than this, the process will be killed.

        Raises
        ------
        VllmServerNotRunningError
            If the vLLM server is not running.
        """

        if self.vllm_server_process is None:
            raise VllmServerNotRunningError

        logger.info("Stopping vLLM server...")

        self.vllm_server_process.terminate()

        try:
            self.vllm_server_process.wait(timeout=timeout)
        except TimeoutExpired:
            logger.warning(
                f"vLLM server not terminated after {timeout}s. Sending kill signal"
            )
            self.vllm_server_process.kill()

        self.vllm_server_process = None

        logger.info("vLLM server stopped.")

    @_check_context_manager_entered
    def _get_vllm_server_status(self, timeout: float = 0.5) -> ServerStatus:
        """Get the status of the vLLM server by trying to list available models.

        Parameters
        ----------
        timeout : float, default=5.0
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

        if self.vllm_server_process is None:
            return "not_started", None

        if self.vllm_server_process.poll() is not None:
            return "exited", None

        try:
            response = httpx.get(f"{self.vllm_server_url}/v1/models", timeout=timeout)
            response.raise_for_status()
            return "online", None
        except ConnectError as e:
            return "not_accepting_connections", str(e)
        except HTTPError as e:
            if 500 <= e.response.status_code < 600:
                return "server_error", str(e)
            else:
                return "other_error", str(e)
