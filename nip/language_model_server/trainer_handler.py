"""A module for handling training jobs in a language model server."""

from typing import Optional, get_args, ClassVar
from datetime import datetime
import logging
from pathlib import Path
from asyncio import wait_for, TimeoutError, TaskGroup, Queue, Lock
from asyncio.subprocess import create_subprocess_exec, STDOUT, Process
from tempfile import TemporaryDirectory
import json

import torch

from filelock import FileLock

from jinja2 import (
    Environment as JinjaEnvironment,
    PackageLoader,
    TemplateNotFound,
    StrictUndefined,
)

from nip.constants import (
    LM_SERVER_TRAINING_LOG_DIR,
    LM_SERVER_TRAINING_STATUS_DIR,
    PACKAGE_ROOT,
    HF_SELF_HOSTED_FINETUNED_REPO_PREFIX,
)
from nip.utils.data import convert_dpo_dataset_to_hugging_face
from nip.language_model_server.types import (
    SubprocessOutputDestination,
    TrainingJobStatus,
    TrainingJobInfo,
    CreateTrainingJobRequest,
)
from nip.utils.env import get_env_var
from nip.language_model_server.exceptions import (
    MaxTrainingJobsReachedError,
    TrainingJobNotFoundServerError,
    AccelerateConfigNotFoundError,
)
from nip.language_model_server.config import Settings


logger = logging.getLogger(__name__)


class TrainingJob:
    """A class representing a training job for fine-tuning a language model.

    Parameters
    ----------
    request : CreateTrainingJobRequest
        A request containing the configuration for the training job, the dataset to be
        used, and an optional job ID suffix.
    settings : Settings
        The configuration settings for the language model server.
    subprocess_output_destination : SubprocessOutputDestination,
    default="stdout_std_err"
        The destination for subprocess output. See :const:`SubprocessOutputDestination
        <nip.language_model_server.types.SubprocessOutputDestination>` for possible
        values.
    """

    @property
    def status_filepath(self) -> Path:
        """The path to the status file for this training job."""
        return LM_SERVER_TRAINING_STATUS_DIR.joinpath(f"{self.id}.status")

    @property
    def status_lock_filepath(self) -> Path:
        """The path to the status lock file for this training job."""
        return LM_SERVER_TRAINING_STATUS_DIR.joinpath(f"{self.id}.status.lock")

    @property
    def error_filepath(self) -> Path:
        """The path to the file with error information for this training job."""
        return LM_SERVER_TRAINING_STATUS_DIR.joinpath(f"{self.id}.error")

    @property
    def error_lock_filepath(self) -> Path:
        """The path to the error lock file for this training job."""
        return LM_SERVER_TRAINING_STATUS_DIR.joinpath(f"{self.id}.error.lock")

    @property
    def temporary_directory_path(self) -> Path:
        """The path to the temporary directory for this training job."""
        return Path(self.temporary_directory.name)

    @property
    def hyperparameters_filepath(self) -> Path:
        """The path to the hyperparameters file for the training script."""
        return self.temporary_directory_path.joinpath("hyperparameters.json")

    @property
    def dataset_filepath(self) -> Path:
        """The path to the dataset file for the training script."""
        return self.temporary_directory_path.joinpath("dataset.jsonl")

    def __init__(
        self,
        request: CreateTrainingJobRequest,
        settings: Settings,
        subprocess_output_destination: SubprocessOutputDestination = "stdout_std_err",
    ):
        self.settings = settings

        self.config = request.config
        self.dataset = convert_dpo_dataset_to_hugging_face(request.dataset)
        self.job_name = request.job_name
        self.subprocess_output_destination = subprocess_output_destination

        self.jinja_environment = JinjaEnvironment(
            loader=PackageLoader("nip", "language_model_server/templates"),
            autoescape=True,
            undefined=StrictUndefined,
        )

        time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        sanitised_model_name = self.config.model_name.replace("/", "_")
        self.id = f"{sanitised_model_name}_{self.config.method}_{time_string}"
        self.repo_name = f"{HF_SELF_HOSTED_FINETUNED_REPO_PREFIX}{self.config.method}"
        if self.job_name:
            self.id += f"_{self.job_name}"
            self.repo_name += f"_{self.job_name}"
        self.repo_name += f"_{time_string}_{sanitised_model_name}"
        self.repo_name = self.repo_name[:96]  # Ensure the repo name is within the limit
        self.new_model_name = (
            f"{get_env_var('HF_SELF_HOSTED_FINETUNE_NAMESPACE')}/{self.repo_name}"
        )

        if self.subprocess_output_destination == "log_file":
            LM_SERVER_TRAINING_LOG_DIR.mkdir(parents=True, exist_ok=True)
            self.log_filepath = LM_SERVER_TRAINING_LOG_DIR.joinpath(f"{self.id}.log")
            self.log_file = open(self.log_filepath, "w")
            logger.info(f"Training job log file created at {self.log_filepath!r}")

        self.temporary_directory = TemporaryDirectory()

        self.process: Optional[Process] = None
        self._status = "pending"

    async def start(self):
        """Start the training job by launching a subprocess."""

        if self._status != "pending":
            logger.warning(
                f"Attempted to start training job {self.id!r}, but its status is "
                f"{self._status!r}."
            )
            return

        with open(self.hyperparameters_filepath, "w") as f:
            f.write(self.config.model_dump_json(indent=4))

        with open(self.dataset_filepath, "w") as f:
            for item in self.dataset:
                f.write(json.dumps(item) + "\n")

        accelerate_config_path = self._get_accelerate_config_path()
        if accelerate_config_path is not None:
            accelerate_args = ["--config_file", str(accelerate_config_path)]
        else:
            accelerate_args = []

        if self.subprocess_output_destination == "log_file":
            output_kwargs = {
                "stdout": self.log_file,
                "stderr": STDOUT,
            }
        else:
            output_kwargs = {}

        logger.info(
            f"Starting training job {self.id!r} with configuration file at "
            f"{self.hyperparameters_filepath!r} and dataset at "
            f"{self.dataset_filepath!r}."
        )

        self.process = await create_subprocess_exec(
            "accelerate",
            "launch",
            *accelerate_args,
            str(PACKAGE_ROOT / "language_model_server" / "trainers" / "dpo.py"),
            "--training-config-path",
            str(self.hyperparameters_filepath),
            "--dataset-path",
            str(self.dataset_filepath),
            "--job-id",
            self.id,
            "--new-model-name",
            self.new_model_name,
            **output_kwargs,
        )

    async def close(self):
        """Perform cleanup after the training job."""

        logger.info(f"Closing training job {self.id!r}.")

        await self.cancel()

        self.temporary_directory.cleanup()

        if self.subprocess_output_destination == "log_file":
            self.log_file.close()

    async def get_status(self, timeout: float = 1.0) -> TrainingJobStatus:
        """Get the current status of the training job.

        Parameters
        ----------
        timeout : float, default=1.0
            The maximum time to wait for the status file to be available.

        Returns
        -------
        status : TrainingJobStatus
            The current status of the training job. See
            :const:`nip.language_model_server.types.TrainingJobStatus
            <TrainingJobStatus>` for possible values.
        """

        if not self.status_filepath.exists():
            if self.process is None:
                if self._status == "pending":
                    return "pending"
                else:
                    logger.warning(
                        f"Status file {self.status_filepath!r} does not exist, but "
                        f"process is None and status is {self._status!r}."
                    )
                    return self._status

            # If the status file does not exist and the process has exited with a
            # non-zero return code, and the job has not been cancelled by the user,
            # we assume the job has crashed.
            returncode = self.process.returncode
            if (
                returncode is not None
                and returncode != 0
                and self._status != "cancelled"
            ):
                self._status = "crashed"
            return self._status

        with FileLock(self.status_lock_filepath, timeout=timeout):
            with open(self.status_filepath, "r") as f:
                status = f.read().strip()
                if status in get_args(TrainingJobStatus):
                    self._status = status
                else:
                    logger.warning(
                        f"Unknown status {status!r} found in {self.status_filepath}."
                    )
                    self._status = "unknown"

        returncode = self.process.returncode
        if returncode is not None:
            # If the process has exited with a non-zero return code but the last
            # reported status is still "running" or "starting", the job must have
            # crashed.
            if returncode != 0 and self._status in ("running", "starting", "unknown"):
                self._status = "crashed"

            elif returncode == 0 and self._status != "succeeded":
                logger.warning(
                    f"Process for job {self.id!r} exited with code 0, but status "
                    f"is {self._status!r} rather than 'succeeded'."
                )

        elif self._status in ("succeeded", "crashed", "interrupted", "cancelled"):
            logger.warning(
                f"Process for job {self.id!r} is still running, but status is "
                f"{self._status!r}."
            )

        return self._status

    async def get_error(self, timeout: float = 1.0) -> str:
        """Get the error message for the training job, if any.

        Parameters
        ----------
        timeout : float, default=1.0
            The maximum time to wait for the error file to be available.

        Returns
        -------
        error_message : str
            The error message if the job has crashed or failed, or an empty string if
            there is no error message.
        """

        if not self.error_filepath.exists():
            return ""

        with FileLock(self.error_lock_filepath, timeout=timeout):
            with open(self.error_filepath, "r") as f:
                return f.read().strip()

    async def cancel(self, timeout: float = 5.0):
        """Cancel the training job.

        This method attempts to terminate the training job process and update its
        status.

        Parameters
        ----------
        timeout : float, default=5.0
            The maximum time to wait for the process to terminate after sending the
            termination signal.
        """

        if self._status == "pending":
            self._status = "cancelled"
            logger.info(f"Cancelling training job {self.id!r} which was not started.")
            return

        if (
            await self.get_status() not in ("running", "starting")
            or self.process.returncode is not None
        ):
            return

        logger.info(f"Cancelling training job {self.id!r}.")

        self.process.terminate()

        try:
            await wait_for(self.process.wait(), timeout=timeout)
        except TimeoutError:
            logger.warning(
                f"Training job {self.id!r} not terminated after {timeout}s. "
                f"Sending kill signal"
            )
            self.process.kill()

        self._status = "cancelled"

        logger.info(f"Training job {self.id!r} cancelled.")

    async def get_info(self) -> TrainingJobInfo:
        """Create a representation of the training job.

        Returns
        -------
        info : TrainingJobInfo
            A data structure containing information about the training job, including
            its ID, status, and configuration.
        """

        return TrainingJobInfo(
            job_id=self.id,
            status=await self.get_status(),
            config=self.config,
            new_model_name=self.new_model_name,
            error_message=await self.get_error(),
        )

    def _get_accelerate_config_path(self) -> Optional[Path]:
        """Get the path to the accelerate configuration file.

        Tries to resolve the path based on the settings, looking first in the current
        working directory, then in the package's templates directory for jinja2
        templates.

        If the filename ends with `.jinja2`, it will be treated as a Jinja2 template and
        rendered.

        If the path is empty, it returns None, indicating that no configuration file is
        specified.

        Returns
        -------
        accelerate_config_path : Path | None
            The path to the accelerate configuration file, or None if no configuration
            file is specified.
        """

        if self.settings.accelerate_config_path == "":
            return None

        if self.settings.parent_script_cwd is not None:
            absolute_path = (
                Path(self.settings.parent_script_cwd)
                / self.settings.accelerate_config_path
            ).resolve()
        else:
            absolute_path = Path(self.settings.accelerate_config_path).resolve()

        if absolute_path.suffix != ".jinja2":
            if absolute_path.is_file():
                return absolute_path
            else:
                raise AccelerateConfigNotFoundError(
                    self.settings.accelerate_config_path
                )

        if absolute_path.is_file():
            with open(absolute_path, "r") as f:
                template = self.jinja_environment.from_string(f.read())
        else:
            try:
                template = self.jinja_environment.get_template(
                    self.settings.accelerate_config_path
                )
            except TemplateNotFound:
                raise AccelerateConfigNotFoundError(
                    self.settings.accelerate_config_path
                )

        num_gpus = torch.cuda.device_count()

        # bfloat16 mixed precision is only available on NVIDIA GPUs with compute
        # capability 8.0 or higher.
        if torch.cuda.get_device_capability()[0] >= 8:
            mixed_precision = "bf16"
        else:
            mixed_precision = "fp16"

        rendered_path = self.temporary_directory_path.joinpath("accelerate_config.yaml")
        with open(rendered_path, "w") as f:
            f.write(
                template.render(
                    num_gpus=num_gpus,
                    mixed_precision=mixed_precision,
                    distributed_type=self.config.distributed_type.upper(),
                )
            )

        return rendered_path


class TrainerHandler:
    """A handler for managing training jobs in the language model server.

    Note
    ----
    This class is designed to be used in an asynchronous context.

    Parameters
    ----------
    settings : Settings
        The configuration settings for the language model server, including
        maximum number of concurrent training jobs and subprocess output destination.
    """

    add_job_lock: ClassVar[Lock] = Lock()
    """A lock to ensure that adding jobs is thread-safe."""

    @property
    def max_running_jobs(self) -> int:
        """The maximum number of training jobs that can run concurrently."""
        return self.settings.max_training_jobs

    @property
    def subprocess_output_destination(self) -> SubprocessOutputDestination:
        """The destination for subprocess output."""
        return self.settings.subprocess_output_destination

    def __init__(self, settings: Settings):
        self.settings = settings

        self.jobs: dict[str, TrainingJob] = {}

    async def close(self):
        """Close all training jobs and perform cleanup."""

        async with TaskGroup() as task_group:
            for job in self.jobs.values():
                task_group.create_task(job.close())

    async def add_job(self, request: CreateTrainingJobRequest) -> TrainingJob:
        """Add a new training job to the handler.

        Parameters
        ----------
        request : CreateTrainingJobRequest
            A request containing the configuration for the training job, the dataset to
            be used, and an optional job ID suffix.
        subprocess_output_destination : SubprocessOutputDestination,
        default="stdout_std_err"
            The destination for subprocess output. See
            :const:`SubprocessOutputDestination
            <nip.language_model_server.types.SubprocessOutputDestination>` for possible
            values.

        Returns
        -------
        job : TrainingJob
            The created training job instance.
        """

        async with self.add_job_lock:

            if await self._count_running_jobs() >= self.max_running_jobs:
                raise MaxTrainingJobsReachedError(self.max_running_jobs)

            job = TrainingJob(
                request, self.settings, self.subprocess_output_destination
            )
            await job.start()
            self.jobs[job.id] = job

            return job

    async def get_job_info(self, job_id: str) -> TrainingJobInfo:
        """Get info about a training job by its ID.

        Parameters
        ----------
        job_id : str
            The unique identifier for the training job.

        Returns
        -------
        job : TrainingJobInfo
            A data structure with information about the job.

        Raises
        ------
        TrainingJobNotFoundError
            If no training job with the specified ID exists.
        """

        if job_id not in self.jobs:
            raise TrainingJobNotFoundServerError(job_id)

        return await self.jobs[job_id].get_info()

    async def cancel_job(self, job_id: str, timeout: float = 5.0):
        """Cancel a training job by its ID.

        Parameters
        ----------
        job_id : str
            The unique identifier for the training job.
        timeout : float, default=5.0
            The maximum time to wait for the job to be cancelled.

        Raises
        ------
        TrainingJobNotFoundError
            If no training job with the specified ID exists.
        """

        try:
            job = self.jobs[job_id]
        except KeyError:
            raise TrainingJobNotFoundServerError(job_id)
        await job.cancel(timeout)

    async def get_training_job_infos(self) -> list[TrainingJobInfo]:
        """List all training jobs.

        Returns
        -------
        jobs : list[TrainingJobInfo]
            A list of ``TrainingJobInfo`` representing every training job.
        """
        return [await job.get_info() for job in self.jobs.values()]

    async def _count_running_jobs(self) -> int:
        """Count the number of currently running jobs.

        Returns
        -------
        count : int
            The number of currently running jobs.
        """

        queue = Queue()

        async def count_job(job: TrainingJob) -> bool:
            """Check if a job is running or starting."""
            if await job.get_status() in ("running", "starting"):
                queue.put_nowait(1)

        async with TaskGroup() as task_group:
            for job in self.jobs.values():
                task_group.create_task(count_job(job))

        return queue.qsize()
