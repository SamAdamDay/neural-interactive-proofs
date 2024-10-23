"""Base classes for all trainers."""

from abc import ABC, abstractmethod
from contextlib import ExitStack
from typing import ContextManager, Callable, Optional
import functools
import inspect
from pathlib import Path
from dataclasses import dataclass
import dataclasses
import json
from logging import getLogger
import sys

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import wandb
from wandb import Artifact

from tqdm import tqdm

from pvg.scenario_base.agents import AgentState
from pvg.parameters import HyperParameters
from pvg.factory import ScenarioInstance
from pvg.experiment_settings import ExperimentSettings
from pvg.utils.hyper_params import get_agent_part_flags
from pvg.constants import (
    EXPERIMENT_STATE_DIR,
    CHECKPOINT_STATE_ARTIFACT_PREFIX,
    CHECKPOINT_STATE_ARTIFACT_TYPE,
)


class CheckPointNotFoundError(Exception):
    """Exception raised when a checkpoint is not found."""


class Trainer(ABC):
    """Base class for all trainers.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    scenario_instance : ScenarioInstance
        The components of the experiment.
    settings : ExperimentSettings
        The instance-specific settings of the experiment, like device, logging, etc.
    """

    @dataclass
    class State:
        """Base class for storing the state of an experiment.

        This class should hold all the data that needs to be saved and restored when
        checkpointing an experiment.

        Parameters
        ----------
        iteration : int
            The current iteration number.
        agents : dict[str, AgentCheckpoint]
            The checkpoints of the agents.
        """

        iteration: int = 0
        agents: dict[str, AgentState] = dataclasses.field(default_factory=dict)

    @property
    def max_message_rounds(self) -> int:
        """The maximum number of message rounds in the protocol."""
        return self.scenario_instance.protocol_handler.max_message_rounds

    @property
    def agent_names(self) -> list[str]:
        """The names of the agents in the scenario."""
        return self.scenario_instance.protocol_handler.agent_names

    def __init__(
        self,
        hyper_params: HyperParameters,
        scenario_instance: ScenarioInstance,
        settings: ExperimentSettings,
    ):
        self.hyper_params = hyper_params
        self.scenario_instance = scenario_instance
        self.settings = settings

        if settings.logger is None:
            self.settings.logger = getLogger(__name__)

        # Try to restore the experiment state from a checkpoint. If no checkpoint is
        # available, initialise the state.
        if settings.do_not_load_checkpoint or settings.test_run:
            self._initialise_state()
        else:
            try:
                self.state = self._load_checkpoint()
                self.settings.logger.info(
                    f"Restoring experiment state from iteration {self.state.iteration}"
                )
            except CheckPointNotFoundError:
                self._initialise_state()

    @abstractmethod
    def train(self):
        """Train the agents."""
        pass

    def save_checkpoint(self, log: bool = True):
        """Save the state of the experiment to a checkpoint.

        Parameters
        ----------
        log : bool, default=True
            Whether to log the checkpointing.
        """

        # If we are running a test run, we don't want to save the rollouts
        if self.settings.test_run:
            return

        # Create the checkpoint directory if it doesn't exist
        self.checkpoint_base_dir.mkdir(parents=True, exist_ok=True)

        with open(self.checkpoint_state_path, "w") as f:
            json.dump(dataclasses.asdict(self.state), f, indent=4)

        # Save the parameters to a separate file
        with open(self.checkpoint_params_path, "w") as f:
            json.dump(self.hyper_params.to_dict(), f, sort_keys=True, indent=4)

        # If using W&B, also log the checkpoint as an artifact
        if self.settings.wandb_run is not None:
            artifact = wandb.Artifact(
                f"{CHECKPOINT_STATE_ARTIFACT_PREFIX}{self.settings.wandb_run.name}",
                CHECKPOINT_STATE_ARTIFACT_TYPE,
            )
            artifact.add_file(self.checkpoint_state_path)
            self.settings.wandb_run.log_artifact(artifact)

        if log:
            self.settings.logger.info(
                f"Saved experiment state to '{self.checkpoint_state_path}'"
            )

    def _initialise_state(self):
        """Initialise the state of the experiment.

        This method should be implemented by subclasses to initialise the state of the
        experiment. This is called at the beginning of training when starting from
        scratch (i.e. not restoring from a checkpoint).
        """

    @classmethod
    def get_checkpoint_base_dir_from_run_id(cls, run_id: str) -> Path:
        """Get the base directory for a checkpoint from a run ID.

        Parameters
        ----------
        run_id : str
            The run ID.

        Returns
        -------
        checkpoint_base_dir : Path
            The path to the base directory for the checkpoint.
        """
        return Path(EXPERIMENT_STATE_DIR, run_id)

    @property
    def checkpoint_base_dir(self) -> Path | None:
        """The path to the directory containing the checkpoint."""

        if self.settings.run_id is None:
            return None

        return self.get_checkpoint_base_dir_from_run_id(self.settings.run_id)

    @property
    def checkpoint_state_path(self) -> Path | None:
        """The path to the checkpoint state file."""

        if self.checkpoint_base_dir is None:
            return None

        return self.checkpoint_base_dir.joinpath("state.json")

    @property
    def checkpoint_params_path(self) -> Path | None:
        """The path to the parameters file for the checkpoint."""

        if self.checkpoint_base_dir is None:
            return None

        return self.checkpoint_base_dir.joinpath("hyper_params.json")

    def _load_checkpoint(self) -> State:
        """Load the experiment state from a checkpoint, if available.

        Returns
        -------
        state : State
            The state of the experiment.

        Raises
        ------
        CheckPointNotFoundError
            If the checkpoint file is not found.
        """

        # If using W&B, try to download the state from the artifact first
        if (
            self.settings.wandb_run is not None
            and not self.checkpoint_state_path.is_file()
        ):
            artifact_name = (
                f"{CHECKPOINT_STATE_ARTIFACT_PREFIX}{self.settings.wandb_run.name}"
                f":latest"
            )
            try:
                artifact: Artifact = self.settings.wandb_run.use_artifact(
                    artifact_name,
                    type=CHECKPOINT_STATE_ARTIFACT_TYPE,
                )
                artifact.download(self.checkpoint_base_dir)
            except wandb.errors.CommError as e:
                # W&B doesn't use subclasses for errors, so we have to check the
                # message. If the error was not that the artifact was not found, we
                # re-raise it.
                if f"artifact '{artifact_name}' not found in" not in e.message:
                    raise e

        if (
            self.checkpoint_state_path is None
            or not self.checkpoint_state_path.exists()
            or not self.checkpoint_state_path.is_file()
        ):
            raise CheckPointNotFoundError

        # Check if the parameters in the checkpoint match the current parameters
        with open(self.checkpoint_params_path, "r") as f:
            if (
                json.dumps(self.hyper_params.to_dict(), sort_keys=True, indent=4)
                != f.read()
            ):
                print(  # noqa: T201
                    "The parameters in the checkpoint do not match the current "
                    "parameters."
                )
                while True:
                    response = input("Do you want to continue? [Y/n]: ")
                    if response.lower() == "y" or response == "":
                        break
                    elif response.lower() == "n":
                        sys.exit(1)
                    else:
                        print(  # noqa: T201
                            "Invalid response. Please enter 'y' or 'n'."
                        )

        with open(self.checkpoint_state_path, "r") as f:
            state_dict = json.load(f)

        return self.State(**state_dict)

    def _add_file_to_wandb_artifact(
        self, artifact_name: str, artifact_type: str, file_path: Path
    ):
        """Add a file to a W&B artifact, creating the artifact if it doesn't exist.

        If the artifact already exists, we add the file to the existing artifact,
        creating a new version.

        Parameters
        ----------
        artifact_name : str
            The name of the artifact to add the file to. This should not contain an
            alias or version, as we always add to the latest version.
        artifact_type : str
            The type of the artifact.
        file_path : Path
            The path to the file to add to the artifact.
        """

        try:
            saved_artifact: Artifact = self.settings.wandb_run.use_artifact(
                f"{artifact_name}:latest",
                type=artifact_type,
            )
            artifact = saved_artifact.new_draft()
        except wandb.errors.CommError as e:
            # W&B doesn't use subclasses for errors, so we have to check the message. If
            # the error was not that the artifact was not found, we re-raise it.
            if f"artifact '{artifact_name}:latest' not found in" not in e.message:
                raise e
            artifact = Artifact(name=artifact_name, type=artifact_type)

        artifact.add_file(file_path)
        self.settings.wandb_run.log_artifact(artifact)

    def get_total_num_iterations(self) -> int:
        """Get the total number of iterations that the trainer will run for.

        This is the sum of the number of iterations declared by methods decorated with
        `attach_progress_bar`.

        Returns
        -------
        total_iterations : int
            The total number of iterations.
        """
        total_iterations = 0
        for _, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, "_num_iterations_func"):
                total_iterations += method._num_iterations_func(self)
        return total_iterations


class TensorDictTrainer(Trainer, ABC):
    """Base class for trainers that use tensors and TensorDicts."""

    def __init__(
        self,
        hyper_params: HyperParameters,
        scenario_instance: ScenarioInstance,
        settings: ExperimentSettings,
    ):
        super().__init__(hyper_params, scenario_instance, settings)

        self.device = self.settings.device

        # Check if we need a critic and if it shares a body with the actor
        self.use_critic, self.use_single_body, _ = get_agent_part_flags(hyper_params)

    def _build_train_context(self, stack: ExitStack) -> list[ContextManager]:
        """Builds the context manager ExitStack for training.

        Takes as input an ExitStack and adds the appropriate context managers to it,
        then returns the context managers.

        Parameters
        ----------
        stack : ExitStack
            The ExitStack to add the context managers to. Note that this is modified
            in-place.

        Returns
        -------
        context_managers : list[ContextManager]
            The target context managers to be used in the training loop.
        """

        context_managers = []

        def add_context_manager(context_manager):
            context_managers.append(stack.enter_context(context_manager))

        # When running on the CPU we need to use the MATH backend in order to calculate
        # the derivative of the scaled dot product.
        if self.settings.device.type == "cpu":
            add_context_manager(sdpa_kernel(SDPBackend.MATH))

        # When running on the GPU we enable all backends, except when we want to disable
        # the efficient attention.
        elif not self.settings.enable_efficient_attention:
            add_context_manager(
                sdpa_kernel(
                    [
                        SDPBackend.MATH,
                        SDPBackend.CUDNN_ATTENTION,
                        SDPBackend.FLASH_ATTENTION,
                    ]
                )
            )

        else:
            add_context_manager(
                sdpa_kernel(
                    [
                        SDPBackend.MATH,
                        SDPBackend.CUDNN_ATTENTION,
                        SDPBackend.EFFICIENT_ATTENTION,
                        SDPBackend.FLASH_ATTENTION,
                    ]
                )
            )

        return context_managers

    def _build_test_context(self, stack: ExitStack) -> list[ContextManager]:
        """Builds the context manager ExitStack for testing.

        Takes as input an ExitStack and adds the appropriate context managers to it,
        then returns the context managers.

        Parameters
        ----------
        stack : ExitStack
            The ExitStack to add the context managers to. Note that this is modified
            in-place.

        Returns
        -------
        context_managers : list[ContextManager]
            The target context managers to be used in the testing loop.
        """

        context_managers = []

        def add_context_manager(context_manager):
            context_managers.append(stack.enter_context(context_manager))

        # When testing we enable all backends, except when we want to disable the
        # efficient attention.
        if not self.settings.enable_efficient_attention:
            add_context_manager(
                sdpa_kernel(
                    [
                        SDPBackend.MATH,
                        SDPBackend.CUDNN_ATTENTION,
                        SDPBackend.FLASH_ATTENTION,
                    ]
                )
            )

        else:
            add_context_manager(
                sdpa_kernel(
                    [
                        SDPBackend.MATH,
                        SDPBackend.CUDNN_ATTENTION,
                        SDPBackend.EFFICIENT_ATTENTION,
                        SDPBackend.FLASH_ATTENTION,
                    ]
                )
            )

        # We don't need gradients for testing.
        add_context_manager(torch.no_grad())

        return context_managers


class IterationContext:
    """Context manager for methods that run for a certain number of iterations.

    This context manager should be used in conjunction with the `attach_progress_bar`
    decorator. It manages the progress bar and ensures that the correct number of
    iterations are run.

    Parameters
    ----------
    trainer : Trainer
        The trainer instance that the method is called on.
    num_iterations : int
        The number of iterations that the method should run for.
    """

    def __init__(self, trainer: Trainer, num_iterations: int):
        self.trainer = trainer
        self.num_iterations = num_iterations
        self.progress_bar: Optional[tqdm] = None

    def __enter__(self):
        self.progress_bar = self.trainer.settings.tqdm_func(total=self.num_iterations)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.progress_bar.close()

    def set_description(self, description: str):
        """Set the description of the progress bar.

        Parameters
        ----------
        description : str
            The description to set.
        """
        self.progress_bar.set_description(description)

    def step(self):
        """Step the progress bar by one iteration."""
        self.progress_bar.update(1)
        self.trainer.settings.global_tqdm_step_fn()


def attach_progress_bar(
    num_iterations_func: Callable[[Trainer], int]
) -> Callable[[Callable], Callable]:
    """Decorator to attach a progress bar to a Trainer method.

    Decorate a method of a `Trainer` subclass with this decorator to have it run with a
    progress bar and declare the number of iterations that it runs for.

    The supplied function should take a Trainer instance as input and return the number
    of iterations.

    The intention is that once a `Trainer` subclass is instantiated, the number of
    iterations can be determined using `num_iterations_func`.

    This decorator wraps the decorated method in an `IterationContext` and assigns the
    `iteration_context` keyword argument to this context. This allows the method to
    interact with the progress bar and access the number of iterations.

    Note
    ----
    The number of iterations must be calculable as soon as the Trainer subclass is
    instantiated, so it should not depend on any other state.

    Parameters
    ----------
    num_iterations_func : Callable[[Trainer], int]
        A function that takes a Trainer instance as input and returns the number of
        iterations which the decorated method will run for.

    Returns
    -------
    decorator : Callable[[Callable], Callable]
        The decorator to apply to the method of a Trainer subclass.

    Example
    -------
    >>> class MyTrainer(Trainer):
    ...     @attach_progress_bar(lambda self: self.hyper_params.num_iterations)
    ...     def train(self, iteration_context: IterationContext):
    ...         for i in range(iteration_context.num_iterations):
    ...             iteration_context.step()
    """

    def decorator(method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapper(trainer: Trainer, *args, **kwargs):
            num_iterations = num_iterations_func(trainer)
            with IterationContext(trainer, num_iterations) as iteration_context:
                return method(
                    trainer, iteration_context=iteration_context, *args, **kwargs
                )

        # Attach the number of iterations function to the wrapper so that it can be
        # accessed later.
        wrapper._num_iterations_func = num_iterations_func

        return wrapper

    return decorator
