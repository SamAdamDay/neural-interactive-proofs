"""Base class for rollouts sampled during a run and saved to W&B.

This class is used to load rollout samples from W&B and visualise them.
"""

from abc import ABC, abstractmethod
from tempfile import TemporaryDirectory
import os
import pickle
from textwrap import indent
import re
from typing import TypeVar, Optional

import wandb

from pvg.parameters import HyperParameters, ScenarioType
from pvg.constants import ROLLOUT_SAMPLE_ARTIFACT_PREFIX
from pvg.utils.env import get_env_var


class IterationNotFoundError(Exception):
    """Error raised when the iteration is not found in the W&B run."""

    def __init__(self, run_id: str, iteration: int, available_iterations: list[int]):
        self.run_id = run_id
        self.iteration = iteration
        self.available_iterations = available_iterations
        super().__init__(
            f"Iteration {iteration} not found in W&B run {run_id}. "
            f"Available iterations: {available_iterations}"
        )


class RolloutSamples(ABC):
    """A collection of rollout samples loaded from W&B.

    Since this class has some cleanup, it should be used as a context manager, or else
    the `finish` method should be called manually (see examples).

    Parameters
    ----------
    run_id : str
        The ID of the W&B run.
    iteration : int
        The iteration of the rollout samples to load.
    wandb_entity : str, optional
        The W&B entity to load the rollout samples from. Defaults to the default entity.
    wandb_project : str, optional
        The W&B project to load the rollout samples from. Defaults to the default
        project.
    silence_wandb : bool, default=True
        Whether to suppress W&B output.

    Examples
    --------
    Using the `RolloutSamples` class as a context manager: >>> with
    RolloutSamples(run_id, iteration) as rollout_samples: ...
    rollout_samples.visualise()

    Or manually calling the `finish` method: >>> rollout_samples =
    RolloutSamples(run_id, iteration) >>> rollout_samples.visualise() >>>
    rollout_samples.finish()
    """

    def __init__(
        self,
        run_id: str,
        iteration: int,
        wandb_entity: Optional[str] = None,
        wandb_project: Optional[str] = None,
        silence_wandb: bool = True,
    ):
        self.run_id = run_id
        self.iteration = iteration

        if wandb_entity is None:
            wandb_entity = get_env_var("WANDB_ENTITY")
        if wandb_project is None:
            wandb_project = get_env_var("WANDB_PROJECT")

        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project

        if silence_wandb:
            os.environ["WANDB_SILENT"] = "true"

        # Load the W&B run
        self._wandb_run = wandb.init(
            id=run_id, entity=wandb_entity, project=wandb_project, resume="must"
        )

        # Load the agent names in order
        agent_config = self._wandb_run.config["agents"]
        if "_agent_order" in agent_config:
            self.agent_names: list[str] = self._wandb_run.config["agents"][
                "_agent_order"
            ]
        else:
            self.agent_names = list(agent_config.keys())

        # Load the rollout samples from W&B
        artifact = self._wandb_run.use_artifact(
            f"{ROLLOUT_SAMPLE_ARTIFACT_PREFIX}{run_id}:latest"
        )
        with TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, f"iteration_{iteration}")
            artifact.download(root=temp_dir)
            try:
                with open(file_path, "rb") as f:
                    self._rollout_samples = pickle.load(f)
            except FileNotFoundError:
                available_iterations = [
                    int(filename.split("_")[-1])
                    for filename in os.listdir(temp_dir)
                    if re.match(r"iteration_\d+", filename)
                ]
                available_iterations.sort()
                raise IterationNotFoundError(run_id, iteration, available_iterations)

    @abstractmethod
    def visualise(self):
        """Visualise the rollout samples."""
        pass

    def finish(self):
        """Finish the W&B run."""
        self._wandb_run.finish()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finish()

    def __getitem__(self, key):
        return self._rollout_samples[key]

    def __len__(self):
        return len(self._rollout_samples)

    def __iter__(self):
        return iter(self._rollout_samples)

    def __contains__(self, item):
        return item in self._rollout_samples

    def __str__(self):
        return str(self._rollout_samples)

    def __repr__(self):
        run_id_str = indent(f"run_id={self.run_id}")
        iteration_str = indent(f"iteration={self.iteration}")
        wandb_entity_str = indent(f"wandb_entity={self.wandb_entity}")
        wandb_project_str = indent(f"wandb_project={self.wandb_project}")
        num_samples_str = indent(f"num_samples={len(self._rollout_samples)}")
        args = ",\n".join(
            [
                run_id_str,
                iteration_str,
                wandb_entity_str,
                wandb_project_str,
                num_samples_str,
            ]
        )
        return f"RolloutSamples({args})"


ROLLOUT_SAMPLES_CLASS_REGISTRY: dict[ScenarioType, type[RolloutSamples]] = {}

R = TypeVar("R", bound=RolloutSamples)


def register_rollout_samples_class(scenario: ScenarioType):
    """Register a subclass of RolloutSamples with a scenario.

    Parameters
    ----------
    scenario : ScenarioType
        The scenario with which to register the subclass.
    """

    def decorator(cls: type[R]) -> type[R]:
        ROLLOUT_SAMPLES_CLASS_REGISTRY[scenario] = cls
        return cls

    return decorator


def build_rollout_samples(hyper_params: HyperParameters):
    """Build a subclass of RolloutSamples based on the parameters.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters for the experiment.

    Returns
    -------
    RolloutSamples
        The subclass of RolloutSamples for the experiment.
    """
    return ROLLOUT_SAMPLES_CLASS_REGISTRY[hyper_params.scenario](hyper_params)
