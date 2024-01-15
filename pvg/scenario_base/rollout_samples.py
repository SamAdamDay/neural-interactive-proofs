"""Base classes for sampling and visualising rollouts.

Sample rollouts are saved and loaded from W&B. We then visualise them.
"""

from abc import ABC, abstractmethod
from tempfile import TemporaryDirectory
import os
import pickle
from textwrap import indent

import wandb

from pvg.constants import (
    ROLLOUT_SAMPLE_ARTIFACT_PREFIX,
    WANDB_ENTITY,
    WANDB_PROJECT,
)


class RolloutSamples(ABC):
    """A collection of rollout samples loaded from W&B.

    Parameters
    ----------
    run_id : str
        The ID of the W&B run.
    iteration : int
        The iteration of the rollout samples to load.
    wandb_entity : str, default=WANDB_ENTITY
        The W&B entity to load the rollout samples from.
    wandb_project : str, default=WANDB_PROJECT
        The W&B project to load the rollout samples from.
    """

    def __init__(
        self,
        run_id: str,
        iteration: int,
        wandb_entity: str = WANDB_ENTITY,
        wandb_project: str = WANDB_PROJECT,
    ):
        self.run_id = run_id
        self.iteration = iteration
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project

        # Load the W&B API
        wandb_api = wandb.Api()

        # Load the rollout samples from W&B
        artifact = wandb_api.artifact(
            f"{wandb_entity}/{wandb_project}/{ROLLOUT_SAMPLE_ARTIFACT_PREFIX}{run_id}"
            f":latest"
        )
        with TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, f"iteration_{iteration}")
            artifact.download(root=temp_dir)
            with open(file_path, "rb") as f:
                self._rollout_samples = pickle.load(f)

        # Load the W&B run
        self._wandb_run = wandb_api.run(f"{wandb_entity}/{wandb_project}/{run_id}")

        # Load the agent names in order
        self.agent_names = self._wandb_run.config["agents"]["agents_order"]

    @abstractmethod
    def visualise(self):
        """Visualise the rollout samples."""
        pass

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
