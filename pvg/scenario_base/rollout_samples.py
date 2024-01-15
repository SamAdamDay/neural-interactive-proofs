"""Base classes for sampling and visualising rollouts.

Sample rollouts are saved and loaded from W&B. We then visualise them.
"""

from abc import ABC, abstractmethod
from tempfile import TemporaryDirectory
import os
import pickle
from textwrap import indent

import numpy as np

import torch

from tensordict import TensorDict

import wandb

from pvg.experiment_settings import ExperimentSettings
from pvg.utils.data import tensordict_to_numpy_dict
from pvg.constants import (
    ROLLOUT_SAMPLE_ARTIFACT_PREFIX,
    ROLLOUT_SAMPLE_ARTIFACT_TYPE,
    ROLLOUT_SAMPLE_FILENAME,
    WANDB_ENTITY,
    WANDB_PROJECT,
)


class RolloutSampler:
    """Samples rollouts from an environment and saves them to W&B.

    Parameters
    ----------
    settings : ExperimentSettings
        The settings of the experiment.
    """

    def __init__(self, settings: ExperimentSettings):
        self.settings = settings

        # Make sure we have a W&B run
        if settings.wandb_run is None:
            raise ValueError("RolloutSampler requires a W&B run")

        self._wandb_run = settings.wandb_run

        # Load the W&B API
        self._wandb_api = wandb.Api()

        # Create and initial artifact
        self._artifact = wandb.Artifact(
            name=f"{ROLLOUT_SAMPLE_ARTIFACT_PREFIX}{self._wandb_run.name}",
            type=ROLLOUT_SAMPLE_ARTIFACT_TYPE,
        )
        self._wandb_run.use_artifact(self._artifact)

    def sample_and_save_rollouts(self, data: TensorDict, iteration: int):
        """Sample rollouts from the given data and save them to W&B.

        Parameters
        ----------
        data : TensorDict
            The data to sample rollouts from.
        iteration : int
            The iteration of training we are on.
        """

        # Sample rollouts randomly from the data
        bids = torch.where(
            data["next", "done"],
            torch.rand_like(data["next", "done"], dtype=torch.float32) + 1,
            0.0,
        )
        _, index_flat = torch.topk(bids.flatten(), self.settings.num_rollout_samples)
        batch_ids, episode_ids = np.unravel_index(
            index_flat.detach().cpu().numpy(), bids.shape
        )

        rollout_samples: list[TensorDict] = []

        for batch_id, episode_id in zip(batch_ids.flat, episode_ids.flat):
            # Determine the start of the episode
            episode_start_id = episode_id - 1
            while (
                episode_start_id >= 0
                and not data["next", "done"][batch_id, episode_start_id]
            ):
                episode_start_id -= 1
            episode_start_id += 1

            # Get the rollout data
            rollout_td = data[batch_id, episode_start_id : episode_id + 1]
            rollout_samples.append(tensordict_to_numpy_dict(rollout_td))

        # Save the rollout samples to W&B
        self._artifact = self._artifact.new_draft()
        with TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, ROLLOUT_SAMPLE_FILENAME)
            with open(file_path, "wb") as f:
                pickle.dump(rollout_samples, f)
            self._artifact.add_file(file_path, f"iteration_{iteration}")
        self._wandb_run.use_artifact(self._artifact)


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
