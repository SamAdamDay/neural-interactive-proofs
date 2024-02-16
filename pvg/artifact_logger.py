"""Logging training artifacts to W&B every so often.

- Samples rollouts from training and saves them to W&B. We then visualise them.
- Saves the model checkpoints to W&B.
"""

from tempfile import TemporaryDirectory
import os
import pickle
from pathlib import Path
from dataclasses import fields

import numpy as np

import torch

from tensordict import TensorDict

import wandb

from pvg.scenario_base.agents import Agent
from pvg.experiment_settings import ExperimentSettings
from pvg.utils.data import tensordict_to_numpy_dict
from pvg.constants import (
    ROLLOUT_SAMPLE_ARTIFACT_PREFIX,
    ROLLOUT_SAMPLE_ARTIFACT_TYPE,
    ROLLOUT_SAMPLE_FILENAME,
    CHECKPOINT_ARTIFACT_PREFIX,
    CHECKPOINT_ARTIFACT_TYPE,
)


class ArtifactLogger:
    """Samples rollouts from an environment and saves them to W&B.

    Parameters
    ----------
    settings : ExperimentSettings
        The settings of the experiment.
    agents : dict[str, Agent]
        The agents in the experiment.
    """

    def __init__(self, settings: ExperimentSettings, agents: dict[str, Agent]):
        self.settings = settings
        self.agents = agents

        # Make sure we have a W&B run
        if settings.wandb_run is None:
            raise ValueError("ArtifactLogger requires a W&B run")

        self._wandb_run = settings.wandb_run

        # Load the W&B API
        self._wandb_api = wandb.Api()

        # Create an initial artifact to hold the rollout samples
        self._rollout_artifact = wandb.Artifact(
            name=f"{ROLLOUT_SAMPLE_ARTIFACT_PREFIX}{self._wandb_run.name}",
            type=ROLLOUT_SAMPLE_ARTIFACT_TYPE,
        )
        self._wandb_run.use_artifact(self._rollout_artifact)

    def log(self, data: TensorDict, iteration: int):
        """Log artifacts to W&B if it's time to do so.

        Parameters
        ----------
        data : TensorDict
            The data sampled most recently from the data collector.
        iteration : int
            The iteration of training we are on.
        """

        if (iteration + 1) % self.settings.rollout_sample_period == 0:
            self._sample_and_save_rollouts(data, iteration)

        if (iteration + 1) % self.settings.checkpoint_period == 0:
            self._save_checkpoint()

    def _sample_and_save_rollouts(self, data: TensorDict, iteration: int):
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
        self._rollout_artifact = self._rollout_artifact.new_draft()
        with TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, ROLLOUT_SAMPLE_FILENAME)
            with open(file_path, "wb") as f:
                pickle.dump(rollout_samples, f)
            self._rollout_artifact.add_file(file_path, f"iteration_{iteration}")
        self._wandb_run.use_artifact(self._rollout_artifact)

    def _save_checkpoint(self):
        """Save the agent models to W&B."""

        # Create an artifact to hold the checkpoint
        artifact = wandb.Artifact(
            name=f"{CHECKPOINT_ARTIFACT_PREFIX}{self._wandb_run.name}",
            type=CHECKPOINT_ARTIFACT_TYPE,
        )

        # Save all the parts of all the agents to the artifact
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for agent_name, agent in self.agents.items():
                agent_path = temp_path.joinpath(agent_name)
                agent_path.mkdir()
                for field in fields(agent):
                    field_value = getattr(agent, field.name)
                    if isinstance(field_value, torch.nn.Module):
                        torch.save(
                            field_value.state_dict(),
                            agent_path.joinpath(f"{field.name}.pkl"),
                        )
            artifact.add_dir(temp_path)

        # Save the artifact to W&B
        self._wandb_run.log_artifact(artifact)
