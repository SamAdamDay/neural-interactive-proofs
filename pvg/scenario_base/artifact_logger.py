"""Logging training artifacts to W&B every so often.

- Samples rollouts from training and saves them to W&B. We then visualise them.
- Saves the model checkpoints to W&B.
"""

from abc import ABC, abstractmethod
from tempfile import TemporaryDirectory
import os
import pickle
from textwrap import indent
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
    WANDB_ENTITY,
    WANDB_PROJECT,
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
    wandb_entity : str, default=WANDB_ENTITY
        The W&B entity to load the rollout samples from.
    wandb_project : str, default=WANDB_PROJECT
        The W&B project to load the rollout samples from.
    silence_wandb : bool, default=True
        Whether to suppress W&B output.

    Examples
    --------
    Using the `RolloutSamples` class as a context manager:
    >>> with RolloutSamples(run_id, iteration) as rollout_samples:
    ...     rollout_samples.visualise()

    Or manually calling the `finish` method:
    >>> rollout_samples = RolloutSamples(run_id, iteration)
    >>> rollout_samples.visualise()
    >>> rollout_samples.finish()
    """

    def __init__(
        self,
        run_id: str,
        iteration: int,
        wandb_entity: str = WANDB_ENTITY,
        wandb_project: str = WANDB_PROJECT,
        silence_wandb: bool = True,
    ):
        self.run_id = run_id
        self.iteration = iteration
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project

        if silence_wandb:
            os.environ["WANDB_SILENT"] = "true"

        # Load the W&B run
        self._wandb_run = wandb.init(
            id=run_id, entity=wandb_entity, project=wandb_project, resume="must"
        )

        # Load the agent names in order
        self.agent_names: list[str] = self._wandb_run.config["agents"]["_agent_order"]

        # Load the rollout samples from W&B
        artifact = self._wandb_run.use_artifact(
            f"{ROLLOUT_SAMPLE_ARTIFACT_PREFIX}{run_id}:latest"
        )
        with TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, f"iteration_{iteration}")
            artifact.download(root=temp_dir)
            with open(file_path, "rb") as f:
                self._rollout_samples = pickle.load(f)

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
