"""Analyse the rollouts of the Code Validation task using language models."""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json

import numpy as np

import wandb

from pvg import HyperParameters, ExperimentSettings, ScenarioType
from pvg.factory import build_scenario_instance
from pvg.trainers import PureTextEiTrainer, build_trainer
from pvg.scenario_base import ROLLOUT_ANALYSERS
from pvg.constants import (
    CHECKPOINT_STATE_ARTIFACT_PREFIX,
    CHECKPOINT_STATE_ARTIFACT_TYPE,
    ROLLOUTS_ARTIFACT_PREFIX,
    ROLLOUTS_ARTIFACT_TYPE,
)
import pvg.code_validation.rollout_analysis
from pvg.utils.env import get_required_env_var

wandb_entity = get_required_env_var("WANDB_ENTITY")
wandb_cv_project = get_required_env_var("WANDB_CV_PROJECT")

available_analysers = []
for scenario, analyser in ROLLOUT_ANALYSERS.keys():
    if scenario == "code_validation":
        available_analysers.append(analyser)

arg_parser = ArgumentParser(
    description=__doc__,
    formatter_class=ArgumentDefaultsHelpFormatter,
)

arg_parser.add_argument(
    "checkpoint_name",
    type=str,
    help="The name of the checkpoint to analyse.",
)

arg_parser.add_argument(
    "--analysers",
    type=str,
    nargs="*",
    default=available_analysers,
    help="The analysers to run.",
)

arg_parser.add_argument(
    "--model-name",
    type=str,
    default="gpt-4o-mini-2024-07-18",
    help="The name of the model to use for the analysis.",
)

arg_parser.add_argument(
    "--overwrite",
    "-o",
    action="store_true",
    help="Whether to overwrite existing analysis if extant.",
    default=False,
)

arg_parser.add_argument(
    "--dry-run",
    "-d",
    action="store_true",
    help="Whether to do a dry run using a dummy API.",
    default=False,
)

# Get the arguments
cmd_args = arg_parser.parse_args()

# Try to download the checkpoint state
wandb_api = wandb.Api()
checkpoint_dir = PureTextEiTrainer.get_checkpoint_base_dir_from_run_id(
    cmd_args.checkpoint_name
)
artifact_name = (
    f"{wandb_entity}"
    f"/{wandb_cv_project}"
    f"/{CHECKPOINT_STATE_ARTIFACT_PREFIX}{cmd_args.checkpoint_name}"
    f":latest"
)
try:
    artifact: wandb.Artifact = wandb_api.artifact(
        artifact_name, type=CHECKPOINT_STATE_ARTIFACT_TYPE
    )
except wandb.errors.CommError as e:
    # W&B doesn't use subclasses for errors, so we have to check the
    # message. If the error was not that the artifact was not found, we
    # re-raise it.
    if f"artifact '{artifact_name}' not found in" not in e.message:
        raise e
else:
    artifact.download(checkpoint_dir)

# Try to download the rollouts
rollouts_dir = checkpoint_dir.joinpath("rollouts")
artifact_name = (
    f"{wandb_entity}"
    f"/{wandb_cv_project}"
    f"/{ROLLOUTS_ARTIFACT_PREFIX}{cmd_args.checkpoint_name}"
    f":latest"
)
try:
    artifact: wandb.Artifact = wandb_api.artifact(
        artifact_name, type=ROLLOUTS_ARTIFACT_TYPE
    )
except wandb.errors.CommError as e:
    # W&B doesn't use subclasses for errors, so we have to check the
    # message. If the error was not that the artifact was not found, we
    # re-raise it.
    if f"artifact '{artifact_name}' not found in" not in e.message:
        raise e
else:
    artifact.download(rollouts_dir)

# Load the parameters
params_path = checkpoint_dir.joinpath("hyper_params.json")
with open(params_path, "r") as params_file:
    params_dict = json.load(params_file)

hyper_params = HyperParameters.from_dict(params_dict, ignore_extra_keys=True)

# Build the experiment
settings = ExperimentSettings(
    run_id=cmd_args.checkpoint_name, do_not_load_checkpoint=True
)
scenario_instance = build_scenario_instance(hyper_params, settings)
trainer = build_trainer(hyper_params, scenario_instance, settings)

if not isinstance(trainer, PureTextEiTrainer):
    raise ValueError("This script is only for the PureTextEiTrainer.")

# Do the analysis
trainer.run_analysers(
    cmd_args.analysers,
    model_name=cmd_args.model_name,
    overwrite=cmd_args.overwrite,
    dry_run=cmd_args.dry_run,
)
