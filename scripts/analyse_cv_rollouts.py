"""Analyse the rollouts of the Code Validation task using language models."""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json

import numpy as np

from pvg import Parameters, ExperimentSettings, ScenarioType
from pvg.factory import build_scenario_instance
from pvg.trainers import PureTextEiTrainer, build_trainer
from pvg.scenario_base import ROLLOUT_ANALYSERS
import pvg.code_validation.rollout_analysis

available_analysers = []
for scenario, analyser in ROLLOUT_ANALYSERS.keys():
    if scenario == ScenarioType.CODE_VALIDATION:
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

# Load the parameters
checkpoint_dir = PureTextEiTrainer.get_checkpoint_base_dir_from_run_id(
    cmd_args.checkpoint_name
)
params_path = checkpoint_dir.joinpath("params.json")
with open(params_path, "r") as params_file:
    params_dict = json.load(params_file)

params = Parameters.from_dict(params_dict)

# Build the experiment
settings = ExperimentSettings(
    run_id=cmd_args.checkpoint_name, do_not_load_checkpoint=True
)
scenario_instance = build_scenario_instance(params, settings)
trainer = build_trainer(params, scenario_instance, settings)

if not isinstance(trainer, PureTextEiTrainer):
    raise ValueError("This script is only for the PureTextEiTrainer.")

# Do the analysis
trainer.run_analysers(
    cmd_args.analysers,
    model_name=cmd_args.model_name,
    overwrite=cmd_args.overwrite,
    dry_run=cmd_args.dry_run,
)
