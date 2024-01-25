"""Global constants for the PVG project."""

import os
from pathlib import Path

_file_dir = os.path.realpath(__file__)
_root_dir = Path(_file_dir).parent.parent.absolute()

DATA_DIR = os.path.join(_root_dir, "data")
GI_DATA_DIR = os.path.join(DATA_DIR, "graph_isomorphism")
IC_DATA_DIR = os.path.join(DATA_DIR, "image_classification")

CACHED_MODELS_DIR = os.path.join(_root_dir, "model_cache")
CACHED_MODELS_METADATA_FILENAME = "metadata.json"

RESULTS_DIR = os.path.join(_root_dir, "results")
RESULTS_DATA_DIR = os.path.join(RESULTS_DIR, "data")
GI_SOLO_AGENTS_RESULTS_DATA_DIR = os.path.join(RESULTS_DATA_DIR, "solo_gi_agents")

# Weights & Biases defaults
WANDB_PROJECT = "pvg-sandbox"
WANDB_ENTITY = "lrhammond-team"
ROLLOUT_SAMPLE_ARTIFACT_PREFIX = "rollout_sample_"
ROLLOUT_SAMPLE_ARTIFACT_TYPE = "rollout_sample"
ROLLOUT_SAMPLE_FILENAME = "rollout_sample.pkl"
