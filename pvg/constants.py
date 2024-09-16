"""Global constants for the PVG project."""

import os
from pathlib import Path

_file_dir = os.path.realpath(__file__)
_root_dir = Path(_file_dir).parent.parent.absolute()

DATA_DIR = os.path.join(_root_dir, "data")
GI_DATA_DIR = os.path.join(DATA_DIR, "graph_isomorphism")
IC_DATA_DIR = os.path.join(DATA_DIR, "image_classification")
CV_DATA_DIR = os.path.join(DATA_DIR, "code_validation")

CACHED_MODELS_DIR = os.path.join(_root_dir, "model_cache")
CACHED_MODELS_METADATA_FILENAME = "metadata.json"

LOG_DIR = os.path.join(_root_dir, "log")

RESULTS_DIR = os.path.join(_root_dir, "results")
RESULTS_DATA_DIR = os.path.join(RESULTS_DIR, "data")
GI_SOLO_AGENTS_RESULTS_DATA_DIR = os.path.join(RESULTS_DATA_DIR, "solo_gi_agents")

# The user hosting the pretrained models
HF_PRETRAINED_MODELS_USER = "SamAdamDay"

# Weights & Biases defaults
WANDB_PROJECT = "pvg-sandbox"
WANDB_ENTITY = "lrhammond-team"
ROLLOUT_SAMPLE_ARTIFACT_PREFIX = "rollout_sample_"
ROLLOUT_SAMPLE_ARTIFACT_TYPE = "rollout_sample"
ROLLOUT_SAMPLE_FILENAME = "rollout_sample.pkl"
CHECKPOINT_ARTIFACT_PREFIX = "checkpoint_"
CHECKPOINT_ARTIFACT_TYPE = "checkpoint"
WANDB_DUMMY_RUN_PROJECT = "pvg-sandbox"
WANDB_DUMMY_RUN_ENTITY = "lrhammond-team"
WANDB_DUMMY_RUN_NAME = "dummy_run"

# Might need to be smarter about storing this
OPENROUTER_API_KEY = (
    "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"
)
