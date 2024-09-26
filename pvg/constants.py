"""Global constants for the PVG project."""

import os
from pathlib import Path

_file_dir = os.path.realpath(__file__)
_root_dir = Path(_file_dir).parent.parent.absolute()

ENV_FILE = os.path.join(_root_dir, ".env")

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

EXPERIMENT_STATE_DIR = os.path.join(_root_dir, "experiment_checkpoints")

# The user hosting the pretrained models
HF_PRETRAINED_MODELS_USER = "SamAdamDay"

# Weights & Biases defaults
WANDB_PROJECT = "pvg-sandbox"
WANDB_CV_PROJECT = "pvg-code-validation-sandbox"
WANDB_ENTITY = "lrhammond-team"
ROLLOUT_SAMPLE_ARTIFACT_PREFIX = "rollout_sample_"
ROLLOUT_SAMPLE_ARTIFACT_TYPE = "rollout_sample"
ROLLOUT_SAMPLE_FILENAME = "rollout_sample.pkl"
CHECKPOINT_ARTIFACT_PREFIX = "checkpoint_"
CHECKPOINT_ARTIFACT_TYPE = "checkpoint"
WANDB_DUMMY_RUN_PROJECT = "pvg-sandbox"
WANDB_DUMMY_RUN_ENTITY = "lrhammond-team"
WANDB_DUMMY_RUN_NAME = "dummy_run"
WANDB_OPENAI_FINETUNE_PROJECT = "pvg-openai-finetune"
