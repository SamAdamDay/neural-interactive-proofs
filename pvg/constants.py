"""Global constants for the PVG project."""

# TODO: Move the configuration options to the .env file

import os
from pathlib import Path

_file_dir = os.path.realpath(__file__)
_root_dir = Path(_file_dir).parent.parent.absolute()

ENV_FILE = os.path.join(_root_dir, ".env")
"""The path to the file containing environment variables."""

DATA_DIR = os.path.join(_root_dir, "data")
"""The path to the directory containing dataset files"""
GI_DATA_DIR = os.path.join(DATA_DIR, "graph_isomorphism")
"""The path to the directory containing graph isomorphism dataset files"""
IC_DATA_DIR = os.path.join(DATA_DIR, "image_classification")
"""The path to the directory containing image classification dataset files"""
CV_DATA_DIR = os.path.join(DATA_DIR, "code_validation")
"""The path to the directory containing code validation dataset files"""

CACHED_MODELS_DIR = os.path.join(_root_dir, "model_cache")
"""The path to the directory containing cached model files"""
CACHED_MODELS_METADATA_FILENAME = "metadata.json"
"""The filename for the metadata file in the cached models directory"""

LOG_DIR = os.path.join(_root_dir, "log")
"""The path to the directory containing general log files (not used much)"""

EXPERIMENT_STATE_DIR = os.path.join(_root_dir, "experiment_checkpoints")
"""The path to the directory where experiment checkpoints are saved"""

SEEDS = [8144, 820, 4173, 3992, 4506, 9876, 5074, 446, 5147, 9030]
"""The default seeds to use for experiments"""

HF_PRETRAINED_MODELS_USER = "SamAdamDay"
"""The user hosting the pretrained models"""
HF_BUGGY_APPS_REPO = "lrhammond/buggy-apps"
"""The repository containing the buggy apps dataset"""

# Weights & Biases defaults
ROLLOUT_SAMPLE_ARTIFACT_PREFIX = "rollout_sample_"
ROLLOUT_SAMPLE_ARTIFACT_TYPE = "rollout_sample"
ROLLOUT_SAMPLE_FILENAME = "rollout_sample.pkl"
MODEL_CHECKPOINT_ARTIFACT_PREFIX = "checkpoint_"
MODEL_CHECKPOINT_ARTIFACT_TYPE = "checkpoint"
CHECKPOINT_STATE_ARTIFACT_PREFIX = "state_"
CHECKPOINT_STATE_ARTIFACT_TYPE = "state"
ROLLOUTS_ARTIFACT_PREFIX = "full_rollouts_"
ROLLOUTS_ARTIFACT_TYPE = "full_rollouts"
RAW_TRANSCRIPT_ARTIFACT_PREFIX = "raw_transcript_"
RAW_TRANSCRIPT_ARTIFACT_TYPE = "raw_transcript"
PROCESSED_TRANSCRIPT_ARTIFACT_PREFIX = "processed_transcript_"
PROCESSED_TRANSCRIPT_ARTIFACT_TYPE = "processed_transcript"
PROMPTS_ARTIFACT_PREFIX = "prompts_"
PROMPTS_ARTIFACT_TYPE = "prompts"
