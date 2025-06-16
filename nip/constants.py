"""Global constants for the NIP project."""

import os
from pathlib import Path

_file_dir = Path(os.path.realpath(__file__))

REPOSITORY_ROOT = _file_dir.parent.parent.absolute()
"""The root directory of the repository."""

PACKAGE_ROOT = REPOSITORY_ROOT / "nip"
"""The root directory of the NIP package."""

ENV_FILE = REPOSITORY_ROOT / ".env"
"""The path to the file containing environment variables."""

DATA_DIR = REPOSITORY_ROOT / "data"
"""The path to the directory containing dataset files"""
GI_DATA_DIR = DATA_DIR / "graph_isomorphism"
"""The path to the directory containing graph isomorphism dataset files"""
IC_DATA_DIR = DATA_DIR / "image_classification"
"""The path to the directory containing image classification dataset files"""
CV_DATA_DIR = DATA_DIR / "code_validation"
"""The path to the directory containing code validation dataset files"""

CACHED_MODELS_DIR = REPOSITORY_ROOT / "model_cache"
"""The path to the directory containing cached model files"""
CACHED_MODELS_METADATA_FILENAME = "metadata.json"
"""The filename for the metadata file in the cached models directory"""

LOG_DIR = REPOSITORY_ROOT / "log"
"""The path to the directory containing general log files"""
VLLM_LOG_DIR = LOG_DIR / "vllm"
"""The path to the directory containing vLLM log files"""
LM_SERVER_TRAINING_LOG_DIR = LOG_DIR / "lm_server_training"
"""The path to the directory containing logs for the language model server training"""

STATUS_DIR = REPOSITORY_ROOT / "status"
"""The path to the directory files for communicating status between processes"""
LM_SERVER_TRAINING_STATUS_DIR = STATUS_DIR / "lm_server_training"
"""The directory containing status files for language model server training"""

EXPERIMENT_STATE_DIR = REPOSITORY_ROOT / "experiment_checkpoints"
"""The path to the directory where experiment checkpoints are saved"""

DATABASE_DIR = REPOSITORY_ROOT / "databases"
"""The path to the directory where databases are saved"""
LANGUAGE_MODEL_DB_DIR = DATABASE_DIR / "language_models.csv"
"""The path to the database containing language model metadata"""

HF_TRAINER_OUTPUT_DIR = REPOSITORY_ROOT / "hf_trainer_output"
"""The path to the directory where Hugging Face Trainer outputs are saved"""

SEEDS = [8144, 820, 4173, 3992, 4506, 9876, 5074, 446, 5147, 9030]
"""The default seeds to use for experiments"""

HF_PRETRAINED_MODELS_USER = "SamAdamDay"
"""The user hosting the pretrained models"""
HF_BUGGY_APPS_REPO = "lrhammond/buggy-apps"
"""The repository containing the buggy apps dataset"""

HF_SELF_HOSTED_FINETUNED_REPO_PREFIX = "finetune_"

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
