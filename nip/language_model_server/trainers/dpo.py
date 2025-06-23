"""Script for Direct Preference Optimization (DPO) training. :cite:p:`Rafailov2023`."""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json
import logging
import os

from pydantic import TypeAdapter

from datasets import Dataset

from trl import DPOConfig, DPOTrainer
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, AutoPeftModelForCausalLM

from filelock import FileLock

from nip.constants import LM_SERVER_TRAINING_STATUS_DIR, HF_TRAINER_OUTPUT_DIR
from nip.utils.env import get_env_var
from nip.utils.types import HuggingFaceDpoDatasetItem
from nip.utils.hugging_face import is_model_peft
from nip.language_model_server.types import (
    LmTrainingConfig,
    LmLoraAdapterConfig,
    TrainingJobStatus,
)

logger = logging.getLogger(__name__)


def make_parser() -> ArgumentParser:
    """Create an argument parser for DPO training."""

    parser = ArgumentParser(
        description=__doc__.partition("\n\n")[0],
        epilog=__doc__.partition("\n\n")[2],
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--training-config-path",
        required=True,
        type=str,
        help="The path to a json file containing the configuration for DPO training.",
    )

    parser.add_argument(
        "--dataset-path",
        required=True,
        type=str,
        help="The path to a jsonl file containing the dataset.",
    )

    parser.add_argument(
        "--job-id",
        required=True,
        type=str,
        help="The unique identifier for the training job.",
    )

    parser.add_argument(
        "--new-model-name",
        required=True,
        type=str,
        help="The name to be given to the model after training is complete.",
    )

    return parser


def set_status(
    job_id: str,
    status: TrainingJobStatus,
    error_message: str = "",
    ignore_lock: bool = False,
):
    """Update the status file for the training job.

    Parameters
    ----------
    job_id : str
        The unique identifier for the training job.
    status : TrainingJobStatus
        The new status to set for the training job.
    error_message : str, default=""
        An optional error message to include if the status is not "succeeded".
    ignore_lock : bool, default=False
        If True, ignore the file lock when writing the status file.
    """

    status_filepath = LM_SERVER_TRAINING_STATUS_DIR.joinpath(f"{job_id}.status")
    status_lock_filepath = LM_SERVER_TRAINING_STATUS_DIR.joinpath(
        f"{job_id}.status.lock"
    )

    status_filepath.parent.mkdir(parents=True, exist_ok=True)

    if not ignore_lock:
        with FileLock(status_lock_filepath):
            with open(status_filepath, "w") as f:
                f.write(status)
    else:
        with open(status_filepath, "w") as f:
            f.write(status)

    if error_message != "":

        error_filepath = LM_SERVER_TRAINING_STATUS_DIR.joinpath(f"{job_id}.error")
        error_lock_filepath = LM_SERVER_TRAINING_STATUS_DIR.joinpath(
            f"{job_id}.error.lock"
        )

        if not ignore_lock:
            with FileLock(error_lock_filepath):
                with open(error_filepath, "w") as f:
                    f.write(error_message)
        else:
            with open(error_filepath, "w") as f:
                f.write(error_message)

    logging.info(f"Set status for job {job_id!r} to {status!r}.")


def load_config(config_path: str) -> LmTrainingConfig:
    """Load and validate the training config from a JSON file.

    Parameters
    ----------
    config_path : str
        The path to the JSON file containing the training config.

    Returns
    -------
    LmTrainingConfig
        The validated training config as a LmTrainingConfig object.
    """

    with open(config_path, "r") as f:
        config = json.load(f)

    config = LmTrainingConfig(**config)

    if config.method != "dpo":
        raise ValueError(f"Invalid method {config.method}. Only 'dpo' is supported.")

    return config


def load_dataset(dataset_path: str) -> Dataset:
    """Load and convert the dataset for DPO training.

    Parameters
    ----------
    dataset_path : str
        The path to the JSON file containing the dataset.

    Returns
    -------
    dataset: Dataset
        The converted dataset ready for DPO training.
    """

    dataset_item_adapter = TypeAdapter(HuggingFaceDpoDatasetItem)

    dataset: list[HuggingFaceDpoDatasetItem] = []
    with open(dataset_path, "r") as f:
        for line in f:
            item: HuggingFaceDpoDatasetItem = json.loads(line)
            dataset_item_adapter.validate_python(item)
            dataset.append(item)

    return Dataset.from_list(dataset)


def set_environment_variables(job_id: str):
    """Set environment variables required for the training job.

    Parameters
    ----------
    job_id : str
        The unique identifier for the training job.
    """

    os.environ["WANDB_ENTITY"] = get_env_var("WANDB_ENTITY")
    os.environ["WANDB_PROJECT"] = get_env_var("WANDB_SELF_HOSTED_FINETUNE_PROJECT")
    os.environ["WANDB_RUN_ID"] = job_id

    os.environ["HF_TOKEN"] = get_env_var("HF_TOKEN")


def train(config: LmTrainingConfig, dataset: Dataset, job_id: str, new_model_name: str):
    """Train a language model using Direct Preference Optimization (DPO).

    Parameters
    ----------
    config : LmTrainingConfig
        The training config for the training job.
    dataset : Dataset
        The dataset to use for training, in the Hugging Face format.
    job_id : str
        The unique identifier for the training job.
    new_model_name : str
        The name to be given to the model after training is complete.
    """

    ignore_training_lora_config = False

    if not is_model_peft(config.model_name):
        model = AutoModelForCausalLM.from_pretrained(config.model_name)

    else:
        model_lora_config = LoraConfig.from_pretrained(config.model_name)

        # When reusing the LoRA adapter, make sure the model's LoRA configuration is
        # compatible with the training configuration.
        if (
            config.model_already_lora_strategy == "reuse"
            and config.training_lora_config is not None
        ):
            for key in LmLoraAdapterConfig.model_fields.keys():
                if getattr(config.training_lora_config, key) != getattr(
                    model_lora_config, key
                ):
                    raise ValueError(
                        f"Model {config.model_name!r} is already LoRA-adapted and its "
                        f"LoRA configuration is not compatible with the training "
                        f"configuration. The {key!r} field of the training "
                        f"configuration is "
                        f"{getattr(config.training_lora_config, key)!r}, while "
                        f"the model's LoRA configuration has {key!r} set to "
                        f"{getattr(model_lora_config, key)!r}."
                    )

        model = AutoPeftModelForCausalLM.from_pretrained(
            config.model_name, is_trainable=True
        )

        # Sanity check: ensure that exactly the LoRA layers are trainable.
        for name, param in model.named_parameters():
            if param.requires_grad and "lora" not in name:
                logger.warning(
                    f"Parameter {name!r} is trainable (required grad), but it does not "
                    f"have 'lora' in its name. This may mean that wrong parts of the "
                    f"model are being trained, due to a misconfiguration."
                )
            elif not param.requires_grad and "lora" in name:
                logger.warning(
                    f"Parameter {name!r} is not trainable (does not require grad), but "
                    f"it has 'lora' in its name (so probably it is a LoRA layer). This "
                    f"may mean that the LoRA layer is not being trained, due to a "
                    f"misconfiguration."
                )

        if config.model_already_lora_strategy == "reuse":
            # Ignore the LoRA training adapter configuration, because the model is
            # already LoRA-adapted and the trainer will train the existing adapter.
            ignore_training_lora_config = True

    if ignore_training_lora_config or config.training_lora_config is None:
        training_lora_config = None
    else:
        training_lora_config = LoraConfig(**config.training_lora_config.model_dump())
    dpo_config = DPOConfig(
        **config.dpo_config.model_dump(),
        hub_model_id=new_model_name,
        run_name=job_id,
        output_dir=HF_TRAINER_OUTPUT_DIR,
        fp16=config.mixed_precision == "fp16",
        bf16=config.mixed_precision == "bf16",
        gradient_checkpointing=config.gradient_checkpointing,
        per_device_train_batch_size=config.per_device_train_batch_size,
        seed=config.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    trainer = DPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=dpo_config,
        peft_config=training_lora_config,
    )

    logger.info("Starting DPO training")

    trainer.train()

    logger.info("DPO training complete. Pushing model to Hugging Face Hub...")
    trainer.push_to_hub()


def main():
    """Run the DPO training script."""

    parser = make_parser()
    args = parser.parse_args()

    set_status(args.job_id, "starting")

    config = load_config(args.training_config_path)
    dataset = load_dataset(args.dataset_path)

    set_environment_variables(args.job_id)

    set_status(args.job_id, "running")

    try:
        train(config, dataset, job_id=args.job_id, new_model_name=args.new_model_name)
        set_status(args.job_id, "succeeded")

    except Exception as e:
        set_status(args.job_id, "crashed", error_message=str(e))
        raise e

    except KeyboardInterrupt:
        set_status(
            args.job_id,
            "interrupted",
            error_message="KeyboardInterrupt",
            ignore_lock=True,
        )
        raise KeyboardInterrupt("Training interrupted by user.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%x %X",
    )
    main()
