"""Instance-specific settings for the experiment, like torch device, logging, etc.

Changing these settings should not effect the reproducibility of the experiment.
"""

from typing import Optional, ClassVar, Any
from dataclasses import dataclass, field, fields

import torch

from openai import OpenAI

import wandb

from tqdm import tqdm

from pvg.stat_logger import StatLogger, DummyStatLogger
from pvg.utils.types import TorchDevice, LoggingType


def default_global_tqdm_step_fn():
    pass


@dataclass
class ExperimentSettings:
    """Instance-specific settings for the experiment.

    Parameters
    ----------
    device : TorchDevice, default="cpu"
        The device to use for training.
    run_id : str, optional
        The ID of the current run. This can be used to save and restore the state of the
        experiment.
    wandb_run : wandb.wandb_sdk.wandb_run.Run, optional
        The W&B run to log to, if any.
    silence_wandb : bool, default=True
        Whether to suppress W&B output.
    stat_logger : StatLogger, optional
        The logger to use for logging statistics. If not provided, a dummy logger is
        used, which does nothing.
    tqdm_func : Callable, optional
        The tqdm function to use. Defaults to tqdm.
    logger : logging.Logger | logging.LoggerAdapter, optional
        The logger to log to. If None, the trainer will create a logger.
    profiler : torch.profiler.profile, optional
        The PyTorch profiler being used to profile the training, if any.
    ignore_cache : bool, default=False
        If True, the dataset and model cache are ignored and rebuilt.
    num_rollout_samples : int, default=10
        The number of rollout samples to collect and save per iteration of RL training.
        These are useful to visualize the progress of the training.
    rollout_sample_period : int, default=1000
        The frequency with which to collect rollout samples. This is the number of
        iterations of RL training between each collection of rollout samples.
    save_models_period : int, default=1000
        The frequency with which to save the models. This is the number of iterations of
        RL training between each save of the models.
    num_dataset_threads : int, default=8
        The number of threads to use for saving the memory-mapped tensordict.
    num_rollout_workers : int, default=4
        The number of workers to use for collecting rollout samples, when this is done
        in parallel.
    pin_memory : bool, default=True
        Whether to pin the memory of the tensors in the dataloader, and move them to the
        GPU with `non_blocking=True`. This can speed up training. When the device if the
        CPU this setting doesn't do anything and is set to False.
    dataset_on_device : bool, default=False
        Whether store the whole dataset on the device. This can speed up training but
        requires that the dataset fits on the device. This makes `pin_memory` redundant.
    enable_efficient_attention: bool, default=False
        Whether to enable the 'Memory-Efficient Attention' backend for the scaled
        dot-product attention. There may be a bug in this implementation which causes
        NaNs to appear in the backward pass. See
        https://github.com/pytorch/pytorch/issues/119320 for more information.
    global_tqdm_step_fn : Callable, default=lambda: ...
        A function to step the global tqdm progress bar. This is used when there are
        multiple processes running in parallel and each process needs to update the
        global progress bar.
    pretrained_embeddings_batch_size : int, default=256
        The batch size to use when generating embeddings for the pretrained models.
    num_api_generation_timeouts : int, default=100
        The number of timeouts to allow when generating API outputs. If the number of
        timeouts exceeds this value, the experiment will be stopped.
    do_not_load_checkpoint : bool, default=False
        If True, the experiment will not load a checkpoint if one exists.
    test_run : bool, default=False
        If True, the experiment is run in test mode. This means we do the smallest
        number of iterations possible and then exit. This is useful for testing that the
        experiment runs without errors. It doesn't make sense to use this with
        wandb_run.
    """

    device: TorchDevice = torch.device("cpu")
    run_id: Optional[str] = None
    wandb_run: Optional[wandb.wandb_sdk.wandb_run.Run] = None
    silence_wandb: bool = True
    stat_logger: Optional[StatLogger] = field(default_factory=DummyStatLogger)
    tqdm_func: callable = tqdm
    logger: Optional[LoggingType] = None
    profiler: Optional[torch.profiler.profile] = None
    ignore_cache: bool = False
    num_rollout_samples: int = 10
    rollout_sample_period: int = 1000
    checkpoint_period: int = 1000
    num_dataset_threads: int = 8
    num_rollout_workers: int = 4
    pin_memory: bool = True
    dataset_on_device: bool = False
    enable_efficient_attention: bool = False
    global_tqdm_step_fn: callable = default_global_tqdm_step_fn
    pretrained_embeddings_batch_size: int = 256
    num_api_generation_timeouts: int = 100
    do_not_load_checkpoint: bool = False
    test_run: bool = False

    unpicklable_fields: ClassVar[tuple[str, ...]] = ("global_tqdm_step_fn",)

    def __post_init__(self):
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        if self.device.type == "cpu":
            self.pin_memory = False

        if self.test_run and self.wandb_run is not None:
            raise ValueError("test_run cannot be True if wandb_run is not None.")

    def __getstate__(self) -> dict[str, Any]:
        """Get the state of the object for pickling.

        This method is called when the object is pickled. We override it to remove
        fields that are not picklable.

        Returns
        -------
        state : dict[str, Any]
            The state of the object.
        """

        state = {}
        for setting_field in fields(self):
            field_name = setting_field.name
            if field_name in self.unpicklable_fields:
                state[field_name] = setting_field.default
            else:
                state[field_name] = getattr(self, field_name)

        return state

    def __deepcopy__(self, memo) -> "ExperimentSettings":
        """We do not deepcopy this object, as it is a singleton and contains locks."""
        return self
