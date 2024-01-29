"""Utilities for saving and loading models together with their parameters.

The parameters should completely specify the model, up to changes in the codebase.
"""

from pathlib import Path
from shutil import rmtree
from typing import Optional
import time
import platform
import subprocess

import json

import torch

from pvg.parameters import Parameters
from pvg.constants import (
    CACHED_MODELS_DIR,
    CACHED_MODELS_METADATA_FILENAME,
)


class ModelNotFoundError(RuntimeError):
    """Raised when a model is not found"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

    def __repr__(self):
        return f"ModelNotFoundError({self.message})"


def get_cached_models_path(params: Parameters | dict, subdir: str) -> Path:
    """Get the path to the directory containing the models with the given parameters

    Parameters
    ----------
    params : Parameters | dict
        The parameters which specify the models.
    subdir : str
        The subdirectory in which to look for the saved models.

    Returns
    -------
    pathlib.Path
        The path to the directory containing the models with the given parameters.

    Raises
    ------
    ModelNotFoundError
        If the directory is not found.
    """

    if isinstance(params, Parameters):
        params = params.to_dict()

    models_subdir = Path(CACHED_MODELS_DIR).joinpath(subdir)
    if not models_subdir.is_dir():
        raise ModelNotFoundError(f"Models directory {models_subdir} does not exist.")

    for path in models_subdir.iterdir():
        if not path.is_dir():
            continue
        metadata_path = path.joinpath(CACHED_MODELS_METADATA_FILENAME)
        if not metadata_path.is_file():
            continue
        with open(metadata_path) as f:
            metadata = json.load(f)
        if metadata["params"] != params:
            continue
        return path

    raise ModelNotFoundError(
        f"No cached model matching the parameters found in directory {models_subdir}."
    )


def cached_models_exist(params: Parameters | dict, subdir: str) -> bool:
    """Check whether cached models exist with the given parameters

    Parameters
    ----------
    params : Parameters | dict
        The parameters which specify the models.
    subdir : str
        The subdirectory in which to look for the saved models.

    Returns
    -------
    bool
        Whether a cached model exists with the given parameters.
    """
    try:
        get_cached_models_path(params, subdir)
        return True
    except ModelNotFoundError:
        return False


def save_model_state_dicts(
    models: dict[str, torch.nn.Module],
    params: Parameters | dict,
    subdir: str,
    overwrite=True,
):
    """Save a model and its parameters

    The model will be saved in a directory named with a hash of the current time.

    Various metadata will be saved in a file named `metadata.json` in the model
    directory.

    Parameters
    ----------
    models : dict[str, torch.nn.Module]
        The models to save. The keys are the names of the models, and the values are
        the models themselves.
    params : Parameters | dict
        The parameters which specify the models.
    subdir : str
        The subdirectory in which to save the models.
    overwrite : bool
        Whether to overwrite the model if it already exists. If False and the model
        already exists, it will do nothing.
    """

    if isinstance(params, Parameters):
        params = params.to_dict()

    # Remove the model directory if it already exists, and we're overwriting
    try:
        model_dir = get_cached_models_path(params, subdir)
        if overwrite:
            rmtree(model_dir)
        else:
            return
    except ModelNotFoundError:
        pass

    # Create the model directory
    model_dir_name = str(hash(time.time()))
    model_dir = Path(CACHED_MODELS_DIR).joinpath(subdir).joinpath(model_dir_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create the metadata file
    git_revision_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    )
    cuda_devices = [
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    ]
    metadata = dict(
        machine=platform.machine(),
        platform=platform.platform(),
        uname=platform.uname(),
        system=platform.system(),
        processor=platform.processor(),
        git_revision_hash=git_revision_hash,
        python_version=platform.python_version(),
        pytorch_version=torch.__version__,
        cuda_devices=cuda_devices,
        current_cuda_device=torch.cuda.current_device(),
        model_devices={name: str(model.device) for name, model in models.items()},
        params=params,
    )
    metadata_path = model_dir.joinpath(CACHED_MODELS_METADATA_FILENAME)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    # Save the models
    for name, model in models.items():
        model_path = model_dir.joinpath(f"{name}.pt")
        torch.save(model.state_dict(), model_path)


def load_cached_model_state_dicts(
    models: dict[str, torch.nn.Module], params: Parameters | dict, subdir: str
):
    """Load the state dict of cached models into models

    Parameters
    ----------
    models : dict[str, torch.nn.Module]
        The models to load the state dicts into. The keys are the names of the models,
        and the values are the models themselves.
    params : Parameters | dict
        The parameters which specify the cached models to load.
    subdir : str
        The subdirectory in which to look for the saved models.

    Raises
    ------
    ModelNotFoundError
        If a model file is not found.
    """

    model_dir = get_cached_models_path(params, subdir)
    for name, model in models.items():
        model_path = model_dir.joinpath(f"{name}.pt")
        if not model_path.is_file():
            raise ModelNotFoundError(
                f"Model file {model_path} does not exist for model {name}."
            )
        model.load_state_dict(torch.load(model_path, map_location=model.device))
