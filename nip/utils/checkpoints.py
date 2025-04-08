"""Utilities for dealing with saved experiment checkpoints.

Currently these utilities only work with checkpoints produced by pure-text trainers,
where rollouts are stored in a :class:`NestedArrayDict
<nip.utils.nested_array_dict.NestedArrayDict>`.
"""

from typing import Literal, Optional, overload
from pathlib import Path
import pickle

from wandb import Api as WandbApi

from nip.utils.nested_array_dict import NestedArrayDict
from nip.utils.env import get_env_var
from nip.constants import (
    EXPERIMENT_STATE_DIR,
    ROLLOUTS_ARTIFACT_PREFIX,
    ROLLOUTS_ARTIFACT_TYPE,
    RAW_TRANSCRIPT_ARTIFACT_PREFIX,
    RAW_TRANSCRIPT_ARTIFACT_TYPE,
    PROCESSED_TRANSCRIPT_ARTIFACT_PREFIX,
    PROCESSED_TRANSCRIPT_ARTIFACT_TYPE,
    PROMPTS_ARTIFACT_PREFIX,
    PROMPTS_ARTIFACT_TYPE,
)


def download_checkpoint(
    wandb_project: str,
    run_id: str,
    wandb_entity: Optional[str] = None,
    wandb_api: Optional[WandbApi] = None,
    *,
    include_everything: bool = False,
    include_rollouts: bool = True,
    include_processed_transcripts: bool = False,
    include_raw_transcripts: bool = False,
    include_prompts: bool = False,
    handle_existing: Literal["overwrite", "skip", "error"] = "skip",
):
    """Download a checkpoint from wandb to the local checkpoint directory.

    Parameters
    ----------
    wandb_project : str
        The project of the wandb run.
    run_id : str
        The ID of the run to download the checkpoint from.
    wandb_entity : str, optional
        The entity of the wandb run. If not provided, the default entity will be used.
    wandb_api : WandbApi, optional
        The wandb API instance to use. If not provided, a new instance will be created.
    include_everything : bool, default=False
        Whether to download all parts of the checkpoint. If True, all other
        ``include_*`` parameters are ignored.
    include_rollouts : bool, default=True
        Whether to download rollouts in the checkpoint.
    include_processed_transcripts : bool, default=False
        Whether to download processed transcripts in the checkpoint.
    include_raw_transcripts : bool, default=False
        Whether to download raw transcripts in the checkpoint.
    include_prompts : bool, default=False
        Whether to download prompts in the checkpoint.
    handle_existing : {"overwrite", "skip", "error"}, default="skip"
        What to do if the checkpoint or any files already exist in the local directory.
        - "overwrite": overwrite the existing checkpoint. - "skip": skip downloading the
        checkpoint. - "error": raise an error if the checkpoint already exists.

    Raises
    ------
    FileExistsError
        If the a requested part of the checkpoint already exists and `handle_existing`
        is "error".
    ValueError
        If no parts of the checkpoint are requested to be downloaded.
    """

    if wandb_api is None:
        wandb_api = WandbApi()
    if wandb_entity is None:
        wandb_entity = get_env_var("WANDB_ENTITY")

    if include_everything:
        include_rollouts = True
        include_processed_transcripts = True
        include_raw_transcripts = True
        include_prompts = True

    if not any(
        (
            include_rollouts,
            include_processed_transcripts,
            include_raw_transcripts,
            include_prompts,
        )
    ):
        raise ValueError(
            "At least one of `include_rollouts`, `include_processed_transcripts`, "
            "`include_raw_transcripts`, or `include_prompts` must be True."
        )

    checkpoint_path = Path(EXPERIMENT_STATE_DIR, run_id)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    def handle_checkpoint_part_dir(dir_path: Path) -> bool:
        """Check if the file path exists and handle according to ``handle_existing``.

        A new directory is created if it does not exist. If it does exist, the
        behavioUr depends on the ``handle_existing`` parameter:
        - "overwrite": delete the existing directory and create a new one.
        - "skip": do nothing and return False.
        - "error": raise a FileExistsError.
        If the path exists but is not a directory, an error is raised.

        Parameters
        ----------
        dir_path : Path
            The path to check and handle.

        Returns
        -------
        download_files : bool
            Whether to download the files to the directory. Returns True if the
            directory was created, and False if it was skipped.
        """
        if dir_path.is_dir():
            if handle_existing == "error":
                raise FileExistsError(
                    f"File {dir_path} already exists. Use handle_existing='overwrite' "
                    f"to overwrite."
                )
            elif handle_existing == "skip":
                return False
            elif handle_existing == "overwrite":
                for child in rollout_path.iterdir():
                    child.unlink()
        elif dir_path.exists():
            if handle_existing == "overwrite":
                dir_path.unlink()
            else:
                raise FileExistsError(
                    f"File {dir_path!r} already exists and is not a directory. Use "
                    f"handle_existing='overwrite' to overwrite."
                )
        dir_path.mkdir(parents=True, exist_ok=True)
        return True

    rollout_path = checkpoint_path / "rollouts"
    if include_rollouts and handle_checkpoint_part_dir(rollout_path):
        rollout_artifact = wandb_api.artifact(
            f"{wandb_entity}/{wandb_project}/{ROLLOUTS_ARTIFACT_PREFIX}{run_id}:latest",
            ROLLOUTS_ARTIFACT_TYPE,
        )
        rollout_artifact.download(rollout_path)

    processed_transcripts_path = checkpoint_path / "processed_transcripts"
    if include_processed_transcripts and handle_checkpoint_part_dir(
        processed_transcripts_path
    ):
        processed_transcripts_artifact = wandb_api.artifact(
            f"{wandb_entity}/{wandb_project}/"
            f"{PROCESSED_TRANSCRIPT_ARTIFACT_PREFIX}{run_id}:latest",
            PROCESSED_TRANSCRIPT_ARTIFACT_TYPE,
        )
        processed_transcripts_artifact.download(processed_transcripts_path)

    raw_transcripts_path = checkpoint_path / "raw_transcripts"
    if include_raw_transcripts and handle_checkpoint_part_dir(raw_transcripts_path):
        raw_transcripts_artifact = wandb_api.artifact(
            f"{wandb_entity}/{wandb_project}/"
            f"{RAW_TRANSCRIPT_ARTIFACT_PREFIX}{run_id}:latest",
            RAW_TRANSCRIPT_ARTIFACT_TYPE,
        )
        raw_transcripts_artifact.download(raw_transcripts_path)

    prompts_path = checkpoint_path / "prompts"
    if include_prompts and handle_checkpoint_part_dir(prompts_path):
        prompts_artifact = wandb_api.artifact(
            f"{wandb_entity}/{wandb_project}/"
            f"{PROMPTS_ARTIFACT_PREFIX}{run_id}:latest",
            PROMPTS_ARTIFACT_TYPE,
        )
        prompts_artifact.download(prompts_path)


@overload
def load_rollouts(
    run_id: str,
    iterations: int | str,
    download_from_wandb: bool = True,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_api: Optional[WandbApi] = None,
) -> NestedArrayDict: ...


@overload
def load_rollouts(
    run_id: str,
    iterations: list[int | str] | None,
    download_from_wandb: bool = True,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_api: Optional[WandbApi] = None,
) -> list[NestedArrayDict]: ...


def load_rollouts(
    run_id: str,
    iterations: int | str | list[int | str] | None = None,
    download_from_wandb: bool = True,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_api: Optional[WandbApi] = None,
) -> NestedArrayDict | list[NestedArrayDict]:
    """Load the rollouts from a checkpoint.

    The function can download the checkpoint from W&B, storing it locally in the
    checkpoint directory.

    Parameters
    ----------
    run_id : str
        The ID of the run to load the rollouts from.
    iterations : int|str | list[int|str] | None, default=None
        The iteration(s) of the rollouts to load. If None, all iterations are loaded. If
        -1 the last iteration is loaded
    download_from_wandb : bool, default=True
        Whether to download the rollouts from W&B. If False, the function will look for
        the rollouts in the local checkpoint directory.
    wandb_project : str, optional
        The project of the wandb run. Must be provided if ``download_from_wandb`` is
        True.
    wandb_entity : str, optional
        The entity of the wandb run. If not provided, the default entity will be used.
    wandb_api : WandbApi, optional
        The wandb API instance to use. If not provided, a new instance will be created.

    Returns
    -------
    rollouts : NestedArrayDict | list[NestedArrayDict]
        The rollouts loaded from the checkpoint. If ``iteration`` is None or a list,
        this will be a list of NestedArrayDicts. If ``iteration`` is an ``int`` or
        ``str``, this will be a single NestedArrayDict.

    Raises
    ------
    FileNotFoundError
        If the rollouts directory is not found or the requested rollout file is not
        found.
    """

    if download_from_wandb:
        if wandb_project is None:
            raise ValueError(
                "wandb_project must be provided if download_from_wandb is True."
            )
        if wandb_entity is None:
            wandb_entity = get_env_var("WANDB_ENTITY")
        download_checkpoint(
            wandb_project,
            run_id,
            wandb_entity=wandb_entity,
            include_rollouts=True,
            wandb_api=wandb_api,
        )

    checkpoint_path = Path(EXPERIMENT_STATE_DIR, run_id)
    rollouts_path = checkpoint_path / "rollouts"
    if not rollouts_path.is_dir():
        raise FileNotFoundError(f"Rollouts directory {rollouts_path} not found.")

    available_rollouts = [
        rollout_file.stem
        for rollout_file in rollouts_path.iterdir()
        if rollout_file.suffix == ".pt"
    ]

    if iterations is None:
        iterations = available_rollouts

    if isinstance(iterations, (int, str)):
        if iterations == -1:
            iterations = max([int(i) for i in available_rollouts if i.isdigit()])
        rollout_file = rollouts_path / f"{iterations}.pt"
        if not rollout_file.is_file():
            raise FileNotFoundError(f"Rollout file {rollout_file!r} not found.")
        with open(rollout_file, "rb") as f:
            return pickle.load(f)

    elif isinstance(iterations, list):
        rollouts = []
        for iteration in iterations:
            rollout_file = rollouts_path / f"{iteration}.pt"
            if not rollout_file.is_file():
                raise FileNotFoundError(f"Rollout file {rollout_file!r} not found.")
            with open(rollout_file, "rb") as f:
                rollouts.append(pickle.load(f))
        return rollouts

    else:
        raise ValueError(
            f"iterations must be an int, a list of ints, or None. "
            f"Got {type(iterations)}."
        )
