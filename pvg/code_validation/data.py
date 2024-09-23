"""Dataset class for the code validation experiments."""

from abc import ABC, abstractmethod
import os
import shutil
import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

from datasets import load_dataset, load_from_disk, Dataset as HuggingFaceDataset

from pvg.experiment_settings import ExperimentSettings
from pvg.parameters import Parameters, ScenarioType
from pvg.protocols import ProtocolHandler
from pvg.scenario_base.data import Dataset
from pvg.factory import register_scenario_class
from pvg.constants import CV_DATA_DIR
from pvg.utils.nested_array_dict import NestedArrayDict
from pvg.utils.types import NumpyStringDtype


class CodeValidationDataset(Dataset, ABC):
    """Base class for the code validation datasets.

    Works with HuggingFace datasets.
    """

    _main_data: HuggingFaceDataset

    @property
    def dataset_filepath_name(self) -> str:
        """The name of the dataset file."""
        return self.params.dataset.replace("/", "_")

    @property
    def raw_dir(self) -> str:
        """The path to the directory containing the raw data."""
        return os.path.join(CV_DATA_DIR, self.dataset_filepath_name, "raw")

    @property
    def processed_dir(self) -> str:
        """The path to the directory containing the processed data."""
        sub_dir = "train" if self.train else "test"
        return os.path.join(
            CV_DATA_DIR,
            self.dataset_filepath_name,
            "processed",
            sub_dir,
        )

    @abstractmethod
    def _load_raw_dataset(self) -> HuggingFaceDataset:
        """Load the dataset.

        Returns
        -------
        raw_data : HuggingFaceDataset
            The unprocessed dataset.
        """

    def _process_data(self, raw_dataset: HuggingFaceDataset) -> HuggingFaceDataset:
        """Process the dataset.

        Parameters
        ----------
        raw_dataset : HuggingFaceDataset
            The unprocessed dataset.

        Returns
        -------
        processed_dataset : HuggingFaceDataset
            The processed dataset.
        """
        return raw_dataset

    def __init__(
        self,
        params: Parameters,
        settings: ExperimentSettings,
        protocol_handler: ProtocolHandler,
        train: bool = True,
    ):
        super().__init__(params, settings, protocol_handler, train)

        if not os.path.isdir(self.processed_dir) or settings.ignore_cache:

            # Delete the processed directory if it exists
            if os.path.isdir(self.processed_dir) and settings.ignore_cache:
                shutil.rmtree(self.processed_dir)

            # Download and process the dataset, then save it to disk
            raw_dataset = self._load_raw_dataset()
            self._main_data = self._process_data(raw_dataset)
            self._main_data.save_to_disk(self.processed_dir)

        else:

            # Load the processed dataset from disk
            self._main_data = load_from_disk(self.processed_dir)

        self._main_data = self._main_data.with_format("numpy")

    def __len__(self) -> int:
        return len(self._main_data)

    def __getitem__(self, index: Any) -> NestedArrayDict:

        if isinstance(index, str):
            raise TypeError("String indexing is not supported.")

        item_dict: dict[str, NDArray] = self._main_data[index]

        # Get an arbitrary value from the dictionary to determine the batch size
        a_value = next(iter(item_dict.values()))

        if isinstance(a_value, np.generic):
            batch_size = ()
        elif isinstance(a_value, np.ndarray):
            batch_size = a_value.shape
        else:
            raise NotImplementedError(
                f"Unsupported data type returned by indexing the dataset: "
                f"{type(a_value)}"
            )

        # Convert the string arrays to NumpyStringDtype
        for key, value in item_dict.items():
            if value.dtype.type is np.str_:
                item_dict[key] = value.astype(NumpyStringDtype, copy=False)

        return NestedArrayDict(item_dict, batch_size=batch_size)


@register_scenario_class(
    ScenarioType.CODE_VALIDATION, Dataset, filter={"dataset": "codeparrot/apps"}
)
class AppsCodeValidationDataset(CodeValidationDataset):
    """The APPS[^1] dataset for code validation.

    References
    ----------
    [^1] Hendrycks et al. 2021. "Measuring Coding Challenge Competence With APPS".
    NeurIPS-21
    """

    def _load_raw_dataset(self):
        split = "train" if self.train else "test"
        return load_dataset(
            self.params.dataset,
            self.params.code_validation.apps_difficulty,
            split=split,
            data_dir=self.raw_dir,
        )

    def _process_data(self, raw_dataset: HuggingFaceDataset) -> HuggingFaceDataset:

        def filter_instance(instance: dict[str, str | int]) -> bool:
            """Filter out datapoint with no solutions."""

            return instance["solutions"] != ""

        def process_instance(instance: dict[str, str | int]) -> dict[str, str]:
            """Process a single datapoint."""

            # The solutions is a JSON list of strings
            try:
                solutions_list = json.loads(instance["solutions"])
            except json.JSONDecodeError:
                raise ValueError(f"Failed to decode the solutions JSON. {instance}")

            # Get the solution at the specified index
            solution = solutions_list[self.params.code_validation.apps_solution_number]

            # Un-escape the solution
            solution = bytes(solution, "utf-8").decode("unicode_escape")

            return {
                "question": instance["question"],
                "solution": solution,
                "y": 1,
            }

        processed_dataset = raw_dataset.filter(filter_instance)
        processed_dataset = processed_dataset.map(process_instance)
        processed_dataset = processed_dataset.select_columns(
            ["question", "solution", "y"]
        )

        return processed_dataset
