"""Dataset class for the code validation experiments."""

from abc import ABC, abstractmethod
import os
import shutil
import json
from typing import Any, ClassVar
from textwrap import indent
from random import randint

import numpy as np
from numpy.typing import NDArray

from datasets import (
    load_dataset,
    load_from_disk,
    Dataset as HuggingFaceDataset,
    concatenate_datasets,
)

from nip.experiment_settings import ExperimentSettings
from nip.parameters import HyperParameters, ScenarioType
from nip.protocols import ProtocolHandler
from nip.scenario_base.data import Dataset
from nip.factory import register_scenario_class
from nip.constants import CV_DATA_DIR
from nip.utils.nested_array_dict import NestedArrayDict
from nip.utils.types import NumpyStringDtype
from nip.utils.string import get_hash_parity


class CodeValidationDataset(Dataset, ABC):
    """Base class for the code validation datasets.

    Works with HuggingFace datasets.

    The dataset should have the following columns:

    - "question": The question text.
    - "solution": The solution text.
    - "y": The label, 1 for correct solutions and 0 for buggy solutions.

    In addition, each datapoint should receive a "prover_stance" which is the verdict
    that the prover should be arguing for, in single-prover settings under the
    appropriate hyper-parameters. This can be computed from the (hash of the) solution
    text.
    """

    instance_keys: ClassVar[tuple[str]] = ("question", "solution", "prover_stance")

    _main_data: HuggingFaceDataset

    @property
    def dataset_filepath_name(self) -> str:
        """The name of the dataset file."""
        return self.hyper_params.dataset.replace("/", "_")

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
        hyper_params: HyperParameters,
        settings: ExperimentSettings,
        protocol_handler: ProtocolHandler,
        train: bool = True,
    ):
        super().__init__(hyper_params, settings, protocol_handler, train)

        if not os.path.isdir(self.processed_dir) or settings.ignore_cache:

            # Delete the processed directory if it exists
            if os.path.isdir(self.processed_dir) and settings.ignore_cache:
                shutil.rmtree(self.processed_dir)

            # Create the raw directory if it does not exist
            os.makedirs(self.raw_dir, exist_ok=True)

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

    def __repr__(self) -> str:
        output = f"{self.__class__.__name__}(\n"
        output += indent(f"fields={list(self._main_data.features.keys())},\n", " " * 4)
        output += indent(f"num_rows={len(self)},\n", " " * 4)
        output += indent(f"train={self.train},\n", " " * 4)
        output += ")"
        return output


@register_scenario_class("code_validation", Dataset, filter={"dataset": "test"})
class TestCodeValidationDataset(CodeValidationDataset):
    """A test dataset for code validation, with dummy data."""

    def _load_raw_dataset(self) -> HuggingFaceDataset:

        def sample_generator():
            for i in range(10):
                yield {
                    "question": f"Question {i}",
                    "solution": f"Solution {i}",
                    "prover_stance": 1,
                    "y": randint(0, 1),
                    "id": i,
                }

        return HuggingFaceDataset.from_generator(sample_generator)

    def _process_data(self, raw_dataset: HuggingFaceDataset) -> HuggingFaceDataset:
        return raw_dataset


@register_scenario_class(
    "code_validation", Dataset, filter={"dataset": "codeparrot/apps"}
)
class AppsCodeValidationDataset(CodeValidationDataset):
    """The APPS :cite:p:`Hendrycks2021` dataset for code validation."""

    def _load_raw_dataset(self) -> HuggingFaceDataset:
        split = "train" if self.train else "test"
        return load_dataset(
            self.hyper_params.dataset,
            self.hyper_params.code_validation.apps_difficulty,
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
            solution = solutions_list[
                self.hyper_params.code_validation.apps_solution_number
            ]

            # Un-escape the solution
            solution = bytes(solution, "utf-8").decode("unicode_escape")

            # Decide on the verdict that the prover should be arguing for
            prover_stance = get_hash_parity(solution)

            return {
                "question": instance["question"],
                "solution": solution,
                "prover_stance": prover_stance,
                "y": 1,
            }

        processed_dataset = raw_dataset.filter(filter_instance)
        processed_dataset = processed_dataset.map(process_instance)
        processed_dataset = processed_dataset.add_column(
            "id", np.arange(len(processed_dataset))
        )
        processed_dataset = processed_dataset.select_columns(
            ["question", "solution", "prover_stance", "y", "id"]
        )

        return processed_dataset


@register_scenario_class(
    "code_validation", Dataset, filter={"dataset": "lrhammond/buggy-apps"}
)
class BuggyAppsCodeValidationDataset(CodeValidationDataset):
    """An extension of the APPS :cite:p:`Hendrycks2021` dataset with buggy solutions.

    Buggy solutions were generated by asking GPT-4o to introduce bugs into the non-buggy
    solutions from the APPS dataset.
    """

    @property
    def processed_dir(self) -> str:
        """The path to the directory containing the processed data."""
        split_dir = "train" if self.train else "test"
        difficulty_dir = self.hyper_params.code_validation.apps_difficulty
        return os.path.join(
            CV_DATA_DIR,
            self.dataset_filepath_name,
            "processed",
            difficulty_dir,
            split_dir,
        )

    def _load_raw_dataset(self) -> HuggingFaceDataset:
        split = "train" if self.train else "test"
        return load_dataset(
            self.hyper_params.dataset,
            split=split,
        )

    def _process_data(self, raw_dataset: HuggingFaceDataset) -> HuggingFaceDataset:

        def filter_instance(instance: dict[str, str | int]) -> bool:
            """Filter based on the hyper_params."""

            if (
                instance["difficulty"]
                != self.hyper_params.code_validation.apps_difficulty
            ):
                return False

            if (
                self.hyper_params.code_validation.apps_exclude_flagged
                and instance["flags"] != ""
            ):
                return False

            return True

        def get_non_buggy_solution(
            instance: dict[str, str | int], solution_index: int
        ) -> str:
            """Get the non-buggy solution for the given instance and solution index."""
            return {
                "question": instance["question"],
                "solution": instance["solutions"][solution_index]["solution"],
                "prover_stance": get_hash_parity(
                    instance["solutions"][solution_index]["solution"]
                ),
                "y": 1,
            }

        def get_buggy_solution(
            instance: dict[str, str | int], solution_index: int
        ) -> str:
            """Get the buggy solution for the given instance and solution index."""
            return {
                "question": instance["question"],
                "solution": instance["buggy_solutions"][solution_index]["solution"],
                "prover_stance": get_hash_parity(
                    instance["buggy_solutions"][solution_index]["solution"]
                ),
                "y": 0,
            }

        filtered_dataset = raw_dataset.filter(filter_instance)

        # Get all the non-buggy solutions, including when one question has multiple
        # solutions
        non_buggy_datasets = []
        solution_index = 0
        while True:

            # Select those instances that have a solution at the specified index
            non_buggy_dataset = filtered_dataset.filter(
                lambda x: len(x["solutions"]) > solution_index
            )

            if len(non_buggy_dataset) == 0:
                break

            # Extract the non-buggy solution at the specified index
            non_buggy_dataset = non_buggy_dataset.map(
                lambda x: get_non_buggy_solution(x, solution_index)
            )

            non_buggy_datasets.append(non_buggy_dataset)

            solution_index += 1

        # Get all the non-buggy solutions, including when one question has multiple
        # solutions
        buggy_datasets = []
        solution_index = 0
        while True:

            # Select those instances that have a solution at the specified index
            buggy_dataset = filtered_dataset.filter(
                lambda x: len(x["solutions"]) > solution_index
            )

            if len(buggy_dataset) == 0:
                break

            # Extract the non-buggy solution at the specified index
            buggy_dataset = buggy_dataset.map(
                lambda x: get_buggy_solution(x, solution_index)
            )

            buggy_datasets.append(buggy_dataset)

            solution_index += 1

        # Concatenate the non-buggy and buggy datasets
        processed_dataset: HuggingFaceDataset = concatenate_datasets(
            non_buggy_datasets + buggy_datasets
        )

        # Add an ID column
        processed_dataset = processed_dataset.add_column(
            "id", np.arange(len(processed_dataset))
        )

        processed_dataset = processed_dataset.select_columns(
            ["question", "solution", "prover_stance", "y", "id"]
        )

        return processed_dataset
