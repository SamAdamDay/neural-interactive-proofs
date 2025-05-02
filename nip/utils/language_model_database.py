"""Utility for accessing the language model database.

This database holds metadata about each model, along with how to access them.
"""

from typing import Annotated, Optional, get_args
from dataclasses import dataclass
import dataclasses

import numpy as np

import pandas as pd

from wandb import Api as WandbApi

from nip.parameters import HyperParameters, PureTextAgentParameters
from nip.constants import LANGUAGE_MODEL_DB_DIR
from nip.utils.checkpoints import load_run_hyper_parameters


@dataclass
class LanguageModelDbEntry:
    """An entry in the language model database."""

    model_series: Annotated[str, "Model Series"]
    model_name: Annotated[str, "Model Name"]
    developer: Annotated[str, "Developer"]
    uri: Annotated[str, "URI"]
    num_parameters: Annotated[Optional[float], "Parameters (10E+9)"] = None
    training_flops: Annotated[Optional[float], "FLOPs (10E+23)"] = None
    mmlu_pro_score: Annotated[Optional[float], "MMLU-Pro"] = None
    openrouter_input_cost: Annotated[Optional[float], "OpenRouter Input Cost"] = None
    openrouter_output_cost: Annotated[Optional[float], "OpenRouter Output Cost"] = None

    @property
    def provider(self) -> str:
        """The service which provides the model API."""
        return self.uri.partition("/")[0]

    @property
    def model_id(self) -> str:
        """The ID of the model in the model provider."""
        return self.uri.partition("/")[2]
    
    @property
    def display_name(self) -> str:
        """A nice name for the model."""
        if self.model_name != "":
            return f"{self.model_series} {self.model_name}"
        return self.model_series

    @classmethod
    def from_row(cls, row: pd.Series) -> "LanguageModelDbEntry":
        """Create a LanguageModelDbEntry from a database Pandas Series.

        Parameters
        ----------
        row : pd.Series
            The row of the Pandas DataFrame containing the model information

        Returns
        -------
        LanguageModelDbEntry
            The LanguageModelDbEntry object created from the row
        """

        arguments = {}
        for field in dataclasses.fields(cls):
            header = field.type.__metadata__[0]
            value = row[header]
            if pd.isna(value):
                arguments[field.name] = None
            elif isinstance(value, np.floating):
                arguments[field.name] = float(value)
            else:
                arguments[field.name] = value

        return cls(**arguments)


class LanguageModelNotFound(Exception):
    """Raised when a database entry for a language model URI was not found."""

    def __init__(self, uri: str):
        self.uri = uri
        super().__init__(f"Language mode with URI {uri!r} not found in the database")


class LanguageModelDatabase:
    """A class for accessing the language model database.

    This class provides methods to load the database and retrieve information about
    language models.

    """

    def __init__(self):
        self._db = pd.read_csv(LANGUAGE_MODEL_DB_DIR)

    def get_by_agent_params(
        self, agent_params: PureTextAgentParameters
    ) -> LanguageModelDbEntry:
        """Find a language model entry for a given set of agent hyper-parameters.

        Parameters
        ----------
        agent_params : PureTextAgentParameters
            The agent hyper-parameters to search for in the database

        Returns
        -------
        LanguageModelDbEntry
            The language model entry corresponding to the hyper parameters

        Raises
        ------
        LanguageModelNotFound
            If no entry is found in the database for the given hyper-parameters
        """

        uri = f"{agent_params.model_provider}/{agent_params.model_name}"
        if uri not in self._db["URI"].values:
            raise LanguageModelNotFound(uri)
        entry = self._db[self._db["URI"] == uri].iloc[0]

        return LanguageModelDbEntry.from_row(entry)

    def get_by_hyper_params(
        self, hyper_params: HyperParameters, agent_name: str
    ) -> LanguageModelDbEntry:
        """Find a language model entry for a set of hyper-parameters and agent name.

        Parameters
        ----------
        hyper_params : HyperParameters
            The hyper-parameters of the experiment
        agent_name : str
            The name of the agent. The parameters for this agent will be used to search
            the database

        Returns
        -------
        LanguageModelDbEntry
            The language model entry corresponding to the hyper parameters

        Raises
        ------
        LanguageModelNotFound
            If no entry is found in the database for the given hyper-parameters
        """

        return self.get_by_agent_params(hyper_params.agents[agent_name])

    def get_by_run_id(
        self,
        run_id: str,
        wandb_project: str,
        agent_name: str,
        wandb_entity: Optional[str] = None,
        wandb_api: Optional[WandbApi] = None,
    ) -> LanguageModelDbEntry:
        """Find a language model entry for a given W&B run.

        Parameters
        ----------
        run_id : str
            The ID of the W&B run
        wandb_project : str
            The project of the wandb run.
        agent_name : str
            The name of the agent. The parameters for this agent will be used to search
            the database
        wandb_entity : str, optional
            The entity of the wandb run. If not provided, the default entity will be
            used.
        wandb_api : WandbApi, optional
            The wandb API instance to use. If not provided, a new instance will be
            created.

        Returns
        -------
        LanguageModelDbEntry
            The language model entry corresponding to the hyper parameters

        Raises
        ------
        LanguageModelNotFound
            If no entry is found in the database for the given hyper-parameters
        """

        hyper_params = load_run_hyper_parameters(
            run_id, wandb_project, wandb_entity=wandb_entity, wandb_api=wandb_api
        )

        return self.get_by_hyper_params(hyper_params, agent_name)
