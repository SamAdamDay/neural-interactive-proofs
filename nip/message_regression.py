"""Regressor for message data.

Used to test to what extent the label can be inferred purely from the messages sent,
without considering the other data.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Callable
from math import ceil, nan

import torch
from torch.nn import Sequential, Linear, LazyLinear
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tensordict import TensorDictBase

from einops import rearrange

from nip.parameters import HyperParameters, MessageRegressionMethodType
from nip.experiment_settings import ExperimentSettings
from nip.protocols import ProtocolHandler
from nip.utils.torch import ACTIVATION_CLASSES


class MessageRegressor(ABC):
    """Base class for regressors on the message data.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    settings : ExperimentSettings
        The settings of the experiment.
    protocol_handler : ProtocolHandler
        The protocol handler.
    """

    def __init__(
        self,
        hyper_params: HyperParameters,
        settings: ExperimentSettings,
        protocol_handler: ProtocolHandler,
    ):
        self.hyper_params = hyper_params
        self.settings = settings
        self.protocol_handler = protocol_handler

        # Determine the agents to do regression analysis on
        if hyper_params.message_regression.agents is not None:
            self.agent_names = hyper_params.message_regression.agents
        else:
            self.agent_names = self.protocol_handler.agent_names

    def fit_score(self, data: TensorDictBase) -> dict[str, float]:
        """Fit and score the regressor on the data.

        Parameters
        ----------
        data : TensorDictBase
            The data produced in a single iteration.

        Returns
        -------
        scores : dict[str, float]
            The regression scores for each agent.
        """

        if self.hyper_params.message_regression.reset_on_fit:
            self.reset_parameters()

        scores = {}

        for agent_name in self.agent_names:

            # Select the data for the first round in which the agent is active
            round_mask = (
                data["round"]
                == self.protocol_handler.agent_first_active_round[agent_name]
            )
            agent_data = data[round_mask]

            # Split into training and testing data
            num_samples = len(agent_data)
            num_test_samples = ceil(
                num_samples * self.hyper_params.message_regression.test_size
            )
            permutation = torch.randperm(num_samples)
            train_mask = permutation[num_test_samples:]
            test_mask = permutation[:num_test_samples]
            train_data = agent_data[train_mask]
            test_data = agent_data[test_mask]

            # If there is not enough data for training or testing, skip the agent
            if train_data.batch_size[0] == 0 or test_data.batch_size[0] == 0:
                scores[agent_name] = nan
                continue

            # Fit and score the regressor
            scores[agent_name] = self.fit_score_agent(agent_name, train_data, test_data)

        return scores

    @abstractmethod
    def fit_score_agent(
        self, agent_name: str, train_data: TensorDictBase, test_data: TensorDictBase
    ) -> float:
        """Fit and score the regressor on the data for a single agent.

        Parameters
        ----------
        agent_name : str
            The name of the agent.
        train_data : TensorDictBase
            A selection of the data for fitting the regressor.
        test_data : TensorDictBase
            A selection of the data for testing the regressor.

        Returns
        -------
        score : float
            The regression score.
        """

    @abstractmethod
    def reset_parameters(self):
        """Reset the parameters of the models."""


class DummyMessageRegressor(MessageRegressor):
    """Dummy regressor that does nothing.

    Used when regression analysis is disabled.
    """

    def fit_score(self, data: TensorDictBase) -> dict[str, float]:  # noqa: D102
        return {}

    def fit_score_agent(  # noqa: D102
        self, agent_name: str, train_data: TensorDictBase, test_data: TensorDictBase
    ) -> float:
        return 0

    def reset_parameters(self):  # noqa: D102
        pass


MESSAGE_REGRESSORS: dict[MessageRegressionMethodType, MessageRegressor] = {}

R = TypeVar("R", bound=MessageRegressor)


def register_protocol_handler(
    method: MessageRegressionMethodType,
) -> Callable[[type[R]], type[R]]:
    """Register a message regressor.

    Parameters
    ----------
    method : MessageRegressionMethodType
        The method to register the regressor for.

    Returns
    -------
    decorator : Callable[[type[R]], type[R]]
        The decorator to register the regressor.
    """

    def decorator(cls: type[R]) -> type[R]:
        MESSAGE_REGRESSORS[method] = cls
        return cls

    return decorator


def build_message_regressor(
    hyper_params: HyperParameters,
    settings: ExperimentSettings,
    protocol_handler: ProtocolHandler,
) -> MessageRegressor:
    """Build a message regressor based on the parameters."""

    if not hyper_params.message_regression.enabled:
        return DummyMessageRegressor(hyper_params, settings, protocol_handler)

    return MESSAGE_REGRESSORS[hyper_params.message_regression.regression_method](
        hyper_params, settings, protocol_handler
    )


@register_protocol_handler("mlp")
class MlpMessageRegressor(MessageRegressor):
    """Regressor that uses an MLP to regress on the message data.

    The score is the accuracy of the regression.
    """

    def __init__(
        self,
        hyper_params: HyperParameters,
        settings: ExperimentSettings,
        protocol_handler: ProtocolHandler,
    ):
        super().__init__(hyper_params, settings, protocol_handler)

        # Build the models
        self.models: dict[str, Sequential] = {}
        for agent_name in self.agent_names:
            model_layers = []
            activation_class = ACTIVATION_CLASSES[
                hyper_params.message_regression.mlp_activation
            ]
            model_layers.append(
                LazyLinear(hyper_params.message_regression.mlp_hidden_size)
            )
            model_layers.append(activation_class())
            for _ in range(hyper_params.message_regression.mlp_num_layers - 2):
                model_layers.append(
                    Linear(
                        hyper_params.message_regression.mlp_hidden_size,
                        hyper_params.message_regression.mlp_hidden_size,
                    )
                )
                model_layers.append(activation_class())
            model_layers.append(
                Linear(hyper_params.message_regression.mlp_hidden_size, 2)
            )
            self.models[agent_name] = Sequential(*model_layers).to(settings.device)

    def fit_score_agent(
        self, agent_name: str, train_data: TensorDictBase, test_data: TensorDictBase
    ) -> float:
        """Fit and score the regressor on the data for a single agent.

        Parameters
        ----------
        agent_name : str
            The name of the agent.
        train_data : TensorDictBase
            A selection of the data for fitting the regressor.
        test_data : TensorDictBase
            A selection of the data for testing the regressor.

        Returns
        -------
        score : float
            The regression score.
        """

        # Get the flattened message and label data
        message_train = rearrange(train_data["message"], "batch ... -> batch (...)")
        label_train = train_data["y"].squeeze(-1)
        message_test = rearrange(test_data["message"], "batch ... -> batch (...)")
        label_test = test_data["y"].squeeze(-1)

        # Add the linear message to the message data if using
        if self.hyper_params.include_linear_message_space:
            linear_message_train = rearrange(
                train_data["linear_message"], "batch ... -> batch (...)"
            )
            linear_message_test = rearrange(
                test_data["linear_message"], "batch ... -> batch (...)"
            )
            message_train = torch.cat([message_train, linear_message_train], dim=-1)
            message_test = torch.cat([message_test, linear_message_test], dim=-1)

        # Create the dataloaders
        dataset_train = TensorDataset(message_train, label_train)
        dataset_test = TensorDataset(message_test, label_test)
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=self.hyper_params.message_regression.mlp_batch_size,
            shuffle=True,
        )
        dataloader_test = DataLoader(
            dataset_test,
            batch_size=self.hyper_params.message_regression.mlp_batch_size,
            shuffle=False,
        )

        # Get the model
        model = self.models[agent_name]
        model.train()

        # Get the optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.hyper_params.message_regression.mlp_learning_rate,
        )

        # Fit the model
        for _ in range(self.hyper_params.message_regression.mlp_num_epochs):
            for message_batch, label_batch in dataloader_train:
                message_batch = message_batch.to(self.settings.device)
                label_batch = label_batch.to(self.settings.device)
                model.zero_grad()
                output = model(message_batch)
                loss = F.cross_entropy(output, label_batch)
                loss.backward()
                optimizer.step()

        # Score the model
        model.eval()
        correct = 0
        with torch.no_grad():
            for message_batch, label_batch in dataloader_test:
                message_batch = message_batch.to(self.settings.device)
                label_batch = label_batch.to(self.settings.device)
                output = model(message_batch)
                correct += (output.argmax(dim=-1) == label_batch).sum().item()
        score = correct / len(dataset_test)

        return score

    def reset_parameters(self):
        """Reset the parameters of the models."""
        for model in self.models.values():
            for layer in model:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
