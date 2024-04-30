"""Base classes for all trainers."""

from abc import ABC, abstractmethod
from contextlib import ExitStack
from typing import ContextManager

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from pvg.parameters import Parameters
from pvg.scenario_instance import ScenarioInstance
from pvg.experiment_settings import ExperimentSettings
from pvg.utils.params import check_if_critic_and_single_body


class Trainer(ABC):
    """Base class for all trainers.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    scenario_instance : ScenarioInstance
        The components of the experiment.
    settings : ExperimentSettings
        The instance-specific settings of the experiment, like device, logging, etc.
    """

    def __init__(
        self,
        params: Parameters,
        scenario_instance: ScenarioInstance,
        settings: ExperimentSettings,
    ):
        self.params = params
        self.scenario_instance = scenario_instance
        self.settings = settings

        self._agent_names = self.scenario_instance.protocol_handler.agent_names

        self.device = self.settings.device

        # Check if we need a critic and if it shares a body with the actor
        self.use_critic, self.use_single_body = check_if_critic_and_single_body(params)

    @abstractmethod
    def train(self):
        """Train the agents."""
        pass

    def _build_train_context(self, stack: ExitStack) -> list[ContextManager]:
        """Builds the context manager ExitStack for training.

        Takes as input an ExitStack and adds the appropriate context managers to it,
        then returns the context managers.

        Parameters
        ----------
        stack : ExitStack
            The ExitStack to add the context managers to. Note that this is modified
            in-place.

        Returns
        -------
        context_managers : list[ContextManager]
            The target context managers to be used in the training loop.
        """

        context_managers = []

        def add_context_manager(context_manager):
            context_managers.append(stack.enter_context(context_manager))

        # When running on the CPU we need to use the MATH backend in order to calculate
        # the derivative of the scaled dot product.
        if self.settings.device.type == "cpu":
            add_context_manager(sdpa_kernel(SDPBackend.MATH))

        # Otherwise we enable all backends.
        else:
            add_context_manager(
                sdpa_kernel(
                    [
                        SDPBackend.MATH,
                        SDPBackend.CUDNN_ATTENTION,
                        SDPBackend.EFFICIENT_ATTENTION,
                        SDPBackend.FLASH_ATTENTION,
                    ]
                )
            )

        return context_managers

    def _build_test_context(self, stack: ExitStack) -> list[ContextManager]:
        """Builds the context manager ExitStack for testing.

        Takes as input an ExitStack and adds the appropriate context managers to it,
        then returns the context managers.

        Parameters
        ----------
        stack : ExitStack
            The ExitStack to add the context managers to. Note that this is modified
            in-place.

        Returns
        -------
        context_managers : list[ContextManager]
            The target context managers to be used in the testing loop.
        """

        context_managers = []

        def add_context_manager(context_manager):
            context_managers.append(stack.enter_context(context_manager))

        # All backends are enabled for testing.
        add_context_manager(
            sdpa_kernel(
                [
                    SDPBackend.MATH,
                    SDPBackend.CUDNN_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.FLASH_ATTENTION,
                ]
            )
        )

        # We don't need gradients for testing.
        add_context_manager(torch.no_grad())

        return context_managers
