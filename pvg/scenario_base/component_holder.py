from abc import ABC, abstractmethod

import torch

from pvg.parameters import Parameters, ScenarioType
from pvg.scenario_base import Dataset, DataLoader, Agent, AgentsBuilder
from pvg.utils.types import TorchDevice


class ComponentHolder(ABC):
    """A base class for holding the components of an experiment.

    All scenarios should subclass this class and add the class attributes below.

    The principal aim of this class is to abstract away the details of the particular
    experiment being run.

    Parameters
    ----------
    params : Parameters
        The params of the experiment.
    device : TorchDevice
        The device to use for training.

    Attributes
    ----------
    dataset_class : type[Dataset]
        The dataset class for the experiment.
    dataloader_class : type[DataLoader]
        The data loader class to use for the experiment.
    agents_builder_class : type[AgentsBuilder]
        The class for building the agents for the experiment.
    """

    dataset_class : type[Dataset]
    dataloader_class : type[DataLoader]
    agents_builder_class : type[AgentsBuilder]

    def __init__(self, params: Parameters, device: TorchDevice):
        self.params = params
        self.device = device

        self.dataset = self.dataset_class(params, device)
        self.agents = self.agents_builder_class.build(params, device)
