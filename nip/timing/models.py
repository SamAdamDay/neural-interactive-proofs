"""Timeables to check model inference performance."""

from abc import ABC
from itertools import cycle
from typing import ClassVar

import torch

from tensordict import TensorDict
from tensordict.nn import TensorDictSequential

from nip.parameters import (
    HyperParameters,
    SoloAgentParameters,
    ScenarioType,
)
from nip.experiment_settings import ExperimentSettings
from nip.scenario_base import TensorDictDataLoader
from nip.factory import build_scenario_instance
from nip.utils.data import truncated_iterator
from nip.timing.timeables import Timeable, register_timeable
from nip.utils.maths import set_seed


class ModelTimeable(Timeable, ABC):
    """Base class for a timeable that runs a model.

    To subclass, define the class attributes below.

    Parameters
    ----------
    param_scale : float, default=1.0
        Scale factor for key default parameters (currently unused)
    force_cpu : bool, default=False
        Whether to force the model to run on the CPU, even if a GPU is available.
    batch_size : int, default=64
        The batch size to use for the model.
    num_batches : int, default=100
        The number of batches to run the model on.

    Attributes
    ----------
    scenario : ClassVar[ScenarioType]
        The scenario which defines the model architecture and datasets.
    dataset : ClassVar[str]
        The name of the dataset to use.
    agent_name : ClassVar[str]
        The name of the agent to use for the model.
    """

    scenario: ClassVar[ScenarioType]
    dataset: ClassVar[str]
    agent_name: ClassVar[str]

    def __init__(
        self,
        *,
        param_scale: float = 1.0,
        force_cpu: bool = False,
        batch_size: int = 64,
        num_batches: int = 100,
    ):
        super().__init__(param_scale=param_scale)
        self.force_cpu = force_cpu
        self.batch_size = batch_size
        self.num_batches = num_batches

        # Set up the components of the experiment
        self.hyper_params = self._get_params()
        if torch.cuda.is_available() and not force_cpu:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.settings = ExperimentSettings(device=self.device)
        self.scenario_instance = build_scenario_instance(
            self.hyper_params, self.settings
        )

        # Set the random seeds
        set_seed(self.hyper_params.seed)
        self.generator = torch.Generator().manual_seed(self.hyper_params.seed)

        # Set up the full model as the body and head
        self.model = TensorDictSequential(
            self.scenario_instance.agents[self.agent_name].body,
            self.scenario_instance.agents[self.agent_name].solo_head,
        )
        self.model.to(self.device)
        self.model.eval()

    def _get_params(self) -> HyperParameters:
        """Get the parameters which define the experiment containing the model.

        Returns
        -------
        hyper_params : HyperParameters
            The parameters of the experiment.
        """
        return HyperParameters(
            scenario=self.scenario,
            trainer="solo_agent",
            dataset=self.dataset,
            solo_agent=SoloAgentParameters(batch_size=self.batch_size),
        )

    def run(self, profiler: torch.profiler.profile):
        """Run the model.

        Parameters
        ----------
        profiler : torch.profiler.profile
            The profiler to run the model with.
        """

        dataloader = TensorDictDataLoader(
            self.scenario_instance.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=self.generator,
        )
        dataloader = truncated_iterator(cycle(dataloader), self.num_batches)

        with torch.no_grad():
            for data in dataloader:
                data: TensorDict
                data = data.to(self.device)
                self.model(data)


@register_timeable(name="graph_isomorphism_verifier")
class GraphIsomorphismVerifierTimeable(ModelTimeable):
    """Timeable to run the graph isomorphism verifier model."""

    scenario = "graph_isomorphism"
    dataset = "eru10000"
    agent_name = "verifier"
