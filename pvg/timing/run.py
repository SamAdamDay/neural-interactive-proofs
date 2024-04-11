"""Timeables to check performance of a complete run."""

from abc import ABC

import torch

from pvg.parameters import (
    Parameters,
    SoloAgentParameters,
    ScenarioType,
    TrainerType,
    CommonPpoParameters,
)
from pvg.run import run_experiment, prepare_experiment
from pvg.timing.timeables import Timeable, register_timeable


class RunTimeable(Timeable, ABC):
    """Base class for a timeable that performs a complete experiment run.

    Other than the arguments to the constructor, all other experiment params are their
    defaults.

    To subclass, define the class attributes below.

    Parameters
    ----------
    force_cpu : bool, default=False
        Whether to force everything to run on the CPU, even if a GPU is available.
    num_steps : int, default=10
        The number of steps to run the experiment for. Depending on the trainer, this
        could be iterations or epochs.
    pretrain : bool, default=False
        When running an RL experiment, whether to pretrain the model.

    Class Attributes
    ----------------
    scenario : ScenarioType
        Which scenario to use.
    trainer : TrainerType
        The trainer to use.
    dataset : str
        The name of the dataset to use.
    """

    scenario: ScenarioType
    trainer: TrainerType
    dataset: str

    def __init__(
        self, *, force_cpu: bool = False, num_steps: int = 10, pretrain: bool = False
    ):
        self.force_cpu = force_cpu
        self.num_steps = num_steps
        self.pretrain = pretrain

        self.params = self._get_params()

        if torch.cuda.is_available() and not force_cpu:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Prepare the experiment, e.g. build the dataset
        prepare_experiment(self.params)

    def _get_params(self) -> Parameters:
        """Get the parameters which define the experiment.

        Returns
        -------
        params : Parameters
            The parameters of the experiment.
        """

        # Set the number of steps in the appropriate place
        if self.trainer == TrainerType.SOLO_AGENT:
            solo_agent = SoloAgentParameters(num_epochs=self.num_steps)
            ppo = None
        else:
            solo_agent = None
            ppo = CommonPpoParameters(num_iterations=self.num_steps)

        return Parameters(
            scenario=self.scenario,
            trainer=self.trainer,
            dataset=self.dataset,
            ppo=ppo,
            solo_agent=solo_agent,
        )

    def run(self, profiler: torch.profiler.profile):
        """Run the experiment.

        Parameters
        ----------
        profiler : torch.profiler.profile
            The profiler to use.
        """
        run_experiment(self.params, device=self.device)


@register_timeable(name="graph_isomorphism_solo_agent")
class GraphIsomorphismSoloAgentRunTimeable(RunTimeable):
    """Timeable for running the graph isomorphism scenario with solo agents."""

    scenario = ScenarioType.GRAPH_ISOMORPHISM
    trainer = TrainerType.SOLO_AGENT
    dataset = "eru10000"


@register_timeable(name="graph_isomorphism_ppo")
class GraphIsomorphismPpoRunTimeable(RunTimeable):
    """Timeable for running the graph isomorphism scenario with vanilla PPO."""

    scenario = ScenarioType.GRAPH_ISOMORPHISM
    trainer = TrainerType.VANILLA_PPO
    dataset = "eru10000"
