"""Timeables to check performance of a complete run."""

from abc import ABC
from dataclasses import fields
from math import ceil

import torch

from pvg.parameters import (
    Parameters,
    SoloAgentParameters,
    ScenarioType,
    TrainerType,
    CommonPpoParameters,
    PvgProtocolParameters,
)
from pvg.run import run_experiment, prepare_experiment
from pvg.timing.timeables import TrainingTimeable, register_timeable


class RunTimeable(TrainingTimeable, ABC):
    """Base class for a timeable that performs a complete experiment run.

    Other than the arguments to the constructor, all other experiment params are their
    defaults.

    The schedule is as follows:

    1. For the first `wait` steps of training, do nothing.
    2. For each of the `repeat` cycles:
        a. For the first `warmup` steps of the cycle, run the profiler but don't record.
        b. For the next `active` steps of the cycle, run the profiler and record.

    To subclass, define the class attributes below.

    Parameters
    ----------
    param_scale : float, default=1.0
        Scale factor for key default experiment parameters.
    wait : int, default=2
        The number of training steps to wait before starting to profile.
    warmup : int, default=1
        The number of warmup steps in each cycle.
    active : int, default=3
        The number of steps to profile in each cycle.
    repeat : int, default=2
        The number of cycles to repeat.
    force_cpu : bool, default=False
        Whether to force everything to run on the CPU, even if a GPU is available.
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
        self,
        *,
        param_scale: float = 1.0,
        wait: int = 2,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 2,
        force_cpu: bool = False,
        pretrain: bool = False,
    ):
        super().__init__(
            param_scale=param_scale,
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
        )
        self.force_cpu = force_cpu
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
            for field in fields(SoloAgentParameters):
                if field.name == "batch_size":
                    default_batch_size = field.default
            solo_agent = SoloAgentParameters(
                num_epochs=self.num_steps,
                batch_size=ceil(default_batch_size * self.param_scale),
            )
            ppo = None
        else:
            for field in fields(CommonPpoParameters):
                if field.name == "minibatch_size":
                    default_minibatch_size = field.default
                elif field.name == "frames_per_batch":
                    default_frames_per_batch = field.default
            for field in fields(PvgProtocolParameters):
                if field.name == "max_message_rounds":
                    max_message_rounds = field.default
            frames_per_batch = (
                ceil(default_frames_per_batch * self.param_scale / max_message_rounds)
                * max_message_rounds
            )
            solo_agent = None
            ppo = CommonPpoParameters(
                num_iterations=self.num_steps,
                minibatch_size=ceil(default_minibatch_size * self.param_scale),
                frames_per_batch=frames_per_batch,
                num_test_iterations=1,
            )

        return Parameters(
            scenario=self.scenario,
            trainer=self.trainer,
            dataset=self.dataset,
            ppo=ppo,
            solo_agent=solo_agent,
            pretrain_agents=self.pretrain,
        )

    def run(self, profiler: torch.profiler.profile):
        """Run the experiment.

        Parameters
        ----------
        profiler : torch.profiler.profile
            The profiler to use.
        """
        run_experiment(self.params, device=self.device, profiler=profiler)


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
