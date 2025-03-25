"""Classes for training agents in an experiment.

A trainer takes the components of a scenario and trains the agents.
"""

from nip.parameters import HyperParameters
from nip.scenario_instance import ScenarioInstance
from nip.experiment_settings import ExperimentSettings

from .trainer_base import (
    Trainer,
    TensorDictTrainer,
    IterationContext,
    attach_progress_bar,
)
from .rl_tensordict_base import TensorDictRlTrainer
from .vanilla_ppo import VanillaPpoTrainer
from .solo_agent import SoloAgentTrainer
from .spg import SpgTrainer
from .reinforce import ReinforceTrainer
from .rl_pure_text_base import PureTextRlTrainer
from .ei_pure_text import PureTextEiTrainer
from .malt_pure_text import PureTextMaltTrainer
from .registry import register_trainer, TRAINER_REGISTRY


def get_trainer_class(hyper_params: HyperParameters) -> type[Trainer]:
    """Get the trainer class from the hyperparameters.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.

    Returns
    -------
    trainer_class : type[Trainer]
        The trainer class.
    """
    return TRAINER_REGISTRY[hyper_params.trainer]


def build_trainer(
    hyper_params: HyperParameters,
    scenario_instance: ScenarioInstance,
    settings: ExperimentSettings,
) -> Trainer:
    """Build a trainer from parameters (factory function).

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    scenario_instance : ScenarioInstance
        The components of the experiment.
    settings : ExperimentSettings
        The instance-specific settings of the experiment, like device, logging, etc.
    """
    return get_trainer_class(hyper_params)(hyper_params, scenario_instance, settings)
