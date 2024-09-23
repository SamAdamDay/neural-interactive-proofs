"""Expert Iteration (EI) trainer for text-based environments which only use APIs."""

from pvg.trainers.rl_pure_text_base import PureTextRlTrainer
from pvg.trainers.registry import register_trainer
from pvg.parameters import TrainerType


@register_trainer(TrainerType.PURE_TEXT_EI)
class PureTextEiTrainer(PureTextRlTrainer):
    """Expert Iteration (EI) trainer for text-based environments which only use APIs.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.
    scenario_instance : ScenarioInstance
        The components of the experiment.
    settings : ExperimentSettings
        The instance-specific settings of the experiment, like device, logging, etc.
    """

    def train(self):
        pass
