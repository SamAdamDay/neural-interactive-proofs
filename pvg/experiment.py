import torch

from pvg.parameters import Parameters
from pvg.trainers import build_trainer
from pvg.base import build_scenario, load_dataset


class Experiment:
    """The experiment class, which builds and runs an experiment.

    Parameters
    ----------
    params : Parameters
        The params of the experiment.
    device : str | torch.device
        The device to use for training.
    """

    def __init__(self, params: Parameters, device: str | torch.device):
        self.params = params
        self.device = device
        self.trainer = build_trainer(params, device)
        self.scenario = build_scenario(params, device)
        self.dataset = load_dataset(params, device)

    def run(self):
        """Run the experiment."""
        self.trainer.train(self.scenario, self.dataset)
