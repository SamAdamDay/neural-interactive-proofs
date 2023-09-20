import torch

from pvg.parameters import Parameters
from pvg.trainers import build_trainer
from pvg.scenarios import build_scenario
from pvg.data import load_dataset


class Experiment:
    """The experiment class, which builds and runs an experiment.

    Parameters
    ----------
    parameters : Parameters
        The parameters of the experiment.
    device : str | torch.device
        The device to use for training.
    """

    def __init__(self, parameters: Parameters, device: str | torch.device):
        self.parameters = parameters
        self.device = device
        self.trainer = build_trainer(parameters, device)
        self.scenario = build_scenario(parameters, device)
        self.dataset = load_dataset(parameters, device)

    def run(self):
        """Run the experiment."""
        self.trainer.train(self.scenario, self.dataset)
