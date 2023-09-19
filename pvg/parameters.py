from dataclasses import dataclass


@dataclass
class Parameters:
    """Parameters of the experiment.

    Parameters
    ----------
    scenario : str
        The name of the scenario to run, which specifies the domain, task and agents.
    trainer : str
        The RL trainer to use.
    dataset : str
        The dataset to use.
    """

    scenario: str
    trainer: str
    dataset: str
