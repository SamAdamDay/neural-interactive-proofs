from pvg.scenario_base import DataLoader, ComponentHolder
from pvg.graph_isomorphism import GraphIsomorphismDataset, GraphIsomorphismAgentsBuilder


class GraphIsomorphismComponentHolder(ComponentHolder):
    """A class which holds the components of a graph isomorphism experiment.

    Parameters
    ----------
    params : Parameters
        The params of the experiment.
    device : TorchDevice
        The device to use for training.
    """

    dataset_class = GraphIsomorphismDataset
    dataloader_class = DataLoader
    agents_builder_class = GraphIsomorphismAgentsBuilder
