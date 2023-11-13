from dataclasses import dataclass
from abc import ABC
from typing import Optional

import dacite


class BaseParameters(ABC):
    """Base class for parameters objects."""

    @classmethod
    def from_dict(cls, data):
        return dacite.from_dict(data_class=cls, data=data)


class AdditionalParameters(BaseParameters, ABC):
    pass


@dataclass
class GraphIsomorphismParameters(AdditionalParameters):
    """Additional parameters specific to the graph isomorphism experiment.

    Parameters
    ----------
    prover_num_layers : int
        The number of layers in the prover's GNN.
    prover_d_gnn : int
        The dimension of the hidden layers in the prover's GNN and of the attention
        embedding.
    prover_d_gin_mlp : int
        The dimension of the hidden layers in the prover's Graph Isomorphism Network
        MLP.
    prover_num_heads : int
        The number of heads in the prover's attention layer.
    prover_d_node_selector : int
        The dimension of the hidden layer in the prover's MLP which selects a node to
        send as a message.
    prover_use_batch_norm : bool
        Whether to use batch normalization in the prover's global pooling layer.
    prover_noise_sigma : float
        The relative standard deviation of the Gaussian noise added to the prover's
        graph-level representations.
    verifier_num_layers : int
        The number of layers in the verifier's GNN.
    verifier_d_gnn : int
        The dimension of the hidden layers in the verifier's GNN and of the attention
        embedding.
    verifier_d_gin_mlp : int
        The dimension of the hidden layers in the verifier's Graph Isomorphism Network
        MLP.
    verifier_num_heads : int
        The number of heads in the verifier's attention layer.
    verifier_d_node_selector : int
        The dimension of the hidden layer in the verifier's MLP which selects a node to
        send as a message.
    verifier_d_decider : int
        The dimension of the hidden layer in the verifier's MLP which decides whether to
        continue exchanging messages or to make a decision.
    verifier_use_batch_norm : bool
        Whether to use batch normalization in the verifier's global pooling layer.
    verifier_noise_sigma : float
        The relative standard deviation of the Gaussian noise added to the verifier's
        graph-level representations.
    """

    prover_num_layers: int = 5
    prover_d_gnn: int = 16
    prover_d_gin_mlp: int = 64
    prover_num_heads: int = 1
    prover_d_node_selector: int = 16
    prover_use_batch_norm: bool = True
    prover_noise_sigma: float = 0.0
    verifier_num_layers: int = 2
    verifier_d_gnn: int = 16
    verifier_d_gin_mlp: int = 64
    verifier_num_heads: int = 1
    verifier_d_node_selector: int = 16
    verifier_d_decider: int = 16
    verifier_use_batch_norm: bool = True
    verifier_noise_sigma: float = 0.0


@dataclass
class Parameters(BaseParameters):
    """Parameters of the experiment.

    Parameters
    ----------
    scenario : str
        The name of the scenario to run, which specifies the domain, task and agents.
    trainer : str
        The RL trainer to use.
    dataset : str
        The dataset to use.
    max_message_rounds : int
        The maximum number of rounds of messages, where the verifier sends a message,
        and the prover responds with a message.
    graph_isomorphism : GraphIsomorphismParameters, optional
        Additional parameters specific to the graph isomorphism experiment.
    """

    scenario: str
    trainer: str
    dataset: str
    max_message_rounds: int = 8
    graph_isomorphism: Optional[GraphIsomorphismParameters | dict] = None

    def __post_init__(self):
        if self.scenario == "graph_isomorphism":
            if self.graph_isomorphism is None:
                self.graph_isomorphism = GraphIsomorphismParameters()
            elif isinstance(self.graph_isomorphism, dict):
                self.graph_isomorphism = GraphIsomorphismParameters.from_dict(
                    self.graph_isomorphism
                )
