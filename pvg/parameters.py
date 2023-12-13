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
class GraphIsomorphismAgentParameters(AdditionalParameters):
    """Additional parameters for agents in the graph isomorphism experiment.

    Parameters
    ----------
    num_gnn_layers : int
        The number of layers in the agents's GNN.
    d_gnn : int
        The dimension of the hidden layers in the agents's GNN and of the attention
        embedding.
    d_gin_mlp : int
        The dimension of the hidden layers in the agents's Graph Isomorphism Network
        MLP.
    num_heads : int
        The number of heads in the agents's transformer.
    num_transformer_layers : int
        The number of transformer layers
    d_transformer : int
        The dimensionality of the transformer
    d_transformer_mlp
        The hidden dimension of the transformer MLP
    transformer_dropout : float
        The dropout value for the transformer
    d_node_selector : int
        The dimension of the hidden layer in the agents's MLP which selects a node to
        send as a message.
    use_batch_norm : bool
        Whether to use batch normalization in the agents's global pooling layer.
    noise_sigma : float
        The relative standard deviation of the Gaussian noise added to the agents's
        graph-level representations.
    use_pair_invariant_pooling : bool
        Whether to use pair-invariant pooling in the agents's global pooling layer. This
        makes the agents's graph-level representations invariant to the order of the
        graphs in the pair.
    """

    num_gnn_layers: int = 5
    d_gnn: int = 16
    d_gin_mlp: int = 64
    num_heads: int = 4
    num_transformer_layers: int = 4
    d_transformer: int = 16
    d_transformer_mlp: int = 64
    transformer_dropout: float = 0.0
    d_node_selector: int = 16
    d_decider: int = 16
    use_batch_norm: bool = True
    noise_sigma: float = 0.0
    use_pair_invariant_pooling: bool = True


@dataclass
class GraphIsomorphismParameters(AdditionalParameters):
    """Additional parameters specific to the graph isomorphism experiment.

    Parameters
    ----------
    prover : GraphIsomorphismAgentParameters
        Parameters for the prover
    verifier : GraphIsomorphismAgentParameters
        Parameters for the prover
    """

    prover: Optional[GraphIsomorphismAgentParameters | dict] = None
    verifier: Optional[GraphIsomorphismAgentParameters | dict] = None

    def __post_init__(self):
        # if isinstance(self.prover, dict):
        #     self.prover = GraphIsomorphismAgentParameters.from_dict(self.prover)
        # if isinstance(self.verifier, dict):
        #     self.verifier = GraphIsomorphismAgentParameters.from_dict(self.verifier)
        if self.prover is None:
            self.prover = GraphIsomorphismAgentParameters()
        elif isinstance(self.prover, dict):
            self.prover = GraphIsomorphismAgentParameters.from_dict(
                self.prover
            )
        if self.verifier is None:
            self.verifier = GraphIsomorphismAgentParameters()
        elif isinstance(self.verifier, dict):
            self.verifier = GraphIsomorphismAgentParameters.from_dict(
                self.verifier
            )


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
    batch_size : int
        The number of simultaneous environments to run in parallel.
    prover_reward : float
        The reward given to the prover when the verifier guesses "accept".
    verifier_reward : float
        The reward given to the verifier when it guesses correctly.
    verifier_terminated_penalty : float
        The reward given to the verifier if the episode terminates before it guesses.
    """

    scenario: str
    trainer: str
    dataset: str
    max_message_rounds: int = 8
    graph_isomorphism: Optional[GraphIsomorphismParameters | dict] = None

    batch_size: int = 64

    prover_reward: float = 1.0
    verifier_reward: float = 1.0
    verifier_terminated_penalty: float = -1.0

    def __post_init__(self):
        if self.scenario == "graph_isomorphism":
            if self.graph_isomorphism is None:
                self.graph_isomorphism = GraphIsomorphismParameters()
            elif isinstance(self.graph_isomorphism, dict):
                self.graph_isomorphism = GraphIsomorphismParameters.from_dict(
                    self.graph_isomorphism
                )

    def __post_init__(self):
        if self.scenario == "graph_isomorphism":
            if self.graph_isomorphism is None:
                self.graph_isomorphism = GraphIsomorphismParameters()
            elif isinstance(self.graph_isomorphism, dict):
                self.graph_isomorphism = GraphIsomorphismParameters.from_dict(
                    self.graph_isomorphism
                )
