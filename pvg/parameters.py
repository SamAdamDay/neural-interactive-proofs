from dataclasses import dataclass
from abc import ABC
from typing import Optional


class AdditionalParameters(ABC):
    pass


@dataclass
class GraphIsomorphismParameters(AdditionalParameters):
    """Additional parameters specific to the graph isomorphism experiment.

    Parameters
    ----------
    prover_num_layers : int
        The number of layers in the prover's GNN.
    prover_gnn_hidden_size : int
        The dimension of the hidden layers in the prover's GNN.
    prover_num_heads : int
        The number of heads in the prover's attention layer.
    prover_d_attn : int
        The embed dimension in the prover's attention layer.
    prover_d_final_mlp : int
        The dimension of the hidden layer in the prover's final MLP.
    verifier_num_layers : int
        The number of layers in the verifier's GNN.
    verifier_gnn_hidden_size : int
        The dimension of the hidden layers in the verifier's GNN.
    verifier_num_heads : int
        The number of heads in the verifier's attention layer.
    verifier_d_attn : int
        The embed dimension in the verifier's attention layer.
    verifier_d_final_mlp : int
        The dimension of the hidden layer in the verifier's final MLP.
    """

    prover_num_layers: int = 5
    prover_d_gnn: int = 64
    prover_num_heads: int = 1
    prover_d_attn: int = 64
    prover_d_final_mlp: int = 16
    verifier_num_layers: int = 2
    verifier_d_gnn: int = 64
    verifier_num_heads: int = 1
    verifier_d_attn: int = 64
    verifier_d_final_mlp: int = 16


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
    max_num_messages : int
        The maximum number of messages which the verifier can send.
    graph_isomorphism : GraphIsomorphismParameters, optional
        Additional parameters specific to the graph isomorphism experiment.
    """

    scenario: str
    trainer: str
    dataset: str
    max_num_messages: int = 8
    graph_isomorphism: Optional[GraphIsomorphismParameters] = None
