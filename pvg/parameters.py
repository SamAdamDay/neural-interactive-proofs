"""The parameters of the experiment.

An experiment should be completely reproducible from its parameters.

The parameters are initialised by constructing a `Parameters` object. This object
completely defines the experiment, and is passed around to all experiment components.

Some experiment parameters are sub-parameters, which are defined in separate classes.
When the `Parameters` object is initialised, these sub-parameters may be initialised as
well, according to the values of the main parameters.

When creating sub-parameters, you can either pass then as an object of the appropriate
sub-parameter class, or as a dictionary. The advantage of the former is that you can use
symbol inspection (e.g. in VS Code) to have easy access to the parameter names and
descriptions. If you pass a dictionary, it will be converted to the appropriate
sub-parameter class.

The whole parameters object can be provided as a dictionary using
`Parameters.from_dict`.

The parameters object can be converted to a dictionary using `Parameters.to_dict`.

Examples
--------
Create a parameters object, using default values for ppo parameters, and others
>>> params = Parameters(
...     scenario=Scenario.GRAPH_ISOMORPHISM,
...     trainer=Trainer.PPO,
...     dataset="eru10000",
...     graph_isomorphism=GraphIsomorphismParameters(
...         prover=GraphIsomorphismAgentParameters(
...             d_gnn=128,
...             use_batch_norm=False,
...         ),
...     ),
... )

Convert the parameters object to a dictionary
>>> params.to_dict()
{'scenario': 'graph_isomorphism', 'trainer': 'ppo', 'dataset': 'eru10000', ...}

Create a parameters object using a dictionary for the ppo parameters
>>> params = Parameters(
...     scenario=Scenario.GRAPH_ISOMORPHISM,
...     trainer=Trainer.PPO,
...     dataset="eru10000",
...     ppo={
...         "num_epochs": 100,
...         "batch_size": 256,
...     },
... )
"""

from dataclasses import dataclass, asdict
from abc import ABC
from typing import Optional
from enum import auto as enum_auto

import dacite

try:
    from enum import StrEnum
except ImportError:
    from pvg.utils.future import StrEnum


class ScenarioType(StrEnum):
    """Enum for the scenario to run."""

    GRAPH_ISOMORPHISM = enum_auto()


class TrainerType(StrEnum):
    """Enum for the RL trainer to use."""

    PPO = enum_auto()
    SOLO_AGENT = enum_auto()


class BaseParameters(ABC):
    """Base class for parameters objects."""

    @classmethod
    def from_dict(cls, data):
        return dacite.from_dict(data_class=cls, data=data)

    def to_dict(self):
        return asdict(self)


class SubParameters(BaseParameters, ABC):
    """Base class for sub-parameters objects."""

    pass


@dataclass
class GraphIsomorphismAgentParameters(SubParameters):
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
    num_node_selector_layers : int
        The number of layers in the agents's node selector MLP.
    d_decider : int
        The dimension of the hidden layer in the agents's MLP which decides whether to
        accept or reject.
    num_decider_layers : int
        The number of layers in the agents's decider MLP.
    d_value : int
        The dimension of the hidden layer in the agents's MLP which estimates the value
        function.
    num_value_layers : int
        The number of layers in the agents's value MLP.
    d_critic : int
        The dimension of the hidden layer in the agents's MLP which estimates the value
        function.
    num_critic_transformer_layers : int
        The number of transformer layers in the agents's critic.
    num_critic_layers : int
        The number of layers in the agents's critic MLP.
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
    num_node_selector_layers: int = 2
    d_decider: int = 16
    num_decider_layers: int = 2
    d_value: int = 16
    num_value_layers: int = 2
    d_critic: int = 16
    num_critic_transformer_layers: int = 2
    num_critic_layers: int = 2
    use_batch_norm: bool = True
    noise_sigma: float = 0.0
    use_pair_invariant_pooling: bool = True


@dataclass
class GraphIsomorphismParameters(SubParameters):
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
            self.prover = GraphIsomorphismAgentParameters.from_dict(self.prover)
        if self.verifier is None:
            self.verifier = GraphIsomorphismAgentParameters()
        elif isinstance(self.verifier, dict):
            self.verifier = GraphIsomorphismAgentParameters.from_dict(self.verifier)


@dataclass
class PpoParameters(SubParameters):
    """Additional parameters for PPO.

    Parameters
    ----------
    frames_per_batch : int
        The number of frames to sample per training iteration.
    num_iterations : int
        The number of sampling and training iterations. `num_iterations *
        frames_per_batch` is the total number of frames sampled during training.
    num_epochs : int
        The number of epochs per training iteration.
    minibatch_size : int
        The size of the minibatches in each optimization step.
    lr : float
        The learning rate.
    max_grad_norm : float
        The maximum norm of the gradients during optimization.
    gamma : float
        The discount factor.
    lmbda : float
        The GAE lambda parameter.
    clip_epsilon : float
        The PPO clip range.
    entropy_eps : float
        The coefficient of the entropy term in the PPO loss.
    body_lr_factor : float
        The learning rate factor for the body part of the model. This allows updating
        the body at a different rate to the rest of the model.
    """

    # Sampling
    frames_per_batch = 1000
    num_iterations: int = 8

    # Training
    num_epochs: int = 4
    minibatch_size: int = 64
    lr = 3e-4
    max_grad_norm = 1.0

    # PPO
    gamma: float = 0.99
    lmbda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_eps: float = 1e-4

    # Agents
    body_lr_factor: float = 1.0


@dataclass
class SoloAgentParameters(SubParameters):
    """Additional parameters for running agents in isolation.

    Parameters
    ----------
    num_epochs : int
        The number of epochs to train for.
    batch_size : int
        The batch size.
    learning_rate : float
        The learning rate.
    body_lr_factor : float
        The learning rate factor for the body part of the model. This allows updating
        the body at a different rate to the rest of the model.
    test_size : float
        The proportion of the dataset to use for testing.
    """

    num_epochs: int = 500
    batch_size: int = 256
    learning_rate: float = 0.001
    body_lr_factor: float = 0.01

    test_size: float = 0.2


@dataclass
class Parameters(BaseParameters):
    """Parameters of the experiment.

    Parameters
    ----------
    scenario : Scenario
        The name of the scenario to run, which specifies the domain, task and agents.
    trainer : Trainer
        The RL trainer to use.
    dataset : str
        The dataset to use.
    max_message_rounds : int
        The maximum number of rounds of messages, where the verifier sends a message,
        and the prover responds with a message.
    batch_size : int
        The number of simultaneous environments to run in parallel.
    prover_reward : float
        The reward given to the prover when the verifier guesses "accept".
    verifier_reward : float
        The reward given to the verifier when it guesses correctly.
    verifier_terminated_penalty : float
        The reward given to the verifier if the episode terminates before it guesses.
    graph_isomorphism : GraphIsomorphismParameters, optional
        Additional parameters specific to the graph isomorphism experiment.
    ppo : PpoParameters, optional
        Additional parameters for PPO.
    solo_agent : SoloAgentParameters, optional
        Additional parameters for running agents in isolation. Used when the trainer is
        "solo_agent".
    """

    scenario: ScenarioType
    trainer: TrainerType
    dataset: str

    seed: int = 6198

    max_message_rounds: int = 8

    prover_reward: float = 1.0
    verifier_reward: float = 1.0
    verifier_terminated_penalty: float = -1.0

    graph_isomorphism: Optional[GraphIsomorphismParameters | dict] = None
    ppo: Optional[PpoParameters | dict] = None
    solo_agent: Optional[SoloAgentParameters | dict] = None

    def __post_init__(self):
        if self.scenario == ScenarioType.GRAPH_ISOMORPHISM:
            if self.graph_isomorphism is None:
                self.graph_isomorphism = GraphIsomorphismParameters()
            elif isinstance(self.graph_isomorphism, dict):
                self.graph_isomorphism = GraphIsomorphismParameters.from_dict(
                    self.graph_isomorphism
                )
        if self.trainer == TrainerType.PPO:
            if self.ppo is None:
                self.ppo = PpoParameters()
            elif isinstance(self.ppo, dict):
                self.ppo = PpoParameters.from_dict(self.ppo)
        elif self.trainer == TrainerType.SOLO_AGENT:
            if self.solo_agent is None:
                self.solo_agent = SoloAgentParameters()
            elif isinstance(self.solo_agent, dict):
                self.solo_agent = SoloAgentParameters.from_dict(self.solo_agent)
