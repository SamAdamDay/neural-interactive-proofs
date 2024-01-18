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

The parameters object can be converted to a dictionary using `Parameters.to_dict`.

Examples
--------
Create a parameters object, using default values for ppo parameters, and others
>>> params = Parameters(
...     scenario=Scenario.GRAPH_ISOMORPHISM,
...     trainer=Trainer.PPO,
...     dataset="eru10000",
...     agents=AgentsParams(
...         [
...             ("prover", GraphIsomorphismAgentParameters(d_gnn=128)),
...             ("verifier", GraphIsomorphismAgentParameters(num_gnn_layers=2)),
...         ],
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

from dataclasses import dataclass, asdict, fields
from abc import ABC
from typing import Optional, ClassVar, OrderedDict
from enum import auto as enum_auto
from textwrap import indent

try:
    from enum import StrEnum
except ImportError:
    from pvg.utils.future import StrEnum


class ScenarioType(StrEnum):
    """Enum for the scenario to run."""

    GRAPH_ISOMORPHISM = enum_auto()
    IMAGE_CLASSIFICATION = enum_auto()


class TrainerType(StrEnum):
    """Enum for the RL trainer to use."""

    PPO = enum_auto()
    SOLO_AGENT = enum_auto()


class BaseParameters(ABC):
    """Base class for parameters objects."""

    def to_dict(self) -> dict:
        """Convert the parameters object to a dictionary.

        Turns enums into strings, and sub-parameters into dictionaries. Includes the
        is_random parameter if it exists.

        Returns
        -------
        params_dict : dict
            A dictionary of the parameters.
        """

        # Add all dataclass fields to the dictionary
        params_dict = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, StrEnum):
                value = value.value
            elif isinstance(value, (BaseParameters, AgentsParameters)):
                value = value.to_dict()
            params_dict[field.name] = value

        return params_dict


class SubParameters(BaseParameters, ABC):
    """Base class for sub-parameters objects."""

    pass


class AgentParameters(SubParameters, ABC):
    """Base class for sub-parameters objects which define agents

    Parameters
    ----------
    body_lr_factor : float
        The learning rate factor for the body part of the model. This allows updating
        the body at a different rate to the rest of the model.
    """

    body_lr_factor: float = 1.0

    is_random: ClassVar[bool] = False

    def to_dict(self) -> dict:
        params_dict = super().to_dict()

        # Add the is_random parameter. This is not a field of the parameters object, but
        # we want to include it in the dictionary.
        params_dict["is_random"] = self.is_random

        return params_dict


@dataclass
class RandomAgentParameters(AgentParameters):
    """Parameters which specify a random agent"""

    is_random: ClassVar[bool] = True


@dataclass
class GraphIsomorphismAgentParameters(AgentParameters):
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
    body_lr_factor : float
        The learning rate factor for the body part of the model. This allows updating
        the body at a different rate to the rest of the model.
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
    body_lr_factor: float = 0.1


class AgentsParameters(OrderedDict[str, AgentParameters]):
    """Parameters which specify the agents in the experiment.

    A subclass of `OrderedDict`. Parameters should be specified as an iterable of
    `(name, parameters)` pairs, where `name` is a string, and `parameters` is a
    `AgentParameters` object.

    The keys are the names of the agents, and the values are the parameters for each
    agent.
    """

    def to_dict(self) -> dict:
        """Convert the parameters object to a dictionary.

        Adds a special key `_agent_order` which is a list of the agent names in the
        order they appear in the dictionary.

        Turns sub-parameters into dictionaries.

        Returns
        -------
        params_dict : dict
            A dictionary of the parameters.
        """
        params_dict = {}
        for param_name, param in self.items():
            params_dict[param_name] = param.to_dict()
        params_dict["_agent_order"] = list(self.keys())
        return params_dict


@dataclass
class PpoParameters(SubParameters):
    """Additional parameters for PPO.

    Parameters
    ----------
    frames_per_batch : int
        The number of frames to sample per training iteration. Should be divisible by
        `max_message_rounds`.
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
    body_lr_factor : float, optional
        The learning rate factor for the body part of the model. If set this overrides
        the `body_lr_factor` parameter of each agent.
    """

    # Sampling
    frames_per_batch: int = 1000
    num_iterations: int = 1000

    # Training
    num_epochs: int = 4
    minibatch_size: int = 64
    lr: float = 3e-4
    max_grad_norm = 1.0

    # PPO
    gamma: float = 0.99
    lmbda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_eps: float = 1e-4

    # Agents
    body_lr_factor: Optional[float] = None


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
    body_lr_factor : float, optional
        The learning rate factor for the body part of the model. If set this overrides
        the `body_lr_factor` parameter of each agent.
    """

    num_epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.001

    # Agents
    body_lr_factor: Optional[float] = None


@dataclass
class ImageClassificationParameters(SubParameters):
    """Additional parameters for the image classification task.

    Parameters
    ----------
    selected_classes : tuple[int, int]
        The indices of the classes of the original dataset to select to use for the
        binary classification task.
    """

    selected_classes: tuple[int, int] = (0, 1)


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
    seed : int
        The random seed.
    max_message_rounds : int
        The maximum number of rounds of messages, where the verifier sends a message,
        and the prover responds with a message.
    pretrain_agents : bool
        Whether to pretrain the agents in isolation before running the main training.
        This pretrains the bodies of the agents using the parameters in `solo_agent`.
    test_size : float
        The proportion of the dataset to use for testing.
    d_representation : int
        The dimension of each agent's body representation output.
    batch_size : int
        The number of simultaneous environments to run in parallel.
    prover_reward : float
        The reward given to the prover when the verifier guesses "accept".
    verifier_reward : float
        The reward given to the verifier when it guesses correctly.
    verifier_terminated_penalty : float
        The reward given to the verifier if the episode terminates before it guesses.
    agents : AgentsParameters | OrderedDict[str, AgentParameters], optional
        Additional parameters for the agents. The keys are the names of the agents, and
        the values are the parameters for each agent. If not provided, the default
        parameters are used for each agent for a given scenario.
    ppo : PpoParameters, optional
        Additional parameters for PPO.
    solo_agent : SoloAgentParameters, optional
        Additional parameters for running agents in isolation. Used when the trainer is
        "solo_agent" or when `pretrain_agents` is `True`.
    image_classification : ImageClassificationParameters, optional
        Additional parameters for the image classification task.
    """

    scenario: ScenarioType
    trainer: TrainerType
    dataset: str

    seed: int = 6198

    max_message_rounds: int = 8
    pretrain_agents: bool = False

    test_size: float = 0.2

    d_representation: int = 16

    prover_reward: float = 1.0
    verifier_reward: float = 1.0
    verifier_terminated_penalty: float = -1.0

    agents: Optional[AgentsParameters | OrderedDict[str, AgentParameters]] = None
    ppo: Optional[PpoParameters | dict] = None
    solo_agent: Optional[SoloAgentParameters | dict] = None
    image_classification: Optional[ImageClassificationParameters | dict] = None

    def __post_init__(self):
        if self.scenario == ScenarioType.GRAPH_ISOMORPHISM:
            self.agents = _process_agents_params(
                self.agents,
                GraphIsomorphismAgentParameters,
                RandomAgentParameters,
            )

        elif self.scenario == ScenarioType.IMAGE_CLASSIFICATION:
            if self.image_classification is None:
                self.image_classification = ImageClassificationParameters()
            elif isinstance(self.image_classification, dict):
                self.image_classification = ImageClassificationParameters(
                    **self.image_classification
                )

        if self.trainer == TrainerType.PPO:
            if self.ppo is None:
                self.ppo = PpoParameters()
            elif isinstance(self.ppo, dict):
                self.ppo = PpoParameters(**self.ppo)

        if self.trainer == TrainerType.SOLO_AGENT or self.pretrain_agents:
            if self.solo_agent is None:
                self.solo_agent = SoloAgentParameters()
            elif isinstance(self.solo_agent, dict):
                self.solo_agent = SoloAgentParameters(**self.solo_agent)


def _process_agents_params(
    agents_params: AgentsParameters | OrderedDict[str, AgentParameters] | None,
    agent_params_class: type[AgentParameters],
    random_agent_params_class: type[RandomAgentParameters],
) -> AgentsParameters:
    """Process agent parameters passed to `Parameters`.

    Fills in missing agent parameters with the default parameters for the scenario.

    Parameters
    ----------
    agents_params : AgentsParameters | OrderedDict[str, AgentParameters] | None
        The agent parameters passed to `Parameters`.
    agent_params_class : type[AgentParameters]
        The class of the agent parameters for the scenario.
    random_agent_params_class : type[RandomAgentParameters]
        The class of the random agent parameters for the scenario.

    Returns
    -------
    new_agents_params : AgentsParameters
        The processed agent parameters.
    """

    # If no agent parameters are provided, use the default parameters for the scenario
    if agents_params is None:
        return AgentsParameters(
            [
                ("prover", agent_params_class()),
                ("verifier", agent_params_class()),
            ]
        )

    if not isinstance(agents_params, OrderedDict):
        raise ValueError(
            f"Agent parameters must be a (subclass of) OrderedDict, not"
            f" {type(agents_params)}."
        )

    new_agents_params = AgentsParameters()

    for agent_name, agent_params in agents_params.items():
        # If the agent parameters are a dictionary, convert them to the appropriate
        # class
        if isinstance(agent_params, dict):
            is_random = False
            if "is_random" in agent_params:
                is_random = agent_params.pop("is_random")
            if is_random:
                new_agents_params[agent_name] = random_agent_params_class(
                    **agent_params
                )
            else:
                new_agents_params[agent_name] = GraphIsomorphismAgentParameters(
                    **agent_params
                )

        elif isinstance(agent_params, (agent_params_class, RandomAgentParameters)):
            new_agents_params[agent_name] = agent_params

        else:
            raise ValueError(
                f"Agent parameters for agent {agent_name} are not a dictionary"
                f" nor {agent_params_class}."
            )

    return new_agents_params
