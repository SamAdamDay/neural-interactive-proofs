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
from typing import Optional, ClassVar, OrderedDict, Iterable
from enum import auto as enum_auto
from textwrap import indent
from itertools import product

try:
    from enum import StrEnum
except ImportError:
    from pvg.utils.future import StrEnum


class ScenarioType(StrEnum):
    """Enum for the scenario to run."""

    GRAPH_ISOMORPHISM = enum_auto()
    IMAGE_CLASSIFICATION = enum_auto()


class SpgVariant(StrEnum):
    """Enum for SPG variants."""

    SPG = enum_auto()
    PSPG = enum_auto()
    LOLA = enum_auto()
    POLA = enum_auto()
    SOS = enum_auto()  # TODO
    PSOS = enum_auto()  # TODO


class IhvpVariant(StrEnum):
    CONJ_GRAD = enum_auto()
    NEUMANN = enum_auto()
    NYSTROM = enum_auto()


class TrainerType(StrEnum):
    """Enum for the RL trainer to use."""

    VANILLA_PPO = enum_auto()
    SOLO_AGENT = enum_auto()
    SPG = enum_auto()


class BinarificationMethodType(StrEnum):
    """Enum for ways of turning a multi-class classification task into a binary one.

    Enums
    -----
    SELECT_TWO
        Select two classes from the original dataset to use for the binary
        classification task.
    MERGE
        Merge all classes from the original dataset into two classes.
    RANDOM
        Select classes completely at random.
    """

    SELECT_TWO = enum_auto()
    MERGE = enum_auto()
    RANDOM = enum_auto()


class ActivationType(StrEnum):
    """Enum for the activation function to use.

    To add a new activation function, add it to this enum, and add the corresponding
    function to `ACTIVATION_CLASSES` in `pvg.utils.torch_modules`.
    """

    RELU = enum_auto()
    TANH = enum_auto()
    SIGMOID = enum_auto()


class InteractionProtocolType(StrEnum):
    """Enum for the interaction protocol to use in the environment.

    Enums
    -----
    PVG
        The full Prover-Verifier Game protocol.
    ABSTRACT_DECISION_PROBLEM
        The Abstract Decision Problem protocol.
    DEBATE
        The Debate protocol.
    MERLIN_ARTHUR
        The Merlin-Arthur classifier protocol.
    """

    PVG = enum_auto()
    ABSTRACT_DECISION_PROBLEM = enum_auto()  # TODO
    DEBATE = enum_auto()  # TODO
    MERLIN_ARTHUR = enum_auto()  # TODO


class MinMessageRoundsSchedulerType(StrEnum):
    """Enum for the scheduler to use for the minimum number of message rounds.

    Enums
    -----
    CONSTANT
        Use a constant `min_message_rounds` minimum number of message rounds.
    LINEAR_DECREASE
        Linearly increase the minimum number of message rounds over time, starting with
        `min_message_rounds` and ending with 1.
    LINEAR_INCREASE
        Linearly decrease the minimum number of message rounds over time, starting with
        1 and ending with `min_message_rounds`.
    LINEAR_INCREASE_DECREASE
        Linearly increase the minimum number of message rounds over time, starting with
        `min_message_rounds` and ending with 1, then linearly decrease the minimum
        number of message rounds over time, starting with 1 and ending with
        `min_message_rounds`.
    """

    CONSTANT = enum_auto()
    LINEAR_DECREASE = enum_auto()
    LINEAR_INCREASE = enum_auto()
    LINEAR_INCREASE_DECREASE = enum_auto()


DEFAULT_AGENT_NAMES = {
    InteractionProtocolType.PVG: ("verifier", "prover"),
    InteractionProtocolType.ABSTRACT_DECISION_PROBLEM: ("prover", "verifier"),
    InteractionProtocolType.DEBATE: ("prover1", "prover2", "verifier"),
    InteractionProtocolType.MERLIN_ARTHUR: ("prover", "verifier"),
}

DEFAULT_STACKELBERG_SEQUENCE = {
    InteractionProtocolType.PVG: (("verifier",), ("prover",)),
    InteractionProtocolType.ABSTRACT_DECISION_PROBLEM: (("verifier",), ("prover",)),
    InteractionProtocolType.DEBATE: (("verifier",), ("prover1", "prover2")),
    InteractionProtocolType.MERLIN_ARTHUR: (("verifier",), ("prover",)),
}


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


@dataclass
class AgentParameters(SubParameters, ABC):
    """Base class for sub-parameters objects which define agents

    Parameters
    ----------
    agent_lr_factor : float
        The learning rate factor for the whole agent compared with the base learning
        rate. This allows updating the agents at different rates.
    body_lr_factor : float
        The learning rate factor for the body part of the model compared with with whole
        agent. This allows updating the body at a different rate to the rest of the
        model.
    """

    agent_lr_factor: float = 1.0
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
    activation_function : ActivationType
        The activation function to use.
    num_gnn_layers : int
        The number of layers in the agents's GNN.
    d_gnn : int
        The dimension of the hidden layers in the agents's GNN and of the attention
        embedding.
    d_gin_mlp : int
        The dimension of the hidden layers in the agents's Graph Isomorphism Network
        MLP.
    gnn_output_digits : int, optional
        The number of digits in the output of the agents's GNN. If not provided, the
        output is not rounded.
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
    include_round_in_decider : bool
        Whether to include the round number in the agents's decider MLP.
    d_value : int
        The dimension of the hidden layer in the agents's MLP which estimates the value
        function.
    num_value_layers : int
        The number of layers in the agents's value MLP.
    include_round_in_value : bool
        Whether to include the round number in the agents's value MLP.
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

    activation_function: ActivationType = ActivationType.TANH

    num_gnn_layers: int = 5
    d_gnn: int = 16
    d_gin_mlp: int = 64
    gnn_output_digits: Optional[int] = 4

    num_heads: int = 4
    num_transformer_layers: int = 4
    d_transformer: int = 16
    d_transformer_mlp: int = 64
    transformer_dropout: float = 0.0

    d_node_selector: int = 16
    num_node_selector_layers: int = 2

    d_decider: int = 16
    num_decider_layers: int = 2
    include_round_in_decider: bool = True

    d_value: int = 16
    num_value_layers: int = 2
    include_round_in_value: bool = True

    use_batch_norm: bool = True
    noise_sigma: float = 0.0
    use_pair_invariant_pooling: bool = True

    body_lr_factor: float = 0.1


@dataclass
class ImageClassificationAgentParameters(AgentParameters):
    """Additional parameters for agents in the image classification experiment.

    Parameters
    ----------
    activation_function : ActivationType
        The activation function to use.
    num_convs_per_group : int
        The number of convolutional layers in each group in the agents's CNN.
    kernel_size : int
        The kernel size of the convolutional layers in the agents's CNN.
    stride : int
        The stride of the convolutional layers in the agents's CNN.
    d_latent_pixel_selector : int
        The dimension of the hidden layer in the agents's MLP which selects a latent
        pixel to send as a message.
    num_latent_pixel_selector_layers : int
        The number of layers in the agents's latent pixel selector MLP.
    d_decider : int
        The dimension of the hidden layer in the agents's MLP which decides whether
        continue exchanging messages or to guess.
    num_decider_layers : int
        The number of layers in the agents's decider MLP.
    include_round_in_decider : bool
        Whether to include the round number in the agents's decider MLP.
    d_value : int
        The dimension of the hidden layer in the agents's MLP which estimates the value
        function.
    num_value_layers : int
        The number of layers in the agents's value MLP.
    include_round_in_value : bool
        Whether to include the round number in the agents's value MLP.
    """

    activation_function: ActivationType = ActivationType.TANH

    num_convs_per_group: int = 2
    kernel_size: int = 3
    stride: int = 1

    d_latent_pixel_selector: int = 16
    num_latent_pixel_selector_layers: int = 2

    d_decider: int = 16
    num_decider_layers: int = 2
    include_round_in_decider: bool = True

    d_value: int = 16
    num_value_layers: int = 2
    include_round_in_value: bool = True


class AgentsParameters(OrderedDict[str, AgentParameters]):
    """Parameters which specify the agents in the experiment.

    A subclass of `OrderedDict`. Parameters should be specified as an iterable of
    `(name, parameters)` pairs, where `name` is a string, and `parameters` is a
    `AgentParameters` object.

    The keys are the names of the agents, and the values are the parameters for each
    agent.

    Agent names must not be substrings of each other.
    """

    def __init__(self, other=(), /, **kwds):
        super().__init__(other, **kwds)

        # Check that the agent names are not substrings of each other
        for name_1, name_2 in product(self.keys(), repeat=2):
            if name_1 != name_2 and name_1 in name_2:
                raise ValueError(
                    f"Agent names must not be substrings of each other, but {name_1}"
                    f" is a substring of {name_2}."
                )

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
class CommonPpoParameters(SubParameters):
    """Common parameters for PPO trainers.

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
    normalize_advantage : bool
        Whether to normalise the advantages in the PPO loss.
    body_lr_factor : float, optional
        The learning rate factor for the body part of the model. If set this overrides
        the `body_lr_factor` parameter of each agent.
    num_test_iterations : int
        The number of iterations to run the test for.
    """

    # Sampling
    frames_per_batch: int = 1000
    num_iterations: int = 1000

    # Training
    num_epochs: int = 4
    minibatch_size: int = 64
    lr: float = 0.003
    max_grad_norm = 1.0

    # PPO
    gamma: float = 0.9
    lmbda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_eps: float = 0.001
    normalize_advantage: bool = True

    # Agents
    body_lr_factor: Optional[float] = None

    # Testing
    num_test_iterations: int = 10


@dataclass
class VanillaPpoParameters(SubParameters):
    """Additional parameters for the vanilla PPO trainer."""


@dataclass
class SpgParameters(SubParameters):
    """Additional parameters for SPG and its variants.

    Parameters
    ----------
    variant : SpgVariant
        The variant of SPG to use.
    stackelberg_sequence : tuple[tuple[str]]
        The sequence of agents to use in the Stackelberg game. The leaders first then
        their respective followers, and so forth.
    names : tuple[str]
        The names of the agents in the Stackelberg game, in the order they were created
        (to enable mapping between agent names and indices).
    ihvp_variant : IhvpVariant
        The variant of IHVP to use.
    ihvp_num_iterations : int
        The number of iterations to use in the IHVP approximation.
    ihvp_rank : int
        The rank of the approximation to use in the IHVP approximation.
    ihvp_rho : float
        The damping factor to use in the IHVP approximation.
    """

    variant: SpgVariant = SpgVariant.SPG
    stackelberg_sequence: tuple[tuple[int]] = (("verifier",), ("prover",))

    # IHVP
    ihvp_variant: IhvpVariant = IhvpVariant.NYSTROM
    ihvp_num_iterations: int = 5  # Default value taken from hypergrad package example
    ihvp_rank: int = 5  # Default value taken from hypergrad package example
    ihvp_rho: float = 0.1  # Default value taken from hypergrad package example


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
    num_conv_groups : int
        The number of groups of convolutional layers in each agents's CNN.
    initial_num_channels : int
        The number of channels in the first convolutional layer in each agents's CNN.
    """

    num_conv_groups: int = 1
    initial_num_channels: int = 16


@dataclass
class DatasetParameters(SubParameters):
    """Additional parameters for the dataset.

    Parameters
    ----------
    binarification_method : BinarificationMethodType
        The method to use to turn the multi-class classification task into a binary
        classification task.
    selected_classes : tuple[int, int], optional
        When selecting two classes from the original dataset, the indices of the classes
        to select. If not provided, the default for the dataset is used.
    binarification_seed : int, optional
        The seed used when doing a randomised binarification. If not provided, the
        default for the dataset is used.
    make_balanced : bool
        Whether to make sure the dataset is balanced.
    """

    binarification_method: BinarificationMethodType = BinarificationMethodType.MERGE
    selected_classes: Optional[tuple[int, int]] = None
    binarification_seed: Optional[int] = None
    make_balanced: bool = True


@dataclass
class PvgProtocolParameters(SubParameters):
    """Additional parameters for the PVG protocol.

    Parameters
    ----------
    prover_reward : float
        The reward given to the prover when the verifier guesses "accept".
    verifier_reward : float
        The reward given to the verifier when it guesses correctly.
    verifier_incorrect_penalty : float
        The penalty given to the verifier when it guesses incorrectly.
    verifier_terminated_penalty : float
        The reward given to the verifier if the episode terminates before it guesses.
    verifier_no_guess_reward : float
        The reward given to the verifier if it does not make a guess in a round.
    shared_reward : bool
        Whether to use a shared reward function, where the prover gets the same reward
        as the verifier. This overrides `prover_reward`.
    """

    prover_reward: float = 1.0
    verifier_reward: float = 1.0
    verifier_incorrect_penalty: float = -1.0
    verifier_terminated_penalty: float = -1.0
    verifier_no_guess_reward: float = 0.0

    shared_reward: bool = False


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
    protocol : Protocol
        The interaction protocol between the agents.
    seed : int
        The random seed.
    max_message_rounds : int
        The maximum number of rounds of the game. Each round corresponds to one move by
        one agent.
    min_message_rounds : int
        The minimum number of rounds of messages. Before this point, verifier cannot
        guess.
    min_message_rounds_scheduler : MinMessageRoundsScheduler
        The scheduler to use for the minimum number of message rounds, allowing it to
        change over time. TODO: not currently implemented.
    pretrain_agents : bool
        Whether to pretrain the agents in isolation before running the main training.
        This pretrains the bodies of the agents using the parameters in `solo_agent`.
    test_size : float
        The proportion of the dataset to use for testing.
    d_representation : int
        The dimension of each agent's body representation output.
    batch_size : int
        The number of simultaneous environments to run in parallel.
    agents : AgentsParameters | OrderedDict[str, AgentParameters], optional
        Additional parameters for the agents. The keys are the names of the agents, and
        the values are the parameters for each agent. If not provided, the default
        parameters are used for each agent for a given scenario.
    ppo : PpoParameters, optional
        Common parameters for PPO trainers.
    vanilla_ppo : VanillaPpoParameters, optional
        Additional parameters for the vanilla PPO trainer.
    spg : SpgParameters, optional
        Additional parameters for SPG and its variants.
    solo_agent : SoloAgentParameters, optional
        Additional parameters for running agents in isolation. Used when the trainer is
        "solo_agent" or when `pretrain_agents` is `True`.
    image_classification : ImageClassificationParameters, optional
        Additional parameters for the image classification task.
    dataset_options : DatasetParameters, optional
        Additional parameters for the dataset.
    """

    scenario: ScenarioType
    trainer: TrainerType
    dataset: str

    interaction_protocol: InteractionProtocolType = InteractionProtocolType.PVG

    seed: int = 6198

    max_message_rounds: int = 8
    min_message_rounds: int = 0
    min_message_rounds_scheduler: MinMessageRoundsSchedulerType = (
        MinMessageRoundsSchedulerType.CONSTANT
    )
    pretrain_agents: bool = False

    test_size: float = 0.2

    d_representation: int = 16

    agents: Optional[AgentsParameters | OrderedDict[str, AgentParameters]] = None

    ppo: Optional[CommonPpoParameters | dict] = None
    vanilla_ppo: Optional[VanillaPpoParameters | dict] = None
    spg: Optional[SpgParameters | dict] = None
    solo_agent: Optional[SoloAgentParameters | dict] = None

    image_classification: Optional[ImageClassificationParameters | dict] = None

    dataset_options: Optional[DatasetParameters | dict] = None

    pvg_protocol: Optional[PvgProtocolParameters | dict] = None

    def __post_init__(self):
        # Convert graph isomorphism agent parameters to the appropriate class
        if self.scenario == ScenarioType.GRAPH_ISOMORPHISM:
            self._process_agents_params(
                GraphIsomorphismAgentParameters,
                RandomAgentParameters,
            )

        # Convert image classification agent parameters to the appropriate class, and
        # convert the image classification parameters to the appropriate class
        elif self.scenario == ScenarioType.IMAGE_CLASSIFICATION:
            self._process_agents_params(
                ImageClassificationAgentParameters,
                RandomAgentParameters,
            )
            if self.image_classification is None:
                self.image_classification = ImageClassificationParameters()
            elif isinstance(self.image_classification, dict):
                self.image_classification = ImageClassificationParameters(
                    **self.image_classification
                )

        # Add common PPO parameters if they are not provided
        if self.trainer == TrainerType.VANILLA_PPO or self.trainer == TrainerType.SPG:
            if self.ppo is None:
                self.ppo = CommonPpoParameters()
            elif isinstance(self.ppo, dict):
                self.ppo = CommonPpoParameters(**self.ppo)

        # Convert PPO parameters for specific variants to the appropriate class
        if self.trainer == TrainerType.VANILLA_PPO:
            if self.spg is None:
                self.spg = VanillaPpoParameters()
            elif isinstance(self.spg, dict):
                self.spg = VanillaPpoParameters(**self.spg)
        if self.trainer == TrainerType.SPG:
            if self.spg is None:
                self.spg = SpgParameters(
                    stackelberg_sequence=DEFAULT_STACKELBERG_SEQUENCE[
                        self.interaction_protocol
                    ]
                )
            elif isinstance(self.spg, dict):
                self.spg = SpgParameters(**self.spg)

        # Convert solo agent parameters to SoloAgentParameters
        if self.trainer == TrainerType.SOLO_AGENT or self.pretrain_agents:
            if self.solo_agent is None:
                self.solo_agent = SoloAgentParameters()
            elif isinstance(self.solo_agent, dict):
                self.solo_agent = SoloAgentParameters(**self.solo_agent)
                self.solo_agent = SoloAgentParameters.from_dict(self.solo_agent)

        # Convert interaction protocol parameters to InteractionProtocolParameters
        if self.interaction_protocol == InteractionProtocolType.PVG:
            if self.pvg_protocol is None:
                self.pvg_protocol = PvgProtocolParameters()
            elif isinstance(self.pvg_protocol, dict):
                self.pvg_protocol = PvgProtocolParameters(**self.pvg_protocol)

        # Convert dataset options to DatasetParameters
        if self.dataset_options is None:
            self.dataset_options = DatasetParameters()
        elif isinstance(self.dataset_options, dict):
            self.dataset_options = DatasetParameters(**self.dataset_options)

    def _process_agents_params(
        self,
        agent_params_class: type[AgentParameters],
        random_agent_params_class: type[RandomAgentParameters],
    ) -> AgentsParameters:
        """Process agent parameters passed to `Parameters`.

        Fills in missing agent parameters with the default parameters for the scenario.

        Parameters
        ----------
        agent_params_class : type[AgentParameters]
            The class of the agent parameters for the scenario.
        random_agent_params_class : type[RandomAgentParameters]
            The class of the random agent parameters for the scenario.
        protocol_params_class : type[InteractionProtocolParameters]
            The class of the interaction protocol parameters for the scenario.
        """

        # If no agent parameters are provided, use the default parameters for the
        # protocol and scenario
        if self.agents is None:
            self.agents = AgentsParameters(
                [
                    (name, agent_params_class())
                    for name in DEFAULT_AGENT_NAMES[self.interaction_protocol]
                ]
            )

        if not isinstance(self.agents, OrderedDict):
            raise ValueError(
                f"Agent parameters must be a (subclass of) OrderedDict, not"
                f" {type(self.agents)}."
            )

        new_agents_params = AgentsParameters()

        for agent_name, agent_params in self.agents.items():
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

        self.agents = new_agents_params
