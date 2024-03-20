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

from dataclasses import dataclass, fields
from abc import ABC
from typing import Optional, ClassVar, Iterable, NamedTuple
from enum import auto as enum_auto
from itertools import product

try:
    from enum import StrEnum
except ImportError:
    from pvg.utils.future import StrEnum

from pvg.constants import WANDB_ENTITY, WANDB_PROJECT


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
    SOS = enum_auto()
    PSOS = enum_auto()


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
    ABSTRACT_DECISION_PROBLEM = enum_auto()
    DEBATE = enum_auto()
    MERLIN_ARTHUR = enum_auto()
    MARKET_MAKING = enum_auto()  # TODO


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


# The agent names required for each protocol
AGENT_NAMES = {
    InteractionProtocolType.PVG: ("verifier", "prover"),
    InteractionProtocolType.ABSTRACT_DECISION_PROBLEM: ("verifier", "prover"),
    InteractionProtocolType.DEBATE: ("prover0", "prover1", "verifier"),
    InteractionProtocolType.MERLIN_ARTHUR: ("prover0", "prover1", "verifier"),
    InteractionProtocolType.MARKET_MAKING: ("verifier", "prover"),
}

DEFAULT_STACKELBERG_SEQUENCE = {
    InteractionProtocolType.PVG: (("verifier",), ("prover",)),
    InteractionProtocolType.ABSTRACT_DECISION_PROBLEM: (("verifier",), ("prover",)),
    InteractionProtocolType.DEBATE: (("verifier",), ("prover0", "prover1")),
    InteractionProtocolType.MERLIN_ARTHUR: (("verifier",), ("prover0", "prover1")),
    InteractionProtocolType.MARKET_MAKING: (("verifier",), ("prover",)),
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
    use_manual_architecture : bool
        Whether to use a manually defined architecture for the agent, which implements a
        hand-specified (non-learned) algorithm designed to maximise reward. This
        algorithm can be different depending on the environment. This is useful to test
        if the other agents can learn to work with a fixed optimum agent. It usually
        makes sense to set `agent_lr_factor` to 0 in this case.
    normalize_message_history : bool
        Whether to normalise the message history before passing it through the GNN
        encoder. Message histories are normalised to have zero mean and unit variance
        assuming that all episode lengths are equally frequent. (While this is probably
        not a realistic assumption, it's the most reasonable one we can make without
        knowing the true distribution of episode lengths. It's unlikely to make a big
        difference to the normalisation, and it's probably better than not normalising
        at all.)
    load_checkpoint_and_parameters : bool
        Whether to load the agent model checkpoint and parameters from W&B. In this
        case, all agent parameters are replaced by the parameters from the checkpoint.
        Otherwise, the model is randomly initialised. If `True`, the `checkpoint_run_id`
        parameter must be set.
    checkpoint_entity : str
        The entity of the W&B run to load the checkpoint from. If not provided, the
        default is used.
    checkpoint_project : str
        The project of the W&B run to load the checkpoint from. If not provided, the
        default is used.
    checkpoint_run_id: str, optional
        The ID of the W&B run to load the checkpoint from. Must be provided if
        `load_checkpoint` is `True`.
    checkpoint_version: str
        The version of the checkpoint to load. If not provided, the latest version is
        used.
    """

    agent_lr_factor: float = 1.0
    body_lr_factor: float = 1.0

    use_manual_architecture: bool = False

    normalize_message_history: bool = True

    load_checkpoint_and_parameters: bool = False
    checkpoint_entity: str = WANDB_ENTITY
    checkpoint_project: str = WANDB_PROJECT
    checkpoint_run_id: Optional[str] = None
    checkpoint_version: str = "latest"

    # The parameters which are preserved when loading from W&B config
    LOAD_PRESERVED_PARAMETERS: ClassVar[list[str]] = [
        "load_checkpoint_and_parameters",
        "checkpoint_entity",
        "checkpoint_project",
        "checkpoint_run_id",
        "checkpoint_version",
    ]

    is_random: ClassVar[bool] = False

    def to_dict(self) -> dict:
        params_dict = super().to_dict()

        # Add the is_random parameter. This is not a field of the parameters object, but
        # we want to include it in the dictionary.
        params_dict["is_random"] = self.is_random

        return params_dict

    def load_from_wandb_config(self, wandb_config: dict):
        """Load the parameters from a W&B config dictionary.

        Parameters
        ----------
        wandb_config : dict
            The W&B config dictionary for this agent (e.g.
            `wandb_run.config["agents"][agent_name]`).
        """
        for field in fields(self):
            if field.name in self.LOAD_PRESERVED_PARAMETERS:
                continue
            if field.name in wandb_config:
                setattr(self, field.name, wandb_config[field.name])
        setattr(self, "is_random", wandb_config["is_random"])


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
    use_dual_gnn : bool
        Whether to run two copies of the GNN in parallel, where on the first we take the
        features as the message history and on the second the features are all zeros.
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
        The learning rate factor for the body part of the model. The final LR for the
        body is obtained by multiplying this factor by the agent LR factor and the base
        LR. This allows updating the body at a different rate to the rest of the model.
    gnn_lr_factor : float
        The learning rate factor for the GNN part of the model. The final LR for the GNN
        is obtained by multiplying this factor by the body LR. This allows updating the
        GNN at a different rate to the rest of the model.
    """

    activation_function: ActivationType = ActivationType.TANH

    num_gnn_layers: int = 5
    d_gnn: int = 16
    d_gin_mlp: int = 64
    gnn_output_digits: Optional[int] = None
    use_dual_gnn: bool = True

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

    body_lr_factor: float = 1.0
    gnn_lr_factor: float = 0.1


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


class AgentsParameters(dict[str, AgentParameters]):
    """Parameters which specify the agents in the experiment.

    A subclass of `dict` which contains the parameters for each agent in the experiment.

    The keys are the names of the agents, and the values are the parameters for each
    agent.

    Agent names must not be substrings of each other.
    """

    def to_dict(self) -> dict:
        """Convert the parameters object to a dictionary.

        Turns sub-parameters into dictionaries.

        Returns
        -------
        params_dict : dict
            A dictionary of the parameters.
        """
        params_dict = {}
        for param_name, param in self.items():
            params_dict[param_name] = param.to_dict()
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


SosParams = NamedTuple("SosParams", [("a", float), ("b", float)])


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
    additional_lola_term : bool
        Whether to add an additional term to the SPG loss to make it equivalent to the later version of LOLA (first introduced implicitly in LOLA-DICE) as opposed to the original version.
    sos_params : NamedTuple
        The parameters for the SOS loss.
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
    additional_lola_term: bool = True
    sos_params: NamedTuple = SosParams(
        a=0.5, b=0.1
    )  # Default values taken from the original paper

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
class CommonProtocolParameters(SubParameters):
    """Common additional parameters for the interaction protocol.

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
class LongProtocolParameters(SubParameters, ABC):
    """Additional parameters for interaction protocols with multiple rounds.

    Parameters
    ----------
    max_message_rounds : int
        The maximum number of rounds of the game. Each round corresponds to one move by
        one or more agents.
    min_message_rounds : int
        The minimum number of rounds of messages. Before this point, the verifier's
        guesses are not registered.
    min_message_rounds_scheduler : MinMessageRoundsScheduler
        The scheduler to use for the minimum number of message rounds, allowing it to
        change over time. TODO: not currently implemented.
    """

    max_message_rounds: int = 8
    min_message_rounds: int = 0
    min_message_rounds_scheduler: MinMessageRoundsSchedulerType = (
        MinMessageRoundsSchedulerType.CONSTANT
    )

    def __post_init__(self):
        # Convert the scheduler to an enum type
        if not isinstance(
            self.min_message_rounds_scheduler, MinMessageRoundsSchedulerType
        ):
            self.min_message_rounds_scheduler = MinMessageRoundsSchedulerType[
                self.min_message_rounds_scheduler
            ]


@dataclass
class PvgProtocolParameters(LongProtocolParameters):
    """Additional parameters for the PVG interaction protocol.

    Parameters
    ----------
    verifier_first : bool
        Whether the verifier goes first in the PVG protocol.
    max_message_rounds : int
        The maximum number of rounds of the game. Each round corresponds to one move by
        one or more agents.
    min_message_rounds : int
        The minimum number of rounds of messages. Before this point, the verifier's
        guesses are not registered.
    min_message_rounds_scheduler : MinMessageRoundsScheduler
        The scheduler to use for the minimum number of message rounds, allowing it to
        change over time. TODO: not currently implemented.
    """

    verifier_first: bool = True


@dataclass
class DebateProtocolParameters(LongProtocolParameters):
    """Additional parameters for the debate interaction protocol.

    Parameters
    ----------
    max_message_rounds : int
        The maximum number of rounds of the game. Each round corresponds to one move by
        one or more agents.
    min_message_rounds : int
        The minimum number of rounds of messages. Before this point, the verifier's
        guesses are not registered.
    min_message_rounds_scheduler : MinMessageRoundsScheduler
        The scheduler to use for the minimum number of message rounds, allowing it to
        change over time. TODO: not currently implemented.
    """


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
    interaction_protocol : Protocol
        The interaction protocol between the agents.
    seed : int
        The random seed.
    pretrain_agents : bool
        Whether to pretrain the agents in isolation before running the main training.
        This pretrains the bodies of the agents using the parameters in `solo_agent`.
    test_size : float
        The proportion of the dataset to use for testing.
    d_representation : int
        The dimension of each agent's body representation output.
    batch_size : int
        The number of simultaneous environments to run in parallel.
    agents : AgentsParameters | dict[str, AgentParameters], optional
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

    pretrain_agents: bool = False

    test_size: float = 0.2

    d_representation: int = 16

    agents: Optional[AgentsParameters | dict[str, AgentParameters]] = None

    ppo: Optional[CommonPpoParameters | dict] = None
    vanilla_ppo: Optional[VanillaPpoParameters | dict] = None
    spg: Optional[SpgParameters | dict] = None
    solo_agent: Optional[SoloAgentParameters | dict] = None

    image_classification: Optional[ImageClassificationParameters | dict] = None

    dataset_options: Optional[DatasetParameters | dict] = None

    protocol_common: Optional[CommonProtocolParameters | dict] = None
    pvg_protocol: Optional[PvgProtocolParameters | dict] = None
    debate_protocol: Optional[DebateProtocolParameters | dict] = None

    def __post_init__(self):
        # Convert any strings to enums
        if not isinstance(self.scenario, ScenarioType):
            self.scenario = ScenarioType[self.scenario]
        if not isinstance(self.trainer, TrainerType):
            self.trainer = TrainerType[self.trainer]
        if not isinstance(self.interaction_protocol, InteractionProtocolType):
            self.interaction_protocol = InteractionProtocolType[
                self.interaction_protocol
            ]

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
            if self.vanilla_ppo is None:
                self.vanilla_ppo = VanillaPpoParameters()
            elif isinstance(self.vanilla_ppo, dict):
                self.vanilla_ppo = VanillaPpoParameters(**self.vanilla_ppo)
        elif self.trainer == TrainerType.SPG:
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

        # Add common interaction protocol parameters if they are not provided
        if self.protocol_common is None:
            self.protocol_common = CommonProtocolParameters()
        elif isinstance(self.protocol_common, dict):
            self.protocol_common = CommonProtocolParameters(**self.protocol_common)

        # Convert PVG and debate protocol parameters to the appropriate class
        if self.interaction_protocol == InteractionProtocolType.PVG:
            if self.pvg_protocol is None:
                self.pvg_protocol = PvgProtocolParameters()
            elif isinstance(self.pvg_protocol, dict):
                self.pvg_protocol = PvgProtocolParameters(**self.pvg_protocol)
        elif self.interaction_protocol == InteractionProtocolType.DEBATE:
            if self.debate_protocol is None:
                self.debate_protocol = DebateProtocolParameters()
            elif isinstance(self.debate_protocol, dict):
                self.debate_protocol = DebateProtocolParameters(**self.debate_protocol)

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
        Also validates the agent parameters.

        Parameters
        ----------
        agent_params_class : type[AgentParameters]
            The class of the agent parameters for the scenario.
        random_agent_params_class : type[RandomAgentParameters]
            The class of the random agent parameters for the scenario.
        protocol_params_class : type[ProtocolParameters]
            The class of the interaction protocol parameters for the scenario.
        """

        # If no agent parameters are provided, use the default parameters for the
        # protocol and scenario
        if self.agents is None:
            self.agents = AgentsParameters(
                **{
                    name: agent_params_class()
                    for name in AGENT_NAMES[self.interaction_protocol]
                }
            )

        if not isinstance(self.agents, dict):
            raise ValueError(
                f"Agent parameters must be a (subclass of) dict, not"
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
                    new_agents_params[agent_name] = agent_params_class(**agent_params)

            elif isinstance(agent_params, (agent_params_class, RandomAgentParameters)):
                new_agents_params[agent_name] = agent_params

            else:
                raise ValueError(
                    f"Agent parameters for agent {agent_name} are not a dictionary"
                    f" nor {agent_params_class}."
                )

        self.agents = new_agents_params

        # Make sure the agent names match the agent names expected by the protocol
        agent_names = tuple(self.agents.keys())
        if set(agent_names) != set(AGENT_NAMES[self.interaction_protocol]):
            raise ValueError(
                f"Agent names {agent_names} do not match the agent names expected"
                f"by interaction protocol {self.interaction_protocol}: "
                f"{AGENT_NAMES[self.interaction_protocol]}."
            )
