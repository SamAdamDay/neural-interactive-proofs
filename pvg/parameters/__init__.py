"""The parameters of the experiment.

An experiment should be completely reproducible from its parameters (up to hardware
quirks).

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
1. Create a parameters object, using default values for ppo parameters, and others

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

2. Convert the parameters object to a dictionary

>>> params.to_dict()
{'scenario': 'graph_isomorphism', 'trainer': 'ppo', 'dataset': 'eru10000', ...}

3. Create a parameters object using a dictionary for the ppo parameters

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

from typing import Optional
from dataclasses import dataclass

from .base import BaseParameters, SubParameters, ParameterValue
from .types import (
    ScenarioType,
    SpgVariant,
    IhvpVariant,
    Guess,
    TrainerType,
    PpoLossType,
    BinarificationMethodType,
    ActivationType,
    InteractionProtocolType,
    MinMessageRoundsSchedulerType,
    ImageBuildingBlockType,
    MessageRegressionMethodType,
)
from .agents import (
    LrFactors,
    AgentParameters,
    RandomAgentParameters,
    GraphIsomorphismAgentParameters,
    ImageClassificationAgentParameters,
    CodeValidationAgentParameters,
    AgentsParameters,
)
from .trainers import (
    RlTrainerParameters,
    CommonPpoParameters,
    VanillaPpoParameters,
    SpgParameters,
    ReinforceParameters,
    SoloAgentParameters,
    EiParameters,
)
from .protocol import (
    CommonProtocolParameters,
    PvgProtocolParameters,
    DebateProtocolParameters,
    MnipProtocolParameters,
    ZkProtocolParameters,
)
from .scenario import ImageClassificationParameters, CodeValidationParameters
from .dataset import DatasetParameters
from .update_schedule import (
    AgentUpdateSchedule,
    ConstantUpdateSchedule,
    ContiguousPeriodicUpdateSchedule,
    AlternatingPeriodicUpdateSchedule,
)
from .message_regression import MessageRegressionParameters

# The agent names required for each protocol
AGENT_NAMES: dict[InteractionProtocolType, tuple[str, ...]] = {
    InteractionProtocolType.PVG: ("verifier", "prover"),
    InteractionProtocolType.ABSTRACT_DECISION_PROBLEM: ("verifier", "prover"),
    InteractionProtocolType.DEBATE: ("prover0", "prover1", "verifier"),
    InteractionProtocolType.MERLIN_ARTHUR: ("prover0", "prover1", "verifier"),
    InteractionProtocolType.MARKET_MAKING: ("verifier", "prover"),
    InteractionProtocolType.MULTI_CHANNEL_TEST: (
        "verifier",
        "prover0",
        "prover1",
        "prover2",
    ),
}

DEFAULT_STACKELBERG_SEQUENCE: dict[
    InteractionProtocolType, tuple[tuple[str, ...], ...]
] = {
    InteractionProtocolType.PVG: (("verifier",), ("prover",)),
    InteractionProtocolType.ABSTRACT_DECISION_PROBLEM: (("verifier",), ("prover",)),
    InteractionProtocolType.DEBATE: (("verifier",), ("prover0", "prover1")),
    InteractionProtocolType.MERLIN_ARTHUR: (("verifier",), ("prover0", "prover1")),
    InteractionProtocolType.MARKET_MAKING: (("verifier",), ("prover",)),
    InteractionProtocolType.MULTI_CHANNEL_TEST: (
        ("verifier",),
        ("prover0", "prover1", "prover2"),
    ),
}


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
    functionalize_modules : bool
        Whether to functionalize the modules in the agents. This allows some additional
        features which we don't currently use, and comes with a big speed cost.
        Disabling it also prevents batch norm from tracking running statistics in eval
        mode, which might have a small effect on performance (unknown). Furthermore,
        disabling this prevents freezing parameters using `requires_grad` when doing a
        non-constant agent update schedule. Otherwise we get "RuntimeError: LSE is not
        correctly aligned".
    pretrain_agents : bool
        Whether to pretrain the agents in isolation before running the main training.
        This pretrains the bodies of the agents using the parameters in `solo_agent`.
    test_size : float
        The proportion of the dataset to use for testing.
    d_representation : int
        The dimension of each agent's body representation output.
    message_size : int
        The size of the message sent by agents. This is a dimension of the message
        space and effectively allows sending multiple messages simultaneously.
    include_linear_message_space : bool
        Whether to include a 1-dimensional message space in addition to the message
        space specified by the scenario. This allows sending a single number as a
        message, in addition to the normal message. This can be useful for debugging
        issues with the message space, especially when combined with shared reward,
        since it should be easier to learn to send a single number when both agents want
        to cooperate.
    d_linear_message_space : int
        The dimension of the linear message space (i.e. the number of possible messages
        which can sent). This is only used if `include_linear_message_space` is `True`.
    batch_size : int
        The number of simultaneous environments to run in parallel.
    agents : AgentsParameters | dict[str, AgentParameters], optional
        Additional parameters for the agents. The keys are the names of the agents, and
        the values are the parameters for each agent. If not provided, the default
        parameters are used for each agent for a given scenario.
    rl : RlTrainerParams, optional
        Common parameters for all RL trainers.
    ppo : PpoParameters, optional
        Common parameters for PPO trainers.
    vanilla_ppo : VanillaPpoParameters, optional
        Additional parameters for the vanilla PPO trainer.
    spg : SpgParameters, optional
        Additional parameters for SPG and its variants.
    reinforce : ReinforceParameters, optional
        Additional parameters for the REINFORCE trainer.
    solo_agent : SoloAgentParameters, optional
        Additional parameters for running agents in isolation. Used when the trainer is
        "solo_agent" or when `pretrain_agents` is `True`.
    image_classification : ImageClassificationParameters, optional
        Additional parameters for the image classification task.
    code_validation : CodeValidationParameters, optional
        Additional parameters for the code validation task.
    dataset_options : DatasetParameters, optional
        Additional parameters for the dataset.
    """

    scenario: ScenarioType
    trainer: TrainerType
    dataset: str

    interaction_protocol: InteractionProtocolType = InteractionProtocolType.PVG

    seed: int = 6198

    functionalize_modules: bool = False

    pretrain_agents: bool = False

    test_size: float = 0.2

    d_representation: int = 16

    message_size: int = 1
    include_linear_message_space: bool = False
    d_linear_message_space: int = 2

    agents: Optional[AgentsParameters | dict[str, AgentParameters]] = None

    rl: Optional[RlTrainerParameters | dict] = None
    ppo: Optional[CommonPpoParameters | dict] = None
    vanilla_ppo: Optional[VanillaPpoParameters | dict] = None
    spg: Optional[SpgParameters | dict] = None
    reinforce: Optional[ReinforceParameters | dict] = None
    solo_agent: Optional[SoloAgentParameters | dict] = None
    ei: Optional[EiParameters | dict] = None

    image_classification: Optional[ImageClassificationParameters | dict] = None
    code_validation: Optional[CodeValidationParameters | dict] = None

    dataset_options: Optional[DatasetParameters | dict] = None

    protocol_common: Optional[CommonProtocolParameters | dict] = None
    pvg_protocol: Optional[PvgProtocolParameters | dict] = None
    debate_protocol: Optional[DebateProtocolParameters | dict] = None
    mnip_protocol: Optional[MnipProtocolParameters | dict] = None
    zk_protocol: Optional[ZkProtocolParameters | dict] = None

    message_regression: Optional[MessageRegressionParameters | dict] = None

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

        if self.scenario == ScenarioType.GRAPH_ISOMORPHISM:
            self._process_agents_params(
                GraphIsomorphismAgentParameters,
                RandomAgentParameters,
            )

        elif self.scenario == ScenarioType.IMAGE_CLASSIFICATION:
            self._process_agents_params(
                ImageClassificationAgentParameters,
                RandomAgentParameters,
            )

        elif self.scenario == ScenarioType.CODE_VALIDATION:
            self._process_agents_params(
                CodeValidationAgentParameters,
                RandomAgentParameters,
            )

        # Add PPO parameters for specific variants to the appropriate class
        if self.trainer == TrainerType.SPG:
            if self.spg is None:
                self.spg = SpgParameters(
                    stackelberg_sequence=DEFAULT_STACKELBERG_SEQUENCE[
                        self.interaction_protocol
                    ]
                )

        super().__post_init__()

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
                **{name: agent_params_class() for name in get_agent_names(self)}
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
        if set(agent_names) != set(get_agent_names(self)):
            raise ValueError(
                f"Agent names {agent_names} do not match the agent names expected"
                f"by interaction protocol {self.interaction_protocol}: "
                f"{get_agent_names(self)}."
            )


def get_agent_names(params: Parameters) -> list[str]:
    """Get the agent names required for the protocol.

    Parameters
    ----------
    params : Parameters
        The parameters of the experiment.

    Returns
    -------
    list[str]
        The agent names required for the protocol.
    """
    agent_names = list(AGENT_NAMES[params.interaction_protocol])
    if params.protocol_common.zero_knowledge:
        agent_names.extend(["simulator", "adversarial_verifier"])
    return agent_names
