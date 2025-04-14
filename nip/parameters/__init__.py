"""The hyper-parameters of the experiment.

An experiment should be completely reproducible from its hyper-parameters (up to
hardware quirks and model API non-reproducibility).

The parameters are initialised by constructing a ``HyperParameters`` object. This object
completely defines the experiment, and is passed around to all experiment components.

Some experiment parameters are sub-parameters, which are defined in separate classes.
When the ``HyperParameters`` object is initialised, these sub-parameters may be
initialised as well, according to the values of the main parameters.

When creating sub-parameters, you can either pass then as an object of the appropriate
sub-parameter class, or as a dictionary. The advantage of the former is that you can use
symbol inspection (e.g. in VS Code) to have easy access to the parameter names and
descriptions. If you pass a dictionary, it will be converted to the appropriate
sub-parameter class.

The parameters object can be converted to a dictionary using
``HyperParameters.to_dict``.

Examples
--------
1. Create a parameters object, using default values for ppo parameters, and others

>>> hyper_params = HyperParameters(
...     scenario="graph_isomorphism",
...     trainer="ppo",
...     dataset="eru10000",
...     agents=AgentsParams(
...         prover=GraphIsomorphismAgentParameters(d_gnn=128),
...         verifier=GraphIsomorphismAgentParameters(num_gnn_layers=2),
...     ),
... )

2. Convert the parameters object to a dictionary

>>> hyper_params.to_dict()
{'scenario': 'graph_isomorphism', 'trainer': 'ppo', 'dataset': 'eru10000', ...}

3. Create a parameters object using a dictionary for the ppo parameters

>>> hyper_params = HyperParameters(
...     scenario="graph_isomorphism",
...     trainer="ppo",
...     dataset="eru10000",
...     agents=AgentsParams(
...         prover={"d_gnn": 128},
...         verifier={"num_gnn_layers": 2},
...     ),
...     ppo={
...         "num_epochs": 100,
...         "batch_size": 256,
...     },
... )
"""

from typing import Optional
import typing
from dataclasses import dataclass, fields

from nip.utils.version import get_package_name, get_version

from .parameters_base import (
    BaseHyperParameters,
    SubParameters,
    ParameterValue,
    register_parameter_class,
)
from .types import (
    ScenarioType,
    SpgVariantType,
    IhvpVariantType,
    GuessType,
    TrainerType,
    TestSchemeType,
    PpoLossType,
    BinarificationMethodType,
    ActivationType,
    InteractionProtocolType,
    VerifierDecisionScaleType,
    ImageBuildingBlockType,
    MessageRegressionMethodType,
    BaseRunType,
)
from .agents import (
    LrFactors,
    AgentParameters,
    RandomAgentParameters,
    GraphIsomorphismAgentParameters,
    ImageClassificationAgentParameters,
    PureTextAgentParameters,
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
    PureTextEiParameters,
    TextRlParameters,
    PureTextMaltParameters,
)
from .protocol import (
    CommonProtocolParameters,
    NipProtocolParameters,
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
from .base_run import BaseRunParameters, BaseRunPreserve
from .version import convert_hyper_param_dict


@register_parameter_class
@dataclass
class HyperParameters(BaseHyperParameters):
    """The hyper-parameters of the experiment.

    An experiment should be completely reproducible from its hyper-parameters (up to
    hardware quirks).

    Parameters
    ----------
    scenario : ScenarioType
        The name of the scenario to run, which specifies the domain, task and agents.
    trainer : TrainerType
        The RL trainer to use.
    dataset : str
        The dataset to use.
    interaction_protocol : InteractionProtocolType
        The interaction protocol between the agents.
    seed : int
        The random seed.
    functionalize_modules : bool
        Whether to functionalize the modules in the agents. This allows some additional
        features which we don't currently use, and comes with a big speed cost.
        Disabling it also prevents batch norm from tracking running statistics in eval
        mode, which might have a small effect on performance (unknown). Furthermore,
        disabling this prevents freezing parameters using ``requires_grad`` when doing a
        non-constant agent update schedule. Otherwise we get "RuntimeError: LSE is not
        correctly aligned".
    pretrain_agents : bool
        Whether to pretrain the agents in isolation before running the main training.
        This pretrains the bodies of the agents using the parameters in ``solo_agent``.
    test_size : float
        The proportion of the dataset to use for testing.
    d_representation : int
        The dimension of each agent's body representation output.
    message_size : int
        The size of the message sent by agents. This is a dimension of the message space
        and effectively allows sending multiple messages simultaneously.
    include_linear_message_space : bool
        Whether to include a 1-dimensional message space in addition to the message
        space specified by the scenario. This allows sending a single number as a
        message, in addition to the normal message. This can be useful for debugging
        issues with the message space, especially when combined with shared reward,
        since it should be easier to learn to send a single number when both agents want
        to cooperate.
    d_linear_message_space : int
        The dimension of the linear message space (i.e. the number of possible messages
        which can sent). This is only used if ``include_linear_message_space`` is
        ``True``.
    agents : AgentsParameters | dict[str, AgentParameters], optional
        Parameters for the agents. The keys are the names of the agents, and the values
        are the parameters for each agent. If not provided, the default parameters are
        used for each agent for a given scenario.
    rl : RlTrainerParams, optional
        Common parameters for all RL trainers.
    ppo : PpoParameters, optional
        Common parameters for PPO trainers.
    vanilla_ppo : VanillaPpoParameters, optional
        Parameters for the vanilla PPO trainer.
    spg : SpgParameters, optional
        Parameters for SPG and its variants.
    reinforce : ReinforceParameters, optional
        Parameters for the REINFORCE trainer.
    solo_agent : SoloAgentParameters, optional
        Parameters for running agents in isolation. Used when the trainer is
        "solo_agent" or when ``pretrain_agents`` is ``True``.
    pure_text_ei : PureTextEiParameters, optional
        Parameters for the expert iteration (EI) trainer which works with agents that call a
        text-based API.
    pure_text_malt : PureTextMaltParameters, optional
        Parameters for the multi-agent LLM training (MALT) trainer which works with agents that call a text-based API.
    image_classification : ImageClassificationParameters, optional
        Parameters for the image classification task.
    code_validation : CodeValidationParameters, optional
        Parameters for the code validation task.
    dataset_options : DatasetParameters, optional
        Parameters for the dataset.
    protocol_common : CommonProtocolParameters, optional
        Parameters common to all protocols.
    nip_protocol : NipProtocolParameters, optional
        Parameters for the NIP protocol.
    debate_protocol : DebateProtocolParameters, optional
        Parameters for the debate protocol.
    mnip_protocol : MnipProtocolParameters, optional
        Parameters for the MNIP protocol.
    zk_protocol : ZkProtocolParameters, optional
        Parameters for zero-knowledge protocols.
    message_regression : MessageRegressionParameters, optional
        Parameters for doing regression analysis on the messages.
    base_run : BaseRunParameters, optional
        Parameters for basing the current experiment on a previous W&B run.
    """

    scenario: ScenarioType
    trainer: TrainerType
    dataset: str

    interaction_protocol: InteractionProtocolType = "nip"

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
    text_rl: Optional[TextRlParameters | dict] = None
    pure_text_ei: Optional[PureTextEiParameters | dict] = None
    pure_text_malt: Optional[PureTextMaltParameters | dict] = None

    image_classification: Optional[ImageClassificationParameters | dict] = None
    code_validation: Optional[CodeValidationParameters | dict] = None

    dataset_options: Optional[DatasetParameters | dict] = None

    protocol_common: Optional[CommonProtocolParameters | dict] = None
    nip_protocol: Optional[NipProtocolParameters | dict] = None
    debate_protocol: Optional[DebateProtocolParameters | dict] = None
    mnip_protocol: Optional[MnipProtocolParameters | dict] = None
    zk_protocol: Optional[ZkProtocolParameters | dict] = None

    message_regression: Optional[MessageRegressionParameters | dict] = None

    base_run: Optional[BaseRunParameters | dict] = None

    def __post_init__(self):

        # Determine whether the protocol is zero-knowledge
        for protocol_common_field in fields(CommonProtocolParameters):
            if protocol_common_field.name == "zero_knowledge":
                default_zero_knowledge = protocol_common_field.default
                break
        else:
            raise RuntimeError("CommonProtocolParameters has no zero_knowledge field.")
        if isinstance(self.protocol_common, CommonProtocolParameters):
            zero_knowledge = self.protocol_common.zero_knowledge
        elif isinstance(self.protocol_common, dict):
            zero_knowledge = self.protocol_common.get(
                "zero_knowledge", default_zero_knowledge
            )
        else:
            zero_knowledge = default_zero_knowledge

        if self.scenario == "graph_isomorphism":
            self._process_agents_params(
                GraphIsomorphismAgentParameters,
                RandomAgentParameters,
                zero_knowledge,
            )

        elif self.scenario == "image_classification":
            self._process_agents_params(
                ImageClassificationAgentParameters,
                RandomAgentParameters,
                zero_knowledge,
            )

        elif self.scenario == "code_validation":
            self._process_agents_params(
                CodeValidationAgentParameters,
                RandomAgentParameters,
                zero_knowledge,
            )

        super().__post_init__()

    def _process_agents_params(
        self,
        agent_params_class: type[AgentParameters],
        random_agent_params_class: type[RandomAgentParameters],
        zero_knowledge: bool,
    ) -> AgentsParameters:
        """Process agent parameters passed to ``HyperParameters``.

        Fills in missing agent parameters with the default parameters for the scenario.
        Also validates the agent parameters.

        Parameters
        ----------
        agent_params_class : type[AgentParameters]
            The class of the agent parameters for the scenario.
        random_agent_params_class : type[RandomAgentParameters]
            The class of the random agent parameters for the scenario.
        zero_knowledge : bool
            Whether the protocol is zero-knowledge.
        """

        # If no agent parameters are provided, use the default parameters for the
        # protocol and scenario for all agents
        if self.agents is None:
            self.agents = AgentsParameters(_default=agent_params_class())

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

    def to_dict(self, include_package_meta: bool = False) -> dict:
        """Convert the parameters object to a dictionary.

        Turns enums into strings, and sub-parameters into dictionaries. Includes the
        is_random parameter if it exists.

        Parameters
        ----------
        include_package_meta : bool, default=False
            Whether to include metadata about the NIP experiments package. This will set
            the "_package_version" and "_package_name" fields.

        Returns
        -------
        params_dict : dict
            A dictionary of the parameters.
        """

        as_dict = super().to_dict()

        if include_package_meta:
            as_dict["_package_version"] = get_version(as_tuple=False)
            as_dict["_package_name"] = get_package_name()

        return as_dict
