"""The NIP experiments package."""

import importlib.metadata

__version__ = importlib.metadata.version(__name__)

from .parameters import (
    HyperParameters,
    LrFactors,
    AgentsParameters,
    RandomAgentParameters,
    GraphIsomorphismAgentParameters,
    ImageClassificationAgentParameters,
    CodeValidationAgentParameters,
    SoloAgentParameters,
    RlTrainerParameters,
    CommonPpoParameters,
    VanillaPpoParameters,
    SpgParameters,
    ReinforceParameters,
    TextRlParameters,
    PureTextEiParameters,
    ImageClassificationParameters,
    CodeValidationParameters,
    DatasetParameters,
    CommonProtocolParameters,
    NipProtocolParameters,
    DebateProtocolParameters,
    MessageRegressionParameters,
    BaseRunParameters,
    ScenarioType,
    TrainerType,
    PpoLossType,
    ActivationType,
    BinarificationMethodType,
    ImageBuildingBlockType,
    SpgVariantType,
    IhvpVariantType,
    GuessType,
    InteractionProtocolType,
    MessageRegressionMethodType,
    AgentUpdateSchedule,
    ConstantUpdateSchedule,
    ContiguousPeriodicUpdateSchedule,
    AlternatingPeriodicUpdateSchedule,
    TestSchemeType,
)
from .experiment_settings import ExperimentSettings
from .run import run_experiment, prepare_experiment, PreparedExperimentInfo
