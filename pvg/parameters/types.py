"""Enum types for experiment parameters.

Used when a parameter can only take on a specific set of values.
"""

from enum import auto as enum_auto

try:
    from enum import StrEnum
except ImportError:
    from pvg.utils.future import StrEnum


class ScenarioType(StrEnum):
    """Enum for the scenario to run."""

    GRAPH_ISOMORPHISM = enum_auto()
    IMAGE_CLASSIFICATION = enum_auto()
    TEXT_CLASSIFICATION = enum_auto()


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


class Guess(StrEnum):
    """Enum for the possible guesses of the verifier in binary cases."""

    ZERO = enum_auto()
    ONE = enum_auto()
    Y = enum_auto()


class TrainerType(StrEnum):
    """Enum for the RL trainer to use."""

    VANILLA_PPO = enum_auto()
    SOLO_AGENT = enum_auto()
    SPG = enum_auto()
    REINFORCE = enum_auto()
    PURE_TEXT_EI = enum_auto()


class PpoLossType(StrEnum):
    """Enum for the PPO loss function to use."""

    CLIP = enum_auto()
    KL_PENALTY = enum_auto()


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
    MULTI_CHANNEL_TEST
        A protocol for testing multi-channel communication.
    """

    PVG = enum_auto()
    ABSTRACT_DECISION_PROBLEM = enum_auto()
    DEBATE = enum_auto()
    MERLIN_ARTHUR = enum_auto()
    MARKET_MAKING = enum_auto()  # TODO
    MULTI_CHANNEL_TEST = enum_auto()


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


class ImageBuildingBlockType(StrEnum):
    """Enum for the type of building block to use in the image classification network.

    Enums
    -----
    CONV2D
        A standard 2D convolutional layer.
    RESIDUAL_BASIC
        A basic residual block used in the ResNet architecture.
    """

    CONV2D = enum_auto()
    RESIDUAL_BASIC = enum_auto()


class MessageRegressionMethodType(StrEnum):
    """Enum for the method to use to do regression analysis on the messages.

    Enums
    -----
    MLP
        Use a multi-layer perceptron to do the regression.
    """

    MLP = enum_auto()
