"""Enum types for experiment parameters.

Used when a parameter can only take on a specific set of values.
"""

from typing import Literal


ScenarioType = Literal["graph_isomorphism", "image_classification", "code_validation"]
"""Type for the scenario to run."""

SpgVariantType = Literal["spg", "pspg", "lola", "pola", "sos", "psos"]
"""Type for SPG variants."""


IhvpVariantType = Literal["conj_grad", "neumann", "nystrom"]
"""Type for the variants of the inverse Hessian-vector product computation.

Possible Values
---------------
conj_grad
    Use the conjugate gradient method.
neumann
    Use the Neumann series method.
nystrom
    Use the Nyström method.
"""

GuessType = Literal["zero", "one", "y"]
"""Type for the possible guesses of the verifier in binary cases."""


TrainerType = Literal["vanilla_ppo", "solo_agent", "spg", "reinforce", "pure_text_ei"]
"""Type for the RL trainer to use."""

PpoLossType = Literal["clip", "kl_penalty"]
"""Type for the PPO loss function to use."""


BinarificationMethodType = Literal["select_two", "merge", "random"]
"""Type for ways of turning a multi-class classification task into a binary one.

Possible Values
---------------
select_two
    Select two classes from the original dataset to use for the binary classification
    task.
merge
    Merge all classes from the original dataset into two classes.
random
    Select classes completely at random.
"""

ActivationType = Literal["relu", "tanh", "sigmoid"]
"""Type for the activation function to use.

To add a new activation function, add it to this type, and add the corresponding
function to `ACTIVATION_CLASSES` in `pvg.utils.torch_modules`.

Possible Values
---------------
relu
    Rectified Linear Unit.
tanh
    Hyperbolic Tangent.
sigmoid
    Sigmoid.
"""

InteractionProtocolType = Literal[
    "pvg",
    "abstract_decision_problem",
    "debate",
    "merlin_arthur",
    "mnip",
    "solo_verifier",
    "market_making",
    "multi_channel_test",
]
"""Type for the interaction protocol to use in the environment.

Possible Values
---------------
pvg
    The full Prover-Verifier Game protocol.
abstract_decision_problem
    The Abstract Decision Problem protocol.
debate
    The Debate protocol.
merlin_arthur
    The Merlin-Arthur classifier protocol.
mnip
    The Prover-Verifier Game protocol with two provers.
solo_verifier
    A protocol consisting of a single verifier, which makes a decision without
    interacting with a prover.
market_making
    A protocol for market making. (TODO: not implemented)
multi_channel_test
    A protocol for testing multi-channel communication.
"""

MinMessageRoundsSchedulerType = Literal[
    "constant", "linear_decrease", "linear_increase", "linear_increase_decrease"
]
"""Type for the scheduler to use for the minimum number of message rounds.

Possible Values
---------------
constant
    Use a constant `min_message_rounds` minimum number of message rounds.
linear_decrease
    Linearly increase the minimum number of message rounds over time, starting with
    `min_message_rounds` and ending with 1.
linear_increase
    Linearly decrease the minimum number of message rounds over time, starting with
    1 and ending with `min_message_rounds`.
linear_increase_decrease
    Linearly increase the minimum number of message rounds over time, starting with
    `min_message_rounds` and ending with 1, then linearly decrease the minimum
    number of message rounds over time, starting with 1 and ending with
    `min_message_rounds`.
"""

ImageBuildingBlockType = Literal["conv2d", "residual_basic"]
"""Type for the type of building block to use in the image classification network.

Possible Values
---------------
conv2d
    A standard 2D convolutional layer.
residual_basic
    A basic residual block used in the ResNet architecture.
"""

MessageRegressionMethodType = Literal["mlp", "cnn"]
"""Type for the method to use to do regression analysis on the messages.

Possible Values
---------------
mlp
    Use a multi-layer perceptron to do the regression.
cnn
    Use a convolutional neural network to do the regression.
"""
