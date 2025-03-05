"""Enum types for experiment parameters.

Used when a parameter can only take on a specific set of values.
"""

from typing import Literal, TypeAlias


ScenarioType: TypeAlias = Literal[
    "graph_isomorphism", "image_classification", "code_validation"
]
"""Type for the scenario to run."""

SpgVariantType: TypeAlias = Literal["spg", "pspg", "lola", "pola", "sos", "psos"]
"""Type for SPG variants.

Possible Values
---------------
spg
    Stackelberg Policy Gradient  :cite:p:`Fiez2020`.
pspg
    SPG with clipped PPO loss.
lola
    Learning with Opponent-Learning Awareness :cite:p:`Foerster2018`.
pola
    LOLA with clipped PPO loss.
sos
    Stable Opponent Shaping :cite:p:`Letcher2019`.
psos
    SOS with clipped PPO loss.
"""


IhvpVariantType: TypeAlias = Literal["conj_grad", "neumann", "nystrom"]
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

GuessType: TypeAlias = Literal["zero", "one", "y"]
"""Type for the possible guesses of the verifier in binary cases."""


TrainerType: TypeAlias = Literal[
    "vanilla_ppo", "solo_agent", "spg", "reinforce", "pure_text_ei", "pure_text_malt"
]
"""Type for the RL trainer to use.

Possible Values
---------------
vanilla_ppo
    The Proximal Policy Optimization trainer, with each agent training independently.
solo_agent
    A trainer that trains a single agent to solve the task using supervised learning.
spg
    Stackelberg Policy Gradient :cite:p:`Fiez2020` and its variants.
reinforce
    The REINFORCE algorithm.
pure_text_ei
    Expert Iteration :cite:p:`Anthony2017` for text-based tasks, where agents are run
    through text-based APIs (i.e. we don't run them locally, so everything can be
    represented as text).
pure_text_malt
     Multi-Agent LLM Training (MALT) :cite:p:`Motwani2024` for text-based tasks, where agents are run through text-based APIs (i.e. we don't run them locally, so everything can be represented as text).
"""

TestSchemeType = Literal["none", "all", "last", "first_and_last"]
"""Enum specifying on which iterations to test the model during training."""

PpoLossType: TypeAlias = Literal["clip", "kl_penalty"]
"""Type for the PPO loss function to use."""


BinarificationMethodType: TypeAlias = Literal["select_two", "merge", "random"]
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

ActivationType: TypeAlias = Literal["relu", "tanh", "sigmoid"]
"""Type for the activation function to use.

To add a new activation function, add it to this type, and add the corresponding
function to ``ACTIVATION_CLASSES`` in ``nip.utils.torch_modules``.

Possible Values
---------------
relu
    Rectified Linear Unit.
tanh
    Hyperbolic Tangent.
sigmoid
    Sigmoid.
"""

InteractionProtocolType: TypeAlias = Literal[
    "nip",
    "adp",
    "debate",
    "merlin_arthur",
    "mnip",
    "solo_verifier",
    "multi_channel_test",
]
"""Type for the interaction protocol to use in the environment.

Possible Values
---------------
nip
    The Neural Interactive Proof (NIP) protocol that we introduce.
adp
    The Abstract Decision Problem (ADP) protocol.
debate
    The Debate protocol.
merlin_arthur
    The Merlin-Arthur Classifier (MAC) protocol.
mnip
    The Neural Interactive Proof protocol with multiple (in this case, two) provers (MNIP).
solo_verifier
    A protocol consisting of a single verifier, which makes a decision without
    interacting with a prover.
multi_channel_test
    A protocol for testing multi-channel communication.
"""

ImageBuildingBlockType: TypeAlias = Literal["conv2d", "residual_basic"]
"""Type for the type of building block to use in the image classification network.

Possible Values
---------------
conv2d
    A standard 2D convolutional layer.
residual_basic
    A basic residual block used in the ResNet architecture.
"""

MessageRegressionMethodType: TypeAlias = Literal["mlp", "cnn"]
"""Type for the method to use to do regression analysis on the messages.

Possible Values
---------------
mlp
    Use a multi-layer perceptron to do the regression.
cnn
    Use a convolutional neural network to do the regression.
"""

BaseRunType: TypeAlias = Literal["none", "parameters", "rerun_tests"]
"""Enum for how to base the current experiment on a previous W&B run.

Values
------
none
    Do not base the current experiment on a previous run.
parameters
    Use the hyper-parameters of a previous run to initialize the current experiment.
rerun_tests
    Rerun the tests of a previous run. The hyper-parameters controlling the tests
    can be different.
"""
