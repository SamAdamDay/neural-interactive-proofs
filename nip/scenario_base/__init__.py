"""Base classes for all NIP scenario components.

A scenario consists of a dataset and a definition of the agents.

Contains base classes for:

- Handling data
- The RL environment
- Building agents
- Holding the components of a scenario

Scenarios should subclass the ``ScenarioInstance`` class, and its ``build`` factory class
method is used to build the agents using the given parameters.
"""

from .data import (
    Dataset,
    TensorDictDataset,
    NestedArrayDictDataset,
    TensorDictDataLoader,
    NestedArrayDictDataLoader,
)
from .environment import Environment, TensorDictEnvironment, PureTextEnvironment
from .agents import (
    AgentHooks,
    AgentPart,
    TensorDictAgentPartMixin,
    TensorDictDummyAgentPartMixin,
    WholeAgent,
    PureTextWholeAgent,
    PureTextSharedModelGroup,
    PureTextSharedModelGroupState,
    RandomWholeAgent,
    AgentBody,
    AgentHead,
    DummyAgentBody,
    AgentPolicyHead,
    RandomAgentPolicyHead,
    AgentValueHead,
    ConstantAgentValueHead,
    SoloAgentHead,
    CombinedAgentPart,
    CombinedTensorDictAgentPart,
    CombinedWhole,
    PureTextCombinedWhole,
    CombinedBody,
    CombinedPolicyHead,
    CombinedValueHead,
    Agent,
    AgentState,
)
from .rollout_samples import (
    RolloutSamples,
    register_rollout_samples_class,
    build_rollout_samples,
)
from .pretrained_models import (
    PretrainedModel,
    register_pretrained_model_class,
    get_pretrained_model_class,
)
from .rollout_analysis import (
    RolloutAnalyser,
    ROLLOUT_ANALYSERS,
    register_rollout_analyser,
    PureTextRolloutAnalyser,
)
