"""All components for the image classification task."""

from .data import (
    DATASET_WRAPPER_CLASSES,
    ImageClassificationDataset,
)
from .agents import (
    ImageClassificationAgentPart,
    ImageClassificationAgentBody,
    ImageClassificationAgentPolicyHead,
    ImageClassificationAgentValueHead,
    ImageClassificationSoloAgentHead,
)

from .environment import ImageClassificationEnvironment
