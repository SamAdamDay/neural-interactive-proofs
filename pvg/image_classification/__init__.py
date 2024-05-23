"""All components for the image classification task."""

from .data import (
    IMAGE_DATASETS,
    TorchVisionDatasetProperties,
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
