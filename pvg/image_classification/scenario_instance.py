"""A class which holds the components of a image classification experiment."""

from pvg.parameters import ScenarioType
from pvg.scenario_base import DataLoader, ScenarioInstance
from pvg.image_classification.data import ImageClassificationDataset
from pvg.image_classification.agents import (
    ImageClassificationAgentBody,
    ImageClassificationDummyAgentBody,
    ImageClassificationAgentPolicyHead,
    ImageClassificationRandomAgentPolicyHead,
    ImageClassificationAgentValueHead,
    ImageClassificationConstantAgentValueHead,
    ImageClassificationSoloAgentHead,
    ImageClassificationCombinedBody,
    ImageClassificationCombinedPolicyHead,
    ImageClassificationCombinedValueHead,
)

from pvg.image_classification.environment import ImageClassificationEnvironment


class ImageClassificationScenarioInstance(ScenarioInstance):
    """A class which holds the components of a image classification experiment.

    Attributes
    ----------
    dataset : Dataset
        The dataset for the experiment.
    dataloader_class : type[DataLoader]
        The data loader class to use for the experiment.
    agents : dict[str, Agent]
        The agents for the experiment.
    environment : Optional[Environment]
        The environment for the experiment, if the experiment is RL.
    combined_body : Optional[CombinedBody]
        The combined body of the agents, if the agents are combined.
    combined_policy_head : Optional[CombinedPolicyHead]
        The combined policy head of the agents, if the agents are combined.
    combined_value_head : Optional[CombinedValueHead]
        The combined value head of the agents, if the agents are combined.

    Parameters
    ----------
    params : Parameters
        The params of the experiment.
    device : TorchDevice
        The device to use for training.
    """

    scenario = ScenarioType.IMAGE_CLASSIFICATION

    dataset_class = ImageClassificationDataset
    dataloader_class = DataLoader

    environment_class = ImageClassificationEnvironment

    body_class = ImageClassificationAgentBody
    dummy_body_class = ImageClassificationDummyAgentBody
    policy_head_class = ImageClassificationAgentPolicyHead
    random_policy_head_class = ImageClassificationRandomAgentPolicyHead
    value_head_class = ImageClassificationAgentValueHead
    constant_value_head_class = ImageClassificationConstantAgentValueHead
    solo_head_class = ImageClassificationSoloAgentHead
    combined_body_class = ImageClassificationCombinedBody
    combined_policy_head_class = ImageClassificationCombinedPolicyHead
    combined_value_head_class = ImageClassificationCombinedValueHead
