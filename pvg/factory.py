"""A factory function for the components of an experiment scenario

This is where the logic for creating the agents and environments lives.

The `ScenarioInstance` class holds the components of a scenario, which serves to
abstract away the details of the particular experiment being run.

Scenarios register their components using the `register_scenario_class` registry, and
the factory `build_scenario_instance` function builds the correct components based on
the parameters.
"""

from tempfile import TemporaryDirectory
from dataclasses import dataclass, fields
from typing import Optional, Callable, Any
from pathlib import Path
import os
from typing import TypeVar
from collections import defaultdict

import numpy as np

import torch
from torch import nn

import wandb

from pvg.parameters import Parameters, ScenarioType, TrainerType
from pvg.experiment_settings import ExperimentSettings
from pvg.scenario_base.data import (
    Dataset,
    TensorDictDataset,
    CachedPretrainedEmbeddingsNotFound,
)
from pvg.scenario_base.agents import (
    AgentPart,
    WholeAgent,
    RandomWholeAgent,
    AgentBody,
    DummyAgentBody,
    AgentPolicyHead,
    RandomAgentPolicyHead,
    AgentValueHead,
    ConstantAgentValueHead,
    SoloAgentHead,
    CombinedBody,
    CombinedPolicyHead,
    CombinedValueHead,
    CombinedWhole,
    Agent,
)
from pvg.scenario_base.environment import Environment
from pvg.scenario_base.pretrained_models import get_pretrained_model_class
from pvg.protocols import ProtocolHandler, build_protocol_handler
from pvg.message_regression import MessageRegressor, build_message_regressor
from pvg.constants import CHECKPOINT_ARTIFACT_PREFIX
from pvg.utils.params import get_agent_part_flags
from pvg.utils.maths import set_seed

T = TypeVar("T")


class ParameterSelector:
    """A data structure for storing and retrieving classes based on parameter values"""

    # Ordered by specificity, with the most specific filters first
    filter_matchers: list[tuple[dict[str, str], type]]

    def __init__(self):
        self.filter_matchers = []

    def add(self, cls: type, filter: dict[str, str] = {}):
        """Add a class to the parameter selector.

        Parameters
        ----------
        cls : type
            The class to store.
        filter : dict[str, str], default={}
            The set of addresses and values to match the parameter value.
            `filter[address] = value` means that the parameter value at `address` must
            be `value`. All must match for the class to be retrieved. An empty
            dictionary will always match.
        """

        # Insert the class into the list at the first position where the filter is a
        # superset of the filter currently at that position, to maintain the order of
        # specificity
        for i, (other_filters, _) in enumerate(self.filter_matchers):
            if other_filters.items() <= filter.items():
                self.filter_matchers.insert(i, (filter, cls))
                break
        else:
            self.filter_matchers.append((filter, cls))

    def select(self, params: Parameters) -> type:
        """Get a class from the parameter selector based on the parameters.

        Parameters
        ----------
        params : Parameters
            The parameters of the experiment.

        Returns
        -------
        cls : type
            The class that matches the parameters.
        """

        # Find the first class that matches the parameters
        for filter, cls in self.filter_matchers:
            for address, value in filter.items():
                if params.get(address) != value:
                    break
            else:
                return cls

        raise NotImplementedError("No class found for the parameters.")


SCENARIO_CLASS_REGISTRY: defaultdict[tuple[ScenarioType, type], ParameterSelector] = (
    defaultdict(ParameterSelector)
)


def register_scenario_class(
    scenario: ScenarioType, base_class: type, filter: dict[str, str] = {}
):
    """Register a component with a scenario.

    Parameters
    ----------
    scenario : ScenarioType
        The scenario with which to register the component.
    base_class : type
        The base class of the component being registered.
    filter : dict[str, str], default={}
        The filter to use to select when to use the class based on the parameters. This
        is a set of addresses and values to match the parameter value. `filter[address]
        = value` means that the parameter value at `address` must be `value`. All must
        match for the class to be retrieved. An empty dictionary will always match.
    """

    def decorator(cls: type[T]) -> type[T]:
        SCENARIO_CLASS_REGISTRY[(scenario, base_class)].add(cls, filter)
        return cls

    return decorator


@dataclass
class ScenarioInstance:
    """A dataclass for holding the components of an experiment.

    The principal aim of this class is to abstract away the details of the particular
    experiment being run.

    Attributes
    ----------
    train_dataset : Dataset
        The train dataset for the experiment.
    test_dataset : Dataset
        The test dataset for the experiment.
    protocol_handler : ProtocolHandler
        The interaction protocol handler for the experiment.
    message_regressor : MessageRegressor
        The message regressor for the experiment, which is used to test if the label can
        be inferred purely from the messages.
    agents : dict[str, Agent]
        The agents for the experiment. Each 'agent' is a dictionary containing all of
        the agent parts.
    train_environment : Optional[Environment]
        The train environment for the experiment, if the experiment is RL.
    test_environment : Optional[Environment]
        The environment for testing the agents, which uses the test dataset.
    combined_whole : Optional[CombinedWholeAgent]
        If the agents are not split into parts, this holds the combination of the whole
        agents.
    combined_body : Optional[CombinedBody]
        The combined body of the agents, if the agents are combined the actor and critic
        share the same body.
    combined_policy_body : Optional[CombinedBody]
        The combined policy body of the agents, if the agents are combined and the actor
        and critic have separate bodies.
    combined_value_body : Optional[CombinedBody]
        The combined value body of the agents, if the agents are combined and the actor
        and critic have separate bodies.
    combined_policy_head : Optional[CombinedPolicyHead]
        The combined policy head of the agents, if the agents are combined.
    combined_value_head : Optional[CombinedValueHead]
        The combined value head of the agents, if the agents are combined.
    """

    train_dataset: Dataset
    test_dataset: Dataset
    agents: dict[str, Agent]
    protocol_handler: ProtocolHandler
    message_regressor: MessageRegressor
    train_environment: Optional[Environment] = None
    test_environment: Optional[Environment] = None
    combined_whole: Optional[CombinedWhole] = None
    combined_body: Optional[CombinedBody] = None
    combined_policy_body: Optional[CombinedBody] = None
    combined_value_body: Optional[CombinedBody] = None
    combined_policy_head: Optional[CombinedPolicyHead] = None
    combined_value_head: Optional[CombinedValueHead] = None


def build_scenario_instance(
    params: Parameters, settings: ExperimentSettings
) -> ScenarioInstance:
    """Factory function for building a scenario instance from parameters.

    The `ScenarioInstance` class holds the components of a scenario, which serves to
    abstract away the details of the particular experiment being run.

    Parameters
    ----------
    params : Parameters
        The params of the experiment.
    settings : ExperimentSettings
        The settings of the experiment.

    Returns
    -------
    scenario_instance : ScenarioInstance
        The constructed scenario instance, which holds the components of the scenario.
    """

    def get_scenario_class(base_class: type, agent_name: str | None = None) -> type:
        """Get the class for a component based on the scenario and base class.

        Parameters
        ----------
        base_class : type
            The base class of the component to get.
        agent_name : str, default=None
            If not None, we get a component for this specific agent.

        Returns
        -------
        scenario_class : type
            The class for the component.
        """

        if (params.scenario, base_class) not in SCENARIO_CLASS_REGISTRY:
            raise NotImplementedError(
                f"Scenario {params.scenario} does not have a class for {base_class}."
            )

        param_selector = SCENARIO_CLASS_REGISTRY[(params.scenario, base_class)]

        if agent_name is not None:
            params_to_select = params.agents[agent_name]
        else:
            params_to_select = params

        try:
            return param_selector.select(params_to_select)
        except NotImplementedError as e:
            raise NotImplementedError(
                f"No class found for {params.scenario} and {base_class} matching any "
                f"filter."
            ) from e

    # Set the random seed
    set_seed(params.seed)

    # Silence W&B if requested
    if settings.silence_wandb:
        os.environ["WANDB_SILENT"] = "true"

    # Create the protocol handler
    protocol_handler = build_protocol_handler(params, settings)

    # Create the message regressor
    message_regressor = build_message_regressor(params, settings, protocol_handler)

    # Create the datasets
    train_dataset = get_scenario_class(Dataset)(
        params, settings, protocol_handler, train=True
    )
    test_dataset = get_scenario_class(Dataset)(
        params, settings, protocol_handler, train=False
    )

    # Build the agents
    agents = _build_agents(params, settings, protocol_handler, get_scenario_class)

    # Add pretrained embeddings to the datasets
    if isinstance(train_dataset, TensorDictDataset) and isinstance(
        test_dataset, TensorDictDataset
    ):
        _add_pretrained_embeddings_to_datasets(
            params, settings, agents, train_dataset, test_dataset
        )

    # Build additional components if the trainer is an RL trainer
    if (
        params.trainer == TrainerType.VANILLA_PPO
        or params.trainer == TrainerType.SPG
        or params.trainer == TrainerType.REINFORCE
        or params.trainer == TrainerType.PURE_TEXT_EI
    ):
        additional_rl_components = _build_components_for_rl_trainer(
            params=params,
            settings=settings,
            protocol_handler=protocol_handler,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            agents=agents,
            get_scenario_class=get_scenario_class,
        )
    else:
        additional_rl_components = {}

    return ScenarioInstance(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        agents=agents,
        protocol_handler=protocol_handler,
        message_regressor=message_regressor,
        **additional_rl_components,
    )


def _build_agents(
    params: Parameters,
    settings: ExperimentSettings,
    protocol_handler: ProtocolHandler,
    get_scenario_class: Callable[[type, Optional[str]], type],
) -> dict[str, Agent]:
    """Build the agents for the experiment.

    Parameters
    ----------
    params : Parameters
        The parameters for the experiment.
    settings : ExperimentSettings
        The settings for the experiment.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    get_scenario_class : Callable[[type], type]
        A function to get the class for a component based on the scenario and base
        class.

    Returns
    -------
    agents : dict[str, Agent]
        The agents for the experiment.
    """

    device = settings.device

    # Check if we need a critic, if it shares a body with the actor, and if the agents
    # are whole agents
    use_critic, use_single_body, use_whole_agent = get_agent_part_flags(params)

    # Create the agents
    agents: dict[str, Agent] = {}
    for agent_name, agent_params in params.agents.items():
        agent_dict = {}

        # If we're loading a checkpoint and parameters, get the run and replace the
        # parameters with the ones from the checkpoint
        if agent_params.load_checkpoint_and_parameters:
            checkpoint_wandb_run = wandb.init(
                id=agent_params.checkpoint_run_id,
                entity=agent_params.checkpoint_entity,
                project=agent_params.checkpoint_project,
                resume="must",
            )
            agent_params.load_from_wandb_config(
                checkpoint_wandb_run.config["agents"][agent_name]
            )

        # Set the random seed based on the agent name
        agent_seed = (params.seed + hash(agent_name)) % (2**32)
        set_seed(agent_seed)

        # Get the names of the bodies
        if use_single_body:
            body_names = ["body"]
        else:
            body_names = ["policy_body", "value_body"]

        def build_part(base_class: type[T]) -> T:
            return get_scenario_class(base_class, agent_name=agent_name)(
                params=params,
                settings=settings,
                protocol_handler=protocol_handler,
                agent_name=agent_name,
            )

        # Random agents have a dummy body
        if not use_whole_agent:
            for name in body_names:
                if agent_params.is_random:
                    agent_dict[name] = build_part(DummyAgentBody)
                else:
                    agent_dict[name] = build_part(AgentBody)

        if (
            params.trainer == TrainerType.VANILLA_PPO
            or params.trainer == TrainerType.SPG
            or params.trainer == TrainerType.REINFORCE
        ):
            if agent_params.is_random:
                agent_dict["policy_head"] = build_part(RandomAgentPolicyHead)
                if use_critic:
                    agent_dict["value_head"] = build_part(ConstantAgentValueHead)
            else:
                agent_dict["policy_head"] = build_part(AgentPolicyHead)
                if use_critic:
                    agent_dict["value_head"] = build_part(AgentValueHead)
        if params.trainer == TrainerType.SOLO_AGENT or (
            params.pretrain_agents and not agent_params.is_random
        ):
            if agent_params.is_random:
                raise ValueError("Cannot use random agents with solo agent trainer.")
            agent_dict["solo_head"] = build_part(SoloAgentHead)
        if use_whole_agent:
            agent_dict["whole"] = build_part(WholeAgent)

        agents[agent_name] = get_scenario_class(Agent, agent_name=agent_name)(
            params=params, agent_name=agent_name, **agent_dict
        )

        # Load the agent checkpoint if requested
        if agent_params.load_checkpoint_and_parameters:
            # Select the artifact to load
            artifact = checkpoint_wandb_run.use_artifact(
                f"{CHECKPOINT_ARTIFACT_PREFIX}{agent_params.checkpoint_run_id}:"
                f"{agent_params.checkpoint_version}"
            )

            # Download the artifact and load the agent checkpoint into the agent
            with TemporaryDirectory() as temp_dir:
                artifact.download(root=temp_dir)
                agent_path = Path(temp_dir).joinpath(agent_name)
                for field in fields(agents[agent_name]):
                    field_value = getattr(agents[agent_name], field.name)
                    if isinstance(field_value, torch.nn.Module):
                        state_dict = torch.load(
                            agent_path.joinpath(f"{field.name}.pkl"),
                            map_location=device,
                        )
                        field_value.load_state_dict(state_dict)

            # Make sure to finish the W&B run
            checkpoint_wandb_run.finish()

    return agents


def _add_pretrained_embeddings_to_datasets(
    params: Parameters,
    settings: ExperimentSettings,
    agents: dict[str, Agent],
    train_dataset: Dataset,
    test_dataset: Dataset,
):
    """Add embeddings to the datasets from pretrained models.

    Agent parts can request embeddings from pretrained models using the
    `required_pretrained_models` attribute.

    This function collects the names of the pretrained models required by the agents,
    and for each one, if the embeddings are not already cached, it generates them and
    adds them to the datasets.

    Parameters
    ----------
    params : Parameters
        The parameters for the experiment.
    settings : ExperimentSettings
        The settings for the experiment.
    agents : dict[str, Agent]
        The agents for the experiment.
    train_dataset : Dataset
        The train dataset for the experiment.
    test_dataset : Dataset
        The test dataset for the experiment.
    """

    datasets = dict(train=train_dataset, test=test_dataset)

    # Get the names of the pretrained models required by the agents
    required_pretrained_models = set()
    for agent in agents.values():
        for field in fields(agent):
            field_value = getattr(agent, field.name)
            if isinstance(field_value, AgentPart):
                required_pretrained_models |= set(
                    field_value.required_pretrained_models
                )

    for base_model_name in required_pretrained_models:

        # Load the pretrained model class
        pretrained_model_class = get_pretrained_model_class(base_model_name, params)

        # Get the full model name, which may differ from the model name specified by the
        # parameters, if the latter is a shorthand
        model_name = pretrained_model_class.name

        # Determine which datasets need embeddings generated from them, by checking if
        # the embeddings are already cached
        if settings.ignore_cache:
            datasets_to_generate = datasets
        else:
            datasets_to_generate = {}
            for dataset_name, dataset in datasets.items():
                try:
                    dataset.load_pretrained_embeddings(model_name)
                except CachedPretrainedEmbeddingsNotFound:
                    datasets_to_generate[dataset_name] = dataset

        if len(datasets_to_generate) == 0:
            continue

        # Generate the embeddings
        pretrained_model = pretrained_model_class(params, settings)
        embeddings = pretrained_model.generate_dataset_embeddings(datasets)

        # Add the embeddings to the datasets
        for dataset_name, dataset in datasets_to_generate.items():
            dataset.add_pretrained_embeddings(
                model_name,
                embeddings[dataset_name],
                overwrite_cache=settings.ignore_cache,
            )


def _build_components_for_rl_trainer(
    params: Parameters,
    settings: ExperimentSettings,
    protocol_handler: ProtocolHandler,
    train_dataset: Dataset,
    test_dataset: Dataset,
    agents: dict[str, Agent],
    get_scenario_class: Callable[[type], type],
) -> dict[str, Environment | CombinedBody | CombinedPolicyHead | CombinedValueHead]:
    """Build the additional components needed for an RL trainer.

    Parameters
    ----------
    params : Parameters
        The parameters for the experiment.
    settings : ExperimentSettings
        The settings for the experiment.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    train_dataset : Dataset
        The train dataset for the experiment.
    test_dataset : Dataset
        The test dataset for the experiment.
    agents : dict[str, Agent]
        The agents for the experiment.
    get_scenario_class : Callable[[type], type]
        A function to get the class for a component based on the scenario and base
        class.

    Returns
    -------
    additional_rl_components : dict[str, Any]
        The additional components needed for an RL trainer.
    """

    # Check if we need a critic and if it shares a body with the actor
    use_critic, use_single_body, use_whole_agent = get_agent_part_flags(params)

    additional_rl_components = {}

    # Create the environments
    additional_rl_components["train_environment"] = get_scenario_class(Environment)(
        params=params,
        settings=settings,
        dataset=train_dataset,
        protocol_handler=protocol_handler,
        train=True,
    )
    additional_rl_components["test_environment"] = get_scenario_class(Environment)(
        params=params,
        settings=settings,
        dataset=test_dataset,
        protocol_handler=protocol_handler,
        train=False,
    )

    # Create the combined agents
    if use_whole_agent:
        additional_rl_components["combined_whole"] = get_scenario_class(CombinedWhole)(
            params=params,
            settings=settings,
            protocol_handler=protocol_handler,
            wholes={name: agents[name].whole for name in params.agents},
        )
    else:
        if use_single_body:
            additional_rl_components["combined_body"] = get_scenario_class(
                CombinedBody
            )(
                params=params,
                settings=settings,
                protocol_handler=protocol_handler,
                bodies={name: agents[name].body for name in params.agents},
            )
        else:
            additional_rl_components["combined_policy_body"] = get_scenario_class(
                CombinedBody
            )(
                params=params,
                settings=settings,
                protocol_handler=protocol_handler,
                bodies={name: agents[name].policy_body for name in params.agents},
            )
            if use_critic:
                additional_rl_components["combined_value_body"] = get_scenario_class(
                    CombinedBody
                )(
                    params=params,
                    settings=settings,
                    protocol_handler=protocol_handler,
                    bodies={name: agents[name].value_body for name in params.agents},
                )
        additional_rl_components["combined_policy_head"] = get_scenario_class(
            CombinedPolicyHead
        )(
            params=params,
            settings=settings,
            protocol_handler=protocol_handler,
            policy_heads={name: agents[name].policy_head for name in params.agents},
        )
        if use_critic:
            additional_rl_components["combined_value_head"] = get_scenario_class(
                CombinedValueHead
            )(
                params=params,
                settings=settings,
                protocol_handler=protocol_handler,
                value_heads={name: agents[name].value_head for name in params.agents},
            )

    return additional_rl_components
