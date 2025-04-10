"""A factory function for the components of an experiment scenario.

This is where the logic for creating the agents and environments lives.

The ``ScenarioInstance`` class holds the components of a scenario, which serves to
abstract away the details of the particular experiment being run.

Scenarios register their components using the ``register_scenario_class`` registry, and
the factory ``build_scenario_instance`` function builds the correct components based on
the parameters.
"""

from tempfile import TemporaryDirectory
from dataclasses import fields
from typing import Optional, Callable
from pathlib import Path
import os
from typing import TypeVar
from collections import defaultdict

import torch

import wandb

from nip.parameters import (
    HyperParameters,
    ScenarioType,
    PureTextAgentParameters,
)
from nip.experiment_settings import ExperimentSettings
from nip.scenario_base.data import (
    Dataset,
    TensorDictDataset,
    CachedPretrainedEmbeddingsNotFound,
)
from nip.scenario_base.agents import (
    AgentPart,
    WholeAgent,
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
    PureTextSharedModelGroup,
)
from nip.scenario_base.environment import Environment
from nip.scenario_base.pretrained_models import get_pretrained_model_class
from nip.scenario_instance import ScenarioInstance
from nip.protocols import ProtocolHandler, build_protocol_handler
from nip.trainers import get_trainer_class
from nip.trainers.trainer_base import Trainer, TensorDictTrainer
from nip.message_regression import build_message_regressor
from nip.constants import MODEL_CHECKPOINT_ARTIFACT_PREFIX
from nip.utils.hyper_params import get_agent_part_flags
from nip.utils.maths import set_seed

T = TypeVar("T")


class _ParameterSelector:
    """A data structure for storing and retrieving classes based on parameter values."""

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
            ``filter[address] = value`` means that the parameter value at ``address``
            must be ``value``. All must match for the class to be retrieved. An empty
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

    def select(self, hyper_params: HyperParameters) -> type:
        """Get a class from the parameter selector based on the parameters.

        Parameters
        ----------
        hyper_params : HyperParameters
            The parameters of the experiment.

        Returns
        -------
        cls : type
            The class that matches the parameters.
        """

        # Find the first class that matches the parameters
        for filter, cls in self.filter_matchers:
            for address, value in filter.items():
                if hyper_params.get(address) != value:
                    break
            else:
                return cls

        raise NotImplementedError("No class found for the parameters.")


SCENARIO_CLASS_REGISTRY: defaultdict[tuple[ScenarioType, type], _ParameterSelector] = (
    defaultdict(_ParameterSelector)
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
        is a set of addresses and values to match the parameter value. ``filter[address]
        = value`` means that the parameter value at ``address`` must be ``value``. All
        must match for the class to be retrieved. An empty dictionary will always match.
    """

    def decorator(cls: type[T]) -> type[T]:
        SCENARIO_CLASS_REGISTRY[(scenario, base_class)].add(cls, filter)
        return cls

    return decorator


def build_scenario_instance(
    hyper_params: HyperParameters, settings: ExperimentSettings
) -> ScenarioInstance:
    """Build a scenario instance from parameters.

    The ``ScenarioInstance`` class holds the components of a scenario, which serves to
    abstract away the details of the particular experiment being run.

    Parameters
    ----------
    hyper_params : HyperParameters
        The hyper_params of the experiment.
    settings : ExperimentSettings
        The settings of the experiment.

    Returns
    -------
    scenario_instance : ScenarioInstance
        The constructed scenario instance, which holds the components of the scenario.
    """

    def get_scenario_class(
        base_class: type[T], agent_name: str | None = None
    ) -> type[T]:
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

        if (hyper_params.scenario, base_class) not in SCENARIO_CLASS_REGISTRY:
            raise NotImplementedError(
                f"Scenario {hyper_params.scenario} does not have a class for "
                f"{base_class}."
            )

        param_selector = SCENARIO_CLASS_REGISTRY[(hyper_params.scenario, base_class)]

        if agent_name is not None:
            params_to_select = hyper_params.agents[agent_name]
        else:
            params_to_select = hyper_params

        try:
            return param_selector.select(params_to_select)
        except NotImplementedError as e:
            raise NotImplementedError(
                f"No class found for {hyper_params.scenario} and {base_class} matching "
                f"any filter."
            ) from e

    # Set the random seed
    set_seed(hyper_params.seed)

    # Silence W&B if requested
    if settings.silence_wandb:
        os.environ["WANDB_SILENT"] = "true"

    # Create the protocol handler
    protocol_handler = build_protocol_handler(hyper_params, settings)

    # Get the class for the trainer
    trainer_class = get_trainer_class(hyper_params)

    # Create the message regressor
    message_regressor = build_message_regressor(
        hyper_params, settings, protocol_handler
    )

    # Create the datasets
    train_dataset = get_scenario_class(Dataset)(
        hyper_params, settings, protocol_handler, train=True
    )
    test_dataset = get_scenario_class(Dataset)(
        hyper_params, settings, protocol_handler, train=False
    )

    # Build the agents
    agents = _build_agents(
        hyper_params, settings, protocol_handler, get_scenario_class, trainer_class
    )

    # Build the shared model groups if applicable
    if all(
        isinstance(agent_params, PureTextAgentParameters)
        for agent_params in hyper_params.agents.values()
    ):

        agent_whole_groups: defaultdict[str, list[WholeAgent]] = defaultdict(list)
        scenario_classes: dict[str, type[PureTextSharedModelGroup]] = {}

        for agent_name in protocol_handler.agent_names:

            agent_params: PureTextAgentParameters = hyper_params.agents[agent_name]

            # If the ``shared_model_group`` is None, the group name is the agent name
            if agent_params.shared_model_group is None:
                group_name = agent_name
            else:
                group_name = agent_params.shared_model_group

            agent_whole_groups[group_name].append(agents[agent_name].whole)

            # Get the class used to build the shared model group, and check that all
            # agents in the group use the same class
            if group_name not in scenario_classes:
                scenario_classes[group_name] = get_scenario_class(
                    PureTextSharedModelGroup, agent_name=agent_name
                )
            elif scenario_classes[group_name] != get_scenario_class(
                PureTextSharedModelGroup, agent_name=agent_name
            ):
                raise ValueError(
                    f"Shared model group {group_name!r} has different "
                    f"`PureTextSharedModelGroup` classes registered for different "
                    f"agents."
                )

        shared_model_groups: dict[str, PureTextSharedModelGroup] = {}
        for group_name, group_agent_wholes in agent_whole_groups.items():
            shared_model_groups[group_name] = scenario_classes[group_name](
                hyper_params=hyper_params,
                settings=settings,
                protocol_handler=protocol_handler,
                agent_wholes=group_agent_wholes,
                group_name=group_name,
            )
    else:
        shared_model_groups = None

    # Add pretrained embeddings to the datasets
    if isinstance(train_dataset, TensorDictDataset) and isinstance(
        test_dataset, TensorDictDataset
    ):
        _add_pretrained_embeddings_to_datasets(
            hyper_params, settings, agents, train_dataset, test_dataset
        )

    # Build additional components if the trainer is an RL trainer
    if trainer_class.trainer_type == "rl":
        additional_rl_components = _build_components_for_rl_trainer(
            hyper_params=hyper_params,
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
        shared_model_groups=shared_model_groups,
        **additional_rl_components,
    )


def _build_agents(
    hyper_params: HyperParameters,
    settings: ExperimentSettings,
    protocol_handler: ProtocolHandler,
    get_scenario_class: Callable[[type, Optional[str]], type],
    trainer_class: type[Trainer],
) -> dict[str, Agent]:
    """Build the agents for the experiment.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters for the experiment.
    settings : ExperimentSettings
        The settings for the experiment.
    protocol_handler : ProtocolHandler
        The protocol handler for the experiment.
    get_scenario_class : Callable[[type], type]
        A function to get the class for a component based on the scenario and base
        class.
    trainer_class : type[Trainer]
        The class of the trainer for the experiment.

    Returns
    -------
    agents : dict[str, Agent]
        The agents for the experiment.
    """

    device = settings.device

    # Check if we need a critic, if it shares a body with the actor, and if the agents
    # are whole agents
    use_critic, use_single_body, use_whole_agent = get_agent_part_flags(hyper_params)

    # Create the agents
    agents: dict[str, Agent] = {}
    for agent_name in protocol_handler.agent_names:
        agent_dict = {}

        agent_params = hyper_params.agents[agent_name]

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
        agent_seed = (hyper_params.seed + hash(agent_name)) % (2**32)
        set_seed(agent_seed)

        # Get the names of the bodies
        if use_single_body:
            body_names = ["body"]
        else:
            body_names = ["policy_body", "value_body"]

        def build_part(base_class: type[T]) -> T:
            return get_scenario_class(base_class, agent_name=agent_name)(
                hyper_params=hyper_params,
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

        if trainer_class.trainer_type == "rl" and issubclass(
            trainer_class, TensorDictTrainer
        ):
            if agent_params.is_random:
                agent_dict["policy_head"] = build_part(RandomAgentPolicyHead)
                if use_critic:
                    agent_dict["value_head"] = build_part(ConstantAgentValueHead)
            else:
                agent_dict["policy_head"] = build_part(AgentPolicyHead)
                if use_critic:
                    agent_dict["value_head"] = build_part(AgentValueHead)
        if trainer_class.trainer_type == "solo_agent" or (
            hyper_params.pretrain_agents and not agent_params.is_random
        ):
            if agent_params.is_random:
                raise ValueError("Cannot use random agents with solo agent trainer.")
            agent_dict["solo_head"] = build_part(SoloAgentHead)
        if use_whole_agent:
            agent_dict["whole"] = build_part(WholeAgent)

        agents[agent_name] = get_scenario_class(Agent, agent_name=agent_name)(
            hyper_params=hyper_params, agent_name=agent_name, **agent_dict
        )

        # Load the agent checkpoint if requested
        if agent_params.load_checkpoint_and_parameters:
            # Select the artifact to load
            artifact = checkpoint_wandb_run.use_artifact(
                f"{MODEL_CHECKPOINT_ARTIFACT_PREFIX}{agent_params.checkpoint_run_id}:"
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
    hyper_params: HyperParameters,
    settings: ExperimentSettings,
    agents: dict[str, Agent],
    train_dataset: Dataset,
    test_dataset: Dataset,
):
    """Add embeddings to the datasets from pretrained models.

    Agent parts can request embeddings from pretrained models using the
    ``required_pretrained_models`` attribute.

    This function collects the names of the pretrained models required by the agents,
    and for each one, if the embeddings are not already cached, it generates them and
    adds them to the datasets.

    Parameters
    ----------
    hyper_params : HyperParameters
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
        pretrained_model_class = get_pretrained_model_class(
            base_model_name, hyper_params
        )

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
        pretrained_model = pretrained_model_class(hyper_params, settings)
        embeddings = pretrained_model.generate_dataset_embeddings(datasets)

        # Add the embeddings to the datasets
        for dataset_name, dataset in datasets_to_generate.items():
            dataset.add_pretrained_embeddings(
                model_name,
                embeddings[dataset_name],
                overwrite_cache=settings.ignore_cache,
            )


def _build_components_for_rl_trainer(
    hyper_params: HyperParameters,
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
    hyper_params : HyperParameters
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
    use_critic, use_single_body, use_whole_agent = get_agent_part_flags(hyper_params)

    additional_rl_components = {}

    # Create the environments
    additional_rl_components["train_environment"] = get_scenario_class(Environment)(
        hyper_params=hyper_params,
        settings=settings,
        dataset=train_dataset,
        protocol_handler=protocol_handler,
        train=True,
    )
    additional_rl_components["test_environment"] = get_scenario_class(Environment)(
        hyper_params=hyper_params,
        settings=settings,
        dataset=test_dataset,
        protocol_handler=protocol_handler,
        train=False,
    )

    # Create the combined agents
    if use_whole_agent:
        additional_rl_components["combined_whole"] = get_scenario_class(CombinedWhole)(
            hyper_params=hyper_params,
            settings=settings,
            protocol_handler=protocol_handler,
            wholes={name: agents[name].whole for name in protocol_handler.agent_names},
        )
    else:
        if use_single_body:
            additional_rl_components["combined_body"] = get_scenario_class(
                CombinedBody
            )(
                hyper_params=hyper_params,
                settings=settings,
                protocol_handler=protocol_handler,
                bodies={
                    name: agents[name].body for name in protocol_handler.agent_names
                },
            )
        else:
            additional_rl_components["combined_policy_body"] = get_scenario_class(
                CombinedBody
            )(
                hyper_params=hyper_params,
                settings=settings,
                protocol_handler=protocol_handler,
                bodies={
                    name: agents[name].policy_body
                    for name in protocol_handler.agent_names
                },
            )
            if use_critic:
                additional_rl_components["combined_value_body"] = get_scenario_class(
                    CombinedBody
                )(
                    hyper_params=hyper_params,
                    settings=settings,
                    protocol_handler=protocol_handler,
                    bodies={
                        name: agents[name].value_body
                        for name in protocol_handler.agent_names
                    },
                )
        additional_rl_components["combined_policy_head"] = get_scenario_class(
            CombinedPolicyHead
        )(
            hyper_params=hyper_params,
            settings=settings,
            protocol_handler=protocol_handler,
            policy_heads={
                name: agents[name].policy_head for name in protocol_handler.agent_names
            },
        )
        if use_critic:
            additional_rl_components["combined_value_head"] = get_scenario_class(
                CombinedValueHead
            )(
                hyper_params=hyper_params,
                settings=settings,
                protocol_handler=protocol_handler,
                value_heads={
                    name: agents[name].value_head
                    for name in protocol_handler.agent_names
                },
            )

    return additional_rl_components
