"""A dataclass for building and holding the components of a scenario and a registry

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

import numpy as np

import torch

import wandb

from pvg.parameters import Parameters, ScenarioType, TrainerType
from pvg.experiment_settings import ExperimentSettings
from pvg.scenario_base.data import Dataset, CachedPretrainedEmbeddingsNotFound
from pvg.scenario_base.agents import (
    AgentPart,
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
    Agent,
)
from pvg.scenario_base.environment import Environment
from pvg.scenario_base.pretrained_models import get_pretrained_model_class
from pvg.protocols import ProtocolHandler, build_protocol_handler
from pvg.constants import CHECKPOINT_ARTIFACT_PREFIX
from pvg.utils.params import check_if_critic_and_single_body


SCENARIO_CLASS_REGISTRY: dict[ScenarioType, dict[type, type]] = {}

T = TypeVar("T")


def register_scenario_class(scenario: ScenarioType, base_class: type):
    """Register a component with a scenario.

    Parameters
    ----------
    scenario : ScenarioType
        The scenario with which to register the component.
    base_class : type
        The base class of the component being registered.
    """

    def decorator(cls: type[T]) -> type[T]:
        if scenario not in SCENARIO_CLASS_REGISTRY:
            SCENARIO_CLASS_REGISTRY[scenario] = {}
        SCENARIO_CLASS_REGISTRY[scenario][base_class] = cls
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
    agents : dict[str, Agent]
        The agents for the experiment. Each 'agent' is a dictionary containing all of
        the agent parts.
    train_environment : Optional[Environment]
        The train environment for the experiment, if the experiment is RL.
    test_environment : Optional[Environment]
        The environment for testing the agents, which uses the test dataset.
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
    train_environment: Optional[Environment] = None
    test_environment: Optional[Environment] = None
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
    device : TorchDevice
        The device to use for training.

    Returns
    -------
    scenario_instance : ScenarioInstance
        The constructed scenario instance, which holds the components of the scenario.
    """

    def get_scenario_class(base_class: type) -> type:
        """Get the class for a component based on the scenario and base class.

        Parameters
        ----------
        base_class : type
            The base class of the component to get.

        Returns
        -------
        scenario_class : type
            The class for the component.
        """
        if base_class not in SCENARIO_CLASS_REGISTRY[params.scenario]:
            raise NotImplementedError(
                f"Scenario {params.scenario} does not have a class for {base_class}."
            )
        return SCENARIO_CLASS_REGISTRY[params.scenario][base_class]

    # Set the random seed
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)

    # Silence W&B if requested
    if settings.silence_wandb:
        os.environ["WANDB_SILENT"] = "true"

    # Create the protocol handler
    protocol_handler = build_protocol_handler(params)

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
    _add_pretrained_embeddings_to_datasets(
        params, settings, agents, train_dataset, test_dataset
    )

    # Build additional components if the trainer is an RL trainer
    if (
        params.trainer == TrainerType.VANILLA_PPO
        or params.trainer == TrainerType.SPG
        or params.trainer == TrainerType.REINFORCE
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
        **additional_rl_components,
    )


def _build_agents(
    params: Parameters,
    settings: ExperimentSettings,
    protocol_handler: ProtocolHandler,
    get_scenario_class: Callable[[type], type],
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

    # Check if we need a critic and if it shares a body with the actor
    use_critic, use_single_body = check_if_critic_and_single_body(params)

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
        torch.manual_seed(agent_seed)
        np.random.seed(agent_seed)

        # Get the names of the bodies
        if use_single_body:
            body_names = ["body"]
        else:
            body_names = ["policy_body", "value_body"]

        # Random agents have a dummy body
        for name in body_names:
            if agent_params.is_random:
                agent_dict[name] = get_scenario_class(DummyAgentBody)(
                    params=params,
                    protocol_handler=protocol_handler,
                    device=device,
                    agent_name=agent_name,
                )
            else:
                agent_dict[name] = get_scenario_class(AgentBody)(
                    params=params,
                    protocol_handler=protocol_handler,
                    device=device,
                    agent_name=agent_name,
                )

        if (
            params.trainer == TrainerType.VANILLA_PPO
            or params.trainer == TrainerType.SPG
            or params.trainer == TrainerType.REINFORCE
        ):
            if agent_params.is_random:
                agent_dict["policy_head"] = get_scenario_class(RandomAgentPolicyHead)(
                    params=params,
                    protocol_handler=protocol_handler,
                    device=device,
                    agent_name=agent_name,
                )
                if use_critic:
                    agent_dict["value_head"] = get_scenario_class(
                        ConstantAgentValueHead
                    )(
                        params=params,
                        protocol_handler=protocol_handler,
                        device=device,
                        agent_name=agent_name,
                    )
            else:
                agent_dict["policy_head"] = get_scenario_class(AgentPolicyHead)(
                    params=params,
                    protocol_handler=protocol_handler,
                    device=device,
                    agent_name=agent_name,
                )
                if use_critic:
                    agent_dict["value_head"] = get_scenario_class(AgentValueHead)(
                        params=params,
                        protocol_handler=protocol_handler,
                        device=device,
                        agent_name=agent_name,
                    )
        if params.trainer == TrainerType.SOLO_AGENT or (
            params.pretrain_agents and not agent_params.is_random
        ):
            if agent_params.is_random:
                raise ValueError("Cannot use random agents with solo agent trainer.")
            agent_dict["solo_head"] = get_scenario_class(SoloAgentHead)(
                params=params,
                protocol_handler=protocol_handler,
                device=device,
                agent_name=agent_name,
            )

        # Initialize the relevant weights of the agent orthogonally (with zero bias) if
        # specified
        if agent_params.ortho_init:
            for name in agent_dict:
                for module in agent_dict[name].modules():
                    if hasattr(module, "weight"):
                        if module.weight.dim() >= 2:
                            torch.nn.init.orthogonal_(
                                module.weight, gain=float(agent_params.ortho_init)
                            )
                    if hasattr(module, "bias") and module.bias is not None:
                        torch.nn.init.constant_(module.bias, 0.0)

        agents[agent_name] = get_scenario_class(Agent)(
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
    use_critic, use_single_body = check_if_critic_and_single_body(params)

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
    if use_single_body:
        additional_rl_components["combined_body"] = get_scenario_class(CombinedBody)(
            params,
            protocol_handler,
            {name: agents[name].body for name in params.agents},
        )
    else:
        additional_rl_components["combined_policy_body"] = get_scenario_class(
            CombinedBody
        )(
            params,
            protocol_handler,
            {name: agents[name].policy_body for name in params.agents},
        )
        if use_critic:
            additional_rl_components["combined_value_body"] = get_scenario_class(
                CombinedBody
            )(
                params,
                protocol_handler,
                {name: agents[name].value_body for name in params.agents},
            )
    additional_rl_components["combined_policy_head"] = get_scenario_class(
        CombinedPolicyHead
    )(
        params,
        protocol_handler,
        {name: agents[name].policy_head for name in params.agents},
    )
    if use_critic:
        additional_rl_components["combined_value_head"] = get_scenario_class(
            CombinedValueHead
        )(
            params,
            protocol_handler,
            {name: agents[name].value_head for name in params.agents},
        )

    return additional_rl_components
