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
from typing import Optional
from pathlib import Path
import os

import numpy as np

import torch

import wandb

from pvg.parameters import Parameters, ScenarioType, TrainerType
from pvg.experiment_settings import ExperimentSettings
from pvg.scenario_base.data import Dataset
from pvg.scenario_base.agents import (
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
from pvg.protocols import ProtocolHandler, build_protocol_handler
from pvg.constants import CHECKPOINT_ARTIFACT_PREFIX
from pvg.utils.params import check_if_critic_and_single_body


SCENARIO_CLASS_REGISTRY: dict[ScenarioType, dict[type, type]] = {}


def register_scenario_class(scenario: ScenarioType, base_class: type):
    """Register a component with a scenario.

    Parameters
    ----------
    scenario : ScenarioType
        The scenario with which to register the component.
    base_class : type
        The base class of the component being registered.
    """

    def decorator(cls: type):
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


def build_scenario_instance(params: Parameters, settings: ExperimentSettings):
    """Factory function for building a scenario instance from parameters.

    Parameters
    ----------
    params : Parameters
        The params of the experiment.
    device : TorchDevice
        The device to use for training.
    """

    scenario_classes = SCENARIO_CLASS_REGISTRY[params.scenario]
    device = settings.device

    # Set the random seed
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)

    # Silence W&B if requested
    if settings.silence_wandb:
        os.environ["WANDB_SILENT"] = "true"

    # Check if we need a critic and if it shares a body with the actor
    use_critic, use_single_body = check_if_critic_and_single_body(params)

    # Create the protocol handler
    protocol_handler = build_protocol_handler(params)

    # Create the datasets
    train_dataset = scenario_classes[Dataset](
        params, settings, protocol_handler, train=True
    )
    test_dataset = scenario_classes[Dataset](
        params, settings, protocol_handler, train=False
    )

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
                agent_dict[name] = scenario_classes[DummyAgentBody](
                    params=params,
                    protocol_handler=protocol_handler,
                    device=device,
                    agent_name=agent_name,
                )
            else:
                agent_dict[name] = scenario_classes[AgentBody](
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
                agent_dict["policy_head"] = scenario_classes[RandomAgentPolicyHead](
                    params=params,
                    protocol_handler=protocol_handler,
                    device=device,
                    agent_name=agent_name,
                )
                if use_critic:
                    agent_dict["value_head"] = scenario_classes[ConstantAgentValueHead](
                        params=params,
                        protocol_handler=protocol_handler,
                        device=device,
                        agent_name=agent_name,
                    )
            else:
                agent_dict["policy_head"] = scenario_classes[AgentPolicyHead](
                    params=params,
                    protocol_handler=protocol_handler,
                    device=device,
                    agent_name=agent_name,
                )
                if use_critic:
                    agent_dict["value_head"] = scenario_classes[AgentValueHead](
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
            agent_dict["solo_head"] = scenario_classes[SoloAgentHead](
                params=params,
                protocol_handler=protocol_handler,
                device=device,
                agent_name=agent_name,
            )

        agents[agent_name] = scenario_classes[Agent](
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

    # Build additional components if the trainer is an RL trainer
    train_environment = None
    test_environment = None
    combined_body = None
    combined_policy_body = None
    combined_value_body = None
    combined_policy_head = None
    combined_value_head = None
    if (
        params.trainer == TrainerType.VANILLA_PPO
        or params.trainer == TrainerType.SPG
        or params.trainer == TrainerType.REINFORCE
    ):
        # Create the environments
        train_environment = scenario_classes[Environment](
            params=params,
            settings=settings,
            dataset=train_dataset,
            protocol_handler=protocol_handler,
            train=True,
        )
        test_environment = scenario_classes[Environment](
            params=params,
            settings=settings,
            dataset=train_dataset,
            protocol_handler=protocol_handler,
            train=False,
        )

        # Create the combined agents
        if use_single_body:
            combined_body = scenario_classes[CombinedBody](
                params,
                protocol_handler,
                {name: agents[name].body for name in params.agents},
            )
        else:
            combined_policy_body = scenario_classes[CombinedBody](
                params,
                protocol_handler,
                {name: agents[name].policy_body for name in params.agents},
            )
            if use_critic:
                combined_value_body = scenario_classes[CombinedBody](
                    params,
                    protocol_handler,
                    {name: agents[name].value_body for name in params.agents},
                )
        combined_policy_head = scenario_classes[CombinedPolicyHead](
            params,
            protocol_handler,
            {name: agents[name].policy_head for name in params.agents},
        )
        if use_critic:
            combined_value_head = scenario_classes[CombinedValueHead](
                params,
                protocol_handler,
                {name: agents[name].value_head for name in params.agents},
            )

    return ScenarioInstance(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        agents=agents,
        protocol_handler=protocol_handler,
        train_environment=train_environment,
        test_environment=test_environment,
        combined_body=combined_body,
        combined_policy_body=combined_policy_body,
        combined_value_body=combined_value_body,
        combined_policy_head=combined_policy_head,
        combined_value_head=combined_value_head,
    )
