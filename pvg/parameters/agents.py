"""Parameters which specify agent configurations.

An agent is a neural network, and it is specified by a `AgentParameters` object. Each
scenario has a different `AgentParameters` subclass.

The `AgentsParameters` object is a dictionary of agent names and their corresponding
`AgentParameters` objects.
"""

from abc import ABC
from typing import ClassVar, Optional
from dataclasses import dataclass
import dataclasses

from pvg.constants import WANDB_ENTITY, WANDB_PROJECT
from pvg.parameters.base import SubParameters, ParameterValue
from pvg.parameters.types import ActivationType, ImageBuildingBlockType
from pvg.parameters.update_schedule import (
    AgentUpdateSchedule,
    ConstantUpdateSchedule,
    AlternatingPeriodicUpdateSchedule,
)


@dataclass
class LrFactors(SubParameters, ABC):
    """
    Class representing learning rate factors for the actor and critic models.

    Attributes:
        actor (float): The learning rate factor for the actor model.
        critic (float): The learning rate factor for the critic model.
    """

    actor: float = 1.0
    critic: float = 1.0


@dataclass
class AgentParameters(SubParameters, ABC):
    """Base class for sub-parameters objects which define agents

    Parameters
    ----------
    agent_lr_factor : [LrFactors | dict], optional
        The learning rate factor for the whole agent (split across the actor and the
        critic) compared with the base learning rate. This allows updating the agents at
        different rates.
    body_lr_factor : [LrFactors | dict], optional
        The learning rate factor for the body part of the model (split across the actor
        and the critic) compared with with whole agent. This allows updating the body at
        a different rate to the rest of the model.
    update_schedule : AgentUpdateSchedule
        The schedule for updating the agent weights when doing multi-agent training.
        This specifies on which iterations the agent should be updated by the optimizer.
    use_manual_architecture : bool
        Whether to use a manually defined architecture for the agent, which implements a
        hand-specified (non-learned) algorithm designed to maximise reward. This
        algorithm can be different depending on the environment. This is useful to test
        if the other agents can learn to work with a fixed optimum agent. It usually
        makes sense to set `agent_lr_factor` to {"actor": 0, "critic": 0} in this case.
    normalize_message_history : bool
        Whether to normalise the message history before passing it through the GNN
        encoder. Message histories are normalised to have zero mean and unit variance
        assuming that all episode lengths are equally frequent. (While this is probably
        not a realistic assumption, it's the most reasonable one we can make without
        knowing the true distribution of episode lengths. It's unlikely to make a big
        difference to the normalisation, and it's probably better than not normalising
        at all.)
    load_checkpoint_and_parameters : bool
        Whether to load the agent model checkpoint and parameters from W&B. In this
        case, all agent parameters are replaced by the parameters from the checkpoint.
        Otherwise, the model is randomly initialised. If `True`, the `checkpoint_run_id`
        parameter must be set.
    checkpoint_entity : str
        The entity of the W&B run to load the checkpoint from. If not provided, the
        default is used.
    checkpoint_project : str
        The project of the W&B run to load the checkpoint from. If not provided, the
        default is used.
    checkpoint_run_id: str, optional
        The ID of the W&B run to load the checkpoint from. Must be provided if
        `load_checkpoint` is `True`.
    checkpoint_version: str
        The version of the checkpoint to load. If not provided, the latest version is
        used.
    ortho_init: bool or float, optional
        Whether to use orthogonal initialisation for the weights of the various networks
        with gain ``float`` (and biases initiliased to 0). If ``True``, the gain is set
        to 1.0. If not provided or ``False``, orthogonal initialisation is not used.
    """

    agent_lr_factor: Optional[LrFactors | dict] = None
    body_lr_factor: Optional[LrFactors | dict] = None
    update_schedule: AgentUpdateSchedule = ConstantUpdateSchedule()

    use_manual_architecture: bool = False

    normalize_message_history: bool = False

    load_checkpoint_and_parameters: bool = False
    checkpoint_entity: str = WANDB_ENTITY
    checkpoint_project: str = WANDB_PROJECT
    checkpoint_run_id: Optional[str] = None
    checkpoint_version: str = "latest"

    ortho_init: Optional[bool | float] = True

    # The parameters which are preserved when loading from W&B config
    LOAD_PRESERVED_PARAMETERS: ClassVar[list[str]] = [
        "load_checkpoint_and_parameters",
        "checkpoint_entity",
        "checkpoint_project",
        "checkpoint_run_id",
        "checkpoint_version",
    ]

    is_random: ClassVar[bool] = False

    def to_dict(self) -> dict:
        params_dict = super().to_dict()

        # Add the is_random parameter. This is not a field of the parameters object, but
        # we want to include it in the dictionary.
        params_dict["is_random"] = self.is_random

        return params_dict

    def load_from_wandb_config(self, wandb_config: dict):
        """Load the parameters from a W&B config dictionary.

        Parameters
        ----------
        wandb_config : dict
            The W&B config dictionary for this agent (e.g.
            `wandb_run.config["agents"][agent_name]`).
        """
        for field in dataclasses.fields(self):
            if field.name in self.LOAD_PRESERVED_PARAMETERS:
                continue
            if field.name in wandb_config:
                setattr(self, field.name, wandb_config[field.name])
        setattr(self, "is_random", wandb_config["is_random"])


@dataclass
class RandomAgentParameters(AgentParameters):
    """Parameters which specify a random agent"""

    is_random: ClassVar[bool] = True


@dataclass
class GraphIsomorphismAgentParameters(AgentParameters):
    """Additional parameters for agents in the graph isomorphism experiment.

    Parameters
    ----------
    activation_function : ActivationType
        The activation function to use.
    num_gnn_layers : int
        The number of layers in the agents's GNN.
    d_gnn : int
        The dimension of the hidden layers in the agents's GNN and of the attention
        embedding.
    d_gin_mlp : int
        The dimension of the hidden layers in the agents's Graph Isomorphism Network
        MLP.
    gnn_output_digits : int, optional
        The number of digits in the output of the agents's GNN. If not provided, the
        output is not rounded.
    use_dual_gnn : bool
        Whether to run two copies of the GNN in parallel, where on the first we take the
        features as the message history and on the second the features are all zeros.
    num_heads : int
        The number of heads in the agents's transformer.
    num_transformer_layers : int
        The number of transformer layers
    d_transformer : int
        The dimensionality of the transformer
    d_transformer_mlp
        The hidden dimension of the transformer MLP
    transformer_dropout : float
        The dropout value for the transformer
    d_node_selector : int
        The dimension of the hidden layer in the agents's MLP which selects a node to
        send as a message.
    num_node_selector_layers : int
        The number of layers in the agents's node selector MLP.
    d_decider : int
        The dimension of the hidden layer in the agents's MLP which decides whether to
        accept or reject.
    num_decider_layers : int
        The number of layers in the agents's decider MLP.
    include_round_in_decider : bool
        Whether to include the round number in the agents's decider MLP.
    d_linear_message_selector : int
        The dimension of the hidden layer in the agents's MLP which selects a linear
        message, if we're using the linear message space.
    num_linear_message_selector_layers : int
        The number of layers in the agents's linear message selector MLP.
    d_value : int
        The dimension of the hidden layer in the agents's MLP which estimates the value
        function.
    num_value_layers : int
        The number of layers in the agents's value MLP.
    include_round_in_value : bool
        Whether to include the round number in the agents's value MLP.
    use_batch_norm : bool
        Whether to use batch normalization in the agents's global pooling layer.
    noise_sigma : float
        The relative standard deviation of the Gaussian noise added to the agents's
        graph-level representations.
    use_pair_invariant_pooling : bool
        Whether to use pair-invariant pooling in the agents's global pooling layer. This
        makes the agents's graph-level representations invariant to the order of the
        graphs in the pair.
    body_lr_factor : [LrFactors | dict], optional
        The learning rate factor for the body part of the model. The final LR for the
        body is obtained by multiplying this factor by the agent LR factor and the base
        LR. This allows updating the body at a different rate to the rest of the model.
    gnn_lr_factor : [LrFactors | dict], optional
        The learning rate factor for the GNN part of the model (split across the actor and the critic). The final LR for the GNN is obtained by multiplying this factor by the body LR. This allows updating the GNN at a different rate to the rest of the model.
    """

    activation_function: ActivationType = ActivationType.TANH

    num_gnn_layers: int = 5
    d_gnn: int = 16
    d_gin_mlp: int = 64
    gnn_output_digits: Optional[int] = None
    use_dual_gnn: bool = True

    num_heads: int = 4
    num_transformer_layers: int = 4
    d_transformer: int = 16
    d_transformer_mlp: int = 64
    transformer_dropout: float = 0.0

    d_node_selector: int = 16
    num_node_selector_layers: int = 2

    d_decider: int = 16
    num_decider_layers: int = 2
    include_round_in_decider: bool = True

    d_linear_message_selector: int = 16
    num_linear_message_selector_layers: int = 2

    d_value: int = 16
    num_value_layers: int = 2
    include_round_in_value: bool = True

    use_batch_norm: bool = True
    noise_sigma: float = 0.0
    use_pair_invariant_pooling: bool = True

    body_lr_factor: Optional[LrFactors | dict] = None
    gnn_lr_factor: Optional[LrFactors | dict] = None

    @classmethod
    def construct_test_params(cls) -> "GraphIsomorphismAgentParameters":
        return cls(
            num_gnn_layers=1,
            d_gnn=1,
            d_gin_mlp=1,
            num_heads=2,
            num_transformer_layers=1,
            d_transformer=2,
            d_transformer_mlp=1,
            d_node_selector=1,
            num_node_selector_layers=1,
            d_decider=1,
            num_decider_layers=1,
            d_linear_message_selector=1,
            num_linear_message_selector_layers=1,
            d_value=1,
            num_value_layers=1,
        )


@dataclass
class ImageClassificationAgentParameters(AgentParameters):
    """Additional parameters for agents in the image classification experiment.

    An image classification network is composed of several groups of building blocks,
    such as convolutional layers. Each group contains several building blocks.

    Parameters
    ----------
    activation_function : ActivationType
        The activation function to use.
    building_block_type : ImageBuildingBlockType
        The type of building block to use in the agents's CNN (e.g. convolutional
        layer).
    num_blocks_per_group : int
        The number of building blocks in each group in the agents's CNN.
    kernel_size : int
        The kernel size of the building blocks in the agents's CNN.
    stride : int
        The stride of the building blocks in the agents's CNN.
    pretrained_embeddings_model : str or None
        If not None, specifies a pretrained model to load. This is usually either of the
        form "{hf_user}/{model_name}_{dataset}", where `hf_user` is a HuggingFace Hub
        username, or "{model_name}", which resolves to
        "{HF_PRETRAINED_MODELS_USER}/{model_name}_{params.dataset}", where
        `HF_PRETRAINED_MODELS_USER` is defined in the `constants` module. The last-layer
        embeddings will be included in the model architecture.
    pretrained_embedding_channels : int
        The number of channels used to represent the pretrained embeddings. The
        pretrained embeddings are resized to this number of channels by using a 1x1
        convolution.
    d_latent_pixel_selector : int
        The dimension of the hidden layer in the agents's MLP which selects a latent
        pixel to send as a message.
    num_latent_pixel_selector_layers : int
        The number of layers in the agents's latent pixel selector MLP.
    d_decider : int
        The dimension of the hidden layer in the agents's MLP which decides whether
        continue exchanging messages or to guess.
    num_decider_layers : int
        The number of layers in the agents's decider MLP.
    include_round_in_decider : bool
        Whether to include the round number in the agents's decider MLP.
    d_linear_message_selector : int
        The dimension of the hidden layer in the agents's MLP which selects a linear
        message, if we're using the linear message space.
    num_linear_message_selector_layers : int
        The number of layers in the agents's linear message selector MLP.
    d_value : int
        The dimension of the hidden layer in the agents's MLP which estimates the value
        function.
    num_value_layers : int
        The number of layers in the agents's value MLP.
    include_round_in_value : bool
        Whether to include the round number in the agents's value MLP.
    """

    activation_function: ActivationType = ActivationType.TANH

    building_block_type: ImageBuildingBlockType = ImageBuildingBlockType.CONV2D
    num_blocks_per_group: int = 2
    kernel_size: int = 3
    stride: int = 1

    pretrained_embeddings_model: Optional[str] = None
    pretrained_embedding_channels: int = 64

    d_latent_pixel_selector: int = 16
    num_latent_pixel_selector_layers: int = 2

    d_decider: int = 16
    num_decider_layers: int = 2
    include_round_in_decider: bool = True

    d_linear_message_selector: int = 16
    num_linear_message_selector_layers: int = 2

    d_value: int = 16
    num_value_layers: int = 2
    include_round_in_value: bool = True

    @classmethod
    def construct_test_params(cls) -> "ImageClassificationAgentParameters":
        return cls(
            building_block_type=ImageBuildingBlockType.CONV2D,
            num_blocks_per_group=1,
            d_latent_pixel_selector=1,
            num_latent_pixel_selector_layers=1,
            d_decider=1,
            num_decider_layers=1,
            d_linear_message_selector=1,
            num_linear_message_selector_layers=1,
            d_value=1,
            num_value_layers=1,
        )


class AgentsParameters(dict[str, AgentParameters], ParameterValue):
    """Parameters which specify the agents in the experiment.

    A subclass of `dict` which contains the parameters for each agent in the experiment.

    The keys are the names of the agents, and the values are the parameters for each
    agent.

    Agent names must not be substrings of each other.
    """

    def to_dict(self) -> dict:
        """Convert the parameters object to a dictionary.

        Turns sub-parameters into dictionaries and adds the combined agents update
        schedule representation.

        Returns
        -------
        params_dict : dict
            A dictionary of the parameters.
        """

        params_dict = {}

        for param_name, param in self.items():
            params_dict[param_name] = param.to_dict()

        params_dict["agents_update_repr"] = self._agents_update_repr()

        return params_dict

    def _agents_update_repr(self) -> str:
        """Return a string representation of the combined agents update schedule.

        Returns
        -------
        agents_update_repr : str
            A string representation of the combined agents update schedule.
        """

        # If all agents have the constant update schedule, return "Standard"
        for agent_params in self.values():
            if not isinstance(agent_params.update_schedule, ConstantUpdateSchedule):
                break
        else:
            return "Standard"

        # If all agents have an alternating update schedule with the same properties,
        # return a string representation of the alternating update schedule
        period = None
        first_agent_num_rounds = None
        first_agent_name = None
        second_agent_name = None
        for agent_name, agent_params in self.items():
            update_schedule = agent_params.update_schedule
            if not isinstance(update_schedule, AlternatingPeriodicUpdateSchedule):
                break
            if period is not None and period != update_schedule.period:
                break
            period = update_schedule.period
            if (
                first_agent_num_rounds is not None
                and first_agent_num_rounds != update_schedule.first_agent_num_rounds
            ):
                break
            first_agent_num_rounds = update_schedule.first_agent_num_rounds
            if update_schedule.first_agent:
                if first_agent_name is not None:
                    break
                first_agent_name = agent_name
            else:
                if second_agent_name is not None:
                    break
                second_agent_name = agent_name
        else:
            return (
                f"Alternating({period}, {first_agent_num_rounds}, "
                f"{first_agent_name!r}, {second_agent_name!r})"
            )

        # Otherwise, return "Custom" with the update schedules of all agents
        agents_update_repr = "Custom("
        for agent_name, agent_params in self.items():
            agents_update_repr += f"{agent_name!r}: {agent_params.update_schedule}, "
        agents_update_repr = agents_update_repr[:-2] + ")"
        return agents_update_repr
