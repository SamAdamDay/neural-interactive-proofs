"""Multi-Agent LLM Training (MALT) for text-based environments that only use APIs.

In the MALT protocol :cite:p:`Motwani2024`, we sample multiple responses per timestep
from the agents. This means that for each datapoint we have a tree of responses. For
each agent ``A``, at each decision point for ``A`` we look at the expected reward for
``A`` for each of the responses. We then select preference pairs of responses from these
and train using Direct Preference Optimization :cite:p:`Rafailov2023`. The way pairs are
selected is determined by the ``hyper_params.pure_text_malt.pair_selection_method``
parameter, which can be one of the following:

- "positive_negative": Selects a response where the agent's expected reward is above a
  certain threshold (by default the reward mid-point) and a response where the agent's
  expected reward is below this threshold.
- "interval": Selects a pair of responses where the difference in expected reward is
  above a certain threshold. This threshold is computed as
  ``interval_threshold_proportion`` times the difference between the maximum and minimum
  possible reward for the agent.

It is also possible do some rounds of Expert Iteration (EI) before doing MALT. The
``PureTextMaltTrainer`` class inherits from the ``PureTextEiTrainer`` class, which
implements the EI protocol, and allows running EI for a number of iterations specified
by the ``hyper_params.pure_text_malt.num_initial_ei_iterations`` parameter.
"""

from typing import (
    Optional,
    ClassVar,
    Any,
    Iterable,
    Callable,
    ParamSpec,
    Concatenate,
    TypeVar,
    Self,
)
from dataclasses import dataclass, InitVar
import dataclasses
import itertools
from functools import wraps
from textwrap import indent
from asyncio import TaskGroup
import logging

import torch
from torch import Tensor

import numpy as np

import einops

from jaxtyping import Int, Bool

from nip.parameters import PureTextAgentParameters
from nip.protocols.protocol_base import ProtocolHandler
from nip.trainers.registry import register_trainer
from nip.trainers.ei_pure_text import PureTextEiTrainer
from nip.scenario_base.environment import PureTextEnvironment
from nip.scenario_base.agents import PureTextSharedModelGroup
from nip.utils.nested_array_dict import NestedArrayDict, concatenate_nested_array_dicts
from nip.utils.rollouts import get_pretty_pure_text_round_message


logger = logging.getLogger(__name__)


P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class _PartialRolloutNode:
    """A node in the tree of responses, which is a partially generated rollout."""

    current_env_state: NestedArrayDict
    """The state of the environment at this node.
    
    This is either the initial state of the environment, for root nodes, or the state
    obtained by calling ``PureTextEnvironment.get_next_state_from_state
    <nip.scenario_base.environment.PureTextEnvironment.get_next_state_from_state>`` on
    the parent node's state. In particular, it doesn't contain any of the agent actions
    or the consequences of those actions.
    """

    protocol_handler: ProtocolHandler
    """The protocol handler for the experiment."""

    ended: bool = False
    """Whether the rollout has ended from this point onwards."""

    padding: bool = False
    """Whether this node is a padding node.
    
    Padding nodes are nodes which are not part of the tree, but are used to fill in
    the tree so that it has uniform depth.

    A node is a padding node if and only if its parent node has ended.
    """

    trajectory_env_states: list[NestedArrayDict] = dataclasses.field(
        default_factory=list
    )
    """A list of the environment states in the trajectory leading up to this node."""

    node_id: int = -1
    """The ID of this node, which is unique in the forest of rollouts.
    
    If not set, the ID is set to the next available ID in the shared data. The
    ``node_id_base`` attribute is used to set the base ID, which ensures that the IDs
    are unique across multiple trees.
    """

    parent_partial_rollout: Optional["_PartialRolloutNode"] = None
    """The parent node of this node, or None if this is the root node."""

    child_partial_rollouts: list["_PartialRolloutNode"] = dataclasses.field(
        default_factory=list
    )
    """The child nodes of this node, the one-step continuations of this node."""

    num_branches: int = 0
    """The number of branches passing through this node.
    
    This is computed after the tree is generated, so is not available immediately.
    """

    total_reward_per_agent: np.ndarray | float = 0.0
    """The total reward for each agent at this node and below.
    
    This is computed after the tree is generated, so is not available immediately.
    """

    node_id_base: InitVar[Optional[int]] = None
    """The base ID, which is used to set the initial ID of a root node."""

    _shared_data: ClassVar[dict[str, Any]] = {"next_node_id": 0}
    """Shared data for all nodes in the forest.
    
    Note that this is a class variable, so it is shared between all instances of this
    class. This is used to ensure that the node IDs are unique across all nodes in the
    forest.
    """

    def __post_init__(self, node_id_base: Optional[int]):
        if node_id_base is not None:
            self._shared_data["next_node_id"] = node_id_base
        if self.node_id == -1:
            self.node_id = self._shared_data["next_node_id"]
            self._shared_data["next_node_id"] += 1

    def clone_as_child(self) -> Self:
        """Clone this node as a child of the current node.

        This creates a new node with the same environment state and adds it to the
        current node's list of child nodes.

        Returns
        -------
        cloned_partial_rollout : _PartialRolloutNode
            The cloned node, which is a child of the current node.
        """
        # We deep copy the current environment state, because that will be
        # modified in place. We shallow copy the trajectory list, because the
        # states will be shared between nodes with the same ancestors, but the
        # list itself will not be shared. NOTE: deep copying the environment
        # state results in a small inefficiency, because we only really need to
        # keep the environment state of the leaf nodes. But the slowdown is
        # probably negligible.
        cloned_partial_rollout = type(self)(
            current_env_state=self.current_env_state.clone(),
            protocol_handler=self.protocol_handler,
            ended=self.ended,
            padding=self.ended,
            trajectory_env_states=self.trajectory_env_states.copy(),
            node_id=self._shared_data["next_node_id"],
            parent_partial_rollout=self,
        )
        self._shared_data["next_node_id"] += 1
        self.child_partial_rollouts.append(cloned_partial_rollout)
        return cloned_partial_rollout

    def has_agent_acted(self, agent_name: str) -> bool:
        """Check if the given agent has acted at this node.

        An action means either sending a message or making a decision (for verifiers).

        Parameters
        ----------
        agent_name : str
            The name of the agent to check.

        Returns
        -------
        has_acted : bool
            True if the agent has acted at this node, False otherwise.
        """

        if len(self.trajectory_env_states) == 0 or self.padding:
            return False

        last_env_state = self.trajectory_env_states[-1]
        agent_id = self.protocol_handler.agent_names.index(agent_name)

        if last_env_state["agents", "message"][0, agent_id, :].any():
            return True
        if (
            agent_name in self.protocol_handler.verifier_names
            and last_env_state["agents", "decision"][0, agent_id] != 2
        ):
            return True

        return False

    def visualise(
        self,
        include_messages: bool = True,
        include_expected_reward: bool = True,
        include_pair_info: bool = True,
        include_padding_nodes: bool = False,
        tab_size: int = 2,
    ) -> str:
        """Get a recursive string representation of the rollout tree.

        Returns
        -------
        tree_string : str
            A representation of the rollout tree starting from this node.
        include_messages : bool, default=True
            Whether to include the messages and decisions sent be each agent at each
            node.
        include_expected_reward : bool, default=True
            Whether to include the expected reward for each agent.
        include_pair_info : bool, default=True
            Whether to indicate whether a node is the positive or negative example in a
            preference pair.
        include_padding_nodes : bool, default=False
            Whether to include padding nodes in the output. Padding nodes are nodes
            which are not part of the tree, but are used to fill in the tree so that it
            has uniform depth.
        tab_size : int, default=2
            The number of spaces to indent each level of the tree.
        """

        tree_string = f"Node ID: {self.node_id}"

        if len(self.trajectory_env_states) > 0:
            last_env_state = self.trajectory_env_states[-1]
        else:
            last_env_state = None

        if self.padding:
            tree_string += " (padding)"
        elif include_messages and last_env_state is not None:
            pretty_message_dict = get_pretty_pure_text_round_message(
                protocol_handler=self.protocol_handler,
                decision=last_env_state["agents", "decision"][0],
                continuous_decision=last_env_state["agents", "continuous_decision"][0],
                raw_decision=last_env_state["agents", "raw_decision"][0],
                message=last_env_state["agents", "message"][0],
            )
            for key, value in pretty_message_dict.items():
                tree_string += f"\n{key}: {value}"

        include_expected_reward = (
            include_expected_reward
            and last_env_state is not None
            and ("agents", "expected_reward") in last_env_state.keys()
        )
        include_pair_info = (
            include_pair_info
            and last_env_state is not None
            and ("agents", "is_pair_positive") in last_env_state.keys()
            and ("agents", "is_pair_negative") in last_env_state.keys()
        )
        if include_expected_reward or include_pair_info:
            for agent_id, agent_name in enumerate(self.protocol_handler.agent_names):
                agent_string = ""
                if include_expected_reward:
                    expected_reward = last_env_state["agents", "expected_reward"][0]
                    agent_string += (
                        f"\nExpected reward: {expected_reward[agent_id]:.2f}"
                    )
                if include_pair_info:
                    if last_env_state["agents", "is_pair_positive"][0][agent_id]:
                        agent_string += "\n[POSITIVE EXAMPLE]"
                    if last_env_state["agents", "is_pair_negative"][0][agent_id]:
                        agent_string += "\n[NEGATIVE EXAMPLE]"
                tree_string += f"\n{agent_name}: {indent(agent_string, ' ' * tab_size)}"

        child_strings = []
        for child in self.child_partial_rollouts:
            if not include_padding_nodes and child.padding:
                continue
            child_string = child.visualise(
                include_messages=include_messages,
                include_padding_nodes=include_padding_nodes,
                tab_size=tab_size,
            )
            child_strings.append(indent(child_string, " " * tab_size))
        if len(child_strings) > 0:
            tree_string += "\nChildren:\n"
            tree_string += "\n".join(child_strings)

        return tree_string


def _tree_iter(
    partial_rollouts_by_level: list[list[_PartialRolloutNode]],
    include_level: bool = False,
    leaves_first: bool = False,
    include_root: bool = False,
):
    """Iterate over the tree of responses, either downwards or upwards.

    Parameters
    ----------
    partial_rollouts_by_level : list[list[_PartialRolloutNode]]
        The tree of responses, stratified by level.
    include_level : bool, default=False
        Whether to include the level in the output. In this case, the output is a tuple
        of the level and the partial rollout.
    leaves_first : bool, default=False
        Whether to iterate from the leaves upwards.
    include_root : bool, default=False
        Whether to include the root node in the output.

    Yields
    ------
    level : int, optional
        The level in the tree of responses.
    partial_rollout : _PartialRolloutNode
        The next node in the tree of responses.
    """

    if include_root:
        first_level = 0
    else:
        first_level = 1
    if leaves_first:
        for level in range(len(partial_rollouts_by_level) - 1, first_level - 1, -1):
            for partial_rollout in partial_rollouts_by_level[level]:
                if include_level:
                    yield level, partial_rollout
                else:
                    yield partial_rollout
    else:
        for level in range(first_level, len(partial_rollouts_by_level)):
            for partial_rollout in partial_rollouts_by_level[level]:
                if include_level:
                    yield level, partial_rollout
                else:
                    yield partial_rollout


def _dispatch_to_trainer(
    method: Callable[Concatenate["PureTextMaltTrainer", P], R],
) -> Callable[Concatenate["PureTextMaltTrainer", P], R]:
    """Dispatch a method to the appropriate trainer.

    This decorator dispatches a method either to the ``PureTextMaltTrainer``
    implementation or to the ``PureTextEiTrainer
    <nip.trainers.ei_pure_text.PureTextEiTrainer>`` implementation, depending on the
    iteration number. This allows doing some rounds of Expert Iteration (EI) before
    doing MALT.

    Parameters
    ----------
    method : Callable
        The method to dispatch. This should be a method of the ``PureTextMaltTrainer``
        class.

    Returns
    -------
    dispatch_method : Callable
        The dispatched method, which will call either `method` or the base class
        implementation of the method, depending on the iteration number.
    """

    @wraps(method)
    def dispatch_method(
        self: "PureTextMaltTrainer",
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        if (
            self.state.iteration
            < self.hyper_params.pure_text_malt.num_initial_ei_iterations
            or self.state.train_loop_stage in ("test", "test_during_training")
        ):
            return getattr(super(type(self), self), method.__name__)(*args, **kwargs)
        else:
            return method(self, *args, **kwargs)

    return dispatch_method


@register_trainer("pure_text_malt")
class PureTextMaltTrainer(PureTextEiTrainer):
    """Multi-Agent LLM Training (MALT) for text-based environments that only use APIs.

    In the MALT protocol :cite:p:`Motwani2024`, we sample multiple responses per
    timestep from the agents. This means that for each datapoint we have a tree of
    responses. For each agent ``A``, at each decision point for ``A`` we look at the
    expected reward for ``A`` for each of the responses. We then select preference pairs
    of responses from these and train using Direct Preference Optimization
    :cite:p:`Rafailov2023`. The way pairs are selected is determined by the
    ``hyper_params.pure_text_malt.pair_selection_method`` parameter, which can be one of
    the following:

    - "positive_negative": Selects a response where the agent's expected reward is above
      a certain threshold (by default the reward mid-point) and a response where the
      agent's expected reward is below this threshold.
    - "interval": Selects a pair of responses where the difference in expected reward is
      above a certain threshold. This threshold is computed as
      ``interval_threshold_proportion`` times the difference between the maximum and
      minimum possible reward for the agent.

    It is also possible do some rounds of Expert Iteration (EI) before doing MALT. The
    ``PureTextMaltTrainer`` class inherits from the ``PureTextEiTrainer`` class, which
    implements the EI protocol, and allows running EI for a number of iterations
    specified by the ``hyper_params.pure_text_malt.num_initial_ei_iterations``
    parameter.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    scenario_instance : ScenarioInstance
        The components of the experiment.
    settings : ExperimentSettings
        The instance-specific settings of the experiment, like device, logging, etc.
    """

    @_dispatch_to_trainer
    async def _stage_create_fine_tune_jobs(self, rollouts: NestedArrayDict):
        """Training stage: create fine-tune jobs for each agent.

        Parameters
        ----------
        rollouts : NestedArrayDict, optional
            The rollouts sampled in this iteration.
        """

        timesteps = self._get_unique_timesteps(rollouts)

        # Order by parent node ID, so that in particular sibling nodes are next to each
        # other.
        timesteps = timesteps[np.argsort(timesteps["_parent_node_id"])]

        is_pair_positive: Bool[np.ndarray, "timestep agent"] = timesteps[
            "agents", "is_pair_positive"
        ]
        is_pair_negative: Bool[np.ndarray, "timestep agent"] = timesteps[
            "agents", "is_pair_negative"
        ]

        async def create_fine_tune_job(
            shared_model_group: PureTextSharedModelGroup,
            group_name: str,
            positive_examples_per_agent: dict[str, NestedArrayDict],
            negative_examples_per_agent: dict[str, NestedArrayDict],
        ):
            await shared_model_group.create_dpo_fine_tune_job(
                positive_examples_per_agent,
                negative_examples_per_agent,
                job_name=self._get_fine_tune_job_name(shared_model_group),
            )
            logger.info(f"Created fine-tune job for group {group_name!r}")

        async with TaskGroup() as task_group:
            for group_name, shared_model_group in self.shared_model_groups.items():

                if shared_model_group.shared_agent_params.freeze_agent:
                    continue

                # Get the positive and negative examples. These line up with each other
                # because we have ordered the timesteps by parent node ID, so that
                # sibling nodes are next to each other.
                positive_examples_per_agent: dict[str, NestedArrayDict] = {}
                negative_examples_per_agent: dict[str, NestedArrayDict] = {}
                for agent_id, agent_name in shared_model_group.agent_ids_and_names():
                    positive_examples_per_agent[agent_name] = timesteps[
                        is_pair_positive[:, agent_id]
                    ]
                    negative_examples_per_agent[agent_name] = timesteps[
                        is_pair_negative[:, agent_id]
                    ]

                task_group.create_task(
                    create_fine_tune_job(
                        shared_model_group,
                        group_name,
                        positive_examples_per_agent,
                        negative_examples_per_agent,
                    )
                )

    def _get_iteration_begin_message(self) -> str:
        """Get the message to log at the beginning of each iteration.

        Returns
        -------
        message : str
            The message to log at the beginning of each iteration.
        """
        if (
            self.state.iteration
            < self.hyper_params.pure_text_malt.num_initial_ei_iterations
        ):
            return (
                f"Initial EI iteration {self.state.iteration+1}/"
                f"{self.hyper_params.pure_text_malt.num_initial_ei_iterations} begins."
            )
        else:
            return "MALT iteration begins."

    @_dispatch_to_trainer
    def _previous_compatible_iterations(self) -> Iterable[int]:
        """Get the previous iterations which are combinable with the current iteration.

        The method is used when combining rollouts from different iterations, and
        returns an iterable of the previous iteration numbers which are able to be
        combined with the current iteration.

        When doing initial EI iterations, on the iterations where we do MALT, we
        combine only the rollouts which also do MALT, not the ones which do EI. This is
        because the rollouts are not compatible, and we don't want to mix them.

        Returns
        -------
        previous_iterations : Iterable[int]
            The previous iterations which are combinable with the current iteration.
        """

        return range(
            self.hyper_params.pure_text_malt.num_initial_ei_iterations,
            self.state.iteration,
        )

    @_dispatch_to_trainer
    async def _sample_rollouts_for_single_environment(
        self,
        environment: PureTextEnvironment,
        data_batch: Optional[NestedArrayDict] = None,
    ) -> list[NestedArrayDict]:
        """Sample rollouts for a single environment.

        A single environment is associated with a single datapoint. This method samples
        rollouts from it.

        To implement the MALT training scheme, we need to sample multiple responses per
        timestep from the agents, and generate a tree of responses.

        We also do additional processing and compute various statistics for each node in
        the tree of responses. It's more efficient and easier to do this now rather than
        later, because we have access to the full tree structure. While it's possible
        recover this later, it takes a bit of work because the rollouts are stored in
        arrays.

        1. We compute the expected reward for each agent at each node of the tree by
           summing up the total reward for all descendants, proceeding from the leaves
           to the root, and dividing by the number of branches passing through the node.
           This is stored in the ``("agents", "expected_reward")`` field of the
           rollouts.

        2. We look at each node and check if in its children there a valid preference
           pair. If so, we randomly sample one and set the ``("agents",
           "is_pair_positive")`` and ``("agents", "is_pair_negative")`` fields to
           ``True`` for the positive and negative example, respectively.

        3. Each node in the response tree gets a unique ID, stored in ``_node_id`` which
           has shape ``(max_message_rounds, )``. This allows reconstructing the tree of
           responses later, if required, because if the same node ID appears in two
           different rollouts, then those points in the message history are the same. We
           also store the parent node ID in the ``_parent_node_id`` field. The first
           timesteps of a rollout have a 'pseudo-parent' node ID. This is important
           because the tree may branch immediately at the first timestep.

        Shapes
        ------
        The following are the shapes of the additional fields added to each rollout.

        - ("agents", "expected_reward"): "round agent"
        - ("agents", "is_pair_positive"): "round agent"
        - ("agents", "is_pair_negative"): "round agent"
        - "_node_id": "round"

        Parameters
        ----------
        environment : PureTextEnvironment
            The environment to sample rollouts in.
        data_batch : NestedArrayDict, optional
            The data batch to use for the rollout. If None, the data batch will be
            sampled from the dataset.

        Returns
        -------
        sampled_rollouts = list[NestedArrayDict]
            The list of sampled rollouts, each of which has batch size
            (max_message_rounds, ).
        """

        partial_rollouts_by_level = await self._generate_response_tree(
            environment, data_batch
        )
        self._compute_tree_expected_reward(partial_rollouts_by_level)
        self._sample_positive_and_negative_examples(partial_rollouts_by_level)

        # Concatenate the environment states of the nodes in the last level to get the
        # sampled rollouts
        sampled_rollouts = []
        for partial_rollout in partial_rollouts_by_level[self.max_message_rounds]:
            sampled_rollout = concatenate_nested_array_dicts(
                partial_rollout.trajectory_env_states, dim=0
            )
            sampled_rollouts.append(sampled_rollout)

        return sampled_rollouts

    async def _generate_response_tree(
        self,
        environment: PureTextEnvironment,
        data_batch: Optional[NestedArrayDict] = None,
    ) -> list[list[_PartialRolloutNode]]:
        """Generate the tree of responses for a single datapoint.

        This generates a tree of partial rollouts, where the children of each node are
        the one-step continuations of the node formed by generating multiple different
        responses for each active agent at that time step. At each step we sample
        ``hyper_params.pure_text_malt.num_responses_per_timestep`` responses.

        The output tree is stratified by the level in the tree, with the root node
        (empty partial rollout) at the first level. Note that in general, the tree will
        not be fully generated, because the environment may terminate before the maximum
        number of message rounds is reached.

        Parameters
        ----------
        environment : PureTextEnvironment
            The environment to sample rollouts in.
        data_batch : NestedArrayDict, optional
            The data batch to use for the rollout. If None, the data batch will be
            sampled from the dataset.

        Returns
        -------
        partial_rollouts_by_level : list[list[_PartialRolloutNode]]
            The tree of responses, stratified by level.
        """

        # Reset the environment to get the initial state
        base_env_state = environment.reset(data_batch=data_batch)

        # A shift which is applied to each node ID to make them unique across all
        # datapoints. This is required because the node IDs are only unique within a
        # datapoint.
        datapoint_id = int(base_env_state["datapoint_id"].item())
        max_num_nodes = (
            self.hyper_params.pure_text_malt.num_responses_per_timestep
            * self.protocol_handler.num_agents
        ) ** (self.protocol_handler.max_message_rounds + 1)
        node_id_base = datapoint_id * max_num_nodes

        # This is the tree structure, stratified by the level in the tree. We start with
        # the root node, which is the initial state of the environment.
        partial_rollouts_by_level = [
            [
                _PartialRolloutNode(
                    base_env_state, self.protocol_handler, node_id_base=node_id_base
                )
            ]
        ]

        # Get a mask indicating whether there is an unfrozen active agent in each round
        active_agents_mask: Bool[Tensor, "round agent channel"] = (
            self.protocol_handler.get_active_agents_mask_from_rounds_and_seed(
                torch.arange(self.protocol_handler.max_message_rounds),
                einops.repeat(
                    torch.from_numpy(base_env_state["seed"]),
                    "1 -> round",
                    round=self.protocol_handler.max_message_rounds,
                ),
            )
        )
        unfrozen_active_agent_mask = []
        for round_id in range(self.protocol_handler.max_message_rounds):
            for agent_id, agent_name in enumerate(self.protocol_handler.agent_names):
                agent_params: PureTextAgentParameters = self.hyper_params.agents[
                    agent_name
                ]
                if (
                    active_agents_mask[round_id, agent_id].any()
                    and not agent_params.freeze_agent
                ):
                    unfrozen_active_agent_mask.append(True)
                    break
            else:
                unfrozen_active_agent_mask.append(False)

        frozen_agents_generate_one_response = (
            self.hyper_params.pure_text_malt.frozen_agents_generate_one_response
        )

        # Generate the tree of responses by iterating down level-by-level
        for level in range(self.protocol_handler.max_message_rounds):

            partial_rollouts_by_level.append([])

            for base_partial_rollout in partial_rollouts_by_level[level]:

                if not base_partial_rollout.ended:

                    if (
                        not frozen_agents_generate_one_response
                        or unfrozen_active_agent_mask[level]
                    ):
                        num_children = (
                            self.hyper_params.pure_text_malt.num_responses_per_timestep
                        )
                    else:
                        num_children = 1

                    # Clone the base rollout to create multiple child rollouts, one for
                    # each response per timestep
                    child_partial_rollouts: list[_PartialRolloutNode] = []
                    for _ in range(num_children):
                        child_partial_rollouts.append(
                            base_partial_rollout.clone_as_child()
                        )

                    for child_partial_rollout in child_partial_rollouts:

                        # Run the forward pass on all agents to sample actions for this
                        # child
                        env_state = await self.combined_agent.forward(
                            child_partial_rollout.current_env_state, environment
                        )

                        # Step the environment to get the next state. This writes the
                        # next state in the "next" sub-dictionary.
                        env_state = environment.step(env_state)

                        # Check if the environment is done or terminated. The state has
                        # batch size 1, so we only need to check the first element.
                        child_partial_rollout.ended = (
                            env_state["next", "done"][0]
                            or env_state["next", "terminated"][0]
                        )

                        # Add the ID of the current partial rollout (i.e. node in tree)
                        # to the environment state. This allows reconstructing the tree
                        # of responses later, if required.
                        env_state["_node_id"] = [child_partial_rollout.node_id]
                        env_state["_parent_node_id"] = [base_partial_rollout.node_id]

                        # Append the current state to the environment states
                        child_partial_rollout.trajectory_env_states.append(env_state)

                        # Update the current state to the next state
                        child_partial_rollout.current_env_state = (
                            environment.get_next_state_from_state(env_state)
                        )

                    partial_rollouts_by_level[level + 1].extend(child_partial_rollouts)

                # If we are done, we need to pad the rollout with zero actions
                else:
                    child_partial_rollout = base_partial_rollout.clone_as_child()
                    env_state = child_partial_rollout.current_env_state
                    env_state["padding"] = np.ones(*environment.batch_size, dtype=bool)
                    env_state["_node_id"] = [child_partial_rollout.node_id]
                    env_state["_parent_node_id"] = [base_partial_rollout.node_id]
                    if "next" not in env_state.keys():
                        env_state = environment.add_dummy_actions_and_next_to_state(
                            env_state
                        )
                    child_partial_rollout.trajectory_env_states.append(env_state)
                    partial_rollouts_by_level[level + 1].append(child_partial_rollout)

        return partial_rollouts_by_level

    def _compute_tree_expected_reward(
        self,
        partial_rollouts_by_level: list[list[_PartialRolloutNode]],
    ):
        """Compute the expected reward for each agent at each node of the tree.

        The expected reward in the average reward that an agent receives over all
        branches passing through a node. This is stored in the ``("agents",
        "expected_reward")`` field of the rollouts, which are modified in-place.

        This is computed by summing up the total reward for all descendants, proceeding
        from the leaves to the root, and dividing by the number of branches passing
        through the node.

        Parameters
        ----------
        partial_rollouts_by_level : list[list[_PartialRolloutNode]]
            The tree of responses, stratified by level. These are modified in-place,
            where we add the ``("agents", "expected_reward")`` field containing the
            expected reward for each agent at each node.
        """

        # Compute the expected reward for each agent at each node of the tree by summing
        # up the total reward for all descendants, proceeding from the leaves to the
        # root
        for level, partial_rollout in _tree_iter(
            partial_rollouts_by_level, include_level=True, leaves_first=True
        ):

            # The last state in the partial trajectory. Note that the partial trajectory
            # has length ``level``
            last_env_state = partial_rollout.trajectory_env_states[-1]

            # For leaf nodes, the number of branches passing through the node is 1. For
            # non-leaf nodes, this number will have been computed by previous
            # iterations, where each child node will have added its number of branches
            # to the this node
            if level == self.max_message_rounds:
                partial_rollout.num_branches = 1

            # Add the rewards of the last state in the partial trajectory to the total
            # reward of the partial rollout. This will have already been added to by the
            # descendants of this node
            partial_rollout.total_reward_per_agent += last_env_state[
                "next", "agents", "reward"
            ]

            # The expected reward for each agent is the total reward divided by the
            # number of branches passing through this node
            last_env_state["agents", "expected_reward"] = (
                partial_rollout.total_reward_per_agent / partial_rollout.num_branches
            )

            # Each of the branches passing through this node pass through the parent, so
            # add the number of branches passing through this node to the number of
            # branches passing through the parent node
            partial_rollout.parent_partial_rollout.num_branches += (
                partial_rollout.num_branches
            )

            # Add the total reward of this node to the total reward of the parent node
            partial_rollout.parent_partial_rollout.total_reward_per_agent += (
                partial_rollout.total_reward_per_agent
            )

    def _sample_positive_and_negative_examples(
        self, partial_rollouts_by_level: list[list[_PartialRolloutNode]]
    ):
        """Sample positive and negative examples for each node in the tree of responses.

        The way this is done depends on the ``pair_selection_method`` hyper-parameter,
        which can be one of the following:

        - "positive_negative": We look at each node and check if in its children there
          is a positive and a negative example.
        - "interval": We look at each node and check if in its children there is a pair
          of nodes whose expected rewards differ by more than a certain threshold. This
          threshold is ``interval_threshold_proportion`` times the difference between
          the maximum and minimum possible reward for the agent.

        If we find a valid pair we randomly sample one and set the ``("agents",
        "is_pair_positive")`` and ``("agents", "is_pair_negative")`` fields to True in
        the positive and negative example, respectively.

        Parameters
        ----------
        partial_rollouts_by_level : list[list[_PartialRolloutNode]]
            The tree of responses, stratified by level. These are modified in-place,
            where we add ``("agents", "is_pair_positive")`` and ``("agents",
            "is_pair_negative")`` fields to the rollouts.
        """

        environment_seed = partial_rollouts_by_level[-1][0].current_env_state["seed"]
        rng = np.random.default_rng(seed=self.hyper_params.seed + environment_seed)

        for partial_rollout in _tree_iter(partial_rollouts_by_level):
            # Add the is_pair_positive and is_pair_negative fields to the last state in
            # the partial trajectory. This will be set to True for the positive and
            # negative examples, respectively.
            last_env_state = partial_rollout.trajectory_env_states[-1]
            last_env_state["agents", "is_pair_positive"] = np.zeros(
                (1, self.num_agents), dtype=bool
            )
            last_env_state["agents", "is_pair_negative"] = np.zeros(
                (1, self.num_agents), dtype=bool
            )

        # Sample positive and negative examples for each node in the tree of responses
        for partial_rollout in _tree_iter(partial_rollouts_by_level, include_root=True):

            for agent_id, agent_name in enumerate(self.agent_names):

                if (
                    self.hyper_params.pure_text_malt.pair_selection_method
                    == "positive_negative"
                ):

                    reward_threshold = self.protocol_handler.reward_mid_point_estimate(
                        agent_name
                    )

                    # Check if in its children there is a positive and a negative
                    # example
                    positive_examples: list[_PartialRolloutNode] = []
                    negative_examples: list[_PartialRolloutNode] = []
                    for child_partial_rollout in partial_rollout.child_partial_rollouts:
                        if not child_partial_rollout.has_agent_acted(agent_name):
                            continue
                        last_env_state = child_partial_rollout.trajectory_env_states[-1]
                        if (
                            last_env_state["agents", "expected_reward"][0, agent_id]
                            >= reward_threshold
                        ):
                            positive_examples.append(child_partial_rollout)
                        else:
                            negative_examples.append(child_partial_rollout)

                    # If there are positive and negative examples, set the corresponding
                    # fields and randomly sample a positive and a negative example from
                    # the children
                    if len(positive_examples) > 0 and len(negative_examples) > 0:
                        sampled_positive_partial_rollout: _PartialRolloutNode = (
                            rng.choice(positive_examples)
                        )
                        sampled_negative_partial_rollout: _PartialRolloutNode = (
                            rng.choice(negative_examples)
                        )
                        sampled_positive_partial_rollout.trajectory_env_states[-1][
                            "agents", "is_pair_positive"
                        ][0, agent_id] = True
                        sampled_negative_partial_rollout.trajectory_env_states[-1][
                            "agents", "is_pair_negative"
                        ][0, agent_id] = True

                elif (
                    self.hyper_params.pure_text_malt.pair_selection_method == "interval"
                ):

                    interval_threshold = (
                        self.hyper_params.pure_text_malt.interval_threshold_proportion
                        * (
                            self.protocol_handler.max_reward(agent_name)
                            - self.protocol_handler.min_reward(agent_name)
                        )
                    )

                    possible_pairs: list[
                        tuple[_PartialRolloutNode, _PartialRolloutNode]
                    ] = []
                    for child_1, child_2 in itertools.combinations(
                        partial_rollout.child_partial_rollouts, 2
                    ):
                        if not child_1.has_agent_acted(
                            agent_name
                        ) or not child_2.has_agent_acted(agent_name):
                            continue
                        expected_reward_1 = child_1.trajectory_env_states[-1][
                            "agents", "expected_reward"
                        ][0, agent_id]
                        expected_reward_2 = child_2.trajectory_env_states[-1][
                            "agents", "expected_reward"
                        ][0, agent_id]
                        if expected_reward_1 - expected_reward_2 > interval_threshold:
                            possible_pairs.append((child_1, child_2))
                        elif expected_reward_2 - expected_reward_1 > interval_threshold:
                            possible_pairs.append((child_2, child_1))

                    if len(possible_pairs) > 0:
                        (
                            sampled_positive_partial_rollout,
                            sampled_negative_partial_rollout,
                        ) = rng.choice(possible_pairs)
                        sampled_positive_partial_rollout.trajectory_env_states[-1][
                            "agents", "is_pair_positive"
                        ][0, agent_id] = True
                        sampled_negative_partial_rollout.trajectory_env_states[-1][
                            "agents", "is_pair_negative"
                        ][0, agent_id] = True

                else:
                    raise ValueError(
                        f"Unknown pair selection method: "
                        f"{self.hyper_params.pure_text_malt.pair_selection_method!r}"
                    )

    @_dispatch_to_trainer
    def _get_log_stats(
        self,
        rollouts: NestedArrayDict,
        *,
        train=True,
    ) -> dict:
        """Get the statistics to log for the given rollouts.

        This method extends the base class method to include the MALT-specific
        statistics.

        Parameters
        ----------
        rollouts : NestedArrayDict
            The rollouts to get the statistics for.
        train : bool, default=True
            Whether the rollouts are from the training environment.

        Returns
        -------
        stats : dict
            The statistics to log.
        """

        log_stats = super()._get_log_stats(rollouts, train=train)

        timesteps = self._get_unique_timesteps(rollouts)
        timesteps = timesteps[~timesteps["padding"]]

        is_pair_positive: Bool[np.ndarray, "timestep agent"] = timesteps[
            "agents", "is_pair_positive"
        ]
        datapoint_id: Int[np.ndarray, "timestep"] = timesteps["datapoint_id"]
        next_done: Bool[np.ndarray, "timestep"] = timesteps["next", "done"]

        unique_datapoint_ids, node_count_by_datapoint = np.unique(
            datapoint_id, return_counts=True
        )
        _, non_terminal_node_count_by_datapoint = np.unique(
            datapoint_id[~next_done], return_counts=True
        )

        log_stats["mean_malt_nodes"] = (
            np.sum(node_count_by_datapoint) / unique_datapoint_ids.shape[0]
        )
        log_stats["mean_non_terminal_malt_nodes"] = (
            np.sum(non_terminal_node_count_by_datapoint) / unique_datapoint_ids.shape[0]
        )

        for agent_id, agent_name in enumerate(self.agent_names):
            log_stats[f"{agent_name}.mean_malt_pairs"] = (
                np.sum(is_pair_positive[:, agent_id]) / unique_datapoint_ids.shape[0]
            )
            log_stats[f"{agent_name}.malt_pair_proportion"] = np.sum(
                is_pair_positive[:, agent_id]
            ) / np.sum(~next_done)

        return log_stats

    def _get_unique_timesteps(self, rollouts: NestedArrayDict) -> NestedArrayDict:
        """Break the rollouts into timesteps, and remove duplicate nodes.

        Each timestep is a unique node in the tree of responses.

        Parameters
        ----------
        rollouts : NestedArrayDict
            The rollouts to get the timesteps for. Has batch size (batch round).

        Returns
        -------
        timesteps : NestedArrayDict
            The rollouts, broken into timesteps, with the duplicate nodes removed. Has
            batch size (timestep).
        """

        node_id: Int[np.ndarray, "batch round"] = rollouts["_node_id"]

        _, unique_index = np.unique(
            einops.rearrange(node_id, "batch round -> (batch round)"), return_index=True
        )

        unique_mask = np.zeros((node_id.shape[0] * node_id.shape[1]), dtype=bool)
        unique_mask[unique_index] = True
        unique_mask = einops.rearrange(
            unique_mask, "(batch round) -> batch round", batch=node_id.shape[0]
        )

        return rollouts[unique_mask]
