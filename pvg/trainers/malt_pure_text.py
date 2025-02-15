"""Multi-Agent LLM Training (MALT) for text-based environments which only use APIs.

In the MALT protocol :cite:`Motwani24`, we sample multiple responses per timestep from
the agents. This means that for each datapoint we have a tree of responses. For each
agent `A`, at each decision point for `A` we look at the expected reward for `A` for
each of the responses. We threshold this expected reward to get a binary classification
label for each response. We select good-bad pairs of these, and train using Direct
Preference Optimization :cite:`Rafailov23`.
"""

from typing import Optional, ClassVar, Any
from dataclasses import dataclass, InitVar
import dataclasses

import numpy as np

import pandas as pd

import einops

from jaxtyping import Int, Float, Bool

from pvg.parameters import HyperParameters
from pvg.protocols.protocol_base import ProtocolHandler
from pvg.trainers.rl_pure_text_base import PureTextRlTrainer
from pvg.trainers.registry import register_trainer
from pvg.scenario_base.environment import PureTextEnvironment
from pvg.scenario_base.agents import PureTextCombinedWhole
from pvg.utils.nested_array_dict import NestedArrayDict, concatenate_nested_array_dicts
from pvg.utils.maths import mean_for_unique_keys


@register_trainer("pure_text_malt")
class PureTextMaltTrainer(PureTextRlTrainer):
    """Multi-Agent LLM Training (MALT) for text-based environments which only use APIs.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    scenario_instance : ScenarioInstance
        The components of the experiment.
    settings : ExperimentSettings
        The instance-specific settings of the experiment, like device, logging, etc.
    """

    def _stage_create_fine_tune_jobs(self, rollouts: NestedArrayDict):
        """Training stage: create fine-tune jobs for each agent.

        Parameters
        ----------
        rollouts : NestedArrayDict, optional
            The rollouts sampled in this iteration.
        """

        node_id: Int[np.ndarray, "batch round"] = rollouts["_node_id"]

        # Break the rollouts into timesteps, and remove duplicate nodes
        _, unique_index = np.unique(
            einops.rearrange(node_id, "batch round -> (batch round)"), return_index=True
        )
        unique_mask = np.zeros((node_id.shape[0] * node_id.shape[1]), dtype=bool)
        unique_mask[unique_index] = True
        unique_mask = einops.rearrange(
            unique_mask, "(batch round) -> batch round", batch=node_id.shape[0]
        )
        timesteps = rollouts[unique_mask]

        node_id: Int[np.ndarray, "timestep"] = timesteps["_node_id"]
        has_positive_and_negative: Bool[np.ndarray, "timestep agent"] = timesteps[
            "agents", "has_positive_and_negative"
        ]

        # A mask indicating which agents are active in any channel in the next round at
        # each timestep. Note that for the last round we say that no agents are active
        # in the next round, because there is no next round.
        last_round_id = self.protocol_handler.max_message_rounds - 1
        clipped_next_round = np.minimum(timesteps["round"] + 1, last_round_id)
        agent_active_mask: Bool[np.ndarray, "timestep agent channel"] = (
            self.protocol_handler.get_active_agents_mask_from_rounds_and_seed(
                clipped_next_round,
                timesteps["seed"],
            )
            .cpu()
            .numpy()
        )
        agent_active_mask_any: Bool[np.ndarray, "timestep agent"] = einops.reduce(
            agent_active_mask, "timestep agent channel -> timestep agent", "max"
        )
        agent_active_mask_any[timesteps["round"] == last_round_id] = False

        for group_name, shared_model_group in self.shared_model_groups.items():

            timesteps_per_agent: dict[str, NestedArrayDict] = {}
            positive_examples_per_agent: dict[str, NestedArrayDict] = {}
            negative_examples_per_agent: dict[str, NestedArrayDict] = {}
            for agent_id, agent_name in shared_model_group.agent_ids_and_names():

                # Select the timesteps which have positive and negative examples for the
                # agent
                timesteps_per_agent[agent_name] = timesteps[
                    has_positive_and_negative[:, agent_id]
                    & agent_active_mask_any[:, agent_id]
                ]

                # Get the node IDs of the positive and negative examples
                sampled_positive_example: Int[np.ndarray, "timestep"] = (
                    timesteps_per_agent[agent_name][
                        "agents", "sampled_positive_example"
                    ][:, agent_id]
                )
                sampled_negative_example: Int[np.ndarray, "timestep"] = (
                    timesteps_per_agent[agent_name][
                        "agents", "sampled_negative_example"
                    ][:, agent_id]
                )

                # Get the indices of the positive and negative examples
                node_id_sorter = np.argsort(node_id)
                positive_example_index = node_id_sorter[
                    np.searchsorted(
                        node_id, sampled_positive_example, sorter=node_id_sorter
                    )
                ]
                negative_example_index = node_id_sorter[
                    np.searchsorted(
                        node_id, sampled_negative_example, sorter=node_id_sorter
                    )
                ]

                # Get the positive and negative examples
                positive_examples_per_agent[agent_name] = timesteps[
                    positive_example_index
                ]
                negative_examples_per_agent[agent_name] = timesteps[
                    negative_example_index
                ]

            self.settings.logger.info(
                f"Creating fine-tune job for group {group_name!r}"
            )

            shared_model_group.create_dpo_fine_tune_job(
                timesteps_per_agent,
                positive_examples_per_agent,
                negative_examples_per_agent,
                job_name=self._get_fine_tune_job_name(shared_model_group),
            )

    @staticmethod
    def _sample_rollouts_for_single_environment(
        args: tuple[
            HyperParameters,
            ProtocolHandler,
            PureTextEnvironment,
            PureTextCombinedWhole,
            Optional[NestedArrayDict],
        ]
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
        summing up the total reward for all descendants, proceeding from the leaves to
        the root, and dividing by the number of branches passing through the node. This
        is stored in the `("agents", "expected_reward")` field of the rollouts.

        2. The expected reward is thresholded using an estimate of the reward mid-points
        to get a binary classification label for each response, into 'positive' and
        'negative' examples. This is stored in `("agents", "is_positive_example")`.

        3. We look at each node and check if in its children there is a positive and a
        negative example. If so, we set the `("agents", "has_positive_and_negative")`
        field to True. In this case, we randomly sample a positive and a negative
        example from the children and set the `("agents", "sampled_positive_example")`
        and `("agents", "sampled_negative_example")` fields to the corresponding node
        IDs. Otherwise these fields are set to -1.

        4. Each node in the response tree gets a unique ID, stored in
        `_node_id` which has shape `(max_message_rounds, )`. This allows
        reconstructing the tree of responses later, if required, because if the same
        node ID appears in two different rollouts, then those points in the message
        history are the same.

        Shapes
        ------
        The following are the shapes of the additional fields added to each rollout.

        - ("agents", "expected_reward"): "round agent"
        - ("agents", "is_positive_example"): "round agent"
        - ("agents", "has_positive_and_negative"): "round agent"
        - ("agents", "sampled_positive_example"): "round agent"
        - ("agents", "sampled_negative_example"): "round agent"
        - "_node_id": "round"

        Notes
        -----
        This function is intended to be applied by a pool of workers. As such it must be
        a static function and take all trainer attributes required as arguments.

        Parameters
        ----------
        hyper_params : HyperParameters
            The parameters of the experiment.
        protocol_handler : ProtocolHandler
            The interaction protocol handler for the experiment.
        environment : PureTextEnvironment
            The environment to sample a rollout in.
        combined_agent : PureTextCombinedWhole
            The combined agent to use for the rollout.
        data_batch : NestedArrayDict, optional
            The data batch to use for the rollout. If None, the data batch will be
            sampled from the dataset.

        Returns
        -------
        sampled_rollouts = list[NestedArrayDict]
            The list of sampled rollouts, each of which has batch size
            (max_message_rounds, ).
        """

        hyper_params, protocol_handler, environment, combined_agent, data_batch = args

        partial_rollouts_by_level = _generate_response_tree(
            hyper_params=hyper_params,
            protocol_handler=protocol_handler,
            environment=environment,
            combined_agent=combined_agent,
            data_batch=data_batch,
        )

        _compute_tree_expected_reward(
            partial_rollouts_by_level=partial_rollouts_by_level,
            hyper_params=hyper_params,
            protocol_handler=protocol_handler,
        )

        _sample_positive_and_negative_examples(
            partial_rollouts_by_level=partial_rollouts_by_level,
            hyper_params=hyper_params,
        )

        # Concatenate the environment states of the nodes in the last level to get the
        # sampled rollouts
        sampled_rollouts = []
        for partial_rollout in partial_rollouts_by_level[
            protocol_handler.max_message_rounds
        ]:
            sampled_rollout = concatenate_nested_array_dicts(
                partial_rollout.trajectory_env_states, dim=0
            )
            sampled_rollouts.append(sampled_rollout)

        return sampled_rollouts

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

        datapoint_id: Int[np.ndarray, "rollout"] = rollouts["datapoint_id"][..., 0]
        node_id: Int[np.ndarray, "rollout round"] = rollouts["_node_id"]

        expected_reward: Float[np.ndarray, "rollout round agent"] = rollouts[
            "agents", "expected_reward"
        ]
        is_positive_example: Float[np.ndarray, "rollout round agent"] = rollouts[
            "agents", "is_positive_example"
        ]
        has_positive_and_negative: Bool[np.ndarray, "rollout round agent"] = rollouts[
            "agents", "has_positive_and_negative"
        ]

        # Split the rollouts, grouping by datapoint ID. We do this assuming that
        # datapoint IDs are contiguous, i.e. that all rollouts for the same datapoint
        # are next to each other in the array. TODO: Write a test to make sure this
        # stays the case.
        _, datapoint_index = np.unique(datapoint_id, return_index=True)
        datapoint_index.sort()
        datapoint_index = datapoint_index[1:]
        node_id_by_datapoint = np.split(node_id, datapoint_index)
        expected_reward_by_datapoint = np.split(expected_reward, datapoint_index)
        is_positive_example_by_datapoint = np.split(
            is_positive_example, datapoint_index
        )
        has_positive_and_negative_by_datapoint = np.split(
            has_positive_and_negative, datapoint_index
        )

        # Compute the mean of the various quantities for each datapoint, round and
        # agent. Note the Python loop introduces a some inefficiency, but it's not too
        # bad because the number of datapoints per iteration is usually small.
        mean_shape = (
            len(node_id_by_datapoint),
            self.max_message_rounds,
            self.num_agents,
        )
        mean_expected_reward = np.zeros(mean_shape, dtype=float)
        mean_is_positive_example = np.zeros(mean_shape, dtype=float)
        mean_has_positive_and_negative = np.zeros(mean_shape, dtype=float)
        for (
            i,
            node_id_datapoint,
            expected_reward_datapoint,
            is_positive_example_datapoint,
            has_positive_and_negative_datapoint,
        ) in zip(
            range(len(node_id_by_datapoint)),
            node_id_by_datapoint,
            expected_reward_by_datapoint,
            is_positive_example_by_datapoint,
            has_positive_and_negative_by_datapoint,
        ):
            mean_expected_reward[i] = mean_for_unique_keys(
                expected_reward_datapoint, node_id_datapoint[..., None], axis=0
            )
            mean_is_positive_example[i] = mean_for_unique_keys(
                is_positive_example_datapoint, node_id_datapoint[..., None], axis=0
            )
            mean_has_positive_and_negative[i] = mean_for_unique_keys(
                has_positive_and_negative_datapoint,
                node_id_datapoint[..., None],
                axis=0,
            )

        log_stats[f"mean_expected_reward_by_round"] = pd.DataFrame(
            np.mean(mean_expected_reward, axis=0), columns=self.agent_names
        )
        log_stats[f"std_expected_reward_by_round"] = pd.DataFrame(
            np.std(mean_expected_reward, axis=0), columns=self.agent_names
        )
        log_stats[f"mean_is_positive_example_by_round"] = pd.DataFrame(
            np.mean(mean_is_positive_example, axis=0), columns=self.agent_names
        )
        log_stats[f"std_is_positive_example_by_round"] = pd.DataFrame(
            np.std(mean_is_positive_example, axis=0), columns=self.agent_names
        )
        log_stats[f"mean_has_positive_and_negative_by_round"] = pd.DataFrame(
            np.mean(mean_has_positive_and_negative, axis=0), columns=self.agent_names
        )
        log_stats[f"std_has_positive_and_negative_by_round"] = pd.DataFrame(
            np.std(mean_has_positive_and_negative, axis=0), columns=self.agent_names
        )
        for agent_id, agent_name in enumerate(self.agent_names):
            log_stats[f"{agent_name}.mean_expected_reward"] = np.mean(
                mean_expected_reward[..., agent_id]
            )
            log_stats[f"{agent_name}.std_expected_reward"] = np.std(
                mean_expected_reward[..., agent_id]
            )
            log_stats[f"{agent_name}.mean_is_positive_example"] = np.mean(
                mean_is_positive_example[..., agent_id]
            )
            log_stats[f"{agent_name}.std_is_positive_example"] = np.std(
                mean_is_positive_example[..., agent_id]
            )
            log_stats[f"{agent_name}.mean_has_positive_and_negative"] = np.mean(
                mean_has_positive_and_negative[..., agent_id]
            )
            log_stats[f"{agent_name}.std_has_positive_and_negative"] = np.std(
                mean_has_positive_and_negative[..., agent_id]
            )

        return log_stats


@dataclass
class _PartialRolloutNode:
    """A node in the tree of responses, which is a partially generated rollout."""

    current_env_state: NestedArrayDict
    ended: bool = False
    trajectory_env_states: list[NestedArrayDict] = dataclasses.field(
        default_factory=list
    )
    node_id: int = -1
    parent_partial_rollout: Optional["_PartialRolloutNode"] = None
    child_partial_rollouts: list["_PartialRolloutNode"] = dataclasses.field(
        default_factory=list
    )
    num_branches: int = 0
    total_reward_per_agent: np.ndarray | float = 0.0

    node_id_base: InitVar[Optional[int]] = None

    _shared_data: ClassVar[dict[str, Any]] = {"next_node_id": 0}

    def __post_init__(self, node_id_base: Optional[int]):
        if node_id_base is not None:
            self._shared_data["next_node_id"] = node_id_base
        if self.node_id == -1:
            self.node_id = self._shared_data["next_node_id"]
            self._shared_data["next_node_id"] += 1

    def clone_as_child(self):
        # We deep copy the current environment state, because that will be
        # modified in place. We shallow copy the trajectory list, because the
        # states will be shared between nodes with the same ancestors, but the
        # list itself will not be shared. NOTE: deep copying the environment
        # state results in a small inefficiency, because we only really need to
        # keep the environment state of the leaf nodes. But the slowdown is
        # probably negligible.
        cloned_partial_rollout = type(self)(
            current_env_state=self.current_env_state.clone(),
            ended=self.ended,
            trajectory_env_states=self.trajectory_env_states.copy(),
            node_id=self._shared_data["next_node_id"],
            parent_partial_rollout=self,
        )
        self._shared_data["next_node_id"] += 1
        self.child_partial_rollouts.append(cloned_partial_rollout)
        return cloned_partial_rollout


def _tree_iter(
    partial_rollouts_by_level: list[list[_PartialRolloutNode]],
    include_level: bool = False,
    leaves_first: bool = False,
):
    """Iterate over the tree of responses, either downwards or upwards.

    We omit the root node, because it is not a response.

    Parameters
    ----------
    partial_rollouts_by_level : list[list[_PartialRolloutNode]]
        The tree of responses, stratified by level.
    include_level : bool, default=False
        Whether to include the level in the output. In this case, the output is a tuple
        of the level and the partial rollout.
    leaves_first : bool, default=False
        Whether to iterate from the leaves upwards.

    Yields
    ------
    level : int, optional
        The level in the tree of responses.
    partial_rollout : _PartialRolloutNode
        The next node in the tree of responses.
    """

    if leaves_first:
        for level in range(len(partial_rollouts_by_level) - 1, 0, -1):
            for partial_rollout in partial_rollouts_by_level[level]:
                if include_level:
                    yield level, partial_rollout
                else:
                    yield partial_rollout
    else:
        for level in range(1, len(partial_rollouts_by_level)):
            for partial_rollout in partial_rollouts_by_level[level]:
                if include_level:
                    yield level, partial_rollout
                else:
                    yield partial_rollout


def _generate_response_tree(
    hyper_params: HyperParameters,
    protocol_handler: ProtocolHandler,
    environment: PureTextEnvironment,
    combined_agent: PureTextCombinedWhole,
    data_batch: Optional[NestedArrayDict] = None,
) -> list[list[_PartialRolloutNode]]:
    """Generate the tree of responses for a single datapoint.

    This generates a tree of partial rollouts, where the children of each node are the
    one-step continuations of the node formed by generating multiple different responses
    for each active agent at that time step. At each step we sample
    `hyper_params.pure_text_malt.num_responses_per_timestep` responses.

    The output tree is stratified by the level in the tree, with the root node (empty
    partial rollout) at the first level. Note that in general, the tree will not be
    fully generated, because the environment may terminate before the maximum number of
    message rounds is reached.

    Parameters
    ----------
    hyper_params : HyperParameters
        The parameters of the experiment.
    protocol_handler : ProtocolHandler
        The interaction protocol handler for the experiment.
    environment : PureTextEnvironment
        The environment to sample a rollout in.
    combined_agent : PureTextCombinedWhole
        The combined agent to use for the rollout.
    data_batch : NestedArrayDict, optional
        The data batch to use for the rollout. If None, the data batch will be sampled
        from the dataset.

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
        hyper_params.pure_text_malt.num_responses_per_timestep
        * protocol_handler.num_agents
    ) ** (protocol_handler.max_message_rounds + 1)
    node_id_base = datapoint_id * max_num_nodes

    # This is the tree structure, stratified by the level in the tree. We start with the
    # root node, which is the initial state of the environment.
    partial_rollouts_by_level = [
        [_PartialRolloutNode(base_env_state, node_id_base=node_id_base)]
    ]

    # Generate the tree of responses by iterating down level-by-level
    for level in range(protocol_handler.max_message_rounds):
        partial_rollouts_by_level.append([])
        for base_partial_rollout in partial_rollouts_by_level[level]:
            if not base_partial_rollout.ended:

                # Clone the base rollout to create multiple child rollouts, one for each
                # response per timestep
                child_partial_rollouts: list[_PartialRolloutNode] = []
                for _ in range(hyper_params.pure_text_malt.num_responses_per_timestep):
                    child_partial_rollouts.append(base_partial_rollout.clone_as_child())

                for child_partial_rollout in child_partial_rollouts:

                    # Run the forward pass on all agents to sample actions for this
                    # child
                    env_state = combined_agent.forward(
                        child_partial_rollout.current_env_state, environment
                    )

                    # Step the environment to get the next state. This writes the next
                    # state in the "next" sub-dictionary.
                    env_state = environment.step(env_state)

                    # Check if the environment is done or terminated. The state has
                    # batch size 1, so we only need to check the first element.
                    child_partial_rollout.ended = (
                        env_state["next", "done"][0]
                        or env_state["next", "terminated"][0]
                    )

                    # Add the ID of the current partial rollout (i.e. node in tree) to
                    # the environment state. This allows reconstructing the tree of
                    # responses later, if required.
                    env_state["_node_id"] = [child_partial_rollout.node_id]

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
                if "next" not in env_state.keys():
                    env_state = environment.add_dummy_actions_and_next_to_state(
                        env_state
                    )
                child_partial_rollout.trajectory_env_states.append(env_state)
                partial_rollouts_by_level[level + 1].append(child_partial_rollout)

    return partial_rollouts_by_level


def _compute_tree_expected_reward(
    partial_rollouts_by_level: list[list[_PartialRolloutNode]],
    hyper_params: HyperParameters,
    protocol_handler: ProtocolHandler,
):
    """Compute the expected reward for each agent at each node of the tree.

    The expected reward in the average reward that an agent receives over all branches
    passing through a node. This is stored in the `("agents", "expected_reward")` field
    of the rollouts, which are modified in-place.

    This is computed by summing up the total reward for all descendants, proceeding from
    the leaves to the root, and dividing by the number of branches passing through the
    node.

    We also threshold the expected reward to get a binary classification label for each
    response. This is stored in the `("agents", "is_positive_example")` field.

    Parameters
    ----------
    partial_rollouts_by_level : list[list[_PartialRolloutNode]]
        The tree of responses, stratified by level. These are modified in-place, where
        we add `("agents", "expected_reward")` and `("agents", "is_positive_example")`
        fields containing the expected reward for each agent at each node.
    hyper_params : HyperParameters
        The parameters of the experiment.
    protocol_handler : ProtocolHandler
        The interaction protocol handler for the experiment.
    """

    # The threshold for being a positive example for each agent. This is the midpoint of
    # the reward range for each agent
    agent_reward_thresholds = np.array(
        [
            protocol_handler.reward_mid_point_estimate(agent_name)
            for agent_name in protocol_handler.agent_names
        ]
    )

    # Compute the expected reward for each agent at each node of the tree by summing up
    # the total reward for all descendants, proceeding from the leaves to the root
    for level, partial_rollout in _tree_iter(
        partial_rollouts_by_level, include_level=True, leaves_first=True
    ):

        # The last state in the partial trajectory. Note that the partial trajectory has
        # length `level`
        last_env_state = partial_rollout.trajectory_env_states[-1]

        # For leaf nodes, the number of branches passing through the node is 1. For
        # non-leaf nodes, this number will have been computed by previous iterations,
        # where each child node will have added its number of branches to the this node
        if level == protocol_handler.max_message_rounds:
            partial_rollout.num_branches = 1

        # Add the rewards of the last state in the partial trajectory to the total
        # reward of the partial rollout. This will have already been added to by the
        # descendants of this node
        partial_rollout.total_reward_per_agent += last_env_state[
            "next", "agents", "reward"
        ]

        # The expected reward for each agent is the total reward divided by the number
        # of branches passing through this node
        last_env_state["agents", "expected_reward"] = (
            partial_rollout.total_reward_per_agent / partial_rollout.num_branches
        )

        # Threshold the expected reward to get a binary classification label for each
        # response
        last_env_state["agents", "is_positive_example"] = (
            last_env_state["agents", "expected_reward"] >= agent_reward_thresholds
        ).astype(np.bool)

        # Each of the branches passing through this node pass through the parent, so add
        # the number of branches passing through this node to the number of branches
        # passing through the parent node
        partial_rollout.parent_partial_rollout.num_branches += (
            partial_rollout.num_branches
        )

        # Add the total reward of this node to the total reward of the parent node
        partial_rollout.parent_partial_rollout.total_reward_per_agent += (
            partial_rollout.total_reward_per_agent
        )


def _sample_positive_and_negative_examples(
    partial_rollouts_by_level: list[list[_PartialRolloutNode]],
    hyper_params: HyperParameters,
):
    """Sample positive and negative examples for each node in the tree of responses.

    We look at each node and check if in its children there is a positive and a negative
    example. If so, we set the `("agents", "has_positive_and_negative")` field to True.
    In this case, we randomly sample a positive and a negative example from the children
    and set the `("agents", "sampled_positive_example")` and `("agents",
    "sampled_negative_example")` fields to the corresponding node IDs. Otherwise these
    fields are set to -1.

    Parameters
    ----------
    partial_rollouts_by_level : list[list[_PartialRolloutNode]]
        The tree of responses, stratified by level. These are modified in-place, where
        we add `("agents", "has_positive_and_negative")`, `("agents",
        "sampled_positive_example")`, and `("agents", "sampled_negative_example")`
        fields to the rollouts.
    hyper_params : HyperParameters
        The parameters of the experiment.
    """

    environment_seed = partial_rollouts_by_level[-1][0].current_env_state["seed"]
    rng = np.random.default_rng(seed=hyper_params.seed + environment_seed)

    num_agents = (
        partial_rollouts_by_level[-1][0].current_env_state["agents", "done"].shape[1]
    )

    # Sample positive and negative examples for each node in the tree of responses
    for partial_rollout in _tree_iter(partial_rollouts_by_level):

        has_positive_and_negative = np.zeros((1, num_agents), dtype=bool)
        sampled_positive_example = np.full((1, num_agents), -1)
        sampled_negative_example = np.full((1, num_agents), -1)

        for agent_id in range(num_agents):

            # Check if in its children there is a positive and a negative example
            positive_examples: list[_PartialRolloutNode] = []
            negative_examples: list[_PartialRolloutNode] = []
            for child_partial_rollout in partial_rollout.child_partial_rollouts:
                last_env_state = child_partial_rollout.trajectory_env_states[-1]
                if last_env_state["agents", "is_positive_example"][0, agent_id]:
                    positive_examples.append(child_partial_rollout)
                else:
                    negative_examples.append(child_partial_rollout)

            # If there are positive and negative examples, set the corresponding fields
            # and randomly sample a positive and a negative example from the children
            if len(positive_examples) > 0 and len(negative_examples) > 0:
                has_positive_and_negative[0, agent_id] = True
                sampled_positive_partial_rollout: _PartialRolloutNode = rng.choice(
                    positive_examples
                )
                sampled_negative_partial_rollout: _PartialRolloutNode = rng.choice(
                    negative_examples
                )
                sampled_positive_example[0, agent_id] = (
                    sampled_positive_partial_rollout.node_id
                )
                sampled_negative_example[0, agent_id] = (
                    sampled_negative_partial_rollout.node_id
                )

        partial_rollout.trajectory_env_states[-1][
            "agents", "has_positive_and_negative"
        ] = has_positive_and_negative

        partial_rollout.trajectory_env_states[-1][
            "agents", "sampled_positive_example"
        ] = sampled_positive_example
        partial_rollout.trajectory_env_states[-1][
            "agents", "sampled_negative_example"
        ] = sampled_negative_example
