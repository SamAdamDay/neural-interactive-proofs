"""The data structure and utility for reconstructing a MALT forest."""

from typing import Union, Iterator

from numpy.typing import NDArray

from jaxtyping import Int

from nip.utils.nested_array_dict import NestedArrayDict


class MaltNode:
    """A node in a MALT tree.

    Parameters
    ----------
    env_state : NestedArrayDict
        The state of the environment at this node, which contains all relevant
        information about the rollout at this timestep.
    node_hash : int
        The hash of the node, which is used to identify the node in the tree and forest.
    parent : MaltNode | MaltTree, optional
        The parent node of this node. If not provided, the parent is set to None.
    """

    def __init__(
        self,
        env_state: NestedArrayDict,
        node_hash: int,
        parent: Union["MaltNode", "MaltTree", None] = None,
    ):
        self.env_state = env_state
        self.node_hash = node_hash
        self.parent = parent

        self.children = []

        if parent is not None:
            parent.add_child(self)

    def add_child(self, child: "MaltNode"):
        """Add a child to this node.

        Parameters
        ----------
        child : MaltNode
            The child node to add.
        """
        self.children.append(child)
        child.parent = self


class MaltTree:
    """A tree in a MALT forest, which is a collection children of a root node."""

    def __init__(self):
        self.children = []

    def add_child(self, child: MaltNode):
        """Add a child to this tree.

        Parameters
        ----------
        child : MaltNode
            The child node to add.
        """
        self.children.append(child)
        child.parent = self

    def __getitem__(self, index: int) -> MaltNode:
        """Get a child node by index.

        Parameters
        ----------
        index : int
            The index of the child node to get.

        Returns
        -------
        MaltNode
            The child node at the given index.
        """

        return self.children[index]

    def __len__(self) -> int:
        """Get the number of child nodes.

        Returns
        -------
        int
            The number of child nodes.
        """

        return len(self.children)

    def __iter__(self) -> Iterator[MaltNode]:
        """Iterate over the child nodes.

        Returns
        -------
        Iterator[MaltNode]
            An iterator over the child nodes.
        """

        return iter(self.children)


def reconstruct_malt_forest(rollouts: NestedArrayDict) -> list[MaltTree]:
    """Reconstruct a forest of trees from MALT rollouts.

    The MALT :cite:p:`Motwani2024` trainer samples a set of trees of responses. These
    are stored in a flat array. This function reconstructs the trees from this flat
    array.

    Parameters
    ----------
    rollouts : NestedArrayDict
        The rollouts to reconstruct the trees from.

    Returns
    -------
    malt_forest : list[MaltNode]
        A list of ``MaltNode`` objects representing the root nodes of the trees in the
        forest.
    """

    node_id: Int[NDArray, "rollout round"] = rollouts["_node_id"]
    datapoint_id: Int[NDArray, "rollout"] = rollouts["datapoint_id"][:, 0]
    padding_mask: Int[NDArray, "rollout round"] = rollouts["padding"]

    nodes_by_hash: dict[int, MaltNode] = {}
    trees_by_datapoint_id: dict[int, MaltTree] = {}

    for rollout_id in range(node_id.shape[0]):

        current_datapoint_id = datapoint_id[rollout_id]
        if current_datapoint_id not in trees_by_datapoint_id:
            trees_by_datapoint_id[current_datapoint_id] = MaltTree()
        parent = trees_by_datapoint_id[current_datapoint_id]

        for round_id in range(node_id.shape[1]):
            if padding_mask[rollout_id, round_id]:
                break
            node_hash = hash(node_id[rollout_id, : round_id + 1].tobytes())
            if node_hash not in nodes_by_hash:
                nodes_by_hash[node_hash] = MaltNode(
                    env_state=rollouts[rollout_id, round_id],
                    node_hash=node_hash,
                    parent=parent,
                )
            parent = nodes_by_hash[node_hash]

    return list(trees_by_datapoint_id.values())
