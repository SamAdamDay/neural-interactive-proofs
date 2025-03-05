"""Functions for generating a dataset of pairs of graphs with WL scores.

The main function is ``generate_gi_dataset``, which generates a dataset of pairs of
graphs with WL scores.

Graphs are generated using the Erdős-Rényi model. The dataset is generated in three
steps:

1. Generate ``prop_non_isomorphic`` non-isomorphic graphs. The pairs are divided equally
   between the different graph sizes and edge probabilities. The number of graphs with a
   score of 1, 2 and greater than 2 are divided according to the proportions
   ``non_iso_prop_score_1`` and ``non_iso_prop_score_2``.
2. Generate ``(1 - prop_non_isomorphic) * iso_prop_from_non_iso`` isomorphic graphs, by
   sampling from the non-isomorphic graph pairs and shuffling the nodes.
3. Generate the remaining ``(1 - prop_non_isomorphic) * (1 - iso_prop_from_non_iso)``
   isomorphic graphs, by generating new graphs and shuffling the nodes.
"""

import os
import itertools
from math import floor
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta

import torch
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.utils import dense_to_sparse

from jaxtyping import Int32

import einops

from tqdm import tqdm

from nip.constants import GI_DATA_DIR
from nip.utils.types import TorchDevice

from sympy import sieve


@dataclass
class GraphIsomorphicDatasetConfig:
    """Configuration for a graph isomorphism dataset.

    Graphs are generated using the Erdős-Rényi model. The dataset is generated in three
    steps:

    1. Generate ``prop_non_isomorphic`` non-isomorphic graphs. The pairs are divided
       equally between the different graph sizes and edge probabilities. The number of
       graphs with a score of 1, 2 and greater than 2 are divided according to the
       proportions ``non_iso_prop_score_1`` and ``non_iso_prop_score_2``.
    2. Generate ``(1 - prop_non_isomorphic) * iso_prop_from_non_iso`` isomorphic graphs,
       by sampling from the non-isomorphic graph pairs and shuffling the nodes.
    3. Generate the remaining ``(1 - prop_non_isomorphic) * (1 - iso_prop_from_non_iso)``
       isomorphic graphs, by generating new graphs and shuffling the nodes.

    Parameters
    ----------
    num_samples : int
        The number of graph pairs to generate.
    graph_sizes : list[int], default=[7, 8, 9, 10, 11]
        The sizes of the graphs to generate.
    edge_probabilities : list[float], default=[0.2, 0.4, 0.6, 0.8]
        The probabilities of an edge between two nodes.
    max_iterations : int, default=5
        The maximum number of iterations of the Weisfeiler-Lehman algorithm to run to
        compute the score.
    prop_non_isomorphic : float, default=0.5
        The proportion of graph pairs which are non-isomorphic.
    non_iso_prop_score_1 : float, default=0.1
        The proportion of non-isomorphic graph pairs which have a score of 1.
    non_iso_prop_score_2 : float, default=0.2
        The proportion of non-isomorphic graph pairs which have a score of 2.
    iso_prop_from_non_iso : float, default=0.5
        The proportion of graph pairs which are isomorphic and are generated from the
        non-isomorphic graph pairs.
    """

    num_samples: int
    graph_sizes: list[int] = field(default_factory=lambda: [7, 8, 9, 10, 11])
    edge_probabilities: list[float] = field(
        default_factory=lambda: [0.2, 0.4, 0.6, 0.8]
    )
    max_iterations: int = 5
    prop_non_isomorphic: float = 0.5
    non_iso_prop_score_1: float = 0.1
    non_iso_prop_score_2: float = 0.2
    iso_prop_from_non_iso: float = 0.5


def wl_score(
    adjacency_a: Int32[Tensor, "batch node1 node2"],
    adjacency_b: Int32[Tensor, "batch node1 node2"],
    max_iterations: int = 5,
    hash_size: int = 2**24 - 1,
    device: Optional[TorchDevice] = None,
) -> Int32[Tensor, "batch"]:
    """Compute the Weisfeiler-Lehman scores for a batch of graphs.

    The score is the number of rounds of the Weisfeiler-Lehman algorithm required to
    determine that the graphs are not isomorphic. If the graphs are isomorphic, the
    score is -1.

    Parameters
    ----------
    adjacency_a : Int32[Tensor, "batch node1 node2"]
        The batch of adjacency matrices for the first graph.
    adjacency_b : Int32[Tensor, "batch node1 node2"]
        The batch of adjacency matrices for the second graph.
    max_iterations : int, default=5
        The maximum number of iterations of the Weisfeiler-Lehman algorithm to run.
    hash_size : int, default=2**24 - 1
        The size of the hash table used to store the hashes of the graphs.
    device : TorchDevice, optional
        The device to use for the computation. If not given, defaults to the device of
        ``adjacency_a``.

    Returns
    -------
    scores : Int32[Tensor, "batch"]
        The Weisfeiler-Lehman scores for each graph in the batch.
    """

    if adjacency_a.shape != adjacency_b.shape:
        raise ValueError(
            f"Adjacency matrices must have the same shape, but got {adjacency_a.shape}"
            f" and {adjacency_b.shape}."
        )

    if device is None:
        device = adjacency_a.device

    batch_size = adjacency_a.shape[0]
    num_nodes = adjacency_a.shape[1]

    # Generate primes for hashing
    primes = torch.tensor(sieve.primerange(hash_size), device=device)

    # Initialize labels and scores
    scores = torch.ones(batch_size, dtype=torch.int32, device=device) * -1
    labels = torch.ones((2, batch_size, num_nodes), dtype=torch.long, device=device)

    # Combine adjacency matrices and add self-loops
    # (graph, batch, node, node)
    adjacency_combined = torch.stack((adjacency_a, adjacency_b), dim=0)
    adjacency_combined += torch.eye(
        num_nodes, dtype=adjacency_combined.dtype, device=device
    )

    for i in range(max_iterations):
        # Take the hash of the labels for the neighbours of each node
        labels_repeated = einops.repeat(
            labels, "graph batch node1 -> graph batch node2 node1", node2=num_nodes
        )
        labels_neighbours = labels_repeated * adjacency_combined
        labels_neighbours = primes[labels_neighbours]
        labels_neighbours = einops.reduce(
            labels_neighbours, "graph batch node1 node2 -> graph batch node1", "prod"
        )
        labels = torch.remainder(labels_neighbours, hash_size)

        # Compute the hash of the graph
        graph_hashes = einops.reduce(
            primes[labels], "graph batch node -> graph batch", "prod"
        )

        # Update the scores if the hashes are different
        diff = graph_hashes[0] != graph_hashes[1]
        scores = torch.where(torch.logical_and(scores == -1, diff), i + 1, scores)

    return scores


def generate_er_graphs(
    num_graphs: int,
    graph_size: int,
    edge_probability: float,
    device: TorchDevice = "cpu",
) -> Int32[Tensor, "batch node1 node2"]:
    """Generate a batch of Erdős-Rényi graphs.

    Parameters
    ----------
    num_graphs : int
        The number of graphs to generate.
    graph_size : int
        The number of nodes in each graph.
    edge_probability : float
        The probability of an edge between two nodes.
    device : TorchDevice, default="cpu"
        The device to use for the computation.

    Returns
    -------
    adjacency : Int32[Tensor, "batch node1 node2"]
        The batch of adjacency matrices for the generated graphs.
    """
    adjacency_values = torch.rand(num_graphs, graph_size, graph_size, device=device)
    adjacency = (adjacency_values < edge_probability).to(torch.int32)
    adjacency = adjacency.triu(diagonal=1)
    adjacency += adjacency.transpose(1, 2).clone()
    return adjacency


def shuffle_adjacencies(
    adjacencies: Int32[Tensor, "... node1 node2"], sizes: Int32[Tensor, "..."]
) -> Int32[Tensor, "... node1 node2"]:
    """Shuffle the nodes in a batch of adjacency matrices of varying sizes.

    Allows for arbitrary batch dimensions.

    Parameters
    ----------
    adjacencies : Int32[Tensor, "... node1 node2"]
        The batch of adjacency matrices to shuffle.
    sizes : Int32[Tensor, "..."]
        The number of nodes in each graph.

    Returns
    -------
    adjacencies : Int32[Tensor, "... node1 node2"]
        The batch of adjacency matrices with the nodes shuffled.
    """
    if adjacencies.shape[:-2] != sizes.shape:
        raise ValueError(
            f"The batch dimensions of the adjacency matrices {adjacencies.shape[:-2]}"
            f" and the sizes {sizes.shape} must be the same."
        )

    # Compute a mask on the nodes based on the sizes of the graphs
    expanded_arange = torch.arange(adjacencies.shape[-1]).expand(sizes.shape + (-1,))
    node_mask = expanded_arange < sizes[..., None]

    # Generate a random permutation up to the graph sizes, by generating random numbers
    # and sorting them. For node indices above the corresponding graph size, we use the
    # identity permutation.
    node_permutation_generator = torch.where(
        node_mask,
        torch.rand(adjacencies.shape[:-1]) - 1,
        torch.arange(adjacencies.shape[-1]),
    )
    node_permutation = torch.argsort(node_permutation_generator)

    # Compute the permutations for the adjacency matrices
    node1_permutation = einops.repeat(
        node_permutation,
        "... node1 -> ... node1 node2",
        node2=adjacencies.shape[-1],
    )
    node2_permutation = einops.repeat(
        node_permutation,
        "... node2 -> ... node1 node2",
        node1=adjacencies.shape[-1],
    )

    # Permute the adjacency matrices
    adjacencies = torch.gather(adjacencies, -2, node1_permutation)
    adjacencies = torch.gather(adjacencies, -1, node2_permutation)

    return adjacencies


def _generate_non_isomorphic_graphs(
    config: GraphIsomorphicDatasetConfig, batch_size: int, device: TorchDevice
) -> tuple[
    Int32[Tensor, "pair batch node1 node2"],
    Int32[Tensor, "batch"],
    Int32[Tensor, "batch"],
]:
    """Generate non-isomorphic graphs.

    Parameters
    ----------
    config : GraphIsomorphicDatasetConfig
        The configuration for the dataset.
    batch_size : int
        The number of pairs of graphs to generate.
    device : TorchDevice
        The device to use for the computation. Note that all returned tensors will be
        on the CPU.

    Returns
    -------
    adjacencies : Int32[Tensor, "pair batch node1 node2"], device="cpu"
        The batch of adjacency matrices for the generated graphs.
    sizes : Int32[Tensor, "batch"], device="cpu"
        The number of nodes in each graph.
    scores : Int32[Tensor, "batch"], device="cpu"
        The Weisfeiler-Lehman scores for each pair of graphs.
    """
    # Compute the number of pairs of graphs for each class to generate
    num_non_iso_per = (
        config.num_samples
        * config.prop_non_isomorphic
        / (len(config.graph_sizes * len(config.edge_probabilities)))
    )
    num_non_iso_score_per = {}
    num_non_iso_score_per["1"] = floor(num_non_iso_per * config.non_iso_prop_score_1)
    num_non_iso_score_per["2"] = floor(num_non_iso_per * config.non_iso_prop_score_2)
    num_non_iso_score_per["gt_2"] = (
        floor(num_non_iso_per) - num_non_iso_score_per["1"] - num_non_iso_score_per["2"]
    )

    adjacencies_list = []
    sizes_list = []
    scores_list = []

    num_configs = len(config.graph_sizes) * len(config.edge_probabilities)
    max_graph_size = max(config.graph_sizes)

    iterator = itertools.product(config.graph_sizes, config.edge_probabilities)
    for i, (graph_size, edge_probability) in enumerate(iterator):
        # The number of graphs generated for each score
        score_counts = {key: 0 for key in num_non_iso_score_per.keys()}

        progress_bar = tqdm(
            total=sum(num_non_iso_score_per.values()),
            desc=f"[{i+1}/{num_configs+2}] Non-isomorphic (n={graph_size}, p={edge_probability})",
        )

        while any(
            count < num
            for count, num in zip(score_counts.values(), num_non_iso_score_per.values())
        ):
            # Generate a batch of graphs
            adjacency_1 = generate_er_graphs(
                batch_size, graph_size, edge_probability, device=device
            )
            adjacency_2 = generate_er_graphs(
                batch_size, graph_size, edge_probability, device=device
            )

            # Compute the Weisfeiler-Lehman scores
            score = wl_score(
                adjacency_1,
                adjacency_2,
                max_iterations=config.max_iterations,
                device=device,
            )

            # Pad the adjacency matrices to the maximum graph size
            adjacency_1 = F.pad(adjacency_1, (0, max_graph_size - graph_size) * 2)
            adjacency_2 = F.pad(adjacency_2, (0, max_graph_size - graph_size) * 2)

            # Add graphs according to their scores
            for key in score_counts.keys():
                # Select the graphs with the correct score
                if key == "1":
                    index = score == 1
                elif key == "2":
                    index = score == 2
                elif key == "gt_2":
                    index = score > 2

                # Only add graphs if there are enough left to add
                num_pairs_to_add = min(
                    num_non_iso_score_per[key] - score_counts[key],
                    torch.sum(index).item(),
                )

                # Add the graphs
                adjacencies_list.append(
                    torch.stack(
                        (
                            adjacency_1[index][:num_pairs_to_add],
                            adjacency_2[index][:num_pairs_to_add],
                        )
                    ).cpu()
                )
                sizes_list.append(
                    torch.ones(num_pairs_to_add, dtype=torch.int32) * graph_size
                )
                scores_list.append(score[index][:num_pairs_to_add].cpu())

                # Update
                score_counts[key] += num_pairs_to_add
                progress_bar.update(num_pairs_to_add)

        progress_bar.close()

    # Combine the lists into tensors
    adjacencies = torch.cat(adjacencies_list, dim=1)
    sizes = torch.cat(sizes_list, dim=0)
    scores = torch.cat(scores_list, dim=0)

    return adjacencies, sizes, scores


def _generate_isomorphic_graphs(
    config: GraphIsomorphicDatasetConfig,
    non_iso_adjacencies: Int32[Tensor, "pair batch node1 node2"],
    non_iso_sizes: Int32[Tensor, "batch"],
) -> tuple[Int32[Tensor, "pair batch node1 node2"], Int32[Tensor, "batch"]]:
    """Generate isomorphic graphs.

    Parameters
    ----------
    config : GraphIsomorphicDatasetConfig
        The configuration for the dataset.
    non_iso_adjacencies : Int32[Tensor, "pair batch node1 node2"], device="cpu"
        The batch of adjacency matrices for the non-isomorphic graph pairs.
    non_iso_sizes : Int32[Tensor, "batch"], device="cpu"
        The number of nodes in each non-isomorphic graph pair.

    Returns
    -------
    adjacencies : Int32[Tensor, "pair batch node1 node2"], device="cpu"
        The batch of adjacency matrices for the generated graphs.
    sizes : Int32[Tensor, "batch"], device="cpu"
        The number of nodes in each graph.
    """
    # Compute the number of graphs to generate from the non-isomorphic graphs
    num_from_non_iso = floor(
        config.num_samples
        * (1 - config.prop_non_isomorphic)
        * config.iso_prop_from_non_iso
    )

    num_configs = len(config.graph_sizes) * len(config.edge_probabilities)
    max_graph_size = max(config.graph_sizes)

    # Select pairs of graphs from the non-isomorphic graphs
    print(  # noqa: T201
        f"[{num_configs+1}/{num_configs+2}] Isomorphic from non-isomorphic"
    )
    batch_index = torch.randint(0, non_iso_adjacencies.shape[1], (num_from_non_iso,))
    pair_index = torch.randint(0, 2, (num_from_non_iso,))
    adjacencies_from_non_iso = einops.repeat(
        non_iso_adjacencies[pair_index, batch_index],
        "pair node1 node2 -> 2 pair node1 node2",
    )
    sizes_from_non_iso = non_iso_sizes[batch_index]

    # Compute the number of new graphs to generate, which is the remainder after the
    # rest of the graphs have been generated
    num_new = (
        config.num_samples
        - non_iso_adjacencies.shape[1]
        - adjacencies_from_non_iso.shape[1]
    )
    num_new_per = floor(
        num_new / (len(config.graph_sizes * len(config.edge_probabilities)))
    )

    # Generate new graphs
    print(f"[{num_configs+2}/{num_configs+2}] Isomorphic new")  # noqa: T201
    adjacencies_new_list = []
    sizes_new_list = []
    for i, (graph_size, edge_probability) in enumerate(
        itertools.product(config.graph_sizes, config.edge_probabilities)
    ):
        if i < num_configs - 1:
            num_new_this = num_new_per
        else:
            num_new_this = num_new - num_new_per * (num_configs - 1)
        adjacency = generate_er_graphs(num_new_this, graph_size, edge_probability)
        adjacency = F.pad(adjacency, (0, max_graph_size - graph_size) * 2)
        adjacencies_new_list.append(
            einops.repeat(adjacency, "batch node1 node2 -> 2 batch node1 node2")
        )
        sizes_new_list.append(torch.ones(num_new_this, dtype=torch.int32) * graph_size)

    # Combine the lists into tensors
    adjacencies = torch.cat((adjacencies_from_non_iso, *adjacencies_new_list), dim=1)
    sizes = torch.cat((sizes_from_non_iso, *sizes_new_list), dim=0)

    # Shuffle the nodes in each graph
    adjacencies = shuffle_adjacencies(adjacencies, sizes.expand(2, -1))

    return adjacencies, sizes


def generate_gi_dataset(
    config: GraphIsomorphicDatasetConfig | dict,
    name: str,
    batch_size: int = 800000,
    split_name: str = "train",
    device: TorchDevice = "cpu",
):
    """Generate a dataset of pairs of graphs with WL scores.

    Graphs are generated using the Erdős-Rényi model. The dataset is generated in three
    steps:

    1. Generate non-isomorphic graphs. The pairs are divided equally between the
       different graph sizes and edge probabilities. The number of graphs with a score
       of 1, 2 and greater than 2 are divided according to the proportions
       ``non_iso_prop_score_1`` and ``non_iso_prop_score_2``.
    2. Generate isomorphic graphs by sampling from the non-isomorphic graph pairs and
       shuffling the nodes.
    3. Generate new isomorphic graphs.

    Parameters
    ----------
    config : GraphIsomorphicDatasetConfig or dict
        The configuration for the dataset.
    name : str
        The the dataset to save. This will be the name of the directory in which the
        dataset is saved, under ``nip.constants.GI_DATA_DIR``.
    batch_size : int, default=1000000
        The batch size to use when generating the graphs.
    split_name : str, default="train"
        The name of the split to save the dataset as.
    device : TorchDevice, default="cpu"
        The device to use for the computation.
    """
    start_time = datetime.now()

    if isinstance(config, dict):
        config = GraphIsomorphicDatasetConfig(**config)

    # Generate non-isomorphic graphs
    (
        adjacencies_non_iso,
        sizes_non_iso,
        scores_non_iso,
    ) = _generate_non_isomorphic_graphs(config, batch_size, device)

    # Generate isomorphic graphs
    adjacencies_iso, sizes_iso = _generate_isomorphic_graphs(
        config, adjacencies_non_iso, sizes_non_iso
    )

    # Combine the graphs
    adjacencies = torch.cat((adjacencies_non_iso, adjacencies_iso), dim=1)
    sizes = torch.cat((sizes_non_iso, sizes_iso), dim=0)
    scores = torch.cat(
        (scores_non_iso, -1 * torch.ones(sizes_iso.shape[0], dtype=torch.int32))
    )

    # Turn the adjacency matrices into edge indices
    indices = torch.arange(1, adjacencies.shape[1] + 1)
    indices = einops.rearrange(indices, "batch -> () batch () ()")
    adjacencies_indexed = adjacencies * indices
    edge_indices_a, batch_a = dense_to_sparse(adjacencies_indexed[0])
    edge_indices_b, batch_b = dense_to_sparse(adjacencies_indexed[1])
    max_sizes_cumsum = torch.arange(adjacencies.shape[1]) * adjacencies.shape[2]
    to_subtract_a = max_sizes_cumsum[batch_a - 1]
    to_subtract_b = max_sizes_cumsum[batch_b - 1]
    edge_indices_a -= to_subtract_a[None, :]
    edge_indices_b -= to_subtract_b[None, :]
    slices_a = torch.cumsum(torch.bincount(batch_a), 0)
    slices_b = torch.cumsum(torch.bincount(batch_b), 0)

    # Save the dataset
    data = dict(
        edge_index_a=edge_indices_a,
        edge_index_b=edge_indices_b,
        slices_a=slices_a,
        slices_b=slices_b,
        wl_scores=scores,
        sizes_a=sizes,
        sizes_b=sizes,
    )
    data_dir = os.path.join(GI_DATA_DIR, name, "raw", split_name)
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    dataset_filename = os.path.join(data_dir, "data.pt")
    torch.save(data, dataset_filename)

    # Calculate the elapsed time, rounding microseconds down
    elapsed_time = datetime.now() - start_time
    elapsed_time = timedelta(days=elapsed_time.days, seconds=elapsed_time.seconds)

    print(f"Done in {elapsed_time}")  # noqa: T201
