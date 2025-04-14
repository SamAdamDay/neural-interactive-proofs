"""Utilities for plotting verifier decisions."""

from math import ceil

import numpy as np
from numpy.typing import NDArray

from nip.protocols.verifier_decision_scale import build_verifier_decision_scale_handler
from nip.parameters import HyperParameters
from nip.utils.nested_array_dict import NestedArrayDict
from nip.utils.plotting.rollouts import get_last_timestep_mask


def get_decision_histogram(
    rollouts: NestedArrayDict,
    hyper_params: HyperParameters,
    bins: int | str = "auto",
    density: bool = False,
) -> tuple[NDArray, list[str]]:
    """Compute a histogram of the verifier decisions.

    Depending on the parameters of the experiment, the verifier can make a variety of
    decisions. This function computes a histogram of these decisions.

    Parameters
    ----------
    rollouts : NestedArrayDict
        The rollouts to be analysed. Each rollout is a NestedArrayDict containing the
        verifier decisions.
    hyper_params : HyperParameters
        The hyperparameters of the experiment. This is used to determine the decision
        scale used by the verifier.
    bins : int | str, default="auto"
        The number of bins to use for the histogram. This is passed to the
        :func:`numpy.histogram` function. See the documentation for more details.
    density : bool, default=False
        If True, the histogram is normalized to form a probability density.

    Returns
    -------
    histogram : NDArray
        The histogram of the verifier decisions. This is a 1D array of shape
        ``(n_bins,)``.
    bin_labels : list[str]
        The labels for the bins. This is a list of strings, where each string is the
        label for the corresponding bin in the histogram. If multiple values are grouped
        into a single bin, the labels are of the form "{first_value}-{last_value}". This
        is a list of length ``n_bins``.
    """

    verifier_decision_scale_handler = build_verifier_decision_scale_handler(
        hyper_params
    )

    possible_raw_decisions = verifier_decision_scale_handler.possible_decision_texts
    verifier_raw_decisions = rollouts["agents", "raw_decision"][
        get_last_timestep_mask(rollouts)
    ][:, -1]

    decision_indexes = np.array(
        [possible_raw_decisions.index(decision) for decision in verifier_raw_decisions]
    )

    histogram, bin_edges = np.histogram(
        decision_indexes,
        bins=bins,
        density=density,
    )

    # Create bin labels
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        min_index_in_bin = ceil(bin_edges[i])
        if i < len(bin_edges) - 2:
            max_index_in_bin = ceil(bin_edges[i + 1]) - 1
        else:
            max_index_in_bin = ceil(bin_edges[i + 1])
        if min_index_in_bin == max_index_in_bin:
            bin_labels.append(possible_raw_decisions[min_index_in_bin])
        else:
            bin_labels.append(
                f"{possible_raw_decisions[min_index_in_bin]}-"
                f"{possible_raw_decisions[max_index_in_bin]}"
            )

    return histogram, bin_labels
