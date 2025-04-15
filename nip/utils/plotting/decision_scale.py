"""Utilities for plotting verifier decisions."""

from typing import Literal
from math import ceil

import numpy as np
from numpy.typing import NDArray

import pandas as pd

from nip.protocols.verifier_decision_scale import build_verifier_decision_scale_handler
from nip.parameters import HyperParameters
from nip.utils.nested_array_dict import NestedArrayDict
from nip.utils.plotting.rollouts import get_last_timestep_mask


def get_decision_histogram(
    rollouts: NestedArrayDict,
    hyper_params: HyperParameters,
    bins: int | str = "auto",
    include_no_decision: Literal["yes", "no", "if_nonzero"] = "no",
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
    include_no_decision : Literal["yes", "no", "if_nonzero"], default="no"
        If "yes", the histogram includes the "no_decision" option, for when the verifier
        does not make a decision by the end of the rollout. This is added at the end of
        the histogram with label "no_decision". If "no", these rollouts are ignored. If
        "if_nonzero", the histogram includes the "no_decision" option only if there are
        any rollouts with no decisions.

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

    # Get the indexes of the verifier decisions, and count the number of no decisions if
    # include_no_decision is True
    decision_indexes = []
    num_no_decision = 0
    for decision in verifier_raw_decisions:
        if decision in possible_raw_decisions:
            decision_indexes.append(possible_raw_decisions.index(decision))
        elif include_no_decision == "yes" or include_no_decision == "if_nonzero":
            num_no_decision += 1
    decision_indexes = np.array(decision_indexes)

    histogram, bin_edges = np.histogram(
        decision_indexes, bins=bins, range=(0, len(possible_raw_decisions) - 1)
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

    # Add the no decision count to the histogram if include_no_decision is True
    if include_no_decision == "yes" or (
        include_no_decision == "if_nonzero" and num_no_decision > 0
    ):
        histogram = np.append(histogram, num_no_decision)
        bin_labels.append("No decision")

    return histogram, bin_labels


def get_thresholded_performance(
    rollouts: NestedArrayDict,
    hyper_params: HyperParameters,
) -> pd.DataFrame:
    """Compute the performance of the verifier at different thresholds.

    When the verifier outputs a decision on a scale, we can threshold it to get a binary
    decision at different levels. This function computes the performance of the verifier
    at different thresholds.

    Parameters
    ----------
    rollouts : NestedArrayDict
        The rollouts to be analysed. Each rollout is a NestedArrayDict containing the
        verifier decisions.
    hyper_params : HyperParameters
        The hyperparameters of the experiment. This is used to determine the decision
        scale used by the verifier.

    Returns
    -------
    performance : pd.DataFrame
        The performance of the verifier at different thresholds. This is a pandas
        DataFrame with the following columns:

        - "threshold_text": The text value of the threshold used to compute the
          performance.
        - "threshold_float": The threshold used to compute the performance as a float
          between -1 and 1.
        - "accuracy": The accuracy of the verifier at this threshold.
        - "true_positive_rate": The true positive rate at this threshold.
        - "false_positive_rate": The false positive rate at this threshold.
        - "true_negative_rate": The true negative rate at this threshold.
        - "false_negative_rate": The false negative rate at this threshold.
        - "precision": The precision of the verifier at this threshold.
    """

    verifier_decision_scale_handler = build_verifier_decision_scale_handler(
        hyper_params
    )

    performance = pd.DataFrame(
        columns=[
            "threshold_text",
            "threshold_float",
            "accuracy",
            "true_positive_rate",
            "false_positive_rate",
            "true_negative_rate",
            "false_negative_rate",
            "precision",
        ]
    )

    verifier_continuous_decision = rollouts["agents", "continuous_decision"][
        get_last_timestep_mask(rollouts)
    ][:, -1]
    y = rollouts["y"][:, 0]

    decision_texts_and_outcomes = (
        verifier_decision_scale_handler.decision_texts_and_outcomes
        + [("<infinity>", 1, float("inf"))]
    )

    for i, (threshold_text, _, threshold_float) in enumerate(
        decision_texts_and_outcomes
    ):
        thresholded_guess = verifier_continuous_decision >= threshold_float
        correct_guess = thresholded_guess == y
        true_positive_rate = np.mean(correct_guess[y == 1])
        true_negative_rate = np.mean(correct_guess[y == 0])
        if np.any(thresholded_guess):
            precision = np.mean(correct_guess[thresholded_guess])
        else:
            precision = 1.0
        performance.loc[i] = {
            "threshold_text": threshold_text,
            "threshold_float": threshold_float,
            "accuracy": np.mean(correct_guess),
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": 1 - true_negative_rate,
            "true_negative_rate": true_negative_rate,
            "false_negative_rate": 1 - true_positive_rate,
            "precision": precision,
        }

    return performance
