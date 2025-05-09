"""Utilities for reconstructing and visualising a forest of trees from MALT rollouts.

The MALT :cite:p:`Motwani2024` trainer samples a set of trees of responses. These are
stored in a flat array. This module provides functions to reconstruct the trees from
this flat array.
"""

from .forest import reconstruct_malt_forest, MaltNode
from .visualise import MaltForestVisualiser
