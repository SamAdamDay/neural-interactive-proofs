"""Global state for runs.

Currently unused.
"""

from dataclasses import dataclass


@dataclass
class GlobalState:
    rl_iteration: int = 0
