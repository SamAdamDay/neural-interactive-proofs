"""Classes for training agents in an experiment.

A trainer takes the components of a scenario and trains the agents.
"""

from .base import Trainer, ReinforcementLearningTrainer
from .ppo import PpoTrainer
from .solo_agent import SoloAgentTrainer
