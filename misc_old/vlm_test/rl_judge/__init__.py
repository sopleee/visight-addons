"""
rl_judge package
================

Tools for training a reinforcement learning judge that scores Qwen logo
detections using an IoU-correlated reward.
"""

from .config import TrainingConfig
from .train import train_judge

__all__ = ["TrainingConfig", "train_judge"]
