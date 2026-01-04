from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    s3_bucket: str
    s3_prefix: str
    s3_data_bucket: str
    s3_data_prefix: str
    s3_data_yaml_key: Optional[str]
    data_cache_dir: Path
    force_download_dataset: bool
    log_dir: Path
    run_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    hidden_dim: int
    val_split: float
    seed: int
    device: str
    max_samples: Optional[int]
    baseline_beta: float
    debug_samples: int
