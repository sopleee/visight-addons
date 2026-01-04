from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import TrainingConfig
from .data import (
    RLJudgeDataset,
    S3PredictionRepository,
    collate_batch,
    ensure_local_dataset,
)
from .features import load_brand_lookup
from .model import PolicyNetwork

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(policy: PolicyNetwork, loader: DataLoader, device: torch.device) -> dict[str, float]:
    policy.eval()
    total_abs_err = 0.0
    total_reward = 0.0
    total_count = 0
    with torch.no_grad():
        for features, targets, _ in loader:
            features = features.to(device)
            targets = targets.to(device)
            mean_score = policy.deterministic_score(features)
            reward = -((mean_score - targets) ** 2)
            total_abs_err += torch.abs(mean_score - targets).sum().item()
            total_reward += reward.sum().item()
            total_count += targets.numel()
    policy.train()
    if total_count == 0:
        return {"abs_error": 0.0, "reward": 0.0}
    return {
        "abs_error": total_abs_err / total_count,
        "reward": total_reward / total_count,
    }


def train_judge(config: TrainingConfig) -> None:
    device = torch.device(config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(config.seed)
    local_data_yaml, local_images_dir, local_labels_dir = ensure_local_dataset(
        s3_data_bucket=config.s3_data_bucket,
        s3_data_prefix=config.s3_data_prefix,
        data_cache_dir=config.data_cache_dir,
        s3_data_yaml_key=config.s3_data_yaml_key,
        force_download=config.force_download_dataset,
    )
    brand_lookup = load_brand_lookup(local_data_yaml)
    if not config.s3_bucket:
        raise ValueError("S3 bucket for predictions must be provided.")
    prediction_repo = S3PredictionRepository(config.s3_bucket, config.s3_prefix or "")
    dataset = RLJudgeDataset(
        images_dir=local_images_dir,
        labels_dir=local_labels_dir,
        brand_lookup=brand_lookup,
        prediction_repo=prediction_repo,
        max_samples=config.max_samples,
        debug_samples=config.debug_samples,
    )
    if len(dataset) == 0:
        raise RuntimeError("No training samples were constructed. Check S3 paths and predictions.")

    val_split = min(max(config.val_split, 0.0), 0.9)
    if val_split > 0.0 and len(dataset) > 1:
        val_size = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config.seed),
        )
    else:
        train_dataset = dataset
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)
        if val_dataset
        else None
    )
    policy = PolicyNetwork(input_dim=train_dataset[0][0].shape[0], hidden_dim=config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = config.log_dir / f"{config.run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = dict(config.__dict__)
    config_snapshot.update(
        {
            "local_data_yaml": str(local_data_yaml),
            "local_images_dir": str(local_images_dir),
            "local_labels_dir": str(local_labels_dir),
            "device_actual": str(device),
        }
    )
    (run_dir / "config.json").write_text(json.dumps(config_snapshot, default=str, indent=2))
    writer = SummaryWriter(log_dir=str(run_dir))
    running_baseline = 0.0
    global_step = 0
    for epoch in range(1, config.epochs + 1):
        epoch_reward = 0.0
        epoch_loss = 0.0
        epoch_abs_err = 0.0
        sample_count = 0
        for features, targets, meta in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            dist = policy(features)
            actions = dist.rsample()
            rewards = -((actions - targets) ** 2)
            running_baseline = (1.0 - config.baseline_beta) * running_baseline + config.baseline_beta * rewards.mean().item()
            advantages = rewards - running_baseline
            loss = -(advantages.detach() * dist.log_prob(actions)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * features.size(0)
            epoch_reward += rewards.mean().item() * features.size(0)
            epoch_abs_err += torch.abs(actions - targets).mean().item() * features.size(0)
            sample_count += features.size(0)
            writer.add_scalar("train/reward", rewards.mean().item(), global_step)
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/abs_error", torch.abs(actions - targets).mean().item(), global_step)
            writer.add_histogram("train/actions", actions, global_step)
            global_step += 1
        if sample_count:
            writer.add_scalar("epoch/reward", epoch_reward / sample_count, epoch)
            writer.add_scalar("epoch/loss", epoch_loss / sample_count, epoch)
            writer.add_scalar("epoch/abs_error", epoch_abs_err / sample_count, epoch)
        if val_loader:
            metrics = evaluate(policy, val_loader, device)
            writer.add_scalar("val/abs_error", metrics["abs_error"], epoch)
            writer.add_scalar("val/reward", metrics["reward"], epoch)
    writer.flush()
    writer.close()
