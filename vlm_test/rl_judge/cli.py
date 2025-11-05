from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import TrainingConfig
from .train import train_judge

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a reinforcement learning judge to score Qwen logo detections.")
    parser.add_argument("--s3-bucket", type=str, required=True, help="S3 bucket containing Qwen prediction txt outputs.")
    parser.add_argument("--s3-prefix", type=str, default="", help="Prefix inside the S3 bucket where predictions live.")
    parser.add_argument("--s3-data-bucket", type=str, required=True, help="S3 bucket storing data.yaml, images/, and labels/ for the dataset.")
    parser.add_argument("--s3-data-prefix", type=str, required=True, help="Prefix inside the S3 bucket pointing to the dataset root (contains images/ and labels/).")
    parser.add_argument("--s3-data-yaml-key", type=str, help="Optional explicit s3 key for data.yaml when it lives outside the prefix.")
    parser.add_argument(
        "--data-cache-dir",
        type=Path,
        default=Path(".cache/rl_judge_dataset"),
        help="Local directory used to cache dataset artifacts downloaded from S3.",
    )
    parser.add_argument(
        "--force-download-dataset",
        action="store_true",
        help="Redownload dataset assets from S3 even if the cache already contains them.",
    )
    parser.add_argument("--log-dir", type=Path, default=Path("runs/rl_judge"), help="Where to write TensorBoard events.")
    parser.add_argument("--run-name", type=str, default="rl_judge", help="Name to label this training run.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for policy optimisation.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension for the policy network.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of samples reserved for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="PyTorch device identifier or 'auto'.")
    parser.add_argument("--max-samples", type=int, help="Optional cap on number of samples to load.")
    parser.add_argument("--baseline-beta", type=float, default=0.1, help="EMA factor for the policy baseline.")
    parser.add_argument("--debug-samples", type=int, default=3, help="Number of parsed prediction examples to log.")
    return parser


def parse_args(argv: list[str] | None = None) -> TrainingConfig:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return TrainingConfig(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        s3_data_bucket=args.s3_data_bucket,
        s3_data_prefix=args.s3_data_prefix,
        s3_data_yaml_key=args.s3_data_yaml_key,
        data_cache_dir=args.data_cache_dir,
        force_download_dataset=args.force_download_dataset,
        log_dir=args.log_dir,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        val_split=args.val_split,
        seed=args.seed,
        device=args.device,
        max_samples=args.max_samples,
        baseline_beta=args.baseline_beta,
        debug_samples=args.debug_samples,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    config = parse_args(argv)
    train_judge(config)


if __name__ == "__main__":
    main()
