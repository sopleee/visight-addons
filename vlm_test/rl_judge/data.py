from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

from vlm_test.s3client import s3Client

from .features import aggregate_iou, extract_features, find_image, parse_qwen_predictions, parse_yolo_labels

logger = logging.getLogger(__name__)


class S3PredictionRepository:
    def __init__(self, bucket: str, prefix: str):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = s3Client(buckets=[bucket])

    def _make_key(self, stem: str) -> str:
        if self.prefix:
            return f"{self.prefix}/{stem}.txt"
        return f"{stem}.txt"

    def list_available_ids(self) -> List[str]:
        prefix = self.prefix
        keys = (
            self.client.batch_get_filenames(prefix=prefix)
            if prefix
            else self.client.batch_get_filenames(prefix="")
        )
        stems: List[str] = []
        for key in keys:
            stripped = key.split("/", 1)[1] if "/" in key else key
            stems.append(Path(stripped).stem)
        return stems

    def load_prediction(self, stem: str) -> Optional[str]:
        key = self._make_key(stem)
        try:
            data = self.client.get_object(key, parse_format="txt")
        except Exception:
            return None
        return data


class JudgeSample(Sequence[torch.Tensor]):
    def __init__(self, features: torch.Tensor, target: torch.Tensor, meta: Dict[str, object]):
        self.features = features
        self.target = target
        self.meta = meta

    def __getitem__(self, item):
        if item == 0:
            return self.features
        if item == 1:
            return self.target
        if item == 2:
            return self.meta
        raise IndexError

    def __len__(self) -> int:
        return 3


class RLJudgeDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        brand_lookup: Dict[int, str],
        prediction_repo: S3PredictionRepository,
        max_samples: Optional[int] = None,
        debug_samples: int = 0,
    ):
        self.samples: List[JudgeSample] = []
        available_predictions = set(prediction_repo.list_available_ids())
        label_files = sorted(labels_dir.glob("*.txt"))
        remaining_debug_logs = max(debug_samples, 0)
        for label_path in label_files:
            stem = label_path.stem
            if available_predictions and stem not in available_predictions:
                continue
            image_path = find_image(images_dir, stem)
            if image_path is None:
                continue
            raw_prediction = prediction_repo.load_prediction(stem)
            if raw_prediction is None:
                continue
            with Image.open(image_path) as img:
                width, height = img.size
            pred_boxes = parse_qwen_predictions(raw_prediction, width, height)
            gt_boxes = parse_yolo_labels(label_path, brand_lookup)
            features = torch.tensor(extract_features(pred_boxes), dtype=torch.float32)
            iou = aggregate_iou(pred_boxes, gt_boxes)
            target = torch.tensor(iou, dtype=torch.float32)
            meta = {
                "image_id": stem,
                "num_predictions": len(pred_boxes),
                "num_ground_truth": len(gt_boxes),
                "image_path": str(image_path),
            }
            self.samples.append(JudgeSample(features=features, target=target, meta=meta))
            if remaining_debug_logs > 0:
                logger.info(
                    "Parsed Qwen prediction for %s | boxes=%d | brands=%s | sample_iou=%.4f",
                    stem,
                    len(pred_boxes),
                    [box.brand for box in pred_boxes[:5]],
                    float(iou),
                )
                logger.debug("Raw Qwen response (%s): %s", stem, raw_prediction[:500])
                remaining_debug_logs -= 1
            if max_samples and len(self.samples) >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> JudgeSample:
        return self.samples[index]


def collate_batch(batch: Sequence[JudgeSample]):
    features = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    meta = [item[2] for item in batch]
    return features, targets, meta


def _strip_bucket_prefix(full_path: str, bucket: str) -> str:
    prefix = f"{bucket}/"
    if full_path.startswith(prefix):
        return full_path[len(prefix) :]
    return full_path


def _download_s3_directory(client: s3Client, prefix: str, local_dir: Path, skip_existing: bool = True) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    keys = client.batch_get_filenames(prefix=prefix)
    for entry in keys:
        object_key = _strip_bucket_prefix(entry, client.bucket)
        if not object_key.startswith(prefix):
            continue
        try:
            relative_key = Path(object_key).relative_to(prefix)
        except ValueError:
            continue
        destination = local_dir / relative_key
        if skip_existing and destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(object_key, destination)


def ensure_local_dataset(
    s3_data_bucket: str,
    s3_data_prefix: str,
    data_cache_dir: Path,
    s3_data_yaml_key: Optional[str] = None,
    force_download: bool = False,
) -> Tuple[Path, Path, Path]:
    cache_root = data_cache_dir
    cache_root.mkdir(parents=True, exist_ok=True)
    data_yaml = cache_root / "data.yaml"
    images_dir = cache_root / "images"
    labels_dir = cache_root / "labels"
    dataset_ready = (
        not force_download
        and data_yaml.exists()
        and images_dir.exists()
        and labels_dir.exists()
    )
    if dataset_ready:
        return data_yaml, images_dir, labels_dir
    client = s3Client(buckets=[s3_data_bucket])
    base_prefix = s3_data_prefix.strip("/")

    def _within_cache(path: Path) -> bool:
        try:
            return cache_root.resolve() in path.resolve().parents or path.resolve() == cache_root.resolve()
        except FileNotFoundError:
            return True

    if force_download:
        for path in (images_dir, labels_dir):
            if path.exists() and _within_cache(path):
                shutil.rmtree(path, ignore_errors=True)
        if data_yaml.exists() and _within_cache(data_yaml):
            data_yaml.unlink()

    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    yaml_key = s3_data_yaml_key or f"{base_prefix}/data.yaml"
    if force_download or not data_yaml.exists():
        try:
            client.download_file(yaml_key, data_yaml)
        except Exception as exc:
            raise FileNotFoundError(
                f"Failed to download data.yaml from s3://{s3_data_bucket}/{yaml_key}. "
                "Use --s3-data-yaml-key to point at the correct object."
            ) from exc

    _download_s3_directory(
        client,
        prefix=f"{base_prefix}/images",
        local_dir=images_dir,
        skip_existing=not force_download,
    )
    _download_s3_directory(
        client,
        prefix=f"{base_prefix}/labels",
        local_dir=labels_dir,
        skip_existing=not force_download,
    )
    return data_yaml, images_dir, labels_dir
