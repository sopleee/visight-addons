from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml


@dataclass
class BoundingBox:
    brand: str
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(self.x2 - self.x1, 0.0)

    @property
    def height(self) -> float:
        return max(self.y2 - self.y1, 0.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)


def load_brand_lookup(data_yaml: Path) -> Dict[int, str]:
    data = yaml.safe_load(data_yaml.read_text())
    names = data.get("names", [])
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return {idx: str(name) for idx, name in enumerate(names)}


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_yolo_labels(label_path: Path, brand_lookup: Dict[int, str]) -> List[BoundingBox]:
    boxes: List[BoundingBox] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        if not line:
            continue
        parts = line.replace("\t", " ").split()
        if len(parts) < 5:
            continue
        cls_idx = int(float(parts[0]))
        x_c, y_c, w, h = (float(x) for x in parts[1:5])
        brand = brand_lookup.get(cls_idx, f"class_{cls_idx}")
        x1 = max(x_c - w * 0.5, 0.0)
        y1 = max(y_c - h * 0.5, 0.0)
        x2 = min(x_c + w * 0.5, 1.0)
        y2 = min(y_c + h * 0.5, 1.0)
        boxes.append(BoundingBox(brand=brand, x1=x1, y1=y1, x2=x2, y2=y2))
    return boxes


def _strip_markdown_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped[3:]
        newline_idx = stripped.find("\n")
        if newline_idx != -1:
            stripped = stripped[newline_idx + 1 :]
        stripped = stripped.strip()
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


def parse_qwen_predictions(raw_prediction: str, image_w: int, image_h: int) -> List[BoundingBox]:
    cleaned_prediction = _strip_markdown_code_fences(raw_prediction)
    try:
        payload = json.loads(cleaned_prediction)
    except json.JSONDecodeError:
        return []
    boxes: List[BoundingBox] = []
    for entry in payload:
        brand = entry.get("brand_name", "unknown")
        raw_locations = entry.get("bbox_locations", [])
        if isinstance(raw_locations, (list, tuple)) and raw_locations and isinstance(raw_locations[0], (int, float)):
            coord_sequences = [raw_locations]
        else:
            coord_sequences = raw_locations
        for coords in coord_sequences:
            if not isinstance(coords, (list, tuple)) or len(coords) != 4:
                continue
            x1, y1, x2, y2 = coords
            if image_w <= 0 or image_h <= 0:
                continue
            nx1 = max(min(x1 / image_w, 1.0), 0.0)
            ny1 = max(min(y1 / image_h, 1.0), 0.0)
            nx2 = max(min(x2 / image_w, 1.0), 0.0)
            ny2 = max(min(y2 / image_h, 1.0), 0.0)
            if nx2 <= nx1 or ny2 <= ny1:
                continue
            boxes.append(BoundingBox(brand=brand, x1=nx1, y1=ny1, x2=nx2, y2=ny2))
    return boxes


def compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    ix1 = max(box_a.x1, box_b.x1)
    iy1 = max(box_a.y1, box_b.y1)
    ix2 = min(box_a.x2, box_b.x2)
    iy2 = min(box_a.y2, box_b.y2)
    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = box_a.area + box_b.area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def aggregate_iou(pred_boxes: Sequence[BoundingBox], gt_boxes: Sequence[BoundingBox]) -> float:
    if not gt_boxes:
        return 1.0 if not pred_boxes else 0.0
    matched_gt: set[int] = set()
    iou_sum = 0.0
    for pred in sorted(pred_boxes, key=lambda b: b.area, reverse=True):
        best_idx = -1
        best_iou = 0.0
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            if pred.brand != gt.brand:
                continue
            current_iou = compute_iou(pred, gt)
            if current_iou > best_iou:
                best_iou = current_iou
                best_idx = idx
        if best_idx >= 0:
            matched_gt.add(best_idx)
            iou_sum += best_iou
    return iou_sum / len(gt_boxes)


def safe_stats(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = float(sum(values) / len(values))
    if len(values) == 1:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(max(variance, 0.0))


def brand_entropy(brands: Sequence[str]) -> float:
    if not brands:
        return 0.0
    total = len(brands)
    counts: Dict[str, int] = {}
    for brand in brands:
        counts[brand] = counts.get(brand, 0) + 1
    probs = [count / total for count in counts.values()]
    entropy = -sum(p * math.log(p + 1e-8) for p in probs)
    max_entropy = math.log(len(counts)) if counts else 1.0
    if max_entropy <= 0.0:
        return 0.0
    return entropy / max_entropy


def extract_features(pred_boxes: Sequence[BoundingBox]) -> List[float]:
    count = len(pred_boxes)
    if not pred_boxes:
        return [
            0.0,  # num predictions
            0.0,  # unique brands
            0.0,  # mean area
            0.0,  # std area
            0.0,  # mean aspect ratio
            0.0,  # std aspect ratio
            0.0,  # mean center x
            0.0,  # std center x
            0.0,  # mean center y
            0.0,  # std center y
            0.0,  # pairwise overlap
            0.0,  # brand entropy
        ]
    areas = [box.area for box in pred_boxes]
    widths = [box.width for box in pred_boxes]
    heights = [box.height for box in pred_boxes]
    aspect_ratios = [w / (h + 1e-6) for w, h in zip(widths, heights)]
    centers = [box.center for box in pred_boxes]
    centers_x = [c[0] for c in centers]
    centers_y = [c[1] for c in centers]
    mean_area, std_area = safe_stats(areas)
    mean_ar, std_ar = safe_stats(aspect_ratios)
    mean_cx, std_cx = safe_stats(centers_x)
    mean_cy, std_cy = safe_stats(centers_y)
    overlap_pairs = 0
    overlapping = 0
    for idx in range(count):
        for jdx in range(idx + 1, count):
            overlap_pairs += 1
            if compute_iou(pred_boxes[idx], pred_boxes[jdx]) > 0.1:
                overlapping += 1
    overlap_ratio = overlapping / overlap_pairs if overlap_pairs else 0.0
    entropy = brand_entropy([box.brand for box in pred_boxes])
    return [
        float(count),
        float(len({box.brand for box in pred_boxes})),
        mean_area,
        std_area,
        mean_ar,
        std_ar,
        mean_cx,
        std_cx,
        mean_cy,
        std_cy,
        overlap_ratio,
        entropy,
    ]
