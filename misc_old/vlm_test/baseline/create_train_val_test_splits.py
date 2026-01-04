import json
from pathlib import Path
from typing import List, Dict, Tuple
import random
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file into list of dicts"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Save list of dicts to JSONL file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    logger.info(f"✓ Saved {len(data)} samples to {file_path}")


def stratified_split(
    data: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create stratified train/val/test splits based on validation status.
    Ensures balanced representation of valid/invalid samples in each split.
    """
    assert abs(train_ratio + val_ratio + test_ratio -
               1.0) < 1e-6, "Ratios must sum to 1.0"

    random.seed(random_seed)

    valid_samples = [d for d in data if d['is_valid']]
    invalid_samples = [d for d in data if not d['is_valid']]

    logger.info(f"Total samples: {len(data)}")
    logger.info(
        f"  Valid: {len(valid_samples)} ({len(valid_samples)/len(data)*100:.1f}%)")
    logger.info(
        f"  Invalid: {len(invalid_samples)} ({len(invalid_samples)/len(data)*100:.1f}%)")

    random.shuffle(valid_samples)
    random.shuffle(invalid_samples)

    def split_list(lst, train_r, val_r):
        n = len(lst)
        train_end = int(n * train_r)
        val_end = train_end + int(n * val_r)
        return lst[:train_end], lst[train_end:val_end], lst[val_end:]

    valid_train, valid_val, valid_test = split_list(
        valid_samples, train_ratio, val_ratio)
    invalid_train, invalid_val, invalid_test = split_list(
        invalid_samples, train_ratio, val_ratio)

    train = valid_train + invalid_train
    val = valid_val + invalid_val
    test = valid_test + invalid_test

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def print_split_statistics(split_name: str, data: List[Dict]):
    """Print statistics for a data split"""
    total = len(data)
    valid_count = sum(1 for d in data if d['is_valid'])
    invalid_count = total - valid_count

    error_types = Counter()
    for d in data:
        if not d['is_valid']:
            for error in d.get('errors', []):
                if ':' in error:
                    error_type = error.split(':')[0].strip()
                elif '.' in error:
                    error_type = error.split('.')[0].strip()
                else:
                    error_type = error.strip()
                error_types[error_type] += 1

    brands = Counter()
    total_detections = 0
    for d in data:
        if d['is_valid']:
            for brand in d.get('brands_detected', []):
                brands[brand] += 1
            total_detections += d.get('num_detections', 0)

    print(f"{split_name.upper()} SPLIT")
    print(f"Total samples: {total}")
    print(f"\tValid: {valid_count} ({valid_count/total*100:.1f}%)")
    print(f"\tInvalid: {invalid_count} ({invalid_count/total*100:.1f}%)")

    if valid_count > 0 and total_detections > 0:
        print(f"\nDetection statistics (valid samples only):")
        print(f"\tTotal detections: {total_detections}")
        print(f"\tAvg detections/image: {total_detections/valid_count:.2f}")
        print(f"\tUnique brands: {len(brands)}")

    if error_types:
        print(f"\nTop error types (invalid samples):")
        for error_type, count in error_types.most_common(5):
            print(f"\t{error_type:.<50} {count:>4}")

    if brands:
        print(f"\nTop brands detected (valid samples):")
        for brand, count in brands.most_common(10):
            print(f"\t{brand:.<50} {count:>4}")


def create_splits(
    input_jsonl: str = "validation_results.jsonl",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
):
    """
    Create train/val/test splits and save them locally in current directory
    """
    # Check if input file exists
    input_path = Path(input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_jsonl}")

    # Load data
    logger.info(f"Loading dataset from {input_jsonl}")
    data = load_jsonl(input_jsonl)
    logger.info(f"✓ Loaded {len(data)} samples")

    # Create splits
    logger.info(
        f"\nCreating splits (train: {train_ratio}, val: {val_ratio}, test: {test_ratio})...")
    train, val, test = stratified_split(
        data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )

    # Save splits in current directory
    save_jsonl(train, "train.jsonl")
    save_jsonl(val, "val.jsonl")
    save_jsonl(test, "test.jsonl")

    # Print statistics
    print("SPLIT STATISTICS")

    print_split_statistics("Train", train)
    print_split_statistics("Val", val)
    print_split_statistics("Test", test)

    # Summary
    print("SUMMARY")
    print(f"Total samples: {len(data)}")
    print(f"\tTrain: {len(train)} ({len(train)/len(data)*100:.1f}%)")
    print(f"\tVal: {len(val)} ({len(val)/len(data)*100:.1f}%)")
    print(f"\tTest: {len(test)} ({len(test)/len(data)*100:.1f}%)")

    # Create metadata file
    metadata = {
        "total_samples": len(data),
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_seed": random_seed,
        "source_file": input_jsonl,
        "split_timestamp": __import__('datetime').datetime.utcnow().isoformat()
    }

    with open("split_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    return train, val, test


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create train/val/test splits from validation_results.jsonl")
    parser.add_argument(
        "--input",
        default="validation_results.jsonl",
        help="Path to input JSONL file (default: validation_results.jsonl)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Create splits
    train, val, test = create_splits(
        input_jsonl=args.input,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )
