import json
import re
import boto3
from pathlib import Path
from typing import List, Optional
import logging
from dataclasses import dataclass, asdict
import modal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


S3_SECRET = modal.Secret.from_name(
    "s3-bucket-secret",
    required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install([
        "boto3",
    ])
)

app = modal.App("format-validator", image=image)

F1_BRANDS = {
    "1and1": 0,
    "AMD": 1,
    "ATandT": 2,
    "Alfa Romeo": 3,
    "Alpha Tauri": 4,
    "Alpinestars": 5,
    "Aramco": 6,
    "Arrow Electronics": 7,
    "Aston Martin": 8,
    "BWT AG": 9,
    "Bybit": 10,
    "Cash App": 11,
    "Castrol": 12,
    "Citrix": 13,
    "Claro": 14,
    "CrowdStrike": 15,
    "Crypto-com": 16,
    "DHL": 17,
    "DP World": 18,
    "DuPont": 19,
    "Emirates": 20,
    "Epson": 21,
    "Esso": 22,
    "Ferrari": 23,
    "Force India": 24,
    "Formula 1": 25,
    "Ftx": 26,
    "Haas": 27,
    "Heineken": 28,
    "Honda": 29,
    "Ineos": 30,
    "Infiniti": 31,
    "Inter": 32,
    "JCB": 33,
    "Kaspersky": 34,
    "Liqui Moly": 35,
    "McLaren": 36,
    "Mercedes-Benz": 37,
    "Microsoft Corporation": 38,
    "Mobil": 39,
    "Monster Energy": 40,
    "Oracle": 41,
    "Orlen": 42,
    "Petronas": 43,
    "Pirelli": 44,
    "Puma": 45,
    "RCI Banque": 46,
    "Randstad Holding": 47,
    "Rauch": 48,
    "RayBan": 49,
    "Red Bull": 50,
    "Renault": 51,
    "Richard Mile": 52,
    "Rolex": 53,
    "Royal Dutch Shell": 54,
    "Santander": 55,
    "Splunk": 56,
    "TAG Heuer": 57,
    "TeamViewer": 58,
    "Telcel": 59,
    "Tezos": 60,
    "Tommy Hilfiger": 61,
    "UBS": 62,
    "United Parcel Service": 63,
    "VTB Bank": 64,
    "Velo": 65,
    "Vuse": 66,
    "Williams": 67,
    "Yahoo": 68,
    "Zoom": 69,
    "alpine": 70,
    "aws": 71,
    "bose": 72,
    "darktrace": 73,
    "dell": 74,
    "etihad": 75,
    "fia": 76,
    "fxpro": 77,
    "gulf": 78,
    "hp": 79,
    "lavazza": 80,
    "logitech": 81,
    "stc": 82,
    "ural kali": 83,
    "verti": 84,
    "wallmart": 85,
}

VALID_BRANDS = set(F1_BRANDS.keys())


@dataclass
class ValidationResult:
    filename: str
    is_valid: bool
    errors: List[str]
    parsed_data: Optional[dict]
    raw_text: str
    num_detections: int
    brands_detected: List[str]

    def to_dict(self):
        return asdict(self)


def validate_qwen_format(response_text: str, valid_brands: set) -> dict:
    """
    Validates Qwen VLM response format for logo detection.
    """

    errors = []

    cleaned = re.sub(r'```json\s*|\s*```', '', response_text.strip())

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        errors.append(f"JSON decode error: {str(e)}")
        return {"is_valid": False, "errors": [f"JSON Parse error: {e}"], "parsed_data": None}

    if not isinstance(data, list):
        errors.append("Response is not a list.")
        return {"is_valid": False, "errors": errors, "parsed_data": None}

    for idx, detection in enumerate(data):
        if "brand_name" not in detection:
            errors.append(f"Entry {idx} missing 'brand_name'.")
        if "bbox_locations" not in detection:
            errors.append(f"Entry {idx} missing 'bbox_locations'.")
            continue

        brand = detection.get("brand_name", "")
        if not any(brand.lower().strip() == valid_brand.lower().strip() for valid_brand in valid_brands):
            errors.append(f"Entry {idx} has invalid brand name: '{brand}'.")

        bboxes = detection.get("bbox_locations", [])
        if not isinstance(bboxes, list):
            errors.append(f"Entry {idx} 'bbox_locations' is not a list.")
            continue
        for bbox_idx, bbox in enumerate(bboxes):
            if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(coord, (int, float)) for coord in bbox)):
                errors.append(
                    f"Entry {idx} bbox {bbox_idx} is not a valid list of 4 numbers.")
                continue
            x_min, y_min, x_max, y_max = bbox

            if not all(isinstance(coord, (int, float)) for coord in [x_min, y_min, x_max, y_max]):
                errors.append(
                    f"Entry {idx} bbox {bbox_idx} contains non-numeric coordinates.")

            if x_min >= x_max or y_min >= y_max:
                errors.append(
                    f"Entry {idx} bbox {bbox_idx} has invalid coordinate relationships.")
                continue

            if not all(0 <= coord <= 2000 for coord in bbox):
                errors.append(
                    f"Entry {idx} bbox {bbox_idx} has invalid coordinates.")

    is_valid = len(errors) == 0
    return {"is_valid": is_valid, "errors": errors, "parsed_data": data if is_valid else None}


@app.function(
    secrets=[S3_SECRET],
    timeout=3600,
)
def create_validation_dataset(
    s3_bucket: str = "visight-data-yusufmoola",
    s3_predictions_dir: str = "vlm_inference/qwen2.5_baseline/predictions/",
    output_jsonl_path: str = "vlm_inference/qwen2.5_baseline/validation_results.jsonl",
    output_stats_path: str = "vlm_inference/qwen2.5_baseline/validation_stats.json",
    max_files: Optional[int] = None,
):
    import os

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_REGION"],
    )

    response = s3_client.list_objects_v2(
        Bucket=s3_bucket,
        Prefix=s3_predictions_dir,
    )

    if 'Contents' not in response:
        logger.info("No files found in the specified S3 directory.")
        return

    prediction_files = [
        obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.txt')
    ]

    if max_files:
        prediction_files = prediction_files[:max_files]
        logger.info(f"Limiting to first {max_files} files for validation.")

    validation_results = []
    stats = {
        "total_files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "total_detections": 0,
        "brands_detected": {},
        "error_types": {},
        "validation_config": {
            "num_valid_brands": len(VALID_BRANDS),
        }
    }

    for file_idx, file_key in enumerate(prediction_files):
        try:
            response = s3_client.get_object(Bucket=s3_bucket, Key=file_key)
            raw_text = response['Body'].read().decode('utf-8')

            filename = Path(file_key).stem

            validation = validate_qwen_format(raw_text, VALID_BRANDS)

            num_detections = 0
            brands_detected = []
            if validation["parsed_data"]:
                num_detections = len(validation["parsed_data"])
                brands_detected = [
                    det.get("brand_name", "")
                    for det in validation["parsed_data"]
                ]

            # Create result object
            result = ValidationResult(
                filename=filename,
                is_valid=validation["is_valid"],
                errors=validation["errors"],
                parsed_data=validation["parsed_data"],
                raw_text=raw_text,
                num_detections=num_detections,
                brands_detected=brands_detected
            )

            validation_results.append(result)
            stats["total_files"] += 1
            if result.is_valid:
                stats["valid_files"] += 1
                stats["total_detections"] += num_detections

                for brand in brands_detected:
                    stats["brands_detected"][brand] = stats["brands_detected"].get(
                        brand, 0) + 1
            else:
                stats["invalid_files"] += 1

            for error in result.errors:
                # Extract error category (first part before colon or period)
                if ':' in error:
                    error_type = error.split(":")[0].strip()
                elif '.' in error:
                    error_type = error.split(".")[0].strip()
                else:
                    error_type = error.strip()
                stats["error_types"][error_type] = stats["error_types"].get(
                    error_type, 0) + 1

            logger.info(f"Processed {file_idx + 1}/{len(prediction_files)} files - "
                        f"Valid: {stats['valid_files']}, Invalid: {stats['invalid_files']}")

        except Exception as e:
            logger.error(f"Error processing {file_key}: {e}")
            stats["error_types"]["processing_error"] = stats["error_types"].get(
                "processing_error", 0) + 1

    stats["valid_rate"] = stats["valid_files"] / \
        stats["total_files"] if stats["total_files"] > 0 else 0
    stats["invalid_rate"] = stats["invalid_files"] / \
        stats["total_files"] if stats["total_files"] > 0 else 0
    stats["avg_detections_per_valid_image"] = (
        stats["total_detections"] /
        stats["valid_files"] if stats["valid_files"] > 0 else 0
    )

    logger.info("VALIDATION SUMMARY")
    logger.info(f"Total files processed: {stats['total_files']}")
    logger.info(
        f"Valid files: {stats['valid_files']} ({stats['valid_rate']:.1%})")
    logger.info(
        f"Invalid files: {stats['invalid_files']} ({stats['invalid_rate']:.1%})")
    logger.info(f"Total detections: {stats['total_detections']}")
    logger.info(
        f"Avg detections/image: {stats['avg_detections_per_valid_image']:.2f}")

    # Write JSONL dataset
    logger.info(f"\nWriting dataset to s3://{s3_bucket}/{output_jsonl_path}")

    jsonl_lines = []
    for result in validation_results:
        jsonl_lines.append(json.dumps(result.to_dict()))

    jsonl_content = "\n".join(jsonl_lines)

    s3_client.put_object(
        Bucket=s3_bucket,
        Key=output_jsonl_path,
        Body=jsonl_content.encode('utf-8'),
        ContentType='application/jsonl'
    )

    logger.info(f"Wrote {len(jsonl_lines)} samples to dataset")

    # Write stats
    logger.info(f"Writing stats to s3://{s3_bucket}/{output_stats_path}")

    s3_client.put_object(
        Bucket=s3_bucket,
        Key=output_stats_path,
        Body=json.dumps(stats, indent=2).encode('utf-8'),
        ContentType='application/json'
    )

    logger.info("Wrote validation statistics")

    # Print top error types
    if stats["error_types"]:
        logger.info(f"\n{'-'*70}")
        logger.info("TOP ERROR TYPES")
        logger.info(f"{'-'*70}")
        sorted_errors = sorted(
            stats["error_types"].items(), key=lambda x: x[1], reverse=True)
        for error_type, count in sorted_errors[:10]:
            pct = (count / stats["total_files"]) * 100
            logger.info(f"  {error_type:.<55} {count:>5} ({pct:>5.1f}%)")

    # Print top brands detected
    if stats["brands_detected"]:
        logger.info("TOP BRANDS DETECTED")
        sorted_brands = sorted(
            stats["brands_detected"].items(), key=lambda x: x[1], reverse=True)
        for brand, count in sorted_brands[:20]:
            logger.info(f"  {brand:.<55} {count:>5}")

    logger.info("VALIDATION COMPLETE!")

    return stats


@app.local_entrypoint()
def main(max_files: Optional[int] = None):
    """Run the validation dataset creation"""
    stats = create_validation_dataset.remote(max_files=max_files)
    print("Validation complete - Files created in S3:")
    print("\tvlm_inference/qwen2.5_baseline/validated_dataset.jsonl")
    print("\tvlm_inference/qwen2.5_baseline/validation_stats.json")


if __name__ == "__main__":
    test_cases = [
        # Valid case
        '''[{"brand_name": "Ferrari", "bbox_locations": [[50, 100, 150, 200]]}]''',

        # Markdown wrapped
        '''```json
[{"brand_name": "Red Bull", "bbox_locations": [[50, 50, 150, 150]]}]
```''',

        # Case insensitive match
        '''[{"brand_name": "ferrari", "bbox_locations": [[100, 50, 200, 150]]}]''',

        # Invalid brand
        '''[{"brand_name": "InvalidBrand", "bbox_locations": [[100, 50, 200, 150]]}]''',

        # Missing bbox
        '''[{"brand_name": "Mercedes-Benz"}]''',

        # Invalid coordinates
        '''[{"brand_name": "McLaren", "bbox_locations": [200, 50, 100, 150]}]''',

        # Empty array
        '''[]''',
    ]

    print("Testing validator...\n")
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}:")
        print(f"Input: {test[:80]}...")
        result = validate_qwen_format(test, VALID_BRANDS)
        print(f"Valid: {result['is_valid']}")
        if result['errors']:
            print(f"Errors: {result['errors']}")
        print()
