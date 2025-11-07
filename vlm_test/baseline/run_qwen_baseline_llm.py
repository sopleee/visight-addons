# run_qwen_baseline_modal.py
import modal
import json
from pathlib import Path
from typing import List, Dict
import time

app = modal.App("qwen-baseline-validator")

S3_SECRET = modal.Secret.from_name(
    "s3-bucket-secret",
    required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0", "git"])
    .pip_install([
        "git+https://github.com/huggingface/transformers",
        "accelerate",
        "qwen-vl-utils[decord]==0.0.8",
        "numpy>=1.24,<2.0",
        "torchvision",
    ])
)


@app.function(
    gpu="A10:1",
    cpu=2,
    timeout=3600,
    image=image
)
def validate_test_set(test_data_json: str):
    """Run Qwen validation on test set"""
    import json
    import re
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import torch
    import time

    test_data = json.loads(test_data_json)

    qwen_version = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_version, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(qwen_version)

    validation_system_prompt = """You are an expert at evaluating JSON format validation for logo detection outputs.

Your task: Determine if a given VLM response follows the correct format for logo detection.

Required format:
- Must be a valid JSON array
- Each element must have:
  * "brand_name": string (the logo brand name)
  * "bbox_locations": array of arrays, each with 4 numeric coordinates [x_min, y_min, x_max, y_max]
- Coordinates must satisfy: x_min < x_max and y_min < y_max
- All coordinates must be non-negative numbers

Respond with a JSON object:
{
  "is_valid": true/false,
  "reasoning": "brief explanation of why valid or invalid",
  "confidence": 0.0-1.0
}

Be strict about format requirements but flexible about brand names."""

    validation_results = []
    stats = {
        "total_samples": len(test_data),
        "qwen_valid": 0,
        "qwen_invalid": 0,
        "ground_truth_valid": 0,
        "ground_truth_invalid": 0,
        "agreement": 0,
        "disagreement": 0,
        "total_inference_time": 0.0,
    }

    total_confidence = 0.0

    for idx, sample in enumerate(test_data):
        try:
            start_time = time.time()

            filename = sample['filename']
            raw_text = sample['raw_text']
            ground_truth_valid = sample['is_valid']

            user_prompt = f"""Evaluate this VLM response for logo detection:

Response to validate:
{raw_text}

Is this correctly formatted? Return your evaluation as JSON."""

            messages = [
                {
                    "role": "system",
                    "content": validation_system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            qwen_response = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            inference_time = time.time() - start_time
            stats["total_inference_time"] += inference_time

            try:
                cleaned_response = re.sub(
                    r'```json\s*|\s*```', '', qwen_response.strip())
                validation_result = json.loads(cleaned_response)

                is_valid = validation_result.get("is_valid", False)
                reasoning = validation_result.get(
                    "reasoning", "No reasoning provided")
                confidence = validation_result.get("confidence", 0.5)

            except json.JSONDecodeError:
                print(f"Warning: Qwen returned non-JSON for {filename}")
                is_valid = False
                reasoning = f"Qwen response parsing failed: {qwen_response[:200]}"
                confidence = 0.0

            # Create result
            result = {
                "filename": filename,
                "is_valid": is_valid,
                "qwen_response": qwen_response,
                "validation_reasoning": reasoning,
                "confidence": confidence,
                "inference_time": inference_time,
                "ground_truth_valid": ground_truth_valid
            }

            validation_results.append(result)

            if is_valid:
                stats["qwen_valid"] += 1
            else:
                stats["qwen_invalid"] += 1

            if ground_truth_valid:
                stats["ground_truth_valid"] += 1
            else:
                stats["ground_truth_invalid"] += 1

            if is_valid == ground_truth_valid:
                stats["agreement"] += 1
            else:
                stats["disagreement"] += 1

            total_confidence += confidence

            # Progress logging
            if (idx + 1) % 10 == 0 or (idx + 1) == len(test_data):
                print(f"Processed {idx + 1}/{len(test_data)} - "
                      f"Agreement: {stats['agreement']}, Disagreement: {stats['disagreement']}")

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            result = {
                "filename": sample.get('filename', f'sample_{idx}'),
                "is_valid": False,
                "qwen_response": "",
                "validation_reasoning": f"Processing error: {str(e)}",
                "confidence": 0.0,
                "inference_time": 0.0,
                "ground_truth_valid": sample.get('is_valid', False)
            }
            validation_results.append(result)
            stats["qwen_invalid"] += 1
            if sample.get('is_valid', False):
                stats["ground_truth_valid"] += 1
            else:
                stats["ground_truth_invalid"] += 1
            stats["disagreement"] += 1

    stats["avg_confidence"] = total_confidence / \
        len(test_data) if len(test_data) > 0 else 0
    stats["avg_inference_time"] = stats["total_inference_time"] / \
        len(test_data) if len(test_data) > 0 else 0
    stats["accuracy"] = stats["agreement"] / \
        len(test_data) if len(test_data) > 0 else 0

    # Calculate precision, recall, F1
    true_positives = sum(
        1 for r in validation_results if r["is_valid"] and r["ground_truth_valid"])
    false_positives = sum(
        1 for r in validation_results if r["is_valid"] and not r["ground_truth_valid"])
    false_negatives = sum(
        1 for r in validation_results if not r["is_valid"] and r["ground_truth_valid"])
    true_negatives = sum(
        1 for r in validation_results if not r["is_valid"] and not r["ground_truth_valid"])

    precision = true_positives / \
        (true_positives + false_positives) if (true_positives +
                                               false_positives) > 0 else 0
    recall = true_positives / \
        (true_positives + false_negatives) if (true_positives +
                                               false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0

    stats["precision"] = precision
    stats["recall"] = recall
    stats["f1_score"] = f1
    stats["true_positives"] = true_positives
    stats["false_positives"] = false_positives
    stats["false_negatives"] = false_negatives
    stats["true_negatives"] = true_negatives

    print("QWEN BASELINE RESULTS\n")
    print(f"Accuracy: {stats['accuracy']:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"Avg inference time: {stats['avg_inference_time']:.2f}s")

    return {
        "results": validation_results,
        "stats": stats
    }


@app.local_entrypoint()
def main(test_file: str = "test.jsonl"):
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    # Run on Modal
    result_dict = validate_test_set.remote(json.dumps(test_data))

    validation_results = result_dict["results"]
    stats = result_dict["stats"]

    output_file = "qwen_baseline_results.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in validation_results:
            f.write(json.dumps(result) + '\n')

    print(f"Results saved to {output_file}")

    stats_file = "qwen_baseline_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print(f"Stats saved to {stats_file}")

    print("\nPERFORMANCE SUMMARY")
    print(f"Total test samples: {stats['total_samples']}")
    print(f"\nMetrics:")
    print(f"\tAccuracy: {stats['accuracy']:.3f}")
    print(f"\tPrecision: {stats['precision']:.3f}")
    print(f"\tRecall: {stats['recall']:.3f}")
    print(f"\tF1-Score: {stats['f1_score']:.3f}")
    print(f"\nConfusion Matrix:")
    print(
        f"\tTP: {stats['true_positives']:>3}  FP: {stats['false_positives']:>3}")
    print(
        f"\tFN: {stats['false_negatives']:>3}  TN: {stats['true_negatives']:>3}")
    print(f"\tTotal time: {stats['total_inference_time']:.1f}s")

    # Show disagreement examples
    disagreements = [
        r for r in validation_results if r["is_valid"] != r["ground_truth_valid"]][:5]
    if disagreements:
        print("DISAGREEMENT EXAMPLES (first 5)")
        for i, result in enumerate(disagreements, 1):
            print(f"\n{i}. {result['filename']}")
            print(
                f"\tGround Truth: {'Valid' if result['ground_truth_valid'] else 'Invalid'}")
            print(f"\tQwen: {'Valid' if result['is_valid'] else 'Invalid'}")
            print(f"\tConfidence: {result['confidence']:.2f}")
            print(f"\tReasoning: {result['validation_reasoning'][:100]}...")
