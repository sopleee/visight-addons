from pathlib import Path
import modal
from pydantic import BaseModel
import zipfile
import requests
import re
from typing import Optional
import os
import argparse
from vlm_test.qwen_inference_pipeline import QwenInferencePipeline

# ENV = os.getenv("MODAL_ENV", "dev")

# # ====== CONFIG ======
# INFRASTRUCTURE_CONFIG = {
#     "dev": {
#         "gpu": "T4",
#         "keep_warm": 0,
#         "concurrency_limit": 1,
#         "max_containers": 2,
#     },
#     "prod": {
#         "gpu": "T4",
#         "keep_warm": 1,
#         "concurrency_limit": 10,
#         "max_containers": 20,
#     }
# }

APP_NAME = "qwen-test"
BUCKET_NAME = "visight-data-yusufmoola"
MOUNT_PATH = Path("/bucket")                 
SMOKE_DATA_YAML = MOUNT_PATH / "tmp/smoke_v1" / "data.yaml"
SAVE_DIR = MOUNT_PATH / "models" / "smoke_yolov10n"
STATS_DIR = MOUNT_PATH / "stats" / "training" / "smoke_yolov10n"
RUNS_DIR = Path("/root/data/runs")                

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


S3_SECRET = modal.Secret.from_name( 
    "s3-bucket-secret",
    required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
)

# ====== IMAGE ======
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0", "git"])
    .pip_install([
        "git+https://github.com/huggingface/transformers",
        "accelerate",
        "qwen-vl-utils[decord]==0.0.8",
        "numpy>=1.24,<2.0",
        "pyyaml>=6.0",
        "onnx>=1.14.0", 
        "fastapi", "boto3", "torchvision", #"flash-attn", "packaging"
    ])
    .add_local_file("./junk/sample_labeled.jpg", remote_path="/sample_labeled.jpg")
    .add_local_file("./junk/sample.jpg", remote_path="/sample.jpg")
)
app = modal.App(f"{APP_NAME}", image=image)

@app.function(
    gpu="A10G:1",
    cpu=2,
    timeout=60 * 60 * 6,   # up to 6h
    volumes={MOUNT_PATH: modal.CloudBucketMount(BUCKET_NAME, secret=S3_SECRET)},
    secrets=[S3_SECRET]
)
def inference(run_name): 
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

    test_dir = Path("raw/roboflow/v8/test")
    local_input_dir = MOUNT_PATH / test_dir
    s3_bucket = "visight-data-yusufmoola"
    s3_write_dir = Path(f"vlm_inference/{run_name}")
    
    example_output = "[{\"brand_name\": \"Alpinestars\",\"bbox_locations\": [[20, 10, 80, 30],[150, 3, 170, 70]]},{\"brand_name\": \"UBS\",\"bbox_locations\": [[200, 50, 250, 90]]}]"
    claude_system_prompt = f"You are an expert at identifying logos in images. Your task is to detect all logo instances and return their locations.\nInstructions:\n\t- Each logo may appear multiple times in an image\n\t- Identify the brand name for each logo\n\t- Provide bounding box coordinates in pixels for every occurrence\n\t- Coordinates format: [x_min, y_min, x_max, y_max] where (0,0) is the top-left corner\n\t- Include logos that are: partially obscured, at angles, in backgrounds, stylized, or low contrast\n\nOutput format:\nReturn a JSON array where each element contains:\n\t- \"brand_name\": the brand/company name\n\t- \"bbox_locations\": array of bounding boxes, each with coordinates [x_min, y_min, x_max, y_max]\nExample output:{example_output}\nFormat your response in JSON."
    user_prompt = "Retrieve all logos and return response in JSON format."
    qwen_version = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_version, torch_dtype="auto", device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(qwen_version)
    pipeline = QwenInferencePipeline(
        s3_bucket=s3_bucket, system_prompt=claude_system_prompt, user_prompt=user_prompt, 
        model=model, processor=processor, qwen_version=qwen_version)
    
    print("ITERATIVELY PROMPT QWEN")
    num_imgs, agg_time = pipeline.iteratively_generate(
        vid_dir=local_input_dir, s3_dir=s3_write_dir)
    
    print("WRITE MODEL CARD")
    pipeline.write_model_card(dst_dir=s3_write_dir, model_id=run_name, dataset_path=test_dir, agg_test_time=agg_time, num_tested_elems=num_imgs)

@app.local_entrypoint()
def main(): 
    # parser = argparse.ArgumentParser(description="Run Qwen on test dataset")
    # parser.add_argument("--run_name", action="store_true", help="Skip saving annotated frames")
    # args = parser.parse_args() 
    inference.remote("qwen2.5_baseline")