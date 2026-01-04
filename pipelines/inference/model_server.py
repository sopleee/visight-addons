from pathlib import Path
import modal
from pydantic import BaseModel
import zipfile
import requests
import re
from typing import Optional
import os
import sys

ENV = os.getenv("MODAL_ENV", "dev")

# Resolve the code root both locally (during image build) and in the container.
_here = Path(__file__).resolve()
if len(_here.parents) >= 3:# and _here.parents[2].name == "a5":
    CODE_ROOT = _here.parents[2]
else:
    CODE_ROOT = Path("/root")
sys.path.append(str(CODE_ROOT))  # ensure Python can find the package

# ====== CONFIG ======
INFRASTRUCTURE_CONFIG = {
    "dev": {
        "gpu": "A10",
        "min_containers": 0,
        "max_containers": 2,
    },
    "prod": {
        "gpu": "A10",
        "min_containers": 1,
        "max_containers": 20,
    }
}


APP_NAME = "visight-yolo-test"
BUCKET_NAME = "visight-data-yusufmoola"
MOUNT_PATH = Path("/bucket")                 
SMOKE_DATA_YAML = MOUNT_PATH / "tmp/smoke_v1" / "data.yaml"
SAVE_DIR = MOUNT_PATH / "models" / "smoke_yolov10n"
STATS_DIR = MOUNT_PATH / "stats" / "training" / "smoke_yolov10n"
RUNS_DIR = Path("/root/data/runs")                

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

job_status_dict = modal.Dict.from_name("job-status", create_if_missing=True)
results_volume = modal.Volume.from_name("results-volume", create_if_missing=True)

S3_SECRET = modal.Secret.from_name( 
    "s3-bucket-secret",
    required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
)

# ====== IMAGE ======
# Use NVIDIA TensorRT base to get TRT runtime + CUDA libs; install torch/ultralytics on top.
image = (
    modal.Image.from_registry("nvcr.io/nvidia/tensorrt:24.03-py3")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"])
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install([
        "ultralytics==8.2.68",     # YOLOv8/v10 support + TRT loading
        "opencv-python==4.10.0.84",
        "numpy>=1.24,<2.0",
        "pyyaml>=6.0",
        "onnx>=1.14.0", 
        "fastapi",
        "boto3"
    ])
#    .add_local_dir(CODE_ROOT, remote_path="/root/pipelines")
)
app = modal.App(f"{APP_NAME}-{ENV}", image=image)

def zip_directory(directory_paths, other_paths, zip_path):
    i = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for directory_path in directory_paths:
            for file_path in Path(directory_path).rglob('*'):  # rglob for recursive
                if file_path.is_file():
                    arcname = file_path.relative_to(directory_path)
                    zf.write(file_path, arcname=f"{str(directory_path.stem)}/{arcname}")
                    i += 1
        for path in other_paths: 
            zf.write(path)
            i+=1
    print(f"NUM FILES ZIPPED:", i)
    
    import os
    file_size = os.path.getsize(zip_path)
    print(f"Zip file size: {file_size:,} bytes")
        
    # Verify it's a valid zip
    with zipfile.ZipFile(zip_path, 'r') as verify_zf:
        file_list = verify_zf.namelist()
        print(f"Verified {len(file_list)} files in zip")
        if len(file_list) == 0:
            raise ValueError("Zip file has no contents!")
    # import os
    # os.sync()

def download_from_google_drive(share_link, output_path):
    """
    Download a file from a public Google Drive link.
    
    Args:
        share_link: Google Drive share link (e.g., https://drive.google.com/file/d/FILE_ID/view?usp=sharing)
        output_path: Path where the file will be saved (e.g., 'video.mp4')
    """
    # Extract file ID from the share link
    file_id = extract_file_id(share_link)
    
    if not file_id:
        print("Error: Could not extract file ID from the link")
        return False
    
    # Google Drive direct download URL
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    session = requests.Session()
    try: 
        response = session.get(url, stream=True)
    except Exception as e:
        print(f"\nNetwork related error: {e}")
        return False
    
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if not token:
        for line in response.iter_lines():
            if b'confirm=' in line:
                match = re.search(b'confirm=([^&"]+)', line)
                if match:
                    token = match.group(1).decode()
                    url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={token}"
                    break
        response = session.get(url, stream=True)
    
    if token:
        url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={token}"
        response = session.get(url, stream=True)
    elif not any(key.startswith('download_warning') for key in response.cookies.keys()):
        response = session.get(url, stream=True)
        
        
    if 'text/html' in response.headers.get('Content-Type', ''):
        # This is the virus warning page for large files
        html_content = response.text
        uuid_match = re.search(r'name="uuid" value="([^"]+)"', html_content)
        
        if uuid_match:
            uuid = uuid_match.group(1)
            url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t&uuid={uuid}"
            print(f"Large file detected, using confirmation download...")
        else:
            # Fallback: try without UUID
            url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        response = session.get(url, stream=True)

    # Download the file
    try:
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Show progress
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloading: {percent:.1f}%", end='')
        
        print(f"\n✓ Downloaded successfully to: {output_path}")
        
        with open(output_path, 'rb') as f:
            first_bytes = f.read(100)
            # Check if file is HTML (common issue with Google Drive)
            if b'<html' in first_bytes.lower() or b'<!doctype' in first_bytes.lower():
                print("ERROR: Downloaded file is HTML, not a video!")
                print("The Google Drive link may not be publicly accessible.")
                print(f"First 100 bytes: {first_bytes[:100]}")
                return False
            # Check for valid video file signature
            if not (first_bytes.startswith(b'\x00\x00\x00') or  # MP4/MOV
                    first_bytes.startswith(b'ftyp') or
                    b'ftyp' in first_bytes[:20]):
                print(f"WARNING: File may not be a valid video")
                print(f"First 20 bytes: {first_bytes[:20]}")
        return True
    except Exception as e:
        print(f"\nError downloading file: {e}")
        return False
        
def extract_file_id(share_link):
    """Extract file ID from various Google Drive link formats."""
    patterns = [
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
        r'/d/([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, share_link)
        if match:
            return match.group(1)
    
    return None

def download_generic_url(url: str, output_path: str, min_bytes: int = 1024) -> bool:
    """Download any URL (e.g., S3 presigned) to output_path."""
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            if r.status_code != 200:
                print(f"Download failed status {r.status_code}")
                return False
            total = 0
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total += len(chunk)
            if total < min_bytes:
                print(f"Download too small ({total} bytes); likely invalid file")
                return False
        print(f"✓ Downloaded via generic URL to: {output_path} ({total} bytes)")
        return True
    except Exception as e:
        print(f"Error downloading generic URL: {e}")
        return False

def download_video(url: str, output_path: str) -> bool:
    """Download video from either Drive or generic URL."""
    if "drive.google.com" in url:
        return download_from_google_drive(url, output_path)
    return download_generic_url(url, output_path)

# def extract_file_id(share_link):
#     """Extract file ID from various Google Drive link formats."""
#     patterns = [
#         r'/file/d/([a-zA-Z0-9_-]+)',
#         r'id=([a-zA-Z0-9_-]+)',
#         r'/d/([a-zA-Z0-9_-]+)'
#     ]
    
#     for pattern in patterns:
#         match = re.search(pattern, share_link)
#         if match:
#             return match.group(1)
    
#     return None

class InferenceRequest(BaseModel):
    video_url: str
    fps: Optional[int] = 12
    confidence_threshold: Optional[float] = 0.5
    batch_size: Optional[int] = 200

@app.function()
@modal.fastapi_endpoint(method="POST")
@modal.concurrent(max_inputs=100)
def submit_job(request: InferenceRequest, save_to_s3: bool = False):
    import uuid
    import json
    from datetime import datetime
    job_id = str(uuid.uuid4())
    call = inference.spawn(job_id, request, save_to_s3)
    job_status_dict[job_id] = json.dumps({
        "cur_status": "submitted",
        "cur_status_progress": 100, 
        "updated_at": datetime.now().isoformat(),
        "call_id": getattr(call, "object_id", None)
    })
    print("job_id:", job_id)
    return {"job_id": job_id, "call_id": getattr(call, "object_id", None)}

@app.function()
@modal.fastapi_endpoint(method="GET")
@modal.concurrent(max_inputs=100)
def check_status(job_id: str):
    """Check job status and progress"""

    import json
    from datetime import datetime
    import modal
    from modal.exception import OutputExpiredError
    
    print(f"check_status - job_id: '{job_id}' (len={len(job_id)})")
    print(f"check_status - job_id type: {type(job_id)}")
    print(f"check_status - job_id repr: {repr(job_id)}")

    status_json = job_status_dict.get(job_id)
    
    if status_json is None: return {"error": f"Job id: {job_id} not found"}, 404
    try:
        status_obj = json.loads(status_json)
    except Exception:
        # fallback if value was stored as plain string
        status_obj = {"cur_status": status_json}
    status_obj["job_id"] = job_id
    status_obj.setdefault("checked_at", datetime.now().isoformat())

    # If we already have a terminal status, return early
    if status_obj.get("cur_status") in {"completed", "failed", "expired"}:
        return status_obj

    # If we have a Modal call_id, use it to refine status
    call_id = status_obj.get("call_id")
    if call_id:
        try:
            call = modal.FunctionCall.from_id(call_id)
            try:
                call.get(timeout=0.1)
                status_obj["cur_status"] = "completed"
                status_obj["cur_status_progress"] = 100
                status_obj["updated_at"] = datetime.now().isoformat()
                job_status_dict[job_id] = json.dumps(status_obj)
            except TimeoutError:
                # still running/pending
                if status_obj.get("cur_status") == "submitted":
                    status_obj["cur_status"] = "running"
                status_obj.setdefault("cur_status_progress", 50)
            except OutputExpiredError:
                status_obj["cur_status"] = "expired"
                status_obj["cur_status_progress"] = 0
                status_obj["updated_at"] = datetime.now().isoformat()
                job_status_dict[job_id] = json.dumps(status_obj)
            status_obj["call_id"] = call_id
            status_obj["checked_at"] = datetime.now().isoformat()
        except Exception as e:
            status_obj["cur_status"] = status_obj.get("cur_status", "unknown")
            status_obj["error"] = str(e)

    return status_obj

@app.function(volumes={"/results": results_volume})
@modal.concurrent(max_inputs=100)
@modal.fastapi_endpoint(method="GET")
def download_result(job_id: str):
    """Download completed result"""
    import os
    import json
    from fastapi.responses import FileResponse
    import tempfile
    import shutil
    import zipfile

    status = job_status_dict.get(job_id)
    if status is None: 
        print(f"Job not found: {job_id}")
        return {"error": "Job not found"}, 404
    status = json.loads(status)
    if status["cur_status"] != "completed":
        return {
            "error": "Job not completed",
            "status": status["cur_status"],
            "progress": status["cur_status_progress"]
        }, 400
    
    # Return zip file
    zip_path = f"/results/{job_id}.zip"
    if not os.path.exists(zip_path): 
        print(f"File DNE: {zip_path}")
        return {"error": "Result file not found"}, 404
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            num_files = len(zf.namelist())
            print(f"Zip contains {num_files} files")
            
            if num_files == 0: return {"error": "Zip file has no contents"}, 500
    except zipfile.BadZipFile as e: return {"error": f"Invalid zip file: {e}"}, 500

    # Copy the zip into temp
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    shutil.copy(zip_path, temp_zip.name)
    
    # Keep the zip and status for repeated downloads/debug
    # results_volume.commit()
    
    # Remove status
    # del job_status_dict[job_id]
    
    return FileResponse(
        temp_zip.name,
        media_type="application/zip",
        filename=f"results_{job_id}.zip"
    )

@app.function(volumes={"/results": results_volume})
@modal.fastapi_endpoint(method="GET")
def debug_list_volume_contents():
    """Debug: List everything in the volume"""
    import os
    
    if not os.path.exists("/results"):
        return {"error": "/results doesn't exist", "exists": False}
    
    files = []
    for filename in os.listdir("/results"):
        filepath = os.path.join("/results", filename)
        size = os.path.getsize(filepath) if os.path.isfile(filepath) else 0
        files.append({
            "name": filename,
            "size": size,
            "is_file": os.path.isfile(filepath)
        })
    
    return {
        "directory": "/results",
        "exists": True,
        "total_files": len(files),
        "files": files
    }


@app.function(
    **INFRASTRUCTURE_CONFIG[ENV],
    cpu=2,
    timeout=3000,
    volumes={MOUNT_PATH: modal.CloudBucketMount(BUCKET_NAME, secret=S3_SECRET), 
             "/results": results_volume},
    secrets=[S3_SECRET], 
)
def inference(job_id: str, request: InferenceRequest, save_to_s3: bool = False): 
    from pipelines.inference.pipeline_remote import InferencePipeline
    from pipelines.configs.config import Config
    from datetime import datetime
    import cv2
    from fastapi import HTTPException
    import json
    
    local_vid_path = f"vid_{job_id}.mp4"
    
    # mark job as running
    job_status_dict[job_id] = json.dumps({
        "cur_status": "running",
        "cur_status_progress": 10,
        "updated_at": datetime.now().isoformat()
    })
    logger.info(f"[{job_id}] status set to running")

    logger.info("Downloading video to Modal container")
    # sample_vid = "https://drive.google.com/file/d/1ya6iuzDMhqCSZG8uRpLsvrNeNA77d8Ew/view?usp=sharing"
    ok = download_video(request.video_url, local_vid_path)
    if not ok:
        logger.error(f"Failed to download video from {request.video_url}")
        job_status_dict[job_id] = json.dumps({
            "cur_status": "failed",
            "cur_status_progress": 0,
            "error": "Could not download video",
            "updated_at": datetime.now().isoformat()
        })
        raise HTTPException(status_code=400, detail="Could not download video")
    logger.info("Finished downloading video")
    cur_config = Config(env=ENV)
        
    model_path = MOUNT_PATH / Path(cur_config.model_key)
    engine_path = model_path / "best.engine"
    pt_path = model_path / "best.pt"
    weights_path = engine_path if engine_path.exists() else pt_path
    logger.info(f"Loading model from {weights_path}")

    def build_pipeline(path):
        return InferencePipeline(
            model_path=path,
            fps=request.fps,
            confidence_threshold=request.confidence_threshold,
            logger=logger,
            batch_size=min(60, request.batch_size),
        )

    try:
        pipeline = build_pipeline(weights_path)
    except Exception as e:
        if weights_path == engine_path and pt_path.exists():
            logger.warning(f"Engine load failed during init ({e}); falling back to {pt_path}")
            pipeline = build_pipeline(pt_path)
            weights_path = pt_path
        else:
            raise
    video_id = f"{str(Path(cur_config.model_key).stem)}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    
    try: 
        res_dirs, additional_files = pipeline.run_inference_on_video(
            video_path=local_vid_path, 
            video_id=video_id, 
            s3_bucket=cur_config.s3_bucket if save_to_s3 else save_to_s3,
            save_annotated_frames=True
        )
    except Exception as e: 
        # If using engine and it failed during inference, retry with pt
        if weights_path == engine_path and pt_path.exists():
            logger.warning(f"Engine inference failed ({e}); retrying with {pt_path}")
            fallback_pipeline = build_pipeline(pt_path)
            try:
                res_dirs, additional_files = fallback_pipeline.run_inference_on_video(
                    video_path=local_vid_path, 
                    video_id=video_id, 
                    s3_bucket=cur_config.s3_bucket if save_to_s3 else save_to_s3,
                    save_annotated_frames=True
                )
                weights_path = pt_path
            except Exception as e2:
                print("error!", e2)
                job_status_dict[job_id] = json.dumps({
                    "cur_status": "failed",
                    "cur_status_progress": 0,
                    "error": str(e2),
                    "updated_at": datetime.now().isoformat()
                })
                raise HTTPException(status_code=500, detail=f"Exception occurred during video frame processing or model inference: {e2}")
        else:
            print("error!", e)
            job_status_dict[job_id] = json.dumps({
                "cur_status": "failed",
                "cur_status_progress": 0,
                "error": str(e),
                "updated_at": datetime.now().isoformat()
            })
            raise HTTPException(status_code=500, detail=f"Exception occurred during video frame processing or model inference: {e}")

    zip_path = Path(f"/results/{job_id}.zip")
    logger.info("Started zipping results")  
    # CHANGED: Pass additional_files instead of [res_json_path]
    try: zip_directory(res_dirs, additional_files, zip_path)
    except Exception as e: 
        job_status_dict[job_id] = json.dumps({
            "cur_status": "failed",
            "cur_status_progress": 0,
            "error": f"zip error: {e}",
            "updated_at": datetime.now().isoformat()
        })
        raise HTTPException(status_code=500, detail=f"Error during zip: {e}")

    logger.info("Finished zipping") 
    import os
    file_size = os.path.getsize(zip_path)
    print(f"Zip file size: {file_size:,} bytes")
    if file_size == 0: raise HTTPException(status_code=500, detail="Internal error: zip file is empty after creation!")  
    
    results_volume.commit()
             
    job_status_dict[job_id] = json.dumps({
        "cur_status": "completed",
        "cur_status_progress": 100, 
        "updated_at": datetime.now().isoformat()
    })
    logger.info(f"[{job_id}] status set to completed")
