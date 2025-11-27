from pathlib import Path
import modal
from pydantic import BaseModel
import zipfile
import requests
import re
from typing import Optional
import os

ENV = os.getenv("MODAL_ENV", "dev")

# ====== CONFIG ======
INFRASTRUCTURE_CONFIG = {
    "dev": {
        "gpu": "T4",
        "min_containers": 0,
        "max_containers": 2,
    },
    "prod": {
        "gpu": "T4",
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
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0"])
    .pip_install([
        "ultralytics==8.2.68",     # YOLOv8/v10 support
        "opencv-python==4.10.0.84",
        "numpy>=1.24,<2.0",
        "pyyaml>=6.0",
        "onnx>=1.14.0", 
        "fastapi", "boto3"
    ])
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
    
    if file_size == 0:
        raise ValueError("Zip file is empty after creation!")
    
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
    
    # Handle large files that require confirmation
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={value}"
            response = session.get(url, stream=True)
            break
    
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
    inference.spawn(job_id, request, save_to_s3)
    job_status_dict[job_id] = json.dumps({
        "cur_status": "submitted",
        "cur_status_progress": 100, 
        "updated_at": datetime.now().isoformat()
    })
    print("job_id:", job_id)
    return {"job_id": job_id}

@app.function()
@modal.fastapi_endpoint(method="GET")
@modal.concurrent(max_inputs=100)
def check_status(job_id: str):
    """Check job status and progress"""
    print(f"check_status - job_id: '{job_id}' (len={len(job_id)})")
    print(f"check_status - job_id type: {type(job_id)}")
    print(f"check_status - job_id repr: {repr(job_id)}")

    status_json = job_status_dict.get(job_id)
    
    if status_json is None: return {"error": "Job not found"}, 404
    return status_json

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
    print("status checked", zip_path)
    if not os.path.exists(zip_path): 
        print(f"File DNE: {zip_path}")
        return {"error": "Result file not found"}, 404
    print("file exists?")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            num_files = len(zf.namelist())
            print(f"Zip contains {num_files} files")
            
            if num_files == 0: return {"error": "Zip file has no contents"}, 500
    except zipfile.BadZipFile as e: return {"error": f"Invalid zip file: {e}"}, 500

    print("zip stuff kinda resolved?")
    # Copy the zip into temp
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    shutil.copy(zip_path, temp_zip.name)
    
    os.remove(zip_path)
    results_volume.commit()
    
    print("os stuff finished")
    # Remove status
    # if job_id in job_status_dict: 
    del job_status_dict[job_id]
    
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
    from fastapi import HTTPException
    import json
    
    local_vid_path = "sample_vid.mp4"
    
    logger.info("Downloading video to Modal container")
    # vid_url = "https://drive.google.com/file/d/1ya6iuzDMhqCSZG8uRpLsvrNeNA77d8Ew/view?usp=sharing"
    if(download_from_google_drive(request.video_url, local_vid_path) is False): raise Exception("Failed to download link")
    logger.info("Finished downloading video")
    cur_config = Config(env=ENV)
    
    model_path = MOUNT_PATH / Path(cur_config.model_key)
        
    pipeline = InferencePipeline(
        model_path=model_path / "best.pt",
        fps=request.fps,
        confidence_threshold=request.confidence_threshold, 
        logger=logger, 
        batch_size=min(100, request.batch_size)
    )
        
    video_id = f"{str(Path(cur_config.model_key).stem)}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    
    try: 
        res_dirs, res_json_path = pipeline.run_inference_on_video(
            video_path=local_vid_path, 
            video_id=video_id, 
            s3_bucket=cur_config.s3_bucket if save_to_s3 else save_to_s3,
            save_annotated_frames=True
        )
    except Exception as e: 
        print("error!", e)
        raise HTTPException(status_code=500)

    zip_path = Path(f"/results/{job_id}.zip")
    logger.info("Started zipping results")  
    try: zip_directory(res_dirs, [res_json_path], zip_path)
    except Exception as e: raise Exception(f"Error during zip: {e}")
    logger.info("Finished zipping") 
    import os
    file_size = os.path.getsize(zip_path)
    print(f"Zip file size: {file_size:,} bytes")
    if file_size == 0: raise ValueError("Zip file is empty after creation!")  
    
    results_volume.commit()
             
    job_status_dict[job_id] = json.dumps({
        "cur_status": "completed",
        "cur_status_progress": 100, 
        "updated_at": datetime.now().isoformat()
    })

