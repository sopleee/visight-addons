# Smoke-test YOLO fine-tuning and inference on a tiny S3-backed dataset.

from pathlib import Path, PosixPath
from dataclasses import dataclass
import modal
import shutil
import yaml
from datetime import datetime


def _safe_copy_to_mount(src: Path, dst: Path):
    """
    Copy a file to a CloudBucketMount path without preserving metadata.
    shutil.copy2() tries to set atime/mtime (copystat), which is not
    supported by S3-backed mounts and raises PermissionError.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as s, open(dst, "wb") as d:   
        shutil.copyfileobj(s, d)

# ====== CONFIG ======
APP_NAME = "visight-yolo-smoke"
BUCKET_NAME = "visight-data-yusufmoola"
MOUNT_PATH = PosixPath("/bucket")                 
SMOKE_DATA_YAML = MOUNT_PATH / "tmp/smoke_v1" / "data.yaml"
SAVE_DIR = MOUNT_PATH / "models" / "smoke_yolov10n"
STATS_DIR = MOUNT_PATH / "stats" / "training" / "smoke_yolov10n"
RUNS_DIR = Path("/root/data/runs")                


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
        "onnx>=1.14.0"
    ])
)

app = modal.App(APP_NAME, image=image)

def stage_dataset_locally(s3_dataset_root: Path, local_root: Path) -> Path:
    """
    Copy a YOLO dataset from the S3 mount to a local writable directory and
    write a new data.yaml that references local paths. Returns the path to the local data.yaml.
    """
    # Ensure local root exists
    local_root.mkdir(parents=True, exist_ok=True)

    # Paths on S3 mount
    s3_train = s3_dataset_root / "train"
    s3_val   = s3_dataset_root / "valid"
    s3_test  = s3_dataset_root / "test"
    s3_yaml  = s3_dataset_root / "data.yaml"

    if not s3_yaml.exists():
        raise FileNotFoundError(f"Expected data.yaml at {s3_yaml}")

    # Copy splits if they exist
    if s3_train.exists():
        shutil.copytree(s3_train, local_root / "train", dirs_exist_ok=True)
    if s3_val.exists():
        shutil.copytree(s3_val, local_root / "valid", dirs_exist_ok=True)
    if s3_test.exists():
        shutil.copytree(s3_test, local_root / "test", dirs_exist_ok=True)

    # Read original yaml to carry over nc and names
    with open(s3_yaml, "r", encoding="utf-8") as f:
        orig = yaml.safe_load(f)

    # Build local yaml dict
    local_yaml = {
        "train": str((local_root / "train" / "images").resolve()),
        "val":   str((local_root / "valid" / "images").resolve()),
        "nc":    orig.get("nc"),
        "names": orig.get("names"),
    }
    # Add test if present
    if (local_root / "test" / "images").exists():
        local_yaml["test"] = str((local_root / "test" / "images").resolve())

    # Write local data.yaml
    local_yaml_path = local_root / "data.yaml"
    with open(local_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(local_yaml, f, sort_keys=False)

    return local_yaml_path

# ====== TRAIN FUNCTION ======
@app.function(
    gpu="t4",                                  
    cpu=4,
    timeout=60 * 20,                        
    volumes={MOUNT_PATH: modal.CloudBucketMount(BUCKET_NAME, secret=S3_SECRET)},
)
def train_smoke(model_size: str = "yolov10n.pt", epochs: int = 2, imgsz: int = 640, batch: int = 8, fraction: float = 0.25):
    """
    Fine-tune YOLO on the tiny smoke dataset mounted at /bucket/tmp/smoke_v1/data.yaml.
    Writes weights under /root/data/runs and copies best.pt to /bucket/models/smoke_yolov10n/best.pt.
    Also exports best.onnx and copies it to /bucket/models/smoke_yolov10n/best.onnx.
    """
    from ultralytics import YOLO
    import shutil, time

    print(f"[train] Using data: {SMOKE_DATA_YAML}")
    # Stage the dataset from the S3 mount into a local
    local_ds_root = Path("/root/work/smoke_v1")
    local_yaml = stage_dataset_locally(s3_dataset_root=SMOKE_DATA_YAML.parent, local_root=local_ds_root)
    print(f"[train] Staged dataset locally at {local_ds_root}; using {local_yaml}")

    if not SMOKE_DATA_YAML.exists():
        raise FileNotFoundError(f"Missing smoke data.yaml at {SMOKE_DATA_YAML}")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_size)

    t0 = time.time()
    model.train(
        data=str(local_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        fraction=fraction,       
        device=0,
        workers=2,
        seed=117,
        cache=False,
        project=str(RUNS_DIR),
        name="smoke_run",
        exist_ok=True,
        verbose=True,
    )
    dt = time.time() - t0
    print(f"[train] Train wall time: {dt:.1f}s")

    # Save validation/training CSV metrics to S3 (results.csv)
    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    local_results = RUNS_DIR / "smoke_run" / "results.csv"
    if local_results.exists():
        stats_out_dir = STATS_DIR / run_id
        stats_out_dir.mkdir(parents=True, exist_ok=True)
        _safe_copy_to_mount(local_results, stats_out_dir / "results.csv")
        print(f"[train] Exported results.csv -> s3://{BUCKET_NAME}/stats/training/smoke_yolov10n/{run_id}/results.csv")
    else:
        print("[train] WARNING: results.csv not found; check Ultralytics run folder for metrics.")

    # Copy best.pt to S3 mount
    best = RUNS_DIR / "smoke_run" / "weights" / "best.pt"
    if best.exists():
        dst = SAVE_DIR / "best.pt"
        _safe_copy_to_mount(best, dst)  # avoid copystat on S3 mount
        print(f"[train] Exported weights -> {dst}")

        # Export ONNX and copy to S3 mount
        try:
            from ultralytics import YOLO as _YOLO
            onnx_out = _YOLO(str(best)).export(
                format="onnx",
                dynamic=True,      # allow variable image sizes
                imgsz=imgsz,       # match training/inference size
                opset=12           # safe default for wide compatibility
            )
            onnx_src = Path(onnx_out)
            onnx_dst = SAVE_DIR / "best.onnx"
            _safe_copy_to_mount(onnx_src, onnx_dst)
            print(f"[train] Exported ONNX -> {onnx_dst}")
        except Exception as e:
            print(f"[train] WARNING: ONNX export failed: {e}")
    else:
        print("[train] WARNING: best.pt not found; check Ultralytics run folder structure/logs.")

# ====== INFERENCE / THROUGHPUT CHECK ======
@app.function(
    gpu="t4",
    cpu=2,
    timeout=60 * 10,
    volumes={MOUNT_PATH: modal.CloudBucketMount(BUCKET_NAME, secret=S3_SECRET)},
)
def quick_infer(max_imgs: int = 30, imgsz: int = 640, conf: float = 0.25):
    """
    Load /bucket/models/smoke_yolov10n/best.pt and run inference on up to N images.
    Save annotated images + a CSV of predictions to S3 under /bucket/predictions/smoke_yolov10n/<run_id>/
    """
    from ultralytics import YOLO
    import csv, time

    weights = SAVE_DIR / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights at {weights}. Did train_smoke run successfully?")

    # Prefer test split, fallback to val
    test_dir = MOUNT_PATH / "tmp/smoke_v1" / "test" / "images"
    if not test_dir.exists() or not any(test_dir.iterdir()):
        test_dir = MOUNT_PATH / "tmp/smoke_v1" / "valid" / "images"
    if not test_dir.exists():
        raise FileNotFoundError(f"No images found under {test_dir}")

    img_paths = [p for p in test_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
    img_paths = img_paths[:max_imgs]
    if not img_paths:
        raise RuntimeError(f"No test images found in {test_dir}")

    model = YOLO(str(weights))

    # Local output (safe) -> then copy to S3 mount
    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    local_project = Path("/root/data/preds")
    local_name = f"smoke_{run_id}"
    s3_out = MOUNT_PATH / "predictions" / "smoke_yolov10n" / run_id

    # 1) Run inference with save=True (annotated imgs go to local dir)
    t0 = time.time()
    for p in img_paths:
        _ = model.predict(
            source=str(p),
            imgsz=imgsz,
            conf=conf,
            half=True,
            verbose=False,
            save=True,                       # save annotated images
            project=str(local_project),      # local (POSIX) path
            name=local_name,
            exist_ok=True,
        )
    dt = time.time() - t0
    n = len(img_paths)
    print(f"[infer] {n} images in {dt:.2f}s -> {n/dt:.2f} img/s  ({dt/n*1000:.1f} ms/img)")

    # 2) Also emit a CSV with raw predictions
    csv_path = local_project / local_name / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "class_id", "confidence", "x", "y", "w", "h"])
        for p in img_paths:
            # run once more without saving image to collect structured boxes
            results = model.predict(source=str(p), imgsz=imgsz, conf=conf, half=True, verbose=False, save=False)
            for r in results:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    cls_id = int(b.cls.item())
                    conf_v = float(b.conf.item())
                    # xywh in pixels; convert to int-ish for readability
                    xywh = b.xywh[0].tolist()
                    w.writerow([Path(p).name, cls_id, round(conf_v, 4),
                               round(xywh[0], 2), round(xywh[1], 2),
                               round(xywh[2], 2), round(xywh[3], 2)])

    # 3) Copy local outputs -> S3 mount
    s3_out.mkdir(parents=True, exist_ok=True)
    # copy annotated images
    for f in (local_project / local_name).iterdir():
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            _safe_copy_to_mount(f, s3_out / f.name)
    # copy CSV
    _safe_copy_to_mount(csv_path, s3_out / "predictions.csv")

    print(f"[infer] Artifacts -> s3://{BUCKET_NAME}/predictions/smoke_yolov10n/{run_id}/")

# ====== ENTRYPOINT ======
@app.local_entrypoint()
def main(
    do_train: bool = True,
    epochs: int = 2,
    imgsz: int = 640,
    batch: int = 2,
    fraction: float = 1.0,
    max_infer_imgs: int = 30,                                   
):
    if do_train:
        train_smoke.remote(
            model_size="yolov10n.pt",
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            fraction=fraction,
        )
    quick_infer.remote(max_imgs=max_infer_imgs, imgsz=imgsz, conf=0.25)
