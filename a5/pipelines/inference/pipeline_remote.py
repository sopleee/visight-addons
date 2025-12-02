import argparse
from pathlib import Path
import json
import tempfile
import cv2
import numpy as np
from pipelines.inference.video_processor import VideoProcessor
from tqdm import tqdm
from ultralytics import YOLO
from pathlib import Path
from pipelines.clients.s3_client import s3Client
from typing import Optional, Generator
import logging
import math
import cProfile
import pstats
from io import StringIO
from datetime import datetime, timezone

def profiled(name: Optional[str] = None, stats_limit: int = 50, outdir: Optional[Path] = None):
    """Decorator that profiles a function with cProfile and writes stats."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            try:
                return func(*args, **kwargs)
            finally:
                profiler.disable()
                s = StringIO()
                pstats.Stats(profiler, stream=s).sort_stats('cumulative').print_stats(stats_limit)
                ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                output_dir = outdir or Path("/data/profiling")
                output_dir.mkdir(parents=True, exist_ok=True)
                fname = f"{(name or func.__name__)}_{ts}.txt"
                profile_file = output_dir / fname
                profile_file.write_text(s.getvalue())
                print(f"\n[PROFILING] {func.__name__} profile saved to: {profile_file}")
        return wrapper
    return decorator

class InferencePipeline:
    """
    Optimized Inference pipeline: In-Memory processing (No Disk I/O for raw frames)
    """
    
    def __init__(self, model_path: str, fps: int = None, 
                 confidence_threshold: float = 0.5, logger:Optional[logging.Logger] = None, batch_size: int = 80):
        self.video_processor = VideoProcessor(fps=fps)
        self.confidence_threshold = confidence_threshold        
        self.model = YOLO(model_path, task='detect')
        self.logger = logger
        self.batch_size = batch_size
    
    def run_inference_on_video(self, video_path: str, video_id: str, 
                               s3_bucket: Optional[str] = None, 
                               save_annotated_frames: bool = True) -> tuple:
        
        msg = f"\n{'='*60}\nProcessing video: {video_id} | Batch Size: {self.batch_size}\n{'='*60}\n"
        if self.logger: self.logger.info(msg)
        else: print(msg)
        
        # --- Step 1: Metadata ---
        if self.logger: self.logger.info("Step 1: Extracting video metadata...")
        else: print("Step 1: Extracting video metadata...")
        video_info = self.video_processor.get_video_info(video_path)
        
        # --- Step 2: Upload Video to S3 ---
        s3_client = s3Client(buckets=[s3_bucket]) if s3_bucket else None
        video_s3_path = None
        if s3_client:
            video_key = f"inference/videos/{video_id}"
            if self.logger: self.logger.info("Step 2: Uploading video to S3...")
            video_s3_path = s3_client.upload_file(video_path, video_key, content_type='video/mp4')

        # --- Setup Directories ---
        data_dir = Path("/data")
        # Note: We do NOT create a 'frames' directory anymore, we process in RAM.
        annotated_dir = data_dir / "annotated" if save_annotated_frames else None
        if annotated_dir:
            annotated_dir.mkdir(parents=True, exist_ok=True)

        # --- Step 3, 4, 5: Streaming Inference Loop ---
        # We combine extraction, inference, and annotation into one pass to minimize memory/disk ops
        if self.logger: self.logger.info("\nStep 3-5: Streaming Inference (In-Memory)...")
        else: print("\nStep 3-5: Streaming Inference (In-Memory)...")

        inference_results = []
        frame_buffer = []
        frame_meta_buffer = []
        
        # Open Video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Determine frame skip if fps override is used (simple version)
        # Note: For maximum speed and simplicity, this implementation processes every frame
        # If specific fps sampling is needed, we would add logic to skip cap.read()
        
        pbar = tqdm(total=total_frames, desc="Inference Progress")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add to batch
            frame_buffer.append(frame)
            frame_meta_buffer.append({
                "frame_number": frame_idx,
                "timestamp": frame_idx / video_fps
            })
            
            # Run Batch if full
            if len(frame_buffer) >= self.batch_size:
                batch_results = self._process_batch(
                    frame_buffer, 
                    frame_meta_buffer, 
                    annotated_dir, 
                    video_id,
                    s3_client
                )
                inference_results.extend(batch_results)
                # Clear buffers
                frame_buffer = []
                frame_meta_buffer = []
            
            frame_idx += 1
            pbar.update(1)

        # Process remaining frames
        if frame_buffer:
            batch_results = self._process_batch(
                frame_buffer, 
                frame_meta_buffer, 
                annotated_dir, 
                video_id,
                s3_client
            )
            inference_results.extend(batch_results)
            
        cap.release()
        pbar.close()

        # --- Step 7: Statistics ---
        if self.logger: self.logger.info("\nStep 7: Generating statistics...")
        summary_stats = self._generate_summary_stats(inference_results)
        
        # --- Step 8: Results & Upload ---
        pipeline_results = {
            "video_id": video_id,
            "video_info": video_info,
            "video_s3_path": video_s3_path,
            "confidence_threshold": self.confidence_threshold,
            "total_frames": total_frames,
            # "frames": [], # Removed huge list of raw frames to save JSON size
            "inference_results": inference_results,
            "summary_stats": summary_stats
        }

        # Save JSON
        res_json_path = "results.json"
        
        # Clean up JSON for save (remove local paths if needed)
        # Creating a copy for JSON dump
        json_output = pipeline_results.copy()
        with open(res_json_path, 'w') as f:
            json.dump(json_output, f, indent=4)
            
        if s3_client:
            results_key = f"inference/results/{video_id}/results.json"
            results_bytes = json.dumps(json_output).encode('utf-8')
            s3_client.put_object(results_key, results_bytes, content_type='application/json')
            if self.logger: self.logger.info(f"Results JSON uploaded to S3")

        # --- Step 9: Create Annotated Video ---
        annotated_video_path = None
        if annotated_dir and annotated_dir.exists():
            if self.logger: 
                self.logger.info("\nStep 9: Creating annotated video...")
            else:
                print("\nStep 9: Creating annotated video...")
            
            video_output_path = data_dir / f"{video_id}_annotated.mp4"
            
            success = self.video_processor.create_annotated_video(
                annotated_frames_dir=str(annotated_dir),
                output_video_path=str(video_output_path),
                original_video_path=video_path,
                fps=video_info['fps'],
                include_audio=True
            )
            
            if success:
                annotated_video_path = video_output_path
                if self.logger:
                    self.logger.info(f"✓ Annotated video created: {annotated_video_path}")
                
                # Upload to S3 if enabled
                if s3_client:
                    video_key = f"inference/videos/{video_id}_annotated.mp4"
                    video_s3_path = s3_client.upload_file(
                        str(annotated_video_path), 
                        video_key, 
                        content_type='video/mp4'
                    )
                    if self.logger:
                        self.logger.info(f"✓ Annotated video uploaded to S3: {video_s3_path}")
                    pipeline_results["annotated_video_s3_path"] = video_s3_path
            else:
                if self.logger:
                    self.logger.warning("⚠ Failed to create annotated video")

        msg = f"\nPipeline Complete. Detections: {summary_stats['total_detections']}"
        if self.logger: self.logger.info(msg)
        
        # Return directories to zip (Annotated + Profiling) and files
        profiling_dir = Path("/data/profiling")
        res_dirs = [profiling_dir] if profiling_dir.exists() else []
        
        # Add video to files list if it exists
        additional_files = [res_json_path]
        if annotated_video_path and annotated_video_path.exists():
            additional_files.append(str(annotated_video_path))
        
        return res_dirs, additional_files

    @profiled(name="process_batch", stats_limit=20)
    def _process_batch(self, frames: list[np.ndarray], meta: list[dict], 
                       annotated_dir: Path, video_id: str, s3_client: Optional[s3Client]) -> list:
        """
        Runs inference on a RAM batch, annotates, and handles S3 uploads for annotated frames.
        """
        batch_results = []
        
        # 1. Run Inference (TensorRT happens here)
        # verbose=False speeds up loop significantly
        predictions = self.model(frames, conf=self.confidence_threshold, verbose=False)
        
        # 2. Process Results
        for i, pred in enumerate(predictions):
            frame_meta = meta[i]
            
            # Extract boxes
            dets = []
            if pred.boxes:
                for box in pred.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    cls_name = self.model.names[cls]
                    
                    dets.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "class_name": cls_name
                    })
            
            result_entry = {
                "frame_number": frame_meta["frame_number"],
                "timestamp": frame_meta["timestamp"],
                "detections": dets,
                "detection_count": len(dets)
            }

            # 3. Annotation (Only if folder exists)
            # Optimization: Only save/upload if detections found OR if you strictly need every single frame annotated
            # Currently set to: Save all frames if annotated_dir exists
            if annotated_dir:
                # pred.plot() is highly optimized C++ plotting
                annotated_img = pred.plot() 
                
                filename = f"frame_{frame_meta['frame_number']:06d}.jpg"
                save_path = annotated_dir / filename
                cv2.imwrite(str(save_path), annotated_img)
                result_entry["annotated_frame_path"] = str(save_path)

                # Inline S3 Upload for Annotated Frame (Optional - can be slow)
                if s3_client:
                    annotated_key = f"inference/frames/{video_id}/annotated/{filename}"
                    s3_path = s3_client.upload_file(save_path, annotated_key, content_type='image/jpeg')
                    result_entry["annotated_s3_path"] = s3_path

            batch_results.append(result_entry)
            
        return batch_results

    def _generate_summary_stats(self, inference_results: list) -> dict:
        total_detections = sum(r["detection_count"] for r in inference_results)
        frames_with_detections = sum(1 for r in inference_results if r["detection_count"] > 0)
        
        class_counts = {}
        class_confidences = {}
        
        for result in inference_results:
            for det in result["detections"]:
                cname = det["class_name"]
                class_counts[cname] = class_counts.get(cname, 0) + 1
                if cname not in class_confidences: class_confidences[cname] = []
                class_confidences[cname].append(det["confidence"])
        
        avg_conf = {k: sum(v)/len(v) for k, v in class_confidences.items()}
        
        return {
            "total_frames": len(inference_results),
            "total_detections": total_detections,
            "frames_with_detections": frames_with_detections,
            "detection_rate": frames_with_detections / len(inference_results) if inference_results else 0,
            "class_counts": class_counts,
            "avg_confidence_per_class": avg_conf
        }