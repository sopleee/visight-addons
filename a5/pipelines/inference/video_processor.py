import cv2
import cProfile
import pstats
from io import StringIO
from pathlib import Path
import hashlib
from typing import Optional, List
from datetime import datetime, timezone
import json
import subprocess

class VideoProcessor:
    
    def __init__(self, fps: Optional[int] = None):
        self.target_fps = fps
    
    def extract_frames(self, video_path: str, output_dir: str) -> List[dict]:
        """
        Extract frames from video and save to output_dir. 
        
        Returns list of metadata for each frame:
        - frame_id: unique identifier (hash)
        - frame_number: sequential frame number
        - timestamp: timestamp in video (seconds)
        - file_path: path to saved frame image
        
        PROFILED FUNCTION
        """
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise Exception(f"Failed to open video: {video_path}")
            
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if self.target_fps and self.target_fps < video_fps:
                frame_interval = int(video_fps / self.target_fps)
            else:
                frame_interval = 1
            
            print(f"Video FPS: {video_fps}, Total frames: {total_frames}")
            print(f"Extracting every {frame_interval} frame(s)")
            
            frames_metadata = []
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / video_fps
                    frame_filename = f"frame_{frame_count:06d}.jpg"
                    frame_path = output_path / frame_filename
                    
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    frame_id = hashlib.sha256(f"{video_path}_{frame_count}".encode()).hexdigest()
                    
                    frames_metadata.append({
                        "frame_id": frame_id,
                        "frame_number": frame_count,
                        "timestamp": round(timestamp, 3),
                        "file_path": str(frame_path)
                    })
                    
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            
            print(f"Extracted {extracted_count} frames from {total_frames} total frames")
            
            # Save metadata
            metadata_path = output_path / "frames_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    "video_path": str(video_path),
                    "video_fps": video_fps,
                    "total_frames": total_frames,
                    "extracted_frames": extracted_count,
                    "frame_interval": frame_interval,
                    "frames": frames_metadata
                }, f, indent=2)
            
            return frames_metadata
        finally:
            profiler.disable()
            
            # Save profiling stats
            s = StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(30)  # Top 30 functions
            
            # Save profile to output directory
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            profile_output_dir = Path(output_dir).parent / "profiling"
            profile_output_dir.mkdir(parents=True, exist_ok=True)
            profile_file = profile_output_dir / f"extract_frames_{timestamp}.txt"
            profile_file.write_text(s.getvalue())
            print(f"\n[PROFILING] extract_frames profile saved to: {profile_file}")
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get basic information about a video file
        
        Returns:
            Dict with video metadata (fps, duration, resolution, etc.)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception(f"Failed to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            "fps": fps,
            "total_frames": frame_count,
            "width": width,
            "height": height,
            "duration_seconds": round(duration, 2),
            "resolution": f"{width}x{height}"
        }
    
    def create_annotated_video(
        self,
        annotated_frames_dir: str,
        output_video_path: str,
        original_video_path: str,
        fps: float,
        include_audio: bool = True
    ) -> bool:
        """
        Create a video from annotated frames with optional audio from original video.
        
        Args:
            annotated_frames_dir: Directory containing annotated frames (frame_000000.jpg, etc.)
            output_video_path: Path where the output video will be saved
            original_video_path: Path to original video (for audio extraction)
            fps: Frame rate for the output video
            include_audio: Whether to include audio from original video
            
        Returns:
            True if successful, False otherwise
        """
        annotated_dir = Path(annotated_frames_dir)
        output_path = Path(output_video_path)
        
        if not annotated_dir.exists():
            print(f"Error: Annotated frames directory not found: {annotated_dir}")
            return False
        
        # Check if frames exist
        frame_files = sorted(annotated_dir.glob("frame_*.jpg"))
        if not frame_files:
            print(f"Error: No annotated frames found in {annotated_dir}")
            return False
        
        print(f"Found {len(frame_files)} annotated frames")
        
        try:
            if include_audio:
                # Two-step process: create video, then add audio
                temp_video = output_path.parent / f"{output_path.stem}_temp.mp4"
                
                # Step 1: Create video from frames
                print("Step 1: Creating video from frames...")
                frames_cmd = [
                    'ffmpeg',
                    '-y',  # Overwrite output file
                    '-framerate', str(fps),
                    '-pattern_type', 'glob',
                    '-i', str(annotated_dir / 'frame_*.jpg'),
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-preset', 'medium',
                    '-crf', '23',
                    str(temp_video)
                ]
                
                result = subprocess.run(
                    frames_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode != 0:
                    print(f"FFmpeg frames error: {result.stderr}")
                    return False
                
                print(f"✓ Created video without audio: {temp_video}")
                
                # Step 2: Add audio from original video
                print("Step 2: Adding audio from original video...")
                audio_cmd = [
                    'ffmpeg',
                    '-y',
                    '-i', str(temp_video),
                    '-i', str(original_video_path),
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-map', '0:v:0',  # video from first input
                    '-map', '1:a:0?',  # audio from second input (optional)
                    '-shortest',  # match shortest stream
                    str(output_path)
                ]
                
                result = subprocess.run(
                    audio_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                # Clean up temp file
                if temp_video.exists():
                    temp_video.unlink()
                
                if result.returncode != 0:
                    print(f"FFmpeg audio warning: {result.stderr}")
                    # If audio merge fails, just use the video without audio
                    if temp_video.exists():
                        temp_video.rename(output_path)
                    print("⚠ Created video without audio (audio merge failed)")
                else:
                    print(f"✓ Created video with audio: {output_path}")
            
            else:
                # Direct creation without audio
                print("Creating video without audio...")
                cmd = [
                    'ffmpeg',
                    '-y',
                    '-framerate', str(fps),
                    '-pattern_type', 'glob',
                    '-i', str(annotated_dir / 'frame_*.jpg'),
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-preset', 'medium',
                    '-crf', '23',  # Quality: 0 (lossless) to 51 (worst), 23 is default
                    str(output_path)
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode != 0:
                    print(f"FFmpeg error: {result.stderr}")
                    return False
                
                print(f"✓ Created video: {output_path}")
            
            # Verify the video was created
            if not output_path.exists():
                print(f"Error: Output video not found at {output_path}")
                return False
            
            file_size = output_path.stat().st_size
            print(f"Video size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
            
            return True
            
        except subprocess.TimeoutExpired:
            print("Error: FFmpeg timed out (>5 minutes)")
            return False
        except Exception as e:
            print(f"Error creating video: {e}")
            return False