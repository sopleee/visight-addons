"""
Locust load testing for Visight Video Inference Pipeline

This file contains two user scenarios:
1. LowFrameUser: Tests with videos that produce fewer frames (lower FPS or shorter duration)
2. HighFrameUser: Tests with videos that produce more frames (higher FPS or longer duration)

Metrics tracked:
- Response time (p50, p95, p99)
- Memory usage (via custom metrics)
- Number of concurrent containers (via Modal API)
- Request success/failure rate

Configuration via environment variables:
- ENABLE_LOW_FRAME: Enable low frame scenario (default: true)
- ENABLE_HIGH_FRAME: Enable high frame scenario (default: true)
- LOW_FRAME_FPS: FPS for low frame scenario (default: 6)
- LOW_FRAME_BATCH_SIZE: Batch size for low frame scenario (default: 50)
- HIGH_FRAME_FPS: FPS for high frame scenario (default: 24)
- HIGH_FRAME_BATCH_SIZE: Batch size for high frame scenario (default: 200)
"""

import os
import time
import json
import psutil
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner, WorkerRunner
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment variables
ENABLE_LOW_FRAME = os.getenv("ENABLE_LOW_FRAME", "true").lower() == "true"
ENABLE_HIGH_FRAME = os.getenv("ENABLE_HIGH_FRAME", "true").lower() == "true"
LOW_FRAME_FPS = int(os.getenv("LOW_FRAME_FPS", "6"))
LOW_FRAME_BATCH_SIZE = int(os.getenv("LOW_FRAME_BATCH_SIZE", "50"))
HIGH_FRAME_FPS = int(os.getenv("HIGH_FRAME_FPS", "24"))
HIGH_FRAME_BATCH_SIZE = int(os.getenv("HIGH_FRAME_BATCH_SIZE", "200"))

logger.info(f"Test Configuration:")
logger.info(f"  Low Frame Enabled: {ENABLE_LOW_FRAME}")
if ENABLE_LOW_FRAME:
    logger.info(f"    FPS: {LOW_FRAME_FPS}, Batch Size: {LOW_FRAME_BATCH_SIZE}")
logger.info(f"  High Frame Enabled: {ENABLE_HIGH_FRAME}")
if ENABLE_HIGH_FRAME:
    logger.info(f"    FPS: {HIGH_FRAME_FPS}, Batch Size: {HIGH_FRAME_BATCH_SIZE}")


# Custom metric tracking
class MetricsCollector:
    """Collects custom metrics during load testing"""
    
    def __init__(self):
        self.memory_samples = []
        self.container_counts = []
        
    def record_memory(self):
        """Record current memory usage"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_samples.append({
            'timestamp': time.time(),
            'memory_mb': memory_mb
        })
        
    def record_container_count(self, count):
        """Record number of active containers"""
        self.container_counts.append({
            'timestamp': time.time(),
            'count': count
        })
    
    def get_stats(self):
        """Get aggregated statistics"""
        if not self.memory_samples:
            return {}
            
        memory_values = [s['memory_mb'] for s in self.memory_samples]
        return {
            'memory_avg_mb': sum(memory_values) / len(memory_values),
            'memory_max_mb': max(memory_values),
            'memory_min_mb': min(memory_values),
            'container_max': max([c['count'] for c in self.container_counts]) if self.container_counts else 0,
        }


# Global metrics collector
metrics_collector = MetricsCollector()


class BaseInferenceUser(HttpUser):
    """Base class for inference API users"""
    
    # Mark as abstract so Locust doesn't instantiate it directly
    abstract = True
    
    # Wait time between requests (in seconds)
    # Given that each request takes 1-2 mins, we want significant wait time
    wait_time = between(60, 120)
    
    def on_start(self):
        """Called when a user starts"""
        logger.info(f"User {self.__class__.__name__} starting")
        metrics_collector.record_memory()
    
    def on_stop(self):
        """Called when a user stops"""
        logger.info(f"User {self.__class__.__name__} stopping")
        metrics_collector.record_memory()
    
    def _make_inference_request(self, video_url, fps, confidence_threshold, batch_size, scenario_name):
        """
        Make an inference request to the Modal endpoint
        
        Args:
            video_url: Google Drive share link to video
            fps: Frames per second to extract
            confidence_threshold: Detection confidence threshold
            batch_size: Batch size for inference
            scenario_name: Name for tracking in metrics
        """
        payload = {
            "video_url": video_url,
            "fps": fps,
            "confidence_threshold": confidence_threshold,
            "batch_size": batch_size
        }
        
        # Record memory before request
        metrics_collector.record_memory()
        
        start_time = time.time()
        
        with self.client.post(
            "/",  # Modal FastAPI endpoint is at root
            json=payload,
            params={"save_to_s3": False},  # Don't save to S3 during load testing
            catch_response=True,
            name=f"inference_{scenario_name}"
        ) as response:
            request_time = time.time() - start_time
            
            if response.status_code == 200:
                # Record successful request
                logger.info(f"{scenario_name} request completed in {request_time:.2f}s")
                response.success()
            else:
                logger.error(f"{scenario_name} request failed: {response.status_code} - {response.text}")
                response.failure(f"Status code: {response.status_code}")
        
        # Record memory after request
        metrics_collector.record_memory()
        
        return response


if ENABLE_LOW_FRAME:
    class LowFrameUser(BaseInferenceUser):
        """
        User that sends requests for videos with LOW frame counts
        
        Configuration controlled by environment variables:
        - LOW_FRAME_FPS: FPS for frame extraction
        - LOW_FRAME_BATCH_SIZE: Batch size for inference
        """
        
        weight = 1  # Equal weight with HighFrameUser
        
        @task
        def inference_low_frames(self):
            """Run inference on a low-frame video"""
            
            # Get video URL from environment or use default
            video_url = os.getenv(
                "LOW_FRAME_VIDEO_URL",
                "https://drive.google.com/file/d/1ya6iuzDMhqCSZG8uRpLsvrNeNA77d8Ew/view?usp=sharing"
            )
            
            self._make_inference_request(
                video_url=video_url,
                fps=LOW_FRAME_FPS,
                confidence_threshold=0.5,
                batch_size=LOW_FRAME_BATCH_SIZE,
                scenario_name="low_frames"
            )


if ENABLE_HIGH_FRAME:
    class HighFrameUser(BaseInferenceUser):
        """
        User that sends requests for videos with HIGH frame counts
        
        Configuration controlled by environment variables:
        - HIGH_FRAME_FPS: FPS for frame extraction
        - HIGH_FRAME_BATCH_SIZE: Batch size for inference
        """
        
        weight = 1  # Equal weight with LowFrameUser
        
        @task
        def inference_high_frames(self):
            """Run inference on a high-frame video"""
            
            # Get video URL from environment or use default
            video_url = os.getenv(
                "HIGH_FRAME_VIDEO_URL",
                "https://drive.google.com/file/d/1ya6iuzDMhqCSZG8uRpLsvrNeNA77d8Ew/view?usp=sharing"
            )
            
            self._make_inference_request(
                video_url=video_url,
                fps=HIGH_FRAME_FPS,
                confidence_threshold=0.5,
                batch_size=HIGH_FRAME_BATCH_SIZE,
                scenario_name="high_frames"
            )


# Event handlers for custom metrics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts"""
    logger.info("=" * 80)
    logger.info("LOAD TEST STARTING")
    logger.info("=" * 80)
    logger.info(f"Target host: {environment.host}")
    logger.info(f"Number of users: {environment.runner.target_user_count if hasattr(environment.runner, 'target_user_count') else 'N/A'}")
    logger.info("=" * 80)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the test stops - print custom metrics"""
    logger.info("=" * 80)
    logger.info("LOAD TEST COMPLETE - CUSTOM METRICS")
    logger.info("=" * 80)
    
    stats = metrics_collector.get_stats()
    
    logger.info(f"Memory Usage:")
    logger.info(f"  Average: {stats.get('memory_avg_mb', 0):.2f} MB")
    logger.info(f"  Maximum: {stats.get('memory_max_mb', 0):.2f} MB")
    logger.info(f"  Minimum: {stats.get('memory_min_mb', 0):.2f} MB")
    logger.info(f"Container Stats:")
    logger.info(f"  Maximum Containers: {stats.get('container_max', 0)}")
    logger.info("=" * 80)
    
    # Save metrics to file
    metrics_file = "load_test_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            'custom_metrics': stats,
            'memory_samples': metrics_collector.memory_samples,
            'container_counts': metrics_collector.container_counts,
        }, f, indent=2)
    logger.info(f"Detailed metrics saved to: {metrics_file}")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, context, **kwargs):
    """Called after each request"""
    if exception:
        logger.error(f"Request failed: {name} - {exception}")
    else:
        # Log p95 latency for inference requests
        if "inference" in name:
            logger.info(f"Request: {name} - Response time: {response_time:.2f}ms")
