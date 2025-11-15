"""
Locust load testing for Visight Video Inference Pipeline - Queue-based Version

This version uses the queue-based API to prevent overwhelming the server:
1. Submit job to queue
2. Poll for job completion
3. Download results when ready

Endpoints:
- POST /submit_job: Submit inference job to queue
- GET /check_status?job_id=<id>: Check job status
- GET /download_result?job_id=<id>: Download completed results

Metrics tracked:
- Queue wait time
- Processing time
- Total end-to-end time
- Memory usage
- Request success/failure rate

Configuration via environment variables:
- ENABLE_LOW_FRAME: Enable low frame scenario (default: true)
- ENABLE_HIGH_FRAME: Enable high frame scenario (default: true)
- LOW_FRAME_FPS: FPS for low frame scenario (default: 6)
- LOW_FRAME_BATCH_SIZE: Batch size for low frame scenario (default: 50)
- HIGH_FRAME_FPS: FPS for high frame scenario (default: 24)
- HIGH_FRAME_BATCH_SIZE: Batch size for high frame scenario (default: 200)
- POLL_INTERVAL: Seconds between status checks (default: 10)
- MAX_POLL_TIME: Maximum time to wait for job completion in seconds (default: 600)
"""

import os
import time
import json
import psutil
import requests
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
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "10"))  # Seconds between status checks
MAX_POLL_TIME = int(os.getenv("MAX_POLL_TIME", "600"))  # Max wait time (10 minutes)

# Queue-based endpoint URLs - These are separate Modal endpoints, not paths
# Get base URL from environment and construct full endpoint URLs
BASE_URL = os.getenv("MODAL_ENDPOINT_URL", "")
if BASE_URL:
    # Extract username and app name from base URL
    # Format: https://username--app-name-env.modal.run
    parts = BASE_URL.replace('https://', '').replace('.modal.run', '').split('--')
    if len(parts) >= 2:
        username = parts[0]
        app_base = parts[1]  # e.g., "visight-yolo-test-dev"
        SUBMIT_JOB_URL = f"https://{username}--{app_base}-submit-job.modal.run"
        CHECK_STATUS_URL = f"https://{username}--{app_base}-check-status.modal.run"
        DOWNLOAD_RESULT_URL = f"https://{username}--{app_base}-download-result.modal.run"
    else:
        logger.error("Could not parse Modal endpoint URL")
        SUBMIT_JOB_URL = CHECK_STATUS_URL = DOWNLOAD_RESULT_URL = ""
else:
    SUBMIT_JOB_URL = CHECK_STATUS_URL = DOWNLOAD_RESULT_URL = ""

logger.info(f"Test Configuration:")
logger.info(f"  Low Frame Enabled: {ENABLE_LOW_FRAME}")
if ENABLE_LOW_FRAME:
    logger.info(f"    FPS: {LOW_FRAME_FPS}, Batch Size: {LOW_FRAME_BATCH_SIZE}")
logger.info(f"  High Frame Enabled: {ENABLE_HIGH_FRAME}")
if ENABLE_HIGH_FRAME:
    logger.info(f"    FPS: {HIGH_FRAME_FPS}, Batch Size: {HIGH_FRAME_BATCH_SIZE}")
logger.info(f"  Poll Interval: {POLL_INTERVAL}s")
logger.info(f"  Max Poll Time: {MAX_POLL_TIME}s")
logger.info(f"Endpoint URLs:")
logger.info(f"  Submit: {SUBMIT_JOB_URL}")
logger.info(f"  Status: {CHECK_STATUS_URL}")
logger.info(f"  Download: {DOWNLOAD_RESULT_URL}")


# Custom metric tracking
class MetricsCollector:
    """Collects custom metrics during load testing"""
    
    def __init__(self):
        self.memory_samples = []
        self.container_counts = []
        self.queue_times = []  # Time spent waiting in queue
        self.processing_times = []  # Time spent processing
        
    def record_memory(self):
        """Record current memory usage"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_samples.append({
            'timestamp': time.time(),
            'memory_mb': memory_mb
        })
    
    def record_queue_time(self, queue_time):
        """Record time spent waiting in queue"""
        self.queue_times.append(queue_time)
    
    def record_processing_time(self, processing_time):
        """Record time spent processing"""
        self.processing_times.append(processing_time)
    
    def get_summary(self):
        """Get summary statistics"""
        memory_values = [s['memory_mb'] for s in self.memory_samples]
        return {
            'memory_avg_mb': sum(memory_values) / len(memory_values) if memory_values else 0,
            'memory_max_mb': max(memory_values) if memory_values else 0,
            'memory_min_mb': min(memory_values) if memory_values else 0,
            'queue_time_avg_s': sum(self.queue_times) / len(self.queue_times) if self.queue_times else 0,
            'queue_time_max_s': max(self.queue_times) if self.queue_times else 0,
            'processing_time_avg_s': sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
            'processing_time_max_s': max(self.processing_times) if self.processing_times else 0,
            'total_jobs': len(self.queue_times),
        }


# Global metrics collector
metrics_collector = MetricsCollector()


class BaseInferenceUser(HttpUser):
    """Base class for inference API users"""
    
    # Mark as abstract so Locust doesn't instantiate it directly
    abstract = True
    
    # Wait time between requests (in seconds)
    # With queue-based system, we can be more aggressive
    wait_time = between(30, 60)
    
    def on_start(self):
        """Called when a user starts"""
        logger.info(f"User {self.__class__.__name__} starting")
        metrics_collector.record_memory()
    
    def on_stop(self):
        """Called when a user stops"""
        logger.info(f"User {self.__class__.__name__} stopping")
        metrics_collector.record_memory()
    
    def _submit_and_wait_for_job(self, video_url, fps, confidence_threshold, batch_size, scenario_name):
        """
        Submit job to queue, poll for completion, and download results
        
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
        
        overall_start_time = time.time()
        job_id = None
        
        # Step 1: Submit job
        logger.info(f"[{scenario_name}] Submitting job to {SUBMIT_JOB_URL}...")
        submit_start = time.time()
        try:
            response = requests.post(
                SUBMIT_JOB_URL,
                json=payload,
                timeout=30
            )
            submit_time = (time.time() - submit_start) * 1000
            
            # Report to Locust
            if response.status_code == 200:
                events.request.fire(
                    request_type="POST",
                    name=f"{scenario_name}_submit",
                    response_time=submit_time,
                    response_length=len(response.content),
                    exception=None,
                    context={}
                )
                try:
                    result = response.json()
                    job_id = result.get("job_id")
                    logger.info(f"[{scenario_name}] Job submitted: {job_id}")
                except Exception as e:
                    logger.error(f"[{scenario_name}] Failed to parse submit response: {e}")
                    logger.error(f"Response: {response.text}")
                    events.request.fire(
                        request_type="POST",
                        name=f"{scenario_name}_submit",
                        response_time=submit_time,
                        response_length=len(response.content),
                        exception=e,
                        context={}
                    )
                    return
            else:
                logger.error(f"[{scenario_name}] Submit failed: {response.status_code} - {response.text}")
                events.request.fire(
                    request_type="POST",
                    name=f"{scenario_name}_submit",
                    response_time=submit_time,
                    response_length=len(response.content) if response.content else 0,
                    exception=Exception(f"Status {response.status_code}"),
                    context={}
                )
                return
        except Exception as e:
            submit_time = (time.time() - submit_start) * 1000
            logger.error(f"[{scenario_name}] Submit request failed: {e}")
            events.request.fire(
                request_type="POST",
                name=f"{scenario_name}_submit",
                response_time=submit_time,
                response_length=0,
                exception=e,
                context={}
            )
            return
        
        if not job_id:
            logger.error(f"[{scenario_name}] No job_id received")
            return
        
        # Step 2: Poll for completion
        logger.info(f"[{scenario_name}] Polling for job completion...")
        poll_start_time = time.time()
        job_completed = False
        job_status = None
        
        while time.time() - poll_start_time < MAX_POLL_TIME:
            time.sleep(POLL_INTERVAL)
            
            status_start = time.time()
            try:
                response = requests.get(
                    CHECK_STATUS_URL,
                    params={"job_id": job_id},
                    timeout=30
                )
                status_time = (time.time() - status_start) * 1000
                
                if response.status_code == 200:
                    events.request.fire(
                        request_type="GET",
                        name=f"{scenario_name}_status",
                        response_time=status_time,
                        response_length=len(response.content),
                        exception=None,
                        context={}
                    )
                    try:
                        status_data = response.json()
                        # Handle both possible response formats
                        if isinstance(status_data, str):
                            import json as json_lib
                            status_data = json_lib.loads(status_data)
                        
                        # The API returns "cur_status" not "status"
                        job_status = status_data.get("cur_status", status_data.get("status"))
                        logger.info(f"[{scenario_name}] Job {job_id} status: {job_status}")
                        
                        if job_status == "completed":
                            job_completed = True
                            break
                        elif job_status == "failed":
                            logger.error(f"[{scenario_name}] Job {job_id} failed")
                            return
                        # else: still processing (submitted, processing, etc.), continue polling
                    except Exception as e:
                        logger.error(f"[{scenario_name}] Failed to parse status response: {e}")
                        logger.error(f"Response: {response.text}")
                        return
                else:
                    logger.error(f"[{scenario_name}] Status check failed: {response.status_code} - {response.text}")
                    events.request.fire(
                        request_type="GET",
                        name=f"{scenario_name}_status",
                        response_time=status_time,
                        response_length=len(response.content) if response.content else 0,
                        exception=Exception(f"Status {response.status_code}"),
                        context={}
                    )
                    return
            except Exception as e:
                status_time = (time.time() - status_start) * 1000
                logger.error(f"[{scenario_name}] Status check request failed: {e}")
                events.request.fire(
                    request_type="GET",
                    name=f"{scenario_name}_status",
                    response_time=status_time,
                    response_length=0,
                    exception=e,
                    context={}
                )
                return
        
        queue_time = time.time() - poll_start_time
        
        if not job_completed:
            logger.error(f"[{scenario_name}] Job {job_id} timed out after {MAX_POLL_TIME}s")
            return
        
        # Step 3: Download results
        logger.info(f"[{scenario_name}] Downloading results...")
        download_start_time = time.time()
        
        try:
            response = requests.get(
                DOWNLOAD_RESULT_URL,
                params={"job_id": job_id},
                timeout=60
            )
            download_time = (time.time() - download_start_time) * 1000
            
            if response.status_code == 200:
                logger.info(f"[{scenario_name}] Results downloaded in {download_time/1000:.2f}s")
                events.request.fire(
                    request_type="GET",
                    name=f"{scenario_name}_download",
                    response_time=download_time,
                    response_length=len(response.content),
                    exception=None,
                    context={}
                )
            else:
                logger.error(f"[{scenario_name}] Download failed: {response.status_code} - {response.text}")
                events.request.fire(
                    request_type="GET",
                    name=f"{scenario_name}_download",
                    response_time=download_time,
                    response_length=len(response.content) if response.content else 0,
                    exception=Exception(f"Status {response.status_code}"),
                    context={}
                )
                return
        except Exception as e:
            download_time = (time.time() - download_start_time) * 1000
            logger.error(f"[{scenario_name}] Download request failed: {e}")
            events.request.fire(
                request_type="GET",
                name=f"{scenario_name}_download",
                response_time=download_time,
                response_length=0,
                exception=e,
                context={}
            )
            return
        
        # Record metrics
        total_time = time.time() - overall_start_time
        processing_time = total_time - queue_time
        
        metrics_collector.record_queue_time(queue_time)
        metrics_collector.record_processing_time(processing_time)
        metrics_collector.record_memory()
        
        logger.info(f"[{scenario_name}] Job {job_id} complete:")
        logger.info(f"  Queue time: {queue_time:.2f}s")
        logger.info(f"  Processing time: {processing_time:.2f}s")
        logger.info(f"  Total time: {total_time:.2f}s")
        
        # Stop this user after completing one job
        self.stop()


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
            
            self._submit_and_wait_for_job(
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
            
            self._submit_and_wait_for_job(
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
    logger.info("LOAD TEST STARTING (Queue-based)")
    logger.info("=" * 80)
    logger.info(f"Target host: {environment.host}")
    logger.info(f"Number of users: {environment.runner.target_user_count if hasattr(environment.runner, 'target_user_count') else 'N/A'}")
    logger.info("=" * 80)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the test stops"""
    logger.info("=" * 80)
    logger.info("LOAD TEST COMPLETE - CUSTOM METRICS")
    logger.info("=" * 80)
    
    summary = metrics_collector.get_summary()
    
    logger.info("Memory Usage:")
    logger.info(f"  Average: {summary['memory_avg_mb']:.2f} MB")
    logger.info(f"  Maximum: {summary['memory_max_mb']:.2f} MB")
    logger.info(f"  Minimum: {summary['memory_min_mb']:.2f} MB")
    
    logger.info("Queue Times:")
    logger.info(f"  Average: {summary['queue_time_avg_s']:.2f}s")
    logger.info(f"  Maximum: {summary['queue_time_max_s']:.2f}s")
    
    logger.info("Processing Times:")
    logger.info(f"  Average: {summary['processing_time_avg_s']:.2f}s")
    logger.info(f"  Maximum: {summary['processing_time_max_s']:.2f}s")
    
    logger.info(f"Total Jobs Completed: {summary['total_jobs']}")
    logger.info("=" * 80)
    
    # Save detailed metrics to file
    metrics_file = "load_test_metrics_queue.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            'custom_metrics': summary,
            'memory_samples': metrics_collector.memory_samples,
            'queue_times': metrics_collector.queue_times,
            'processing_times': metrics_collector.processing_times,
        }, f, indent=2)
    logger.info(f"Detailed metrics saved to: {metrics_file}")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, context, **kwargs):
    """Called after each request"""
    if exception:
        logger.error(f"Request failed: {name} - {exception}")
    else:
        logger.info(f"Request: {name} - Response time: {response_time:.2f}ms")
