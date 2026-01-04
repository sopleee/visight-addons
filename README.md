# visight
Visight is a vision powered platform providing Formula One brand sponsorship analytics.

The Formula 1 sponsorship market generates over $2 billion annually, yet sponsors lack precise metrics to calculate their return on investment. They instead rely on estimated viewership and subjective placement assessments with no quantitative measurement of actual screen time. We address this gap by developing an end-to-end computer vision system that automatically detects and quantifies brand logo exposure in F1 race footage with frame-level precision.

## Our solution includes..
- fine-tuned YOLOv8 object detection model trained on a curated dataset of 3,049 annotated F1 frames containing 86 distinct sponsor brands
- serverless inference architecture on Modal Labs that processes videos asynchronously, extracting frames at configurable rates, performing batch inference on NVIDIA T4 GPUs
- generating comprehensive exposure metrics including visibility duration, screen area, and per-brand analytics. 
The system uses Amazon S3 as a versioned data lake for reproducible training pipelines and stores all artifacts—datasets, model weights, and inference results—with built-in disaster recovery through cross-region replication.

## Model inference latency reduction..
We also focused on latency reduction for video inference by compiling model weights to NVIDIA TensorRT engines with FP16 quantization and dynamic shape support. We further eliminated disk I/O bottlenecks by engineering a streaming in-memory inference pipeline. Our approach scales effectively from single-user development to production workloads. Our architectural design demonstrates practical trade-offs between model accuracy, inference latency, and operational cost across 10×, 100×, and 1000× user growth scenarios.

# Set up

## S3 Access
- Create AWS account and S3 bucket to read and write to. Create or use access key and secret access key 
- Edit pipelines/configs/config.py paths for data storage in S3 to match your intended schema

## Modal Access: https://modal.com/docs/guide 
Create modal secret called "S3-bucket-secret" with attributes "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"

## General Requirements
conda env create -f environment.yml
OR install the requirements.txt

This project was developed in python 3.10.

# Running
Initialize the appropriate development environment by setting the MODAL_ENV environment variable to prod or dev (by default is dev).

Start the application server with: (after saving the fine-tuned model weights in tensorRT format to the appropriate paths)
`modal deploy -m pipelines.inference.model_server

Follow web/FRONTEND.md to start up the front-end server where .env.local endpoints are set to the modal application in the previous. 

NOTE: when rerunning the application after canceling, make sure to delete web/.next before rerunning

## Sample endpoint requests to application server (Windows)
Submit a new job with: 
``` curl.exe -X POST https://MODALUSERNAME--visight-yolo-test-dev-submit-job.modal.run -H "Content-Type: application/json" `
>> -d '{\"video_url\": \"INSERT_VIDEO_URL\" }' ```

It outputs the job_id, which is needed to track the job's status and download the results after it completes. 

``` curl.exe -X GET https://MODALUSERNAME--visight-yolo-test-dev-check-status.modal.run?job_id=JOB_ID ```

``` curl.exe -X GET https://MODALUSERNAME--visight-yolo-test-dev-download-result.modal.run?job_id=JOB_ID --output RESULT_DIR.zip```