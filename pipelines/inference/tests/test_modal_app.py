# test_modal_app.py
import pytest
import modal
from datetime import datetime
import json
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import zipfile
import shutil
from fastapi import HTTPException
from pipelines.inference.model_server import app, InferenceRequest
import os

from pipelines.inference.model_server import (
    submit_job, 
    check_status, 
    InferenceRequest,
    job_status_dict,
    download_result, 
    inference
)

@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Clean up test data before/after each test"""
    yield
    # Clear job status dict
    for key in list(job_status_dict.keys()):
        del job_status_dict[key]

@pytest.fixture
def sample_request():
    return InferenceRequest(
        video_url="https://drive.google.com/file/d/test_id/view",
        fps=12,
        confidence_threshold=0.5,
        batch_size=200
    )

@pytest.fixture
def temp_volume_dir(tmp_path):
    """Create a temporary directory to act as volume mount"""
    volume_dir = tmp_path / "results"
    volume_dir.mkdir()
    return volume_dir

@pytest.fixture
def mock_volume():
    """Mock volume object with commit method"""
    volume = Mock()
    volume.commit = Mock()
    return volume

@pytest.fixture(autouse=True)
def patch_volume_paths(temp_volume_dir, mock_volume):
    """Patch both the volume object and file paths"""
    with patch('pipelines.inference.model_server.results_volume', mock_volume):
        # This patches the /results path used in your code
        with patch.dict(os.environ, {'TEST_VOLUME_PATH': str(temp_volume_dir)}):
            yield temp_volume_dir

class TestModalApp: 

    @patch('pipelines.inference.model_server.inference.spawn')
    def test_submit_creates_job_status(self, mock_spawn, sample_request):
        mock_spawn.return_value = 3
        result = submit_job.local(sample_request, save_to_s3=False)
        job_id = result["job_id"]
        
        status = json.loads(job_status_dict[job_id])
        assert status["cur_status"] == "submitted"
        assert status["cur_status_progress"] == 100
    
    def test_check_status_with_call_id(self):
        """Test checking status with a Modal call_id"""
        import uuid
        job_id = str(uuid.uuid4())
        call_id = "fc-123456"
        
        job_status_dict[job_id] = json.dumps({
            "cur_status": "submitted",
            "call_id": call_id,
            "updated_at": datetime.now().isoformat()
        })
        
        with patch('modal.FunctionCall.from_id') as mock_from_id:
            mock_call = MagicMock()
            mock_call.get.side_effect = TimeoutError()  # Simulate still running
            mock_from_id.return_value = mock_call
            
            result = check_status.local(job_id=job_id)
            
            assert result["cur_status"] == "running"
    
    def test_check_status_nonexistent_job(self):
        """Test checking status of non-existent job"""
        result = check_status.local(job_id="nonexistent-job-id")
        print(type(result))
        assert "error" in list(result[0].keys())
        assert result[0]["error"] == "Job id: nonexistent-job-id not found"
    
    def test_download_result_job_not_found(self):
        """Test downloading result for non-existent job"""
        result = download_result.local(job_id="nonexistent-job-id")
        
        assert result[0]["error"] == "Job not found"
        assert result[1] == 404

    @patch('pipelines.inference.model_server.download_from_google_drive')
    @patch('pipelines.inference.pipeline_remote.InferencePipeline')
    def test_download_result_file_and_cleaned(
        self, mock_pipeline_class, mock_download, sample_request, temp_volume_dir, mock_volume):
        """Test successful download of results"""
        from pipelines.inference.model_server import download_result, job_status_dict
        from fastapi.responses import FileResponse
        
        job_id = "test-job-789"
        job_status_dict[job_id] = json.dumps({
            "cur_status": "completed",
            "cur_status_progress": 100,
            "updated_at": "2024-01-01T00:00:00"
        })
        
        expected_check_path = f"/results/{job_id}.zip"
        
        with patch('os.path.exists') as mock_exists:
            mock_exists.side_effect = lambda p: p == expected_check_path
            
            with patch('zipfile.ZipFile') as mock_zipfile:
                mock_zip_context = Mock()
                mock_zip_context.__enter__ = Mock(return_value=Mock(namelist=Mock(return_value=["results.json"])))
                mock_zip_context.__exit__ = Mock(return_value=False)
                mock_zipfile.return_value = mock_zip_context
                
                with patch('shutil.copy') as mock_copy:
                    result = download_result.local(job_id=job_id)
        
        assert isinstance(result, FileResponse)
        assert result.media_type == "application/zip"
        assert result.filename == f"results_{job_id}.zip"
        
        # # Cleanup
        # mock_volume.commit.assert_called()        
        # assert job_id not in job_status_dict
        
    def test_download_result_file_not_found(self, temp_volume_dir, mock_volume):
        """Test error handling when zip file doesn't exist"""
        from pipelines.inference.model_server import download_result, job_status_dict
        
        job_id = "test-job-missing"
        job_status_dict[job_id] = json.dumps({
            "cur_status": "completed",
            "cur_status_progress": 100
        })
        
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            
            result = download_result.local(job_id=job_id)
        
        # Verify error response
        assert isinstance(result, tuple)
        error_dict, status_code = result
        assert status_code == 404
        assert error_dict["error"] == "Result file not found"
        
        # Job status should NOT be removed on error
        assert job_id in job_status_dict
    
    def test_download_result_job_not_completed(self):
        """Test error handling when zip file doesn't exist"""
        from pipelines.inference.model_server import download_result, job_status_dict
        
        job_id = "test-job-missing"
        job_status_dict[job_id] = json.dumps({
            "cur_status": "not completed",
            "cur_status_progress": 100
        })
        
        result = download_result.local(job_id=job_id)
        
        # Verify error response
        assert isinstance(result, tuple)
        error_dict, status_code = result
        assert status_code == 400
        assert error_dict["error"] == "Job not completed"
        
        # Job status should NOT be removed on error
        assert job_id in job_status_dict

    def test_download_result_empty_zip(self, temp_volume_dir, mock_volume):
        """Test error handling for empty zip files"""
        from pipelines.inference.model_server import download_result, job_status_dict
        
        job_id = "test-job-empty"
        job_status_dict[job_id] = json.dumps({
            "cur_status": "completed",
            "cur_status_progress": 100
        })
                
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            with patch('zipfile.ZipFile', wraps=zipfile.ZipFile) as mock_zipfile:
                mock_zip_context = Mock()
                mock_zip_context.__enter__ = Mock(return_value=Mock(namelist=Mock(return_value=[])))
                mock_zip_context.__exit__ = Mock(return_value=False)
                mock_zipfile.return_value = mock_zip_context
                result = download_result.local(job_id=job_id)
        
        # Verify error response
        assert isinstance(result, tuple)
        error_dict, status_code = result
        assert status_code == 500
        assert "Zip file has no contents" in error_dict["error"]

    def test_download_result_corrupted_zip(self, temp_volume_dir, mock_volume):
        """Test error handling for corrupted zip files"""
        from pipelines.inference.model_server import download_result, job_status_dict
        
        job_id = "test-job-corrupted"
        job_status_dict[job_id] = json.dumps({
            "cur_status": "completed",
            "cur_status_progress": 100
        })
        
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            with patch('zipfile.ZipFile', wraps=zipfile.ZipFile) as mock_zipfile:
                mock_zipfile.side_effect = zipfile.BadZipFile("File is not a zip file")
                result = download_result.local(job_id=job_id)
        
        # Verify error response
        assert isinstance(result, tuple)
        error_dict, status_code = result
        assert status_code == 500
        assert "Invalid zip file" in error_dict["error"]

    def test_download_result_multiple_files_in_zip(self, temp_volume_dir, mock_volume):
        """Test downloading zip with multiple result files"""
        from pipelines.inference.model_server import download_result, job_status_dict
        from fastapi.responses import FileResponse
        
        job_id = "test-job-multi"
        job_status_dict[job_id] = json.dumps({
            "cur_status": "completed",
            "cur_status_progress": 100
        })
        
        # Create zip with multiple files
        zip_path = temp_volume_dir / f"{job_id}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("output1.json", json.dumps({"result": 1}))
            zf.writestr("output2.json", json.dumps({"result": 2}))
            zf.writestr("summary.txt", "Summary of results")
            zf.writestr("plots/plot1.png", b"fake image data")
        
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            with patch('zipfile.ZipFile', wraps=zipfile.ZipFile) as mock_zipfile:
                mock_zip_context = Mock()
                mock_zip_context.__enter__ = Mock(return_value=Mock(namelist=Mock(
                    return_value=["output1.json", "output2.json", "summary.txt", "plots/plot1.png"])))
                mock_zip_context.__exit__ = Mock(return_value=False)
                mock_zipfile.return_value = mock_zip_context
                real_copy = shutil.copy
                with patch('shutil.copy') as mock_copy:
                    mock_copy.side_effect = lambda src, dst: real_copy(str(zip_path), dst)
                    
                    with patch('os.remove'):
                        with patch('tempfile.NamedTemporaryFile') as mock_temp:
                            temp_file_path = temp_volume_dir / "temp_multi.zip"
                            mock_temp_file = Mock()
                            mock_temp_file.name = str(temp_file_path)
                            mock_temp.return_value = mock_temp_file
                            
                            result = download_result.local(job_id=job_id)
        
        assert isinstance(result, FileResponse)
        
        # Verify all files are in the zip
        with zipfile.ZipFile(temp_file_path, 'r') as zf:
            files = zf.namelist()
            assert len(files) == 4
            assert "output1.json" in files
            assert "output2.json" in files
            assert "summary.txt" in files
            assert "plots/plot1.png" in files
    
    def test_debug_list_volume_contents_success(self, temp_volume_dir, mock_volume):
        """Test successful listing of volume contents"""
        from pipelines.inference.model_server import debug_list_volume_contents
        
        # Create some test files in the volume
        (temp_volume_dir / "file1.txt").write_text("content1")
        (temp_volume_dir / "file2.json").write_text('{"key": "value"}')
        (temp_volume_dir / "large_file.zip").write_bytes(b"x" * 1024)  # 1KB file
        
        # Create a subdirectory
        subdir = temp_volume_dir / "subdir"
        subdir.mkdir()
        
        with patch('os.path.exists') as mock_exists, \
            patch('os.listdir') as mock_listdir, \
            patch('os.path.join', side_effect=lambda *args: "/".join(args)), \
            patch('os.path.getsize') as mock_getsize, \
            patch('os.path.isfile') as mock_isfile:
            
            mock_exists.return_value = True
            mock_listdir.return_value = ["file1.txt", "file2.json", "large_file.zip", "subdir"]
            
            # Mock file sizes
            def getsize_side_effect(path):
                if "file1.txt" in path: return 8
                elif "file2.json" in path: return 16
                return 1024
            
            mock_getsize.side_effect = getsize_side_effect
            
            # Mock isfile - subdir should return False
            def isfile_side_effect(path):
                return "subdir" not in path
            
            mock_isfile.side_effect = isfile_side_effect
            
            result = debug_list_volume_contents.local()
        
        assert result["directory"] == "/results"
        assert result["exists"] is True
        assert result["total_files"] == 4
        assert len(result["files"]) == 4
        
        # Check file details
        files_by_name = {f["name"]: f for f in result["files"]}
        
        assert files_by_name["file1.txt"]["size"] == 8
        assert files_by_name["file1.txt"]["is_file"] is True
        
        assert files_by_name["file2.json"]["size"] == 16
        assert files_by_name["file2.json"]["is_file"] is True
        
        assert files_by_name["large_file.zip"]["size"] == 1024
        assert files_by_name["large_file.zip"]["is_file"] is True
        
        assert files_by_name["subdir"]["size"] == 0
        assert files_by_name["subdir"]["is_file"] is False

    def test_debug_list_volume_contents_directory_not_exists(self):
        """Test when /results directory doesn't exist"""
        from pipelines.inference.model_server import debug_list_volume_contents
        
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            
            result = debug_list_volume_contents.local()
        
        assert "error" in result
        assert result["error"] == "/results doesn't exist"
        assert result["exists"] is False

    def test_debug_list_volume_contents_empty_directory(self):
        """Test listing an empty volume directory"""
        from pipelines.inference.model_server import debug_list_volume_contents
        
        with patch('os.path.exists') as mock_exists, \
            patch('os.listdir') as mock_listdir:
            
            mock_exists.return_value = True
            mock_listdir.return_value = []
            
            result = debug_list_volume_contents.local()
        
        assert result["directory"] == "/results"
        assert result["exists"] is True
        assert result["total_files"] == 0
        assert result["files"] == []
    
    def test_inference_success(self, temp_volume_dir, mock_volume, sample_request):
        """Test successful inference execution"""
        from pipelines.inference.model_server import inference, job_status_dict
        from pipelines.configs import config as config_module
        
        mock_config = Mock()
        mock_config.model_key = "test_model/v1"
        mock_config.s3_bucket = "test-bucket"
        job_id = "test-job-success"

        with patch('pipelines.inference.model_server.download_from_google_drive') as mock_download, \
             patch('pipelines.inference.pipeline_remote.InferencePipeline') as mock_pipeline_class, \
             patch('pipelines.inference.model_server.zip_directory') as mock_zip, \
             patch('pipelines.inference.model_server.MOUNT_PATH', temp_volume_dir), \
             patch.object(config_module, 'Config', return_value=mock_config) as mock_config_class, \
             patch('os.path.getsize', return_value=1024):
            
            # Mock download
            mock_download.return_value = True
                        
            # Mock InferencePipeline
            mock_pipeline = Mock()
            res_dirs = [temp_volume_dir / "frames", temp_volume_dir / "annotated"]
            res_json_path = temp_volume_dir / "results.json"
            mock_pipeline.run_inference_on_video.return_value = (res_dirs, res_json_path)
            mock_pipeline_class.return_value = mock_pipeline
            
            # Create model directory
            model_dir = temp_volume_dir / "test_model" / "v1"
            model_dir.mkdir(parents=True)
            (model_dir / "best.pt").touch()
            
            # Run inference
            inference.local(job_id=job_id, request=sample_request, save_to_s3=False)
            
            # Verify download was called
            mock_download.assert_called_once_with(sample_request.video_url, "vid_test-job-success.mp4")
            
            # Verify pipeline was created with correct params
            mock_pipeline_class.assert_called_once()
            call_kwargs = mock_pipeline_class.call_args[1]
            assert call_kwargs['fps'] == sample_request.fps
            assert call_kwargs['confidence_threshold'] == sample_request.confidence_threshold
            assert call_kwargs['batch_size'] == min(60, sample_request.batch_size)
            
            # Verify inference was run
            mock_pipeline.run_inference_on_video.assert_called_once()
            
            # Verify zip was created
            mock_zip.assert_called_once()
            
            # Verify volume was committed
            mock_volume.commit.assert_called_once()
            
            # Verify job status was updated
            assert job_id in job_status_dict
            status = json.loads(job_status_dict[job_id])
            assert status["cur_status"] == "completed"
            assert status["cur_status_progress"] == 100
            assert "updated_at" in status
    
    def test_inference_with_s3_save(self, temp_volume_dir, mock_volume, sample_request):
        """Test inference with S3 saving enabled"""
        from pipelines.inference.model_server import inference
        from pipelines.configs import config as config_module

        job_id = "test-job-s3"
        mock_config = MagicMock()
        mock_config.model_key = "test_model"
        mock_config.s3_bucket = "test-bucket"
        
        with patch('pipelines.inference.model_server.download_from_google_drive', return_value=True), \
             patch('pipelines.inference.pipeline_remote.InferencePipeline') as mock_pipeline_class, \
             patch('pipelines.inference.model_server.zip_directory'), \
             patch('pipelines.inference.model_server.MOUNT_PATH', temp_volume_dir), \
             patch.object(config_module, 'Config', return_value=mock_config) as mock_config_class, \
             patch('os.path.getsize', return_value=2048):
             
            
            mock_pipeline = Mock()
            mock_pipeline.run_inference_on_video.return_value = ([temp_volume_dir], temp_volume_dir / "results.json")
            mock_pipeline_class.return_value = mock_pipeline
            
            # Create model
            (temp_volume_dir / "test_model").mkdir()
            (temp_volume_dir / "test_model" / "best.pt").touch()
            
            # Run with S3 enabled
            inference.local(job_id=job_id, request=sample_request, save_to_s3=True)
            
            # Verify S3 bucket was passed to pipeline
            call_args = mock_pipeline.run_inference_on_video.call_args[1]
            assert call_args['s3_bucket'] == "test-bucket"
    
    def test_inference_download_failure(self, temp_volume_dir, sample_request):
        """Test inference when video download fails"""
        from pipelines.inference.model_server import inference
        
        job_id = "test-job-download-fail"
        
        with patch('pipelines.inference.model_server.download_from_google_drive', return_value=False):
            with pytest.raises(HTTPException) as excinfo:
                inference.local(job_id=job_id, request=sample_request, save_to_s3=False)
        
        assert excinfo.value.status_code == 400
        assert excinfo.value.detail == "Could not download video"

    def test_inference_pipeline_exception(self, temp_volume_dir, mock_volume, sample_request):
        """Test inference when pipeline raises an exception"""
        from pipelines.configs import config as config_module
        
        job_id = "test-job-pipeline-error"
        mock_config = Mock()
        mock_config.model_key = "test_model"

        
        with patch('pipelines.inference.model_server.download_from_google_drive', return_value=True), \
             patch('pipelines.inference.pipeline_remote.InferencePipeline') as mock_pipeline_class, \
             patch.object(config_module, 'Config', return_value=mock_config) as mock_config_class, \
             patch('pipelines.inference.model_server.MOUNT_PATH', temp_volume_dir):
            
            # mock_config_class.return_value = mock_config
            
            # Create model
            (temp_volume_dir / "test_model").mkdir()
            (temp_volume_dir / "test_model" / "best.pt").touch()
            
            # Make pipeline raise exception
            mock_pipeline = Mock()
            mock_pipeline.run_inference_on_video.side_effect = Exception("Pipeline failed")
            mock_pipeline_class.return_value = mock_pipeline
            
            with pytest.raises(HTTPException) as exc_info:
                inference.local(job_id=job_id, request=sample_request, save_to_s3=False)
            
            assert exc_info.value.status_code == 500
            assert exc_info.value.detail == f"Exception occurred during video frame processing or model inference: Pipeline failed"
    
    def test_inference_zip_failure(self, temp_volume_dir, mock_volume, sample_request):
        """Test inference when zipping fails"""
        from pipelines.inference.model_server import inference
        from pipelines.configs import config as config_module
        job_id = "test-job-zip-fail"          
        mock_config = Mock()
        mock_config.model_key = "test_model"

        with patch('pipelines.inference.model_server.download_from_google_drive', return_value=True), \
             patch('pipelines.inference.pipeline_remote.InferencePipeline') as mock_pipeline_class, \
             patch('pipelines.inference.model_server.zip_directory') as mock_zip, \
             patch.object(config_module, 'Config', return_value=mock_config) as mock_config_class, \
             patch('pipelines.inference.model_server.MOUNT_PATH', temp_volume_dir):
            
            mock_pipeline = Mock()
            mock_pipeline.run_inference_on_video.return_value = ([temp_volume_dir], temp_volume_dir / "results.json")
            mock_pipeline_class.return_value = mock_pipeline
            
            # Create model
            (temp_volume_dir / "test_model").mkdir()
            (temp_volume_dir / "test_model" / "best.pt").touch()
            
            # Make zip fail
            mock_zip.side_effect = Exception("Zip failed")
            
            with pytest.raises(HTTPException) as exc_info:
                inference.local(job_id=job_id, request=sample_request, save_to_s3=False)
            
            assert exc_info.value.status_code == 500
            assert exc_info.value.detail == f"Error during zip: Zip failed"
    
    def test_inference_empty_zip_file(self, temp_volume_dir, mock_volume, sample_request):
        """Test inference when zip file is empty"""
        from pipelines.inference.model_server import inference
        from pipelines.configs import config as config_module
        
        job_id = "test-job-empty-zip"
        mock_config = Mock()
        mock_config.model_key = "test_model"

        with patch('pipelines.inference.model_server.download_from_google_drive', return_value=True), \
             patch('pipelines.inference.pipeline_remote.InferencePipeline') as mock_pipeline_class, \
             patch('pipelines.inference.model_server.zip_directory'), \
             patch('pipelines.inference.model_server.MOUNT_PATH', temp_volume_dir), \
             patch.object(config_module, 'Config', return_value=mock_config) as mock_config_class, \
             patch('os.path.getsize', return_value=0):  # Empty file
            
            mock_pipeline = Mock()
            mock_pipeline.run_inference_on_video.return_value = ([temp_volume_dir], temp_volume_dir / "results.json")
            mock_pipeline_class.return_value = mock_pipeline
            
            # Create model
            (temp_volume_dir / "test_model").mkdir()
            (temp_volume_dir / "test_model" / "best.pt").touch()
            
            with pytest.raises(HTTPException) as exc_info:
                inference.local(job_id=job_id, request=sample_request, save_to_s3=False)
            
            assert exc_info.value.status_code == 500
            assert exc_info.value.detail == "Internal error: zip file is empty after creation!"
    
    def test_inference_batch_size_capped(self, temp_volume_dir, mock_volume):
        """Test that batch size is capped at 60"""
        from pipelines.inference.model_server import inference, InferenceRequest
        from pipelines.configs import config as config_module
        
        job_id = "test-job-batch-cap"
        large_batch_request = InferenceRequest(
            video_url="https://drive.google.com/file/d/test/view",
            fps=12,
            confidence_threshold=0.5,
            batch_size=500  # Should be capped at 60
        )
        mock_config = Mock()
        mock_config.model_key = "test_model"
        
        with patch('pipelines.inference.model_server.download_from_google_drive', return_value=True), \
             patch('pipelines.inference.pipeline_remote.InferencePipeline') as mock_pipeline_class, \
             patch('pipelines.inference.model_server.zip_directory'), \
             patch('pipelines.inference.model_server.MOUNT_PATH', temp_volume_dir), \
             patch.object(config_module, 'Config', return_value=mock_config) as mock_config_class, \
             patch('os.path.getsize', return_value=1024):
            
            mock_pipeline = Mock()
            mock_pipeline.run_inference_on_video.return_value = ([temp_volume_dir], temp_volume_dir / "results.json")
            mock_pipeline_class.return_value = mock_pipeline
            
            # Create model
            (temp_volume_dir / "test_model").mkdir()
            (temp_volume_dir / "test_model" / "best.pt").touch()
            
            inference.local(job_id=job_id, request=large_batch_request, save_to_s3=False)
            
            # Verify batch size was capped
            call_kwargs = mock_pipeline_class.call_args[1]
            assert call_kwargs['batch_size'] == 60
    
    def test_inference_video_id_format(self, temp_volume_dir, mock_volume, sample_request):
        """Test that video_id is formatted correctly with timestamp"""
        from pipelines.inference.model_server import inference
        from pipelines.configs import config as config_module
        
        job_id = "test-job-video-id"
        mock_config = Mock()
        mock_config.model_key = "models/yolo_v8/weights"
        
        with patch('pipelines.inference.model_server.download_from_google_drive', return_value=True), \
             patch('pipelines.inference.pipeline_remote.InferencePipeline') as mock_pipeline_class, \
             patch('pipelines.inference.model_server.zip_directory'), \
             patch('pipelines.inference.model_server.MOUNT_PATH', temp_volume_dir), \
             patch('os.path.getsize', return_value=1024), \
             patch.object(config_module, 'Config', return_value=mock_config) as mock_config_class, \
             patch('datetime.datetime') as mock_datetime:
            
            # Mock datetime
            mock_now = Mock()
            mock_now.strftime.return_value = "2024_01_15_10_30_45"
            mock_now.isoformat.return_value = "2024-01-15T10:30:45"
            mock_datetime.now.return_value = mock_now
            
            mock_pipeline = Mock()
            mock_pipeline.run_inference_on_video.return_value = ([temp_volume_dir], temp_volume_dir / "results.json")
            mock_pipeline_class.return_value = mock_pipeline
            
            # Create model
            model_dir = temp_volume_dir / "models" / "yolo_v8" / "weights"
            model_dir.mkdir(parents=True)
            (model_dir / "best.pt").touch()
            
            inference.local(job_id=job_id, request=sample_request, save_to_s3=False)
            
            # Verify video_id format
            call_kwargs = mock_pipeline.run_inference_on_video.call_args[1]
            assert "2024_01_15_10_30_45" in call_kwargs['video_id']
    
    def test_inference_saves_annotated_frames(self, temp_volume_dir, mock_volume, sample_request):
        """Test that save_annotated_frames is always True"""
        from pipelines.inference.model_server import inference
        from pipelines.configs import config as config_module
        
        job_id = "test-job-annotated"
        mock_config = Mock()
        mock_config.model_key = "test_model"
        
        with patch('pipelines.inference.model_server.download_from_google_drive', return_value=True), \
             patch('pipelines.inference.pipeline_remote.InferencePipeline') as mock_pipeline_class, \
             patch('pipelines.inference.model_server.zip_directory'), \
             patch('pipelines.inference.model_server.MOUNT_PATH', temp_volume_dir), \
             patch.object(config_module, 'Config', return_value=mock_config) as mock_config_class, \
             patch('os.path.getsize', return_value=1024):
            
            mock_pipeline = Mock()
            mock_pipeline.run_inference_on_video.return_value = ([temp_volume_dir], temp_volume_dir / "results.json")
            mock_pipeline_class.return_value = mock_pipeline
            
            # Create model
            (temp_volume_dir / "test_model").mkdir()
            (temp_volume_dir / "test_model" / "best.pt").touch()
            
            inference.local(job_id=job_id, request=sample_request, save_to_s3=False)
            
            # Verify save_annotated_frames=True
            call_kwargs = mock_pipeline.run_inference_on_video.call_args[1]
            assert call_kwargs['save_annotated_frames'] is True