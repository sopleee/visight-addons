# test_modal_app.py
import pytest
import modal
from datetime import datetime
import json
from pathlib import Path
from unittest.mock import patch, Mock
import zipfile
import shutil

from pipelines.inference.model_server import app, InferenceRequest

import pytest
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
        result = submit_job.local(sample_request, save_to_s3=False)
        job_id = result["job_id"]
        
        status = json.loads(job_status_dict[job_id])
        assert status["cur_status"] == "submitted"
        assert status["cur_status_progress"] == 100
    
    def test_check_status_existing_job(self):
        """Test checking status of an existing job"""
        import uuid
        job_id = str(uuid.uuid4())
        
        # Setup job status
        job_status_dict[job_id] = json.dumps({
            "cur_status": "processing",
            "cur_status_progress": 50,
            "updated_at": datetime.now().isoformat()
        })
        
        # Check status using .local()
        result = check_status.local(job_id=job_id)
        
        status = json.loads(result)
        assert status["cur_status"] == "processing"
        assert status["cur_status_progress"] == 50
    
    def test_check_status_nonexistent_job(self):
        """Test checking status of non-existent job"""
        result = check_status.local(job_id="nonexistent-job-id")
        print(type(result))
        assert "error" in list(result[0].keys())
        assert result[0]["error"] == "Job not found"
    
    def test_download_result_job_not_found(self):
        """Test downloading result for non-existent job"""
        result = download_result.local(job_id="nonexistent-job-id")
        
        assert result[0]["error"] == "Job not found"
        assert result[1] == 404

    @patch('pipelines.inference.model_server.download_from_google_drive')
    @patch('pipelines.inference.pipeline_remote.InferencePipeline')
    def test_download_result_inference_creates_zip_in_volume(
        self, mock_pipeline_class, mock_download, sample_request, temp_volume_dir, mock_volume):
        """Test successful download of results"""
        from pipelines.inference.model_server import download_result, job_status_dict
        
        job_id = "test-job-789"
        job_status_dict[job_id] = json.dumps({
            "cur_status": "completed",
            "cur_status_progress": 100,
            "updated_at": "2024-01-01T00:00:00"
        })
        
        zip_path = temp_volume_dir / f"{job_id}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("results.json", '{"test": "data"}')        
        expected_check_path = f"/results/{job_id}.zip"
        
        with patch('os.path.exists') as mock_exists:
            mock_exists.side_effect = lambda p: p == expected_check_path
            
            with patch('zipfile.ZipFile') as mock_zipfile:
                mock_zip_context = Mock()
                mock_zip_context.__enter__ = Mock(return_value=Mock(namelist=Mock(return_value=["results.json"])))
                mock_zip_context.__exit__ = Mock(return_value=False)
                mock_zipfile.return_value = mock_zip_context
                
                with patch('shutil.copy') as mock_copy:
                    with patch('os.remove') as mock_remove:
                        with patch('tempfile.NamedTemporaryFile') as mock_temp:
                            mock_temp_file = Mock()
                            mock_temp_file.name = str(temp_volume_dir / "temp.zip")
                            mock_temp.return_value = mock_temp_file
                            
                            with zipfile.ZipFile(mock_temp_file.name, 'w') as zf:
                                zf.writestr("test.txt", "data")
                            
                            result = download_result.local(job_id=job_id)
        
        mock_volume.commit.assert_called()        
        assert job_id not in job_status_dict
        
    def test_download_result_returns_file_response(self, temp_volume_dir, mock_volume):
        """Test that download returns a proper FileResponse with correct attributes"""
        from pipelines.inference.model_server import download_result, job_status_dict
        from fastapi.responses import FileResponse
        
        job_id = "test-job-complete"
        job_status_dict[job_id] = json.dumps({
            "cur_status": "completed",
            "cur_status_progress": 100,
            "updated_at": "2024-01-01T00:00:00"
        })
        
        # Create a real zip file with test content
        zip_path = temp_volume_dir / f"{job_id}.zip"
        test_data = {"result": "test_value", "score": 0.95}
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("results.json", json.dumps(test_data))
            zf.writestr("metadata.txt", "Test metadata")
        
        expected_check_path = f"/results/{job_id}.zip"
        
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            with patch('zipfile.ZipFile', wraps=zipfile.ZipFile) as mock_zipfile:
                with patch('shutil.copy') as mock_copy:
                    # Mock copy to actually copy the file
                    def copy_side_effect(src, dst):
                        shutil.copy(str(zip_path), dst)
                    mock_copy.side_effect = copy_side_effect
                    
                    with patch('os.remove') as mock_remove:
                        with patch('tempfile.NamedTemporaryFile') as mock_temp:
                            temp_file_path = temp_volume_dir / "temp_response.zip"
                            mock_temp_file = Mock()
                            mock_temp_file.name = str(temp_file_path)
                            mock_temp.return_value = mock_temp_file
                            
                            result = download_result.local(job_id=job_id)
        
        # Verify it's a FileResponse
        assert isinstance(result, FileResponse)
        assert result.media_type == "application/zip"
        assert result.filename == f"results_{job_id}.zip"
        
        # Verify the file exists and has content
        assert os.path.exists(temp_file_path)
        with zipfile.ZipFile(temp_file_path, 'r') as zf:
            assert "results.json" in zf.namelist()
            assert "metadata.txt" in zf.namelist()
            
            # Verify content is correct
            content = json.loads(zf.read("results.json"))
            assert content == test_data
        
        # Verify cleanup
        mock_volume.commit.assert_called_once()
        mock_remove.assert_called_once_with(expected_check_path)
        assert job_id not in job_status_dict

    # def test_download_result_file_not_found(self, temp_volume_dir, mock_volume):
    #     """Test error handling when zip file doesn't exist"""
    #     from pipelines.inference.model_server import download_result, job_status_dict
        
    #     job_id = "test-job-missing"
    #     job_status_dict[job_id] = json.dumps({
    #         "cur_status": "completed",
    #         "cur_status_progress": 100
    #     })
        
    #     with patch('os.path.exists') as mock_exists:
    #         mock_exists.return_value = False
            
    #         result = download_result.local(job_id=job_id)
        
    #     # Verify error response
    #     assert isinstance(result, tuple)
    #     error_dict, status_code = result
    #     assert status_code == 404
    #     assert error_dict["error"] == "Result file not found"
        
    #     # Job status should NOT be removed on error
    #     assert job_id in job_status_dict

    # def test_download_result_empty_zip(self, temp_volume_dir, mock_volume):
    #     """Test error handling for empty zip files"""
    #     from pipelines.inference.model_server import download_result, job_status_dict
        
    #     job_id = "test-job-empty"
    #     job_status_dict[job_id] = json.dumps({
    #         "cur_status": "completed",
    #         "cur_status_progress": 100
    #     })
        
    #     # Create an empty zip file
    #     zip_path = temp_volume_dir / f"{job_id}.zip"
    #     with zipfile.ZipFile(zip_path, 'w') as zf:
    #         pass  # Empty zip
        
    #     with patch('os.path.exists') as mock_exists:
    #         mock_exists.return_value = True
            
    #         with patch('zipfile.ZipFile', wraps=zipfile.ZipFile):
    #             result = download_result.local(job_id=job_id)
        
    #     # Verify error response
    #     assert isinstance(result, tuple)
    #     error_dict, status_code = result
    #     assert status_code == 500
    #     assert "no contents" in error_dict["error"].lower()

    # def test_download_result_corrupted_zip(self, temp_volume_dir, mock_volume):
    #     """Test error handling for corrupted zip files"""
    #     from pipelines.inference.model_server import download_result, job_status_dict
        
    #     job_id = "test-job-corrupted"
    #     job_status_dict[job_id] = json.dumps({
    #         "cur_status": "completed",
    #         "cur_status_progress": 100
    #     })
        
    #     # Create a corrupted zip file
    #     zip_path = temp_volume_dir / f"{job_id}.zip"
    #     with open(zip_path, 'w') as f:
    #         f.write("This is not a valid zip file!")
        
    #     with patch('os.path.exists') as mock_exists:
    #         mock_exists.return_value = True
            
    #         with patch('zipfile.ZipFile', wraps=zipfile.ZipFile):
    #             result = download_result.local(job_id=job_id)
        
    #     # Verify error response
    #     assert isinstance(result, tuple)
    #     error_dict, status_code = result
    #     assert status_code == 500
    #     assert "Invalid zip file" in error_dict["error"]

    # def test_download_result_job_not_completed(self, mock_volume):
    #     """Test that download fails for incomplete jobs"""
    #     from pipelines.inference.model_server import download_result, job_status_dict
        
    #     job_id = "test-job-running"
    #     job_status_dict[job_id] = json.dumps({
    #         "cur_status": "running",
    #         "cur_status_progress": 50
    #     })
        
    #     result = download_result.local(job_id=job_id)
        
    #     # Verify error response
    #     assert isinstance(result, tuple)
    #     error_dict, status_code = result
    #     assert status_code == 400
    #     assert error_dict["error"] == "Job not completed"
    #     assert error_dict["status"] == "running"
    #     assert error_dict["progress"] == 50
        
    #     # Job should still be in dict
    #     assert job_id in job_status_dict

    # def test_download_result_multiple_files_in_zip(self, temp_volume_dir, mock_volume):
    #     """Test downloading zip with multiple result files"""
    #     from pipelines.inference.model_server import download_result, job_status_dict
    #     from fastapi.responses import FileResponse
        
    #     job_id = "test-job-multi"
    #     job_status_dict[job_id] = json.dumps({
    #         "cur_status": "completed",
    #         "cur_status_progress": 100
    #     })
        
    #     # Create zip with multiple files
    #     zip_path = temp_volume_dir / f"{job_id}.zip"
    #     with zipfile.ZipFile(zip_path, 'w') as zf:
    #         zf.writestr("output1.json", json.dumps({"result": 1}))
    #         zf.writestr("output2.json", json.dumps({"result": 2}))
    #         zf.writestr("summary.txt", "Summary of results")
    #         zf.writestr("plots/plot1.png", b"fake image data")
        
    #     with patch('os.path.exists') as mock_exists:
    #         mock_exists.return_value = True
            
    #         with patch('zipfile.ZipFile', wraps=zipfile.ZipFile):
    #             with patch('shutil.copy') as mock_copy:
    #                 mock_copy.side_effect = lambda src, dst: shutil.copy(str(zip_path), dst)
                    
    #                 with patch('os.remove'):
    #                     with patch('tempfile.NamedTemporaryFile') as mock_temp:
    #                         temp_file_path = temp_volume_dir / "temp_multi.zip"
    #                         mock_temp_file = Mock()
    #                         mock_temp_file.name = str(temp_file_path)
    #                         mock_temp.return_value = mock_temp_file
                            
    #                         result = download_result.local(job_id=job_id)
        
    #     assert isinstance(result, FileResponse)
        
    #     # Verify all files are in the zip
    #     with zipfile.ZipFile(temp_file_path, 'r') as zf:
    #         files = zf.namelist()
    #         assert len(files) == 4
    #         assert "output1.json" in files
    #         assert "output2.json" in files
    #         assert "summary.txt" in files
    #         assert "plots/plot1.png" in files

    # def test_download_result_cleans_up_original_file(self, temp_volume_dir, mock_volume):
    #     """Test that the original zip is removed after successful download"""
    #     from pipelines.inference.model_server import download_result, job_status_dict
        
    #     job_id = "test-job-cleanup"
    #     job_status_dict[job_id] = json.dumps({
    #         "cur_status": "completed",
    #         "cur_status_progress": 100
    #     })
        
    #     zip_path = temp_volume_dir / f"{job_id}.zip"
    #     with zipfile.ZipFile(zip_path, 'w') as zf:
    #         zf.writestr("data.json", '{"test": true}')
        
    #     expected_remove_path = f"/results/{job_id}.zip"
        
    #     with patch('os.path.exists') as mock_exists:
    #         mock_exists.return_value = True
            
    #         with patch('zipfile.ZipFile', wraps=zipfile.ZipFile):
    #             with patch('shutil.copy'):
    #                 with patch('os.remove') as mock_remove:
    #                     with patch('tempfile.NamedTemporaryFile') as mock_temp:
    #                         mock_temp_file = Mock()
    #                         mock_temp_file.name = str(temp_volume_dir / "temp.zip")
    #                         mock_temp.return_value = mock_temp_file
                            
    #                         # Create temp file for FileResponse
    #                         with zipfile.ZipFile(mock_temp_file.name, 'w') as zf:
    #                             zf.writestr("test.txt", "data")
                            
    #                         result = download_result.local(job_id=job_id)
        
    #     # Verify the original file was removed
    #     mock_remove.assert_called_once_with(expected_remove_path)
        
    #     # Verify volume was committed after removal
    #     mock_volume.commit.assert_called_once()
        
        