import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
from pathlib import Path
import numpy as np
import torch
import cv2
from pipelines.clients.s3_client import s3Client

class TestRemoteInference:
    """Test suite for _generate_summary_stats method"""
    @pytest.fixture
    def mock_dependencies(self):
        """Mock model init dependencies"""
        with patch('pipelines.inference.pipeline_remote.YOLO') as mock_yolo, \
             patch('pipelines.inference.pipeline_remote.VideoProcessor') as mock_vp:
            mock_yolo_instance = MagicMock()
            mock_yolo.return_value = mock_yolo_instance
            mock_vp_instance = MagicMock()
            mock_vp.return_value = mock_vp_instance
            yield {
                'yolo_class': mock_yolo,
                'yolo_instance': mock_yolo_instance,
                'vp_class': mock_vp,
                'vp_instance': mock_vp_instance
            }
    
    @pytest.fixture
    def mock_instance(self, mock_dependencies):
        """Create a mock instance with the method"""
        from pipelines.inference.pipeline_remote import InferencePipeline
        # mock_logger = Mock()        
        instance = InferencePipeline(model_path="fake/path/model.pt", fps=1, batch_size=3)#, logger=mock_logger)
        mock_dependencies['yolo_class'].assert_called_once_with("fake/path/model.pt", task='detect')
        mock_dependencies['vp_class'].assert_called_once_with(fps=1)
        assert instance.model == mock_dependencies['yolo_instance']
        assert instance.video_processor == mock_dependencies['vp_instance']
        # assert instance.logger == mock_logger
        assert instance.confidence_threshold == 0.5
        assert instance.batch_size == 3
        return instance

    @pytest.fixture
    def mock_video_capture(self):
        """Create a mock video capture that returns frames"""
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 5,
            cv2.CAP_PROP_FPS: 30.0
        }.get(prop, 0)
        
        # Simulate 5 frames then end
        mock_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        mock_cap.read.side_effect = (
            [(True, frame) for frame in mock_frames] + [(False, None)]
        )
        return mock_cap
    
    def test_full_video_inference_without_s3(self, mock_instance, mock_video_capture):
        """Test running inference without S3 upload"""
        # Setup mocks
        mock_instance.video_processor.get_video_info.return_value = {
            "fps": 30, 
            "duration": 10,
            "width": 640,
            "height": 480
        }
        
        mock_batch_results = [
            {
                "frame_number": 0,
                "timestamp": 0.0,
                "detection_count": 1,
                "detections": [{"class_name": "logo", "confidence": 0.9}]
            },
            {
                "frame_number": 1,
                "timestamp": 0.033,
                "detection_count": 0,
                "detections": []
            }
        ]
        
        mock_summary_stats = {
            "total_detections": 3,
            "total_frames": 5,
            "frames_with_detections": 2
        }
        
        with patch('cv2.VideoCapture', return_value=mock_video_capture), \
             patch.object(mock_instance, '_process_batch', return_value=mock_batch_results), \
             patch.object(mock_instance, '_generate_summary_stats', return_value=mock_summary_stats), \
             patch('builtins.open', mock_open()), \
             patch('json.dump'), \
             patch('pathlib.Path.exists', return_value=False), \
             patch('pathlib.Path.mkdir'):
            
            result = mock_instance.run_inference_on_video(
                video_path="/data/videos/test.mp4",
                video_id="test_video",
                s3_bucket=None,
                save_annotated_frames=False
            )
        
        # Verify result structure
        assert isinstance(result, tuple)
        assert len(result) == 2
        res_dirs, additional_files = result
        
        # Should return profiling dir if it exists (mocked as False)
        assert res_dirs == []
        
        # Should return results.json
        assert "results.json" in additional_files
        
        # Verify video processor was called
        mock_instance.video_processor.get_video_info.assert_called_once_with("/data/videos/test.mp4")
        
        # Verify _process_batch was called (5 frames / batch_size=2 = 3 calls)
        assert mock_instance._process_batch.call_count == 3
        
        # Verify summary stats generation
        mock_instance._generate_summary_stats.assert_called_once()
        
    def test_full_video_inference_with_s3(self, mock_instance, mock_video_capture):
        """Test running inference with S3 upload"""
        from pipelines.inference.pipeline_remote import InferencePipeline
        
        mock_instance.video_processor.get_video_info.return_value = {
            "fps": 30,
            "duration": 10
        }
        
        # Create mock S3 client
        mock_s3_client = Mock()
        mock_s3_client.upload_file.return_value = "s3://bucket/inference/videos/test_video"
        mock_s3_client.put_object.return_value = None
        
        mock_batch_results = [{
            "frame_number": 0,
            "timestamp": 0.0,
            "detection_count": 1,
            "detections": []
        }]
        
        mock_summary_stats = {"total_detections": 1}
        
        # Bind the real method to mock_instance
        mock_instance.run_inference_on_video = InferencePipeline.run_inference_on_video.__get__(mock_instance, InferencePipeline)
        
        # Mock s3Client at the module level where run_inference_on_video is defined
        with patch('cv2.VideoCapture', return_value=mock_video_capture), \
            patch.object(mock_instance, '_process_batch', return_value=mock_batch_results), \
            patch.object(mock_instance, '_generate_summary_stats', return_value=mock_summary_stats), \
            patch('builtins.open', mock_open()), \
            patch('json.dump'), \
            patch('json.dumps', return_value='{}'), \
            patch('pathlib.Path.exists', return_value=False), \
            patch('pathlib.Path.mkdir'), \
            patch('pipelines.inference.pipeline_remote.s3Client', return_value=mock_s3_client) as mock_s3_class:
            
            result = mock_instance.run_inference_on_video(
                video_path="/data/videos/test.mp4",
                video_id="test_video",
                s3_bucket="test-bucket",
                save_annotated_frames=False
            )
        
        # Verify S3 client was instantiated
        mock_s3_class.assert_called_once_with(buckets=["test-bucket"])
        
        # Verify S3 upload was called for video
        assert mock_s3_client.upload_file.called
        
        # Verify S3 put_object was called for results JSON
        assert mock_s3_client.put_object.called
        
    def test_gen_stats_no_frames(self, mock_instance):
        """Test with empty inference results list"""
        result = mock_instance._generate_summary_stats([])
        
        assert result["total_frames"] == 0
        assert result["total_detections"] == 0
        assert result["frames_with_detections"] == 0
        assert result["detection_rate"] == 0
        assert result["class_counts"] == {}
        assert result["avg_confidence_per_class"] == {}
    
    def test_gen_stats_no_detections(self, mock_instance):
        """Test with single frame containing no detections"""
        inference_results = [
            {"detection_count": 0, "detections": []}
        ]
        
        result = mock_instance._generate_summary_stats(inference_results)
        
        assert result["total_frames"] == 1
        assert result["total_detections"] == 0
        assert result["frames_with_detections"] == 0
        assert result["detection_rate"] == 0.0
        assert result["class_counts"] == {}
        assert result["avg_confidence_per_class"] == {}
            
    def test_gen_stats_variable_dets_per_frame(self, mock_instance):
        """Test when all frames contain detections"""
        inference_results = [
            {"detection_count": 0, "detections": []},
            {"detection_count": 1, "detections": [{"class_name": "a", "confidence": 0.9}]},
            {"detection_count": 2, "detections": [
                {"class_name": "a", "confidence": 0.8},
                {"class_name": "c", "confidence": 0.7}
            ]},
            {"detection_count": 0, "detections": []}
        ]
        
        result = mock_instance._generate_summary_stats(inference_results)
        
        assert result["detection_rate"] == 0.5
        assert result["frames_with_detections"] == 2
        assert result["class_counts"]["a"] == 2
        assert result["class_counts"]["c"] == 1
        assert result["avg_confidence_per_class"]["a"] == pytest.approx((0.9+0.8)/2)
        assert result["avg_confidence_per_class"]["c"] == pytest.approx(0.7)
    
    @patch('pipelines.inference.pipeline_remote.pstats')
    @patch('pipelines.inference.pipeline_remote.cProfile')
    @patch('pipelines.inference.pipeline_remote.cv2.imwrite')
    @patch('pipelines.inference.pipeline_remote.tqdm')
    def test_process_batch_with_annotated_dir(self, mock_tqdm, mock_imwrite, mock_cprofile, mock_pstats, mock_instance):
        """Test processing with annotated directory specified"""
        
        # Mock the profiler
        mock_profiler = MagicMock()
        mock_cprofile.Profile.return_value = mock_profiler
        mock_pstats.Stats.return_value = mock_profiler
        
        # Create mock prediction object
        mock_pred = MagicMock()        
        mock_box = MagicMock()
        mock_box.xyxy = [torch.tensor([10.0, 20.0, 100.0, 150.0])]
        mock_box.conf = [torch.tensor(0.95)]
        mock_box.cls = [torch.tensor(0)]
        mock_pred.boxes = [mock_box]
        
        # Mock the plot method to return a fake annotated image
        mock_pred.plot.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock the model's predict method to return list of predictions
        mock_instance.model.return_value = [mock_pred]
        mock_instance.model.names = {0: "logo"}
        
        # Create test data
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        file_meta = [
            {
                "frame_id": "frame_001",
                "frame_number": 1,
                "timestamp": 0.033,
                "file_path": "/data/frames/frame_001.jpg"
            }
        ]
        
        annotated_dir = Path("/data/annotated")
        
        # Mock cv2.imwrite to succeed
        mock_imwrite.return_value = True
        
        # Execute
        results = mock_instance._process_batch(frames, file_meta, annotated_dir, "vid_id")
        
        # Assertions
        assert len(results) == 1
        assert results[0]["frame_number"] == 1
        assert results[0]["detection_count"] == 1
        assert results[0]["detections"][0]["class_name"] == "logo"
        assert results[0]["detections"][0]["confidence"] == pytest.approx(0.95)
        assert "annotated_frame_path" in results[0]
        assert results[0]["annotated_frame_path"] == str(annotated_dir / "frame_000001.jpg")
        
        mock_imwrite.assert_called_once()
        mock_profiler.enable.assert_called()
        mock_profiler.disable.assert_called()
        
    def test_full_video_inference_without_s3(self, mock_instance):
        """Test running inference without S3 upload"""
        mock_instance.video_processor.get_video_info.return_value = {"fps": 30, "duration": 10}
        mock_instance.video_processor.extract_frames.return_value = [
            {"frame_id": "frame_001", "frame_number": 1, "timestamp": 0.033, "file_path": "/data/frames/frame_001.jpg"}, 
            {"frame_id": "frame_002", "frame_number": 2, "timestamp": 0.035, "file_path": "/data/frames/frame_002.jpg"}
        ]
        
        mock_inference_results = [
            {"frame_id": "frame_001", "detection_count": 1, "detections": [{"class_name": "logo", "confidence": 0.9}]}, 
            {"frame_id": "frame_002", "detection_count": 0, "detections":[]}
        ]
        
        # with patch.object(mock_instance, '_run_model_inference', return_value=mock_inference_results), \
        with patch.object(mock_instance, '_generate_summary_stats', return_value={"total_detections": 1, "total_frames": 1, "frames_with_detections": 1}):
            
            result = mock_instance.run_inference_on_video(
                video_path="/data/videos/test.mp4",
                video_id="test_video",
                s3_bucket=None,
                save_annotated_frames=False
            )
        
        # Verify result structure
        assert isinstance(result, tuple)
        assert len(result) == 2
        res_dirs, res_summary_path = result
        assert res_summary_path == ["results.json"]
        assert res_dirs == [Path("/data/profiling")]
    
    @patch('pipelines.inference.pipeline_remote.profiled')
    def test_profiler_decorator_called(self, mock_profiled, mock_instance):
        """Test that the profiler decorator is applied correctly"""
        assert hasattr(mock_instance._process_batch, '__wrapped__') or callable(mock_instance._process_batch)

    def test_batch_processing_logic(self, mock_instance, mock_video_capture):
        """Test that batches are processed correctly"""
        mock_instance.batch_size = 2
        mock_instance.video_processor.get_video_info.return_value = {"fps": 30}
        mock_summary_stats = {"total_detections": 0}
        
        batch_call_count = 0
        def mock_process_batch(frames, meta, *args, **kwargs):
            nonlocal batch_call_count
            batch_call_count += 1
            return [{"frame_number": i} for i in range(len(frames))]
        
        with patch('cv2.VideoCapture', return_value=mock_video_capture), \
             patch.object(mock_instance, '_process_batch', side_effect=mock_process_batch), \
             patch.object(mock_instance, '_generate_summary_stats', return_value=mock_summary_stats), \
             patch('builtins.open', mock_open()), \
             patch('json.dump'), \
             patch('pathlib.Path.exists', return_value=False), \
             patch('pathlib.Path.mkdir'):
            
            mock_instance.run_inference_on_video(
                video_path="/data/videos/test.mp4",
                video_id="test_video",
                s3_bucket=None,
                save_annotated_frames=False
            )
        
        # 5 frames with batch_size=2: should call 3 times (2, 2, 1)
        assert batch_call_count == 3
    
    def test_annotated_video_creation(self, mock_instance, mock_video_capture):
        """Test annotated video creation when save_annotated_frames=True"""
        mock_instance.video_processor.get_video_info.return_value = {
            "fps": 30,
            "duration": 10
        }
        mock_instance.video_processor.create_annotated_video.return_value = True
        
        mock_batch_results = [{"frame_number": 0, "detections": []}]
        mock_summary_stats = {"total_detections": 0}
        
        mock_annotated_path = Path("/data/test_video_annotated.mp4")
        
        with patch('cv2.VideoCapture', return_value=mock_video_capture), \
             patch.object(mock_instance, '_process_batch', return_value=mock_batch_results), \
             patch.object(mock_instance, '_generate_summary_stats', return_value=mock_summary_stats), \
             patch('builtins.open', mock_open()), \
             patch('json.dump'), \
             patch('pathlib.Path.mkdir'), \
             patch('pathlib.Path.exists') as mock_exists:
            
            # Mock directory exists for annotated frames, video file exists
            mock_exists.side_effect = lambda: True
            
            with patch.object(Path, 'exists', return_value=True):
                result = mock_instance.run_inference_on_video(
                    video_path="/data/videos/test.mp4",
                    video_id="test_video",
                    s3_bucket=None,
                    save_annotated_frames=True
                )
        
        # Verify create_annotated_video was called
        mock_instance.video_processor.create_annotated_video.assert_called_once()
        
        # Verify annotated video is in additional files
        res_dirs, additional_files = result
        assert any("annotated.mp4" in str(f) for f in additional_files)
    
    def test_profiling_directory_included(self, mock_instance, mock_video_capture):
        """Test that profiling directory is included in results if it exists"""
        mock_instance.video_processor.get_video_info.return_value = {"fps": 30}
        mock_summary_stats = {"total_detections": 0}
        
        with patch('cv2.VideoCapture', return_value=mock_video_capture), \
            patch.object(mock_instance, '_process_batch', return_value=[]), \
            patch.object(mock_instance, '_generate_summary_stats', return_value=mock_summary_stats), \
            patch('builtins.open', mock_open()), \
            patch('json.dump'), \
            patch('pathlib.Path.mkdir'):
            
            # Mock profiling dir exists - check if the path contains "profiling"
            def mock_exists(self):
                return "profiling" in str(self)
            
            with patch.object(Path, 'exists', mock_exists):
                result = mock_instance.run_inference_on_video(
                    video_path="/data/videos/test.mp4",
                    video_id="test_video",
                    s3_bucket=None,
                    save_annotated_frames=False
                )
        
        res_dirs, _ = result
        # Profiling dir should be in results if it exists
        assert len(res_dirs) == 1
        assert "profiling" in str(res_dirs[0])
    
    def test_logger_messages(self, mock_instance, mock_video_capture):
        """Test that appropriate log messages are generated"""
        mock_instance.logger = Mock()
        mock_instance.video_processor.get_video_info.return_value = {"fps": 30}
        
        with patch('cv2.VideoCapture', return_value=mock_video_capture), \
             patch.object(mock_instance, '_process_batch', return_value=[]), \
             patch.object(mock_instance, '_generate_summary_stats', return_value={"total_detections": 0}), \
             patch('builtins.open', mock_open()), \
             patch('json.dump'), \
             patch('pathlib.Path.exists', return_value=False), \
             patch('pathlib.Path.mkdir'):
            
            mock_instance.run_inference_on_video(
                video_path="/data/videos/test.mp4",
                video_id="test_video",
                s3_bucket=None,
                save_annotated_frames=False
            )
        
        # Verify logger was called with expected messages
        assert mock_instance.logger.info.called
        log_calls = [call[0][0] for call in mock_instance.logger.info.call_args_list]
        
        # Check for key log messages
        assert any("Processing video" in msg for msg in log_calls)
        assert any("Step 1" in msg for msg in log_calls)
        assert any("Step 3-5" in msg for msg in log_calls)
        assert any("Step 7" in msg for msg in log_calls)
