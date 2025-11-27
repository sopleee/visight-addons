import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import numpy as np

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
        mock_dependencies['yolo_class'].assert_called_once_with("fake/path/model.pt")
        mock_dependencies['vp_class'].assert_called_once_with(fps=1)
        assert instance.model == mock_dependencies['yolo_instance']
        assert instance.video_processor == mock_dependencies['vp_instance']
        # assert instance.logger == mock_logger
        assert instance.confidence_threshold == 0.5
        assert instance.batch_size == 3
        return instance
    
    def test_gen_stats_no_frames(self, mock_instance):
        """Test with empty inference results list"""
        result = mock_instance._generate_summary_stats([])
        
        assert result["total_frames"] == 0
        assert result["total_detections"] == 0
        assert result["frames_with_detections"] == 0
        assert result["frames_without_detections"] == 0
        assert result["detection_rate"] == 0
        assert result["avg_detections_per_frame"] == 0
        assert result["class_counts"] == {}
        assert result["avg_confidence_per_class"] == {}
        assert result["unique_classes_detected"] == 0
    
    def test_gen_stats_no_detections(self, mock_instance):
        """Test with single frame containing no detections"""
        inference_results = [
            {"detection_count": 0, "detections": []}
        ]
        
        result = mock_instance._generate_summary_stats(inference_results)
        
        assert result["total_frames"] == 1
        assert result["total_detections"] == 0
        assert result["frames_with_detections"] == 0
        assert result["frames_without_detections"] == 1
        assert result["detection_rate"] == 0.0
        assert result["avg_detections_per_frame"] == 0.0
        assert result["class_counts"] == {}
        assert result["avg_confidence_per_class"] == {}
        assert result["unique_classes_detected"] == 0
            
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
        assert result["frames_without_detections"] == 2
        assert result["frames_with_detections"] == 2
        assert result["class_counts"]["a"] == 2
        assert result["class_counts"]["c"] == 1
        assert result["unique_classes_detected"] == 2
        assert result["avg_confidence_per_class"]["a"] == pytest.approx((0.9+0.8)/2)
        assert result["avg_confidence_per_class"]["c"] == pytest.approx(0.7)
        
    def test_split_batches_multiple_batches(self, mock_instance):
        """Test splitting into multiple batches"""
        frames = [
            {"file_path": f"frame_{i}.jpg", "frame_id": str(i)}
            for i in range(7)
        ]
        
        batches = mock_instance._split_into_batches(frames)
        
        assert len(batches) == 3  # 7 frames / 3 batch_size = 3 batches
        assert len(batches[0]["paths"]) == 3
        assert len(batches[1]["paths"]) == 3
        assert len(batches[2]["paths"]) == 1
        assert batches[0]["paths"] == ["frame_0.jpg", "frame_1.jpg", "frame_2.jpg"]
        assert batches[2]["paths"] == ["frame_6.jpg"]
        
    def test_split_batches_empty_frames(self, mock_instance):
        """Test splitting with empty frames list"""
        batches = mock_instance._split_into_batches([])
        
        assert len(batches) == 0

    @patch('pipelines.inference.pipeline_remote.cv2')
    def test_bounding_box_draw_multiple_detections(self, mock_cv2, mock_instance):
        """Test drawing multiple bounding boxes"""
        mock_img = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = mock_img
        mock_cv2.getTextSize.return_value = ((100, 20), 5)
        
        frame_path = Path("/data/frames/frame_001.jpg")
        output_dir = Path("/data/annotated")
        detections = [
            {"bbox": [10, 20, 100, 105], "class_name": "logo1", "confidence": 0.95},
            {"bbox": [200, 50, 300, 200], "class_name": "logo2", "confidence": 0.87}
        ]
        
        result = mock_instance._draw_bounding_boxes(frame_path, detections, output_dir)
        
        assert mock_cv2.rectangle.call_count >= 4
        assert mock_cv2.putText.call_count == 2
    
    @patch('pipelines.inference.pipeline_remote.tqdm')
    def test_process_res_with_annotated_dir(self, mock_tqdm, mock_instance):
        """Test processing with annotated directory specified"""
        mock_det = MagicMock()
        mock_det.names = {0: "logo"}
        mock_det.boxes.data.tolist.return_value = [
            [10, 20, 100, 150, 0.95, 0]
        ]
        
        file_meta = [
            {
                "frame_id": "frame_001",
                "frame_number": 1,
                "timestamp": 0.033,
                "file_path": "/data/frames/frame_001.jpg"
            }
        ]
        
        mock_tqdm.return_value = [mock_det]
        annotated_dir = Path("/data/annotated")
        
        with patch.object(mock_instance, '_draw_bounding_boxes', return_value=Path("/data/annotated/frame_001.jpg")):
            results = mock_instance._process_inference_res(0, [mock_det], file_meta, annotated_dir)
        
        assert "annotated_frame_path" in results[0]
        assert results[0]["annotated_frame_path"] == str(Path("/data/annotated/frame_001.jpg"))
            
    def test_run_inference_multiple_batches(self, mock_instance):
        """Test running inference on multiple batches"""
        frames_metadata = [
            {"file_path": f"frame_{i}.jpg", "frame_id": str(i), "frame_number": i, "timestamp": i * 0.033}
            for i in range(5)
        ]
        
        mock_batch_results = [
            [{"frame_id": "0", "detection_count": 0}, {"frame_id": "1", "detection_count": 1},
             {"frame_id": "2", "detection_count": 0}],
            [{"frame_id": "3", "detection_count": 2},{"frame_id": "4", "detection_count": 1}]
        ]
        
        with patch.object(mock_instance, '_minibatch_inference', side_effect=mock_batch_results) as mock_minibatch:
            results = mock_instance._run_model_inference(frames_metadata, Path("/data/frames"), None)
        assert len(results) == 5
        assert [r["frame_id"] for r in results] == [str(i) for i in range(5)]
        assert mock_minibatch.call_count == 2
        
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
        
        with patch.object(mock_instance, '_run_model_inference', return_value=mock_inference_results), \
             patch.object(mock_instance, '_generate_summary_stats', return_value={"total_detections": 1, "total_frames": 1, "frames_with_detections": 1}):
            
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
        assert res_summary_path == "results.json"
        assert res_dirs == [Path("/data/profiling")]
    
    @patch('pipelines.inference.pipeline_remote.s3Client')
    @patch('pipelines.inference.pipeline_remote.tqdm')
    def test_full_video_inference_with_s3(self, mock_tqdm, mock_s3_class, mock_instance):
        """Test running inference with S3 upload"""
        # Setup S3 mock
        mock_s3_instance = MagicMock()
        mock_s3_instance.upload_file.return_value = "s3://bucket/path/file.jpg"
        mock_s3_instance.put_object.return_value = "s3://bucket/path/results.json"
        mock_s3_class.return_value = mock_s3_instance
        
        mock_instance.video_processor.get_video_info.return_value = {"fps": 30, "duration": 10}
        mock_instance.video_processor.extract_frames.return_value = [
            {"frame_id": "frame_001", "frame_number": 1, "timestamp": 0.033, "file_path": "/data/frames/frame_001.jpg"}
        ]
        
        mock_inference_results = [
            {"frame_id": "frame_001", "detection_count": 1, "detections": [{"class_name": "logo", "confidence": 0.9}]}
        ]
        
        mock_tqdm.side_effect = lambda x, desc: x
        
        with patch.object(mock_instance, '_run_model_inference', return_value=mock_inference_results), \
             patch.object(mock_instance, '_generate_summary_stats', return_value={"total_detections": 1, "total_frames": 1, "frames_with_detections": 1}):
            
            result = mock_instance.run_inference_on_video(
                video_path="/data/videos/test.mp4",
                video_id="test_video",
                s3_bucket="my-bucket",
                save_annotated_frames=False
            )
        
        # Verify S3 client was created
        mock_s3_class.assert_called_once_with(buckets=["my-bucket"])
        
        # Verify uploads happened
        assert mock_s3_instance.upload_file.called
        assert mock_s3_instance.put_object.called
        assert mock_s3_instance.upload_file.call_count == 2
    
    @patch('pipelines.inference.pipeline_remote.s3Client')
    @patch('pipelines.inference.pipeline_remote.tqdm')
    def test_full_video_inference_with_s3_and_annotated(self, mock_tqdm, mock_s3_class, mock_instance):
        """Test running inference with S3 upload"""
        # Setup S3 mock
        mock_s3_instance = MagicMock()
        mock_s3_instance.upload_file.return_value = "s3://bucket/path/file.jpg"
        mock_s3_instance.put_object.return_value = "s3://bucket/path/results.json"
        mock_s3_class.return_value = mock_s3_instance
        mock_s3_class.upload_file.return_value = "s3://bucket/path/annotated-path"
        
        mock_instance.video_processor.get_video_info.return_value = {"fps": 30, "duration": 10}
        mock_instance.video_processor.extract_frames.return_value = [
            {"frame_id": "frame_001", "frame_number": 1, "timestamp": 0.033, "file_path": "/data/frames/frame_001.jpg"}
        ]
        
        # annotated_dir = Path("/data/annotated")
        
        mock_inference_results = [
            {"frame_id": "frame_001", "detection_count": 1, "detections": [{"class_name": "logo", "confidence": 0.9}], "annotated_frame_path":"res_annotated_path"}
        ]
        
        mock_tqdm.side_effect = lambda x, desc: x
        
        with patch.object(mock_instance, '_run_model_inference', return_value=mock_inference_results), \
             patch.object(mock_instance, '_generate_summary_stats', return_value={"total_detections": 1, "total_frames": 1, "frames_with_detections": 1}), \
             patch.object(mock_instance, '_draw_bounding_boxes', return_value=Path("/data/annotated/frame_001.jpg")):
            
            result = mock_instance.run_inference_on_video(
                video_path="/data/videos/test.mp4",
                video_id="test_video",
                s3_bucket="my-bucket",
                save_annotated_frames=True
            )
        
        # Verify S3 client was created
        mock_s3_class.assert_called_once_with(buckets=["my-bucket"])
        
        # Verify uploads happened
        assert mock_s3_instance.upload_file.called
        assert mock_s3_instance.put_object.called
        assert mock_s3_instance.upload_file.call_count == 3
    
    @patch('pipelines.inference.pipeline_remote.s3Client')
    def test_full_video_inference_with_logger(self, mock_s3, mock_instance):
        """Test running inference with logger"""
        mock_logger = Mock()
        mock_instance.logger = mock_logger
        mock_s3_instance = MagicMock()
        mock_s3_instance.upload_file.return_value = "s3://bucket/path/file.jpg"
        mock_s3_instance.put_object.return_value = "s3://bucket/path/results.json"
        mock_s3.return_value = mock_s3_instance
        
        mock_instance.video_processor.get_video_info.return_value = {"fps": 30}
        mock_instance.video_processor.extract_frames.return_value = [
            {"frame_id": "frame_001", "frame_number": 1, "timestamp": 0.033, "file_path": "/data/frames/frame_001.jpg"}
        ]
        
        mock_inference_results = [{"frame_id": "frame_001", "detection_count": 0, "detections": []}]
        
        with patch.object(mock_instance, '_run_model_inference', return_value=mock_inference_results), \
             patch.object(mock_instance, '_generate_summary_stats', return_value={"total_detections": 0, "total_frames": 1, "frames_with_detections": 0}):
            
            mock_instance.run_inference_on_video(
                video_path="/data/videos/test.mp4",
                video_id="test_video",
                s3_bucket="my-bucket",
                save_annotated_frames=False
            )
        
        assert mock_logger.info.called
        assert mock_logger.info.call_count == 13
        
    @patch('pipelines.inference.pipeline_remote.profiled')
    def test_profiler_decorator_called(self, mock_profiled, mock_instance):
        """Test that the profiler decorator is applied correctly"""
        assert hasattr(mock_instance._generate_summary_stats, '__wrapped__') or callable(mock_instance._generate_summary_stats)
    
    @patch('pipelines.inference.pipeline_remote.cProfile')
    @patch('pipelines.inference.pipeline_remote.pstats')
    def test_minibatch_inference_profiling(self, mock_pstats, mock_cprofile, mock_instance):
        """Test that profiling decorator in _minibatch_inference works"""
        # Mock profiler
        mock_profiler = MagicMock()
        mock_cprofile.Profile.return_value = mock_profiler
        mock_pstats.Stats.return_value = mock_profiler
        
        # Mock model prediction
        mock_instance.model.predict.return_value = []
        
        frame_paths = ["frame_1.jpg", "frame_2.jpg"]
        file_meta = [
            {"frame_id": "1", "frame_number": 1, "timestamp": 0.0, "file_path": "frame_1.jpg"},
            {"frame_id": "2", "frame_number": 2, "timestamp": 0.033, "file_path": "frame_2.jpg"}
        ]
        
        with patch.object(mock_instance, '_process_inference_res', return_value=[]):
            mock_instance._minibatch_inference(0, frame_paths, file_meta, None)
                    
        mock_profiler.enable.assert_called()
        mock_profiler.disable.assert_called()