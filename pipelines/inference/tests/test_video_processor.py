import json
import cv2
import numpy as np
import pytest
from pathlib import Path
from pipelines.inference.video_processor import VideoProcessor


def create_test_video(path: Path, num_frames=10, width=320, height=240, fps=10):
    """
    Creates a small synthetic video for testing.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    for i in range(num_frames):
        # Solid color frames to simplify
        frame = np.full((height, width, 3), (i*20 % 255), dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return path

class TestVideoProcessor: 

    def test_get_video_info(self, tmp_path):
        video_path = tmp_path / "test.mp4"
        create_test_video(video_path, num_frames=12, fps=6)

        vp = VideoProcessor()
        info = vp.get_video_info(str(video_path))

        assert info["fps"] == 6
        assert info["total_frames"] == 12
        assert info["width"] == 320
        assert info["height"] == 240
        assert info["duration_seconds"] == round(12 / 6, 2)
        assert info["resolution"] == "320x240"

    def test_get_video_info_invalid_path(self):
        vp = VideoProcessor()
        with pytest.raises(Exception):
            vp.get_video_info("nonexistent_file.mp4")

    def test_extract_frames_basic(self,tmp_path):
        video_path = tmp_path / "input.mp4"
        output_dir = tmp_path / "frames"

        create_test_video(video_path, num_frames=5, fps=5)

        vp = VideoProcessor()
        frames = vp.extract_frames(str(video_path), str(output_dir))

        # Expect all frames extracted (frame_interval = 1)
        assert len(frames) == 5

        # Validate structure of returned metadata objects
        example = frames[0]
        assert "frame_id" in example
        assert "frame_number" in example
        assert "timestamp" in example
        assert "file_path" in example

        # All frames saved?
        for f in frames:
            assert Path(f["file_path"]).exists()

        # Metadata file created?
        metadata_file = output_dir / "frames_metadata.json"
        assert metadata_file.exists()

        # JSON is valid
        with open(metadata_file) as f:
            metadata_json = json.load(f)
            assert metadata_json["extracted_frames"] == 5
            assert len(metadata_json["frames"]) == 5

    def test_extract_frames_target_fps_reduction(self,tmp_path):
        """
        If target_fps < video_fps, extractor should skip frames.
        """
        video_path = tmp_path / "input_fps.mp4"
        output_dir = tmp_path / "frames2"

        create_test_video(video_path, num_frames=10, fps=10)

        vp = VideoProcessor(fps=2)  # target_fps=2 < actual=10
        frames = vp.extract_frames(str(video_path), str(output_dir))

        # Expected interval = video_fps / target_fps = 10 / 2 = 5
        # So frames at 0, 5 only
        assert len(frames) == 2
        assert frames[0]["frame_number"] == 0
        assert frames[1]["frame_number"] == 5

    def test_extract_frames_invalid_video(self,tmp_path):
        bad_video = tmp_path / "badfile.mp4"
        bad_video.write_bytes(b"not a video")

        vp = VideoProcessor()

        with pytest.raises(Exception):
            vp.extract_frames(str(bad_video), str(tmp_path / "out"))

    def test_profiling_output_created(self,tmp_path):
        video_path = tmp_path / "input3.mp4"
        output_dir = tmp_path / "frames3"
        profiling_dir = tmp_path / "profiling"

        create_test_video(video_path, num_frames=3, fps=3)

        vp = VideoProcessor()
        vp.extract_frames(str(video_path), str(output_dir))

        # Find profiling file
        prof_files = list(profiling_dir.glob("extract_frames_*.txt"))
        assert len(prof_files) == 1
        assert prof_files[0].exists()
        assert prof_files[0].stat().st_size > 0
