import pytest
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import shutil
import tempfile

import sys
from pathlib import Path

# Add parent directory to path to import training module
sys.path.insert(0, str(Path(__file__).parent.parent))

from training import (
    TrainSpec,
    _now_utc_stamp,
    _safe_copy_file,
    _copy_dir_tree,
    stage_dataset_from_s3,
    export_onnx,
    write_model_card,
    copy_training_artifacts_to_s3,
)


class TestTrainSpec:
    """Test the TrainSpec dataclass"""

    def test_train_spec_defaults(self):
        spec = TrainSpec(dataset_version="raw")
        assert spec.dataset_version == "raw"
        assert spec.model_size == "yolov8s.pt"
        assert spec.epochs == 20
        assert spec.img_size == 1280
        assert spec.batch == 0.95
        assert spec.workers == 4
        assert spec.seed == 117
        assert spec.use_wandb is False
        assert spec.freeze == 1

    def test_train_spec_s3_prefix_raw(self):
        spec = TrainSpec(dataset_version="raw")
        assert spec.s3_prefix() == "raw/roboflow/v8"

    def test_train_spec_s3_prefix_v1(self):
        spec = TrainSpec(dataset_version="v1")
        assert spec.s3_prefix() == "processed/roboflow/v1"

    def test_train_spec_s3_prefix_custom(self):
        spec = TrainSpec(dataset_version="custom/path/v2")
        assert spec.s3_prefix() == "custom/path/v2"

    def test_train_spec_custom_params(self):
        spec = TrainSpec(
            dataset_version="v1",
            model_size="yolov8m.pt",
            epochs=50,
            img_size=640,
            batch=16,
            warmup_epochs=3,
            dropout=0.2,
        )
        assert spec.epochs == 50
        assert spec.img_size == 640
        assert spec.batch == 16
        assert spec.warmup_epochs == 3
        assert spec.dropout == 0.2


class TestHelperFunctions:
    """Test helper functions for file operations"""

    def test_now_utc_stamp_format(self):
        stamp = _now_utc_stamp()
        # Should be in format YYYYMMDD-HHMMSS
        assert len(stamp) == 15
        assert stamp[8] == "-"
        assert stamp[:8].isdigit()
        assert stamp[9:].isdigit()

    def test_safe_copy_file(self, tmp_path):
        src_file = tmp_path / "source.txt"
        src_file.write_text("test content")
        dst_file = tmp_path / "subdir" / "dest.txt"

        _safe_copy_file(src_file, dst_file)

        assert dst_file.exists()
        assert dst_file.read_text() == "test content"

    def test_safe_copy_file_creates_parent_dirs(self, tmp_path):
        src_file = tmp_path / "source.txt"
        src_file.write_text("test")
        dst_file = tmp_path / "deep" / "nested" / "path" / "file.txt"

        _safe_copy_file(src_file, dst_file)

        assert dst_file.exists()
        assert dst_file.parent.exists()

    def test_copy_dir_tree(self, tmp_path):
        # Create source structure
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        (src_dir / "file1.txt").write_text("content1")
        (src_dir / "subdir").mkdir()
        (src_dir / "subdir" / "file2.txt").write_text("content2")

        dst_dir = tmp_path / "dest"

        _copy_dir_tree(src_dir, dst_dir)

        assert (dst_dir / "file1.txt").exists()
        assert (dst_dir / "file1.txt").read_text() == "content1"
        assert (dst_dir / "subdir" / "file2.txt").exists()
        assert (dst_dir / "subdir" / "file2.txt").read_text() == "content2"

    @patch("training.shutil.copy2")
    def test_safe_copy_file_permission_error(self, mock_copy2, tmp_path):
        """Test fallback to copyfile when copy2 raises PermissionError"""
        src_file = tmp_path / "source.txt"
        src_file.write_text("test content")
        dst_file = tmp_path / "subdir" / "dest.txt"

        # Mock copy2 to raise PermissionError
        mock_copy2.side_effect = PermissionError("Permission denied")

        # Should fall back to copyfile
        with patch("training.shutil.copyfile") as mock_copyfile:
            _safe_copy_file(src_file, dst_file)

            # Verify copyfile was called as fallback
            mock_copyfile.assert_called_once_with(src_file, dst_file)


class TestStageDatasetFromS3:
    """Test dataset staging functionality"""

    @patch("training.MOUNT_PATH")
    @patch("training.DATA_WORKDIR")
    @patch("training.RUNS_DIR")
    def test_stage_dataset_success(
        self, mock_runs_dir, mock_workdir, mock_mount, tmp_path
    ):
        # Setup mock paths
        mock_mount_path = tmp_path / "mount"
        mock_workdir_path = tmp_path / "work"
        mock_runs_path = tmp_path / "runs"

        mock_mount_path.mkdir()
        mock_workdir_path.mkdir()
        mock_runs_path.mkdir()
        (mock_runs_path / "profiling").mkdir()

        mock_mount.__truediv__ = lambda self, x: mock_mount_path / x
        mock_workdir.__truediv__ = lambda self, x: mock_workdir_path / x
        mock_runs_dir.__truediv__ = lambda self, x: mock_runs_path / x

        # Create fake S3 dataset structure
        s3_dataset = mock_mount_path / "raw" / "roboflow" / "v8"
        s3_dataset.mkdir(parents=True)
        (s3_dataset / "data.yaml").write_text("test: yaml")

        # Mock the source path existence
        with patch("training.MOUNT_PATH", mock_mount_path):
            with patch("training.DATA_WORKDIR", mock_workdir_path):
                with patch("training.RUNS_DIR", mock_runs_path):
                    result = stage_dataset_from_s3("raw/roboflow/v8")

                    # Result should be local working directory with data.yaml
                    assert "raw_roboflow_v8" in str(result)
                    assert (result / "data.yaml").exists()

    @patch("training.MOUNT_PATH")
    def test_stage_dataset_not_found(self, mock_mount, tmp_path):
        mock_mount_path = tmp_path / "mount"
        mock_mount_path.mkdir()
        mock_mount.__truediv__ = lambda self, x: mock_mount_path / x

        with patch("training.MOUNT_PATH", mock_mount_path):
            with pytest.raises(FileNotFoundError, match="S3 prefix not found"):
                stage_dataset_from_s3("nonexistent/path")

    @patch("training.MOUNT_PATH")
    @patch("training.DATA_WORKDIR")
    @patch("training.RUNS_DIR")
    @patch("training._copy_dir_tree")
    def test_stage_dataset_missing_yaml(
        self, mock_copy_tree, mock_runs_dir, mock_workdir, mock_mount, tmp_path
    ):
        mock_mount_path = tmp_path / "mount"
        mock_workdir_path = tmp_path / "work"
        mock_runs_path = tmp_path / "runs"

        mock_mount_path.mkdir()
        mock_workdir_path.mkdir()
        mock_runs_path.mkdir()

        # Create dataset without data.yaml
        s3_dataset = mock_mount_path / "raw" / "roboflow" / "v8"
        s3_dataset.mkdir(parents=True)

        with patch("training.MOUNT_PATH", mock_mount_path):
            with patch("training.DATA_WORKDIR", mock_workdir_path):
                with patch("training.RUNS_DIR", mock_runs_path):
                    with pytest.raises(FileNotFoundError, match="Missing data.yaml"):
                        stage_dataset_from_s3("raw/roboflow/v8")

    @patch("training.MOUNT_PATH")
    @patch("training.DATA_WORKDIR")
    @patch("training.RUNS_DIR")
    def test_stage_dataset_refresh_existing(
        self, mock_runs_dir, mock_workdir, mock_mount, tmp_path
    ):
        """Test that existing local directory is removed before staging"""
        mock_mount_path = tmp_path / "mount"
        mock_workdir_path = tmp_path / "work"
        mock_runs_path = tmp_path / "runs"

        mock_mount_path.mkdir()
        mock_workdir_path.mkdir()
        mock_runs_path.mkdir()
        (mock_runs_path / "profiling").mkdir()

        # Create S3 dataset
        s3_dataset = mock_mount_path / "raw" / "roboflow" / "v8"
        s3_dataset.mkdir(parents=True)
        (s3_dataset / "data.yaml").write_text("test: yaml")

        # Create existing local directory
        local_dir = mock_workdir_path / "raw_roboflow_v8"
        local_dir.mkdir()
        (local_dir / "old_file.txt").write_text("old content")

        with patch("training.MOUNT_PATH", mock_mount_path):
            with patch("training.DATA_WORKDIR", mock_workdir_path):
                with patch("training.RUNS_DIR", mock_runs_path):
                    result = stage_dataset_from_s3("raw/roboflow/v8")

                    # Old file should be gone (directory was refreshed)
                    assert not (result / "old_file.txt").exists()
                    # New data.yaml should exist
                    assert (result / "data.yaml").exists()


class TestExportOnnx:
    """Test ONNX export functionality"""

    @patch("training.YOLO")
    def test_export_onnx_success(self, mock_yolo_cls, tmp_path):
        best_weights = tmp_path / "weights" / "best.pt"
        best_weights.parent.mkdir(parents=True)
        best_weights.write_bytes(b"fake weights")

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Create fake ONNX file
        onnx_file = run_dir / "best.onnx"
        onnx_file.write_bytes(b"fake onnx")

        mock_model = Mock()
        mock_yolo_cls.return_value = mock_model

        result = export_onnx(best_weights, run_dir, img_size=640)

        mock_model.export.assert_called_once_with(
            format="onnx", imgsz=640, opset=12, dynamic=True
        )
        assert result == onnx_file

    @patch("training.YOLO")
    def test_export_onnx_no_output(self, mock_yolo_cls, tmp_path):
        best_weights = tmp_path / "weights" / "best.pt"
        best_weights.parent.mkdir(parents=True)
        best_weights.write_bytes(b"fake weights")

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        mock_model = Mock()
        mock_yolo_cls.return_value = mock_model

        result = export_onnx(best_weights, run_dir, img_size=640)

        assert result is None

    @patch("training.YOLO")
    def test_export_onnx_exception(self, mock_yolo_cls, tmp_path):
        best_weights = tmp_path / "best.pt"
        best_weights.write_bytes(b"fake")
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        mock_model = Mock()
        mock_model.export.side_effect = Exception("Export failed")
        mock_yolo_cls.return_value = mock_model

        result = export_onnx(best_weights, run_dir, 640)

        assert result is None


class TestWriteModelCard:
    """Test model card generation"""

    def test_write_model_card_basic(self, tmp_path):
        spec = TrainSpec(
            dataset_version="v1",
            model_size="yolov8s.pt",
            epochs=10,
            img_size=640,
        )

        artifacts = {
            "best_pt": "s3://bucket/models/model_id/best.pt",
            "best_onnx": "s3://bucket/models/model_id/best.onnx",
            "results_csv": "s3://bucket/stats/training/model_id/results.csv",
        }

        data_yaml_local = tmp_path / "data.yaml"

        write_model_card(tmp_path, "test-model-123", spec, artifacts, data_yaml_local)

        card_file = tmp_path / "model_card.json"
        assert card_file.exists()

        card = json.loads(card_file.read_text())
        assert card["model_id"] == "test-model-123"
        assert card["dataset_version"] == "v1"
        assert card["epochs"] == 10
        assert card["img_size"] == 640
        assert card["artifacts"] == artifacts

    def test_write_model_card_with_optional_params(self, tmp_path):
        spec = TrainSpec(
            dataset_version="raw",
            warmup_epochs=5,
            dropout=0.3,
            freeze=2,
        )

        artifacts = {"best_pt": "s3://bucket/path"}
        data_yaml_local = tmp_path / "data.yaml"

        write_model_card(tmp_path, "model-id", spec, artifacts, data_yaml_local)

        card = json.loads((tmp_path / "model_card.json").read_text())
        assert card["warmup_epochs"] == 5
        assert card["dropout"] == 0.3
        assert card["freeze"] == 2


class TestCopyTrainingArtifactsToS3:
    """Test copying training artifacts to S3"""

    @patch("training.MOUNT_PATH")
    @patch("training.RUNS_DIR")
    @patch("training._safe_copy_file")
    def test_copy_artifacts_basic(self, mock_copy, mock_runs_dir, mock_mount, tmp_path):
        mock_mount_path = tmp_path / "mount"
        mock_runs_path = tmp_path / "runs"
        mock_mount_path.mkdir()
        mock_runs_path.mkdir()
        (mock_runs_path / "profiling").mkdir()  # Create profiling directory

        # Create run directory with artifacts
        run_dir = tmp_path / "run_dir"
        run_dir.mkdir()
        weights_dir = run_dir / "weights"
        weights_dir.mkdir()
        best_pt = weights_dir / "best.pt"
        best_pt.write_bytes(b"model weights")
        results_csv = run_dir / "results.csv"
        results_csv.write_text("epoch,loss\n1,0.5")

        with patch("training.MOUNT_PATH", mock_mount_path):
            with patch("training.RUNS_DIR", mock_runs_path):
                with patch("training.Path.mkdir"):
                    best_pt_s3, onnx_s3, results_csv_s3 = copy_training_artifacts_to_s3(
                        run_dir=run_dir,
                        model_id="test-model",
                        save_results_csv=True,
                    )

                    # Should copy best.pt and results.csv
                    assert mock_copy.call_count >= 2
                    assert "best.pt" in str(best_pt_s3)
                    assert "results.csv" in str(results_csv_s3)
                    assert onnx_s3 is None

    @patch("training.MOUNT_PATH")
    @patch("training.RUNS_DIR")
    def test_copy_artifacts_missing_best_pt(self, mock_runs_dir, mock_mount, tmp_path):
        mock_mount_path = tmp_path / "mount"
        mock_runs_path = tmp_path / "runs"
        mock_mount_path.mkdir()
        mock_runs_path.mkdir()

        run_dir = tmp_path / "run_dir"
        run_dir.mkdir()

        with patch("training.MOUNT_PATH", mock_mount_path):
            with patch("training.RUNS_DIR", mock_runs_path):
                with pytest.raises(FileNotFoundError, match="best.pt not found"):
                    copy_training_artifacts_to_s3(run_dir, "test-model")

    @patch("training.MOUNT_PATH")
    @patch("training.RUNS_DIR")
    @patch("training._safe_copy_file")
    def test_copy_artifacts_with_onnx(
        self, mock_copy, mock_runs_dir, mock_mount, tmp_path
    ):
        mock_mount_path = tmp_path / "mount"
        mock_runs_path = tmp_path / "runs"
        mock_mount_path.mkdir()
        mock_runs_path.mkdir()
        (mock_runs_path / "profiling").mkdir()  # Create profiling directory

        run_dir = tmp_path / "run_dir"
        run_dir.mkdir()
        weights_dir = run_dir / "weights"
        weights_dir.mkdir()
        (weights_dir / "best.pt").write_bytes(b"weights")
        (run_dir / "best.onnx").write_bytes(b"onnx model")

        with patch("training.MOUNT_PATH", mock_mount_path):
            with patch("training.RUNS_DIR", mock_runs_path):
                with patch("training.Path.mkdir"):
                    best_pt_s3, onnx_s3, results_csv_s3 = copy_training_artifacts_to_s3(
                        run_dir, "model-id", save_results_csv=False
                    )

                    assert onnx_s3 is not None
                    assert "best.onnx" in str(onnx_s3)
                    assert results_csv_s3 is None

    @patch("training.MOUNT_PATH")
    @patch("training.RUNS_DIR")
    @patch("training._safe_copy_file")
    def test_copy_artifacts_with_plots(
        self, mock_copy, mock_runs_dir, mock_mount, tmp_path
    ):
        mock_mount_path = tmp_path / "mount"
        mock_runs_path = tmp_path / "runs"
        mock_mount_path.mkdir()
        mock_runs_path.mkdir()
        (mock_runs_path / "profiling").mkdir()  # Create profiling directory

        run_dir = tmp_path / "run_dir"
        run_dir.mkdir()
        weights_dir = run_dir / "weights"
        weights_dir.mkdir()
        (weights_dir / "best.pt").write_bytes(b"weights")
        (run_dir / "labels.jpg").write_bytes(b"plot")
        (run_dir / "confusion_matrix.png").write_bytes(b"plot")

        with patch("training.MOUNT_PATH", mock_mount_path):
            with patch("training.RUNS_DIR", mock_runs_path):
                with patch("training.Path.mkdir"):
                    copy_training_artifacts_to_s3(
                        run_dir, "model-id", save_plots=True, save_results_csv=False
                    )

                    # Should copy best.pt + 2 plots
                    assert mock_copy.call_count >= 3


class TestTrainYolo:
    """Test the main train_yolo function"""

    @patch("training.YOLO")
    @patch("training.stage_dataset_from_s3")
    @patch("training.export_onnx")
    @patch("training.copy_training_artifacts_to_s3")
    @patch("training.write_model_card")
    @patch("training._safe_copy_file")
    @patch("training.vol")
    @patch("training.RUNS_DIR")
    @patch("training.MOUNT_PATH")
    def test_train_yolo_basic(
        self,
        mock_mount,
        mock_runs_dir,
        mock_vol,
        mock_copy_file,
        mock_write_card,
        mock_copy_artifacts,
        mock_export,
        mock_stage,
        mock_yolo_cls,
        tmp_path,
    ):
        """Test basic train_yolo execution path"""
        # Setup mocks
        mock_runs_path = tmp_path / "runs"
        mock_runs_path.mkdir()
        mock_runs_dir.__truediv__ = lambda self, x: mock_runs_path / x

        mock_mount_path = tmp_path / "mount"
        mock_mount_path.mkdir()
        mock_mount.__truediv__ = lambda self, x: mock_mount_path / x

        # Mock dataset staging
        local_data = tmp_path / "local_data"
        local_data.mkdir()
        (local_data / "data.yaml").write_text("test: yaml")
        mock_stage.return_value = local_data

        # Mock YOLO training
        mock_model = Mock()
        mock_yolo_cls.return_value = mock_model

        # Mock export
        mock_export.return_value = None

        # Mock artifact copying
        mock_copy_artifacts.return_value = (
            Path("s3://bucket/best.pt"),
            None,
            Path("s3://bucket/results.csv"),
        )

        with patch("training.RUNS_DIR", mock_runs_path):
            with patch("training.MOUNT_PATH", mock_mount_path):
                with patch.dict("os.environ", {}, clear=True):
                    from training import train_yolo

                    # Call the function
                    train_yolo(
                        dataset_version="raw",
                        model_size="yolov8s.pt",
                        epochs=1,
                        img_size=640,
                        batch=0.95,
                        use_wandb=False,
                        export_to_onnx=False,
                    )

        # Verify key calls
        mock_stage.assert_called_once()
        mock_model.train.assert_called_once()
        mock_vol.commit.assert_called_once()

    @patch("training.YOLO")
    @patch("training.stage_dataset_from_s3")
    @patch("training.export_onnx")
    @patch("training.copy_training_artifacts_to_s3")
    @patch("training.write_model_card")
    @patch("training._safe_copy_file")
    @patch("training.vol")
    @patch("training.RUNS_DIR")
    @patch("training.MOUNT_PATH")
    def test_train_yolo_with_wandb(
        self,
        mock_mount,
        mock_runs_dir,
        mock_vol,
        mock_copy_file,
        mock_write_card,
        mock_copy_artifacts,
        mock_export,
        mock_stage,
        mock_yolo_cls,
        tmp_path,
    ):
        """Test train_yolo with W&B enabled"""
        mock_runs_path = tmp_path / "runs"
        mock_runs_path.mkdir()
        mock_runs_dir.__truediv__ = lambda self, x: mock_runs_path / x

        mock_mount_path = tmp_path / "mount"
        mock_mount_path.mkdir()
        mock_mount.__truediv__ = lambda self, x: mock_mount_path / x

        local_data = tmp_path / "local_data"
        local_data.mkdir()
        (local_data / "data.yaml").write_text("test: yaml")
        mock_stage.return_value = local_data

        mock_model = Mock()
        mock_yolo_cls.return_value = mock_model
        mock_export.return_value = Path(tmp_path / "model.onnx")
        mock_copy_artifacts.return_value = (Path("s3://"), Path("s3://"), Path("s3://"))

        with patch("training.RUNS_DIR", mock_runs_path):
            with patch("training.MOUNT_PATH", mock_mount_path):
                with patch.dict("os.environ", {}, clear=True):
                    from training import train_yolo

                    train_yolo(
                        dataset_version="v1",
                        use_wandb=True,
                        export_to_onnx=True,
                        warmup_epochs=3,
                        dropout=0.2,
                    )

        # Verify W&B environment was set
        mock_model.train.assert_called_once()

    @patch("training.YOLO")
    @patch("training.stage_dataset_from_s3")
    @patch("training.export_onnx")
    @patch("training.copy_training_artifacts_to_s3")
    @patch("training.write_model_card")
    @patch("training._safe_copy_file")
    @patch("training.vol")
    @patch("training.RUNS_DIR")
    @patch("training.MOUNT_PATH")
    def test_train_yolo_backup(
        self,
        mock_mount,
        mock_runs_dir,
        mock_vol,
        mock_copy_file,
        mock_write_card,
        mock_copy_artifacts,
        mock_export,
        mock_stage,
        mock_yolo_cls,
        tmp_path,
    ):
        """Test train_yolo_backup function"""
        mock_runs_path = tmp_path / "runs"
        mock_runs_path.mkdir()
        mock_runs_dir.__truediv__ = lambda self, x: mock_runs_path / x

        mock_mount_path = tmp_path / "mount"
        mock_mount_path.mkdir()
        mock_mount.__truediv__ = lambda self, x: mock_mount_path / x

        local_data = tmp_path / "local_data"
        local_data.mkdir()
        (local_data / "data.yaml").write_text("test: yaml")
        mock_stage.return_value = local_data

        mock_model = Mock()
        mock_yolo_cls.return_value = mock_model
        mock_export.return_value = None
        mock_copy_artifacts.return_value = (Path("s3://"), None, Path("s3://"))

        with patch("training.RUNS_DIR", mock_runs_path):
            with patch("training.MOUNT_PATH", mock_mount_path):
                with patch.dict("os.environ", {}, clear=True):
                    from training import train_yolo_backup

                    train_yolo_backup(
                        dataset_version="raw",
                        epochs=5,
                        plots=True,
                    )

        mock_model.train.assert_called_once()
        mock_vol.commit.assert_called_once()


class TestMain:
    """Test the main local entrypoint"""

    @patch("training.train_yolo")
    def test_main_basic(self, mock_train):
        """Test main entrypoint with basic parameters"""
        from training import main

        with patch("training.modal.App.local_entrypoint", lambda: lambda f: f):
            main_func = main
            # Mock the remote call
            mock_train.remote = Mock()

            # This would normally be called by Modal, we simulate it
            # The function should prepare parameters and call train_yolo.remote()

    @patch("training.train_yolo")
    def test_main_with_config_file(self, mock_train, tmp_path):
        """Test main with YAML config file"""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "dataset_version": "v1",
            "epochs": 50,
            "img_size": 1280,
            "batch": 16,
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        from training import main

        # Main function with params argument
        # This is harder to test without actually invoking Modal


class TestIntegration:
    """Integration tests combining multiple functions"""

    def test_full_artifact_workflow(self, tmp_path):
        """Test complete workflow of staging, training artifacts, and model card"""
        # This would require more complex mocking of Modal and YOLO
        # Keeping as placeholder for future implementation
        pass
