import pytest
import pandas as pd
import hashlib
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from io import StringIO

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Mock the Config before importing the module
from unittest.mock import Mock, MagicMock

# Create a mock Config class
class MockConfig:
    raw_prefix = "raw/roboflow/v8"
    aug_prefix = "processed/roboflow/augmented"
    processed_prefix = "processed/roboflow/v1"
    s3_bucket = "test-bucket"
    catalogue_path = "s3://test-bucket/configs"
    model_key = "models/test-model"

# Patch the Config import before importing the module
sys.modules['pipelines.configs.config'] = Mock(Config=MockConfig)

from pipelines.training_ingestion.image_brand_catalog import (
    get_prefix,
    create_brand_catalogue,
    get_classes_in_img,
    create_image_catalogue,
    get_and_format_split_directories,
    extract,
    transform,
    load,
)


class TestGetPrefix:
    """Test dataset version to S3 prefix mapping"""

    def test_get_prefix_raw(self):
        with patch("pipelines.training_ingestion.image_brand_catalog.Config") as mock_config:
            mock_config.raw_prefix = "raw/roboflow/v8"
            result = get_prefix("raw")
            assert result == "raw/roboflow/v8"

    def test_get_prefix_v1(self):
        with patch("pipelines.training_ingestion.image_brand_catalog.Config") as mock_config:
            mock_config.processed_prefix = "processed/roboflow/v1"
            result = get_prefix("v1")
            assert result == "processed/roboflow/v1"

    def test_get_prefix_aug(self):
        with patch("pipelines.training_ingestion.image_brand_catalog.Config") as mock_config:
            mock_config.aug_prefix = "augmented/roboflow/v1"
            result = get_prefix("aug")
            assert result == "augmented/roboflow/v1"

    def test_get_prefix_invalid(self):
        with pytest.raises(ValueError, match="Unknown version"):
            get_prefix("invalid_version")


class TestCreateBrandCatalogue:
    """Test brand catalogue creation from metadata"""

    def test_create_brand_catalogue_basic(self):
        metadata = {
            "names": ["Brand_A", "Brand_B", "Brand_C"],
            "nc": 3,
        }
        result = create_brand_catalogue(metadata)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ["id", "name"]
        assert list(result["id"]) == [0, 1, 2]
        assert list(result["name"]) == ["Brand_A", "Brand_B", "Brand_C"]

    def test_create_brand_catalogue_large_dataset(self):
        metadata = {
            "names": [f"Brand_{i}" for i in range(100)],
            "nc": 100,
        }
        result = create_brand_catalogue(metadata)

        assert len(result) == 100
        assert result["id"].min() == 0
        assert result["id"].max() == 99

    def test_create_brand_catalogue_missing_names(self):
        metadata = {"nc": 3}
        with pytest.raises(Exception, match="Missing keys from metadata"):
            create_brand_catalogue(metadata)

    def test_create_brand_catalogue_missing_nc(self):
        metadata = {"names": ["Brand_A"]}
        with pytest.raises(Exception, match="Missing keys from metadata"):
            create_brand_catalogue(metadata)

    def test_create_brand_catalogue_empty(self):
        metadata = {"names": [], "nc": 0}
        result = create_brand_catalogue(metadata)
        assert len(result) == 0


class TestGetClassesInImg:
    """Test extracting class IDs from YOLO label files"""

    def test_get_classes_in_img_single_class(self):
        mock_client = Mock()
        mock_client.get_object.return_value = "0 0.5 0.5 0.2 0.2\n"

        label_key, classes = get_classes_in_img(
            mock_client, "train/labels", "image_001"
        )

        assert label_key == "train/labels/image_001.txt"
        assert classes == [0]
        mock_client.get_object.assert_called_once_with(
            "train/labels/image_001.txt", "txt"
        )

    def test_get_classes_in_img_multiple_classes(self):
        mock_client = Mock()
        mock_client.get_object.return_value = """0 0.5 0.5 0.2 0.2
2 0.3 0.3 0.1 0.1
1 0.7 0.7 0.15 0.15
2 0.4 0.4 0.1 0.1"""

        label_key, classes = get_classes_in_img(
            mock_client, "val/labels", "image_002"
        )

        assert classes == [0, 1, 2]  # Sorted and deduplicated

    def test_get_classes_in_img_empty_file(self):
        mock_client = Mock()
        mock_client.get_object.return_value = ""

        label_key, classes = get_classes_in_img(
            mock_client, "train/labels", "empty_img"
        )

        assert classes == []

    def test_get_classes_in_img_whitespace_only(self):
        mock_client = Mock()
        mock_client.get_object.return_value = "   \n  \n  "

        label_key, classes = get_classes_in_img(
            mock_client, "train/labels", "whitespace"
        )

        assert classes == []

    def test_get_classes_in_img_invalid_format(self):
        mock_client = Mock()
        # Invalid YOLO format (not 5 values)
        mock_client.get_object.return_value = "0 0.5 0.5\n1 0.3"

        label_key, classes = get_classes_in_img(
            mock_client, "train/labels", "invalid"
        )

        # Should skip invalid lines
        assert classes == []

    def test_get_classes_in_img_mixed_valid_invalid(self):
        mock_client = Mock()
        mock_client.get_object.return_value = """0 0.5 0.5 0.2 0.2
invalid line
1 0.3 0.3 0.1 0.1
another bad line"""

        label_key, classes = get_classes_in_img(
            mock_client, "train/labels", "mixed"
        )

        assert classes == [0, 1]

    def test_get_classes_in_img_file_not_found(self):
        mock_client = Mock()
        mock_client.get_object.side_effect = Exception("File not found in S3")

        with pytest.raises(Exception, match="Error fetching file"):
            get_classes_in_img(mock_client, "train/labels", "missing")

    def test_get_classes_in_img_float_class_ids(self):
        mock_client = Mock()
        # Sometimes class IDs might be stored as floats
        mock_client.get_object.return_value = "0.0 0.5 0.5 0.2 0.2\n2.0 0.3 0.3 0.1 0.1"

        label_key, classes = get_classes_in_img(
            mock_client, "train/labels", "float_ids"
        )

        assert classes == [0, 2]


class TestCreateImageCatalogue:
    """Test image catalogue creation"""

    def test_create_image_catalogue_basic(self):
        mock_client = Mock()

        split_directories = {"train": "train", "val": "val"}
        images_per_split = {
            "train": ["raw/roboflow/v8/train/images/img1.jpg"],
            "val": ["raw/roboflow/v8/val/images/img2.jpg"],
        }

        # Mock label retrieval
        def mock_get_classes(client, label_dir, stem):
            return f"{label_dir}/{stem}.txt", [0, 1]

        with patch(
            "pipelines.training_ingestion.image_brand_catalog.get_classes_in_img",
            side_effect=mock_get_classes,
        ):
            result = create_image_catalogue(
                mock_client,
                split_directories,
                images_per_split,
                "raw/roboflow/v8",
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert set(result.columns) == {
                "id",
                "s3_img_path",
                "s3_label_path",
                "class_ids",
                "split",
            }
            assert set(result["split"]) == {"train", "val"}

    def test_create_image_catalogue_id_hashing(self):
        mock_client = Mock()

        split_directories = {"train": "train"}
        images_per_split = {"train": ["prefix/train/images/test_image.jpg"]}

        with patch(
            "pipelines.training_ingestion.image_brand_catalog.get_classes_in_img",
            return_value=("label_path", [0]),
        ):
            result = create_image_catalogue(
                mock_client, split_directories, images_per_split, "prefix"
            )

            # Verify ID is SHA256 hash of stem
            expected_id = hashlib.sha256("test_image".encode()).hexdigest()
            assert result.iloc[0]["id"] == expected_id

    def test_create_image_catalogue_multiple_images_per_split(self):
        mock_client = Mock()

        split_directories = {"train": "train"}
        images_per_split = {
            "train": [
                "prefix/train/images/img1.jpg",
                "prefix/train/images/img2.jpg",
                "prefix/train/images/img3.jpg",
            ]
        }

        with patch(
            "pipelines.training_ingestion.image_brand_catalog.get_classes_in_img",
            return_value=("label", [0, 1, 2]),
        ):
            result = create_image_catalogue(
                mock_client, split_directories, images_per_split, "prefix"
            )

            assert len(result) == 3
            assert all(result["split"] == "train")

    def test_create_image_catalogue_empty_splits(self):
        mock_client = Mock()

        split_directories = {}
        images_per_split = {}

        result = create_image_catalogue(
            mock_client, split_directories, images_per_split, "prefix"
        )

        assert len(result) == 0

    def test_create_image_catalogue_preserves_class_order(self):
        mock_client = Mock()

        split_directories = {"train": "train"}
        images_per_split = {"train": ["prefix/train/images/img.jpg"]}

        expected_classes = [2, 5, 1, 10]
        with patch(
            "pipelines.training_ingestion.image_brand_catalog.get_classes_in_img",
            return_value=("label", expected_classes),
        ):
            result = create_image_catalogue(
                mock_client, split_directories, images_per_split, "prefix"
            )

            assert result.iloc[0]["class_ids"] == expected_classes


class TestGetAndFormatSplitDirectories:
    """Test formatting of split directories from metadata"""

    def test_format_split_directories_basic(self):
        metadata = {
            "train": "../train/images",
            "val": "../val/images",
            "test": "../test/images",
        }
        expected_splits = ["train", "val", "test"]

        result = get_and_format_split_directories(metadata, expected_splits)

        assert result == {"train": "train", "val": "val", "test": "test"}

    def test_format_split_directories_relative_paths(self):
        metadata = {
            "train": "./train/images",
            "val": "../../val/images",
        }
        expected_splits = ["train", "val"]

        result = get_and_format_split_directories(metadata, expected_splits)

        # The function strips leading ./ but keeps ../ prefixes
        assert result == {"train": "train", "val": "../val"}

    def test_format_split_directories_missing_split(self, capsys):
        metadata = {"train": "../train/images"}
        expected_splits = ["train", "val", "test"]

        result = get_and_format_split_directories(metadata, expected_splits)

        assert "train" in result
        assert "val" not in result
        assert "test" not in result

        # Check warning message
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "val" in captured.out or "test" in captured.out

    def test_format_split_directories_no_images_suffix(self):
        metadata = {"train": "../train"}
        expected_splits = ["train"]

        result = get_and_format_split_directories(metadata, expected_splits)

        assert result == {"train": "train"}


class TestExtract:
    """Test the extract phase of ETL pipeline"""

    def test_extract_success(self):
        mock_client = Mock()
        mock_client.get_object.return_value = {
            "names": ["Brand_A", "Brand_B"],
            "nc": 2,
            "train": "../train/images",
            "val": "../val/images",
        }
        mock_client.batch_get_filenames.side_effect = [
            ["prefix/train/images/img1.jpg", "prefix/train/images/img2.jpg"],
            ["prefix/val/images/img3.jpg"],
        ]

        result = extract(mock_client, "prefix", expected_splits=["train", "val"])

        assert "metadata" in result
        assert "split_directory" in result
        assert "image_files" in result
        assert len(result["image_files"]["train"]) == 2
        assert len(result["image_files"]["val"]) == 1

    def test_extract_missing_splits(self):
        mock_client = Mock()
        mock_client.get_object.return_value = {
            "names": ["Brand_A"],
            "nc": 1,
            "train": "../train/images",
        }
        mock_client.batch_get_filenames.return_value = ["img1.jpg"]

        result = extract(mock_client, "prefix", expected_splits=["train", "val", "test"])

        assert "train" in result["split_directory"]
        # Missing splits shouldn't be in split_directory

    def test_extract_empty_split(self):
        mock_client = Mock()
        mock_client.get_object.return_value = {
            "names": ["Brand_A"],
            "nc": 1,
            "train": "../train/images",
        }
        mock_client.batch_get_filenames.return_value = []

        result = extract(mock_client, "prefix", expected_splits=["train"])

        # Empty splits should be filtered out
        assert "train" not in result["image_files"]


class TestTransform:
    """Test the transform phase of ETL pipeline"""

    def test_transform_success(self):
        mock_client = Mock()

        retrieved_data = {
            "metadata": {"names": ["Brand_A", "Brand_B"], "nc": 2},
            "split_directory": {"train": "train"},
            "image_files": {"train": ["prefix/train/images/img1.jpg"]},
        }

        with patch(
            "pipelines.training_ingestion.image_brand_catalog.get_classes_in_img",
            return_value=("label", [0, 1]),
        ):
            result = transform(mock_client, retrieved_data, "prefix")

            assert "brand_catalogue" in result
            assert "image_catalogue" in result
            assert isinstance(result["brand_catalogue"], pd.DataFrame)
            assert isinstance(result["image_catalogue"], pd.DataFrame)
            assert len(result["brand_catalogue"]) == 2

    def test_transform_missing_keys(self):
        mock_client = Mock()
        retrieved_data = {"metadata": {}}

        with pytest.raises(Exception, match="Missing keys from Extract step"):
            transform(mock_client, retrieved_data, "prefix")


class TestLoad:
    """Test the load phase of ETL pipeline"""

    def test_load_success(self, tmp_path):
        brand_df = pd.DataFrame({"id": [0, 1], "name": ["Brand_A", "Brand_B"]})
        image_df = pd.DataFrame(
            {
                "id": ["hash1", "hash2"],
                "s3_img_path": ["s3://img1.jpg", "s3://img2.jpg"],
                "s3_label_path": ["s3://lbl1.txt", "s3://lbl2.txt"],
                "class_ids": [[0], [1]],
                "split": ["train", "val"],
            }
        )

        result_data = {"brand_catalogue": brand_df, "image_catalogue": image_df}

        # Mock config to use tmp_path
        with patch("pipelines.training_ingestion.image_brand_catalog.CONFIG") as mock_config:
            mock_config.catalogue_path = str(tmp_path)

            # Create output directory
            (tmp_path / "v1").mkdir(parents=True)

            load(result_data, "v1")

            # Check files were created
            brand_file = tmp_path / "v1" / "brand_catalogue.csv"
            image_file = tmp_path / "v1" / "image_catalogue.csv"

            assert brand_file.exists()
            assert image_file.exists()

            # Verify content
            loaded_brand = pd.read_csv(brand_file)
            assert len(loaded_brand) == 2
            assert list(loaded_brand["name"]) == ["Brand_A", "Brand_B"]

    def test_load_missing_keys(self):
        result_data = {"brand_catalogue": pd.DataFrame()}

        with pytest.raises(Exception, match="Missing keys from Transform step"):
            load(result_data, "v1")


class TestIntegration:
    """Integration tests for full ETL pipeline"""

    def test_full_etl_pipeline(self, tmp_path):
        """Test complete Extract -> Transform -> Load pipeline"""
        mock_client = Mock()

        # Mock extract phase
        mock_client.get_object.return_value = {
            "names": ["Brand_A", "Brand_B"],
            "nc": 2,
            "train": "../train/images",
            "val": "../val/images",
        }
        mock_client.batch_get_filenames.side_effect = [
            ["prefix/train/images/img1.jpg", "prefix/train/images/img2.jpg"],
            ["prefix/val/images/img3.jpg"],
        ]

        # Mock label files
        with patch(
            "pipelines.training_ingestion.image_brand_catalog.get_classes_in_img",
            return_value=("label_path", [0, 1]),
        ):
            # Extract
            extracted = extract(mock_client, "prefix")

            # Transform
            transformed = transform(mock_client, extracted, "prefix")

            # Verify transformed data
            assert len(transformed["brand_catalogue"]) == 2
            assert len(transformed["image_catalogue"]) == 3

            # Load
            with patch("pipelines.training_ingestion.image_brand_catalog.CONFIG") as mock_config:
                mock_config.catalogue_path = str(tmp_path)
                (tmp_path / "v1").mkdir()

                load(transformed, "v1")

                # Verify files created
                assert (tmp_path / "v1" / "brand_catalogue.csv").exists()
                assert (tmp_path / "v1" / "image_catalogue.csv").exists()
