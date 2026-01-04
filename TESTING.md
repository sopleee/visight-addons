# Unit Testing Guide for Training and Ingestion Scripts

This document provides an overview of the unit tests created for the training and ingestion components of the Visight project.

## Test Coverage Overview

### 1. Modal Training Tests (`model_training/tests/test_training.py`)

Tests for the YOLO model training infrastructure on Modal.

**Test Classes:**
- `TestTrainSpec`: Tests for training configuration dataclass
- `TestHelperFunctions`: Tests for file operation utilities
- `TestStageDatasetFromS3`: Tests for S3 dataset staging functionality
- `TestExportOnnx`: Tests for ONNX model export
- `TestWriteModelCard`: Tests for model card generation
- `TestCopyTrainingArtifactsToS3`: Tests for artifact copying to S3
- `TestIntegration`: Integration tests

**Key Features Tested:**
- Configuration management and S3 prefix mapping
- File copying with permission handling
- Dataset staging from S3 to local storage
- ONNX export with error handling
- Model card JSON generation
- Training artifact copying (best.pt, best.onnx, results.csv)
- Profiling output generation

**Total Tests:** 21 test cases

### 2. Image/Brand Catalog Tests (`a5/pipelines/training_ingestion/tests/test_image_brand_catalog.py`)

Tests for the ETL pipeline that creates brand and image catalogues from datasets.

**Test Classes:**
- `TestGetPrefix`: Dataset version to S3 prefix mapping
- `TestCreateBrandCatalogue`: Brand catalogue creation from metadata
- `TestGetClassesInImg`: YOLO label parsing and class extraction
- `TestCreateImageCatalogue`: Image catalogue with label associations
- `TestGetAndFormatSplitDirectories`: Split directory formatting
- `TestExtract`: ETL extract phase
- `TestTransform`: ETL transform phase
- `TestLoad`: ETL load phase
- `TestIntegration`: Full ETL pipeline integration

**Key Features Tested:**
- YOLO format label file parsing
- Brand catalogue creation with ID assignment
- Image ID hashing (SHA256)
- S3 client integration
- Train/val/test split handling
- CSV output generation
- Error handling for missing files

**Total Tests:** 33 test cases

### 3. RL Judge Training Tests (`misc_old/vlm_test/rl_judge/tests/test_train.py`)

Tests for the reinforcement learning judge training pipeline.

**Test Classes:**
- `TestSeedEverything`: Random seed reproducibility
- `TestEvaluate`: Policy network evaluation
- `TestTrainJudge`: Main training loop
- `TestTrainingIntegration`: Multi-epoch training integration
- `TestEdgeCases`: Edge cases and error handling

**Key Features Tested:**
- Random seed setting for reproducibility
- Policy network evaluation metrics (abs_error, reward)
- Device selection (CPU/CUDA auto-detection)
- Dataset validation (empty dataset handling)
- Train/validation split creation
- TensorBoard logging
- Config snapshot generation
- Baseline beta for advantage calculation
- Multi-epoch training loops

**Total Tests:** 18 test cases

## Running the Tests

### Prerequisites

1. **Install Python 3.8+** (Python 3.12 recommended)

2. **Install pytest and testing dependencies:**

```bash
pip install pytest pytest-cov pytest-mock
```

3. **Install project dependencies** (optional, for tests that require actual imports):

```bash
# For Modal training tests
pip install modal ultralytics

# General
pip install -r requirements.txt

# For ETL pipeline tests
pip install pandas pyyaml boto3

# For RL judge tests
pip install torch tensorboard

# For data splitting tests
# (No additional dependencies required)
```

### Running All Tests

From the project root:

```bash
# Use python -m pytest (works on Windows without PATH issues)
python -m pytest

# Run with coverage report
python -m pytest --cov=. --cov-report=html

# Run with verbose output
python -m pytest -v
```

### Running Specific Test Suites

```bash
# Modal training tests (requires modal, ultralytics)
python -m pytest model_training/tests/test_training.py -v

# Image/brand catalog tests (requires pandas, boto3)
python -m pytest pipelines/data_ingestion/tests/test_image_brand_catalog.py -v

# RL judge training tests (requires torch, tensorboard)
python -m pytest misc_old/vlm_test/rl_judge/tests/test_train.py -v
```

### Running Specific Test Classes or Methods

```bash
# Run a specific test class
pytest model_training/tests/test_training.py::TestTrainSpec

# Run a specific test method
pytest model_training/tests/test_training.py::TestTrainSpec::test_train_spec_defaults

# Run tests matching a pattern
pytest -k "test_load" -v
```

## Test Organization

```
visight/
├── modal/
│   └── tests/
│       └── test_training.py              # YOLO training tests
├── pipelines/
│   └── data_ingestion/
│       └── tests/
│           └── test_image_brand_catalog.py  # ETL pipeline tests
│   └── inference/
│       └── tests/
│           └── test_modal_app.py 
│           └── test_model_server.py 
│           └── test_pipeline_remote.py 
│           └── test_video_processor.py 
```

## Testing Patterns Used

### 1. Mocking External Dependencies

Tests use `unittest.mock` to mock external dependencies like:
- S3 clients and file operations
- Modal infrastructure
- YOLO model loading and training
- File system operations

Example:
```python
@patch("training.YOLO")
def test_export_onnx_success(self, mock_yolo_cls, tmp_path):
    mock_model = Mock()
    mock_yolo_cls.return_value = mock_model
    # Test implementation
```

### 2. Temporary Directories

Tests use pytest's `tmp_path` fixture for isolated file operations:

```python
def test_save_jsonl_basic(self, tmp_path):
    jsonl_file = tmp_path / "output.jsonl"
    save_jsonl(data, str(jsonl_file))
    assert jsonl_file.exists()
```

### 3. Reproducibility Testing

Tests verify deterministic behavior with random seeds:

```python
def test_seed_everything_reproducible(self):
    seed_everything(42)
    rand1 = random.random()

    seed_everything(42)
    rand2 = random.random()

    assert rand1 == rand2
```

### 4. Edge Case Coverage

Tests include edge cases like:
- Empty datasets
- Missing files
- Invalid formats
- Extreme class imbalances
- NaN values
- Unicode characters

## Known Limitations

1. **Modal Integration**: Full Modal app testing requires actual Modal infrastructure. Current tests mock Modal components.

2. **S3 Integration**: S3 operations are mocked. Integration tests with actual S3 would require test buckets.

3. **GPU Testing**: CUDA-specific tests are skipped on CPU-only machines using `pytest.mark.skipif`.

4. **YOLO Training**: Actual model training is mocked to avoid long-running tests.

## Future Improvements

1. **Integration Tests**: Add end-to-end tests with test S3 buckets and Modal environments
2. **Performance Tests**: Add benchmarking for data loading and processing
3. **Parameterized Tests**: Use `pytest.parametrize` for testing multiple configurations
4. **Fixtures**: Create shared fixtures in `conftest.py` files
5. **CI/CD Integration**: Set up automated testing in CI pipeline