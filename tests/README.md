# HoneyBee Test Suite

Comprehensive automated tests for the HoneyBee multimodal AI framework.

## Overview

This test suite provides automated testing for all HoneyBee functionality across clinical, pathology, and radiology modalities. The tests are organized into unit tests and integration tests, with extensive use of mocking to ensure fast execution even without model weights or large test data files.

## Directory Structure

```
tests/
├── conftest.py                      # Shared fixtures and pytest configuration
├── pytest.ini                       # pytest settings
├── requirements-test.txt            # Test dependencies
├── README.md                        # This file
├── unit/                            # Unit tests for individual components
│   ├── test_honeybee_api.py        # Core HoneyBee API tests
│   ├── processors/
│   │   ├── test_clinical_processor.py
│   │   ├── test_pathology_processor.py
│   │   └── test_radiology_processor.py
│   ├── models/                      # (Future: model-specific tests)
│   └── loaders/                     # (Future: loader tests)
├── integration/                     # End-to-end workflow tests
│   └── test_complete_workflows.py
└── fixtures/                        # Shared test data
```

## Installation

### 1. Install Test Dependencies

```bash
# Install test requirements
pip install -r tests/requirements-test.txt
```

### 2. Install HoneyBee Dependencies

Make sure you have HoneyBee's main dependencies installed:

```bash
pip install -r requirements.txt
```

## Running Tests

### Run All Tests

```bash
# From project root
pytest tests/

# With verbose output
pytest tests/ -v

# With coverage report
pytest tests/ --cov=honeybee --cov-report=html
```

### Run Specific Test Categories

```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run only clinical tests
pytest tests/ -m clinical

# Run only pathology tests
pytest tests/ -m pathology

# Run only radiology tests
pytest tests/ -m radiology
```

### Run Tests by Speed

```bash
# Run only fast tests (default)
pytest tests/

# Run slow tests (> 5 seconds)
pytest tests/ --runslow

# Run all tests including slow ones
pytest tests/ --runslow
```

### Run Tests by Requirements

```bash
# Run tests that require GPU
pytest tests/ --rungpu

# Run tests that require model weights
pytest tests/ --runmodels

# Run tests that require sample data files
pytest tests/ -m requires_sample_data
```

### Run Specific Test Files

```bash
# Test HoneyBee API
pytest tests/unit/test_honeybee_api.py

# Test ClinicalProcessor
pytest tests/unit/processors/test_clinical_processor.py

# Test PathologyProcessor
pytest tests/unit/processors/test_pathology_processor.py

# Test RadiologyProcessor
pytest tests/unit/processors/test_radiology_processor.py

# Test complete workflows
pytest tests/integration/test_complete_workflows.py
```

### Run Specific Test Classes or Functions

```bash
# Run specific test class
pytest tests/unit/test_honeybee_api.py::TestHoneyBeeInitialization

# Run specific test function
pytest tests/unit/test_honeybee_api.py::TestHoneyBeeInitialization::test_init_without_config

# Run tests matching a pattern
pytest tests/ -k "clinical"
```

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests (> 5 seconds)
- `@pytest.mark.gpu` - Tests requiring GPU/CUDA
- `@pytest.mark.requires_models` - Tests requiring downloaded model weights
- `@pytest.mark.requires_sample_data` - Tests requiring sample WSI/DICOM files
- `@pytest.mark.clinical` - Clinical processing tests
- `@pytest.mark.pathology` - Pathology/WSI processing tests
- `@pytest.mark.radiology` - Radiology/DICOM processing tests

## Test Coverage

### Unit Tests

#### Core HoneyBee API (`test_honeybee_api.py`)
- ✅ HoneyBee initialization with/without config
- ✅ `generate_embeddings()` for all modalities
- ✅ `integrate_embeddings()` for multimodal fusion
- ✅ `process_clinical()` for documents and text
- ✅ `process_clinical_batch()` for batch processing
- ✅ `predict_survival()` placeholder

#### ClinicalProcessor (`test_clinical_processor.py`)
- ✅ Initialization and configuration
- ✅ Text processing and document analysis
- ✅ Entity extraction (cancer-specific, biomarkers, staging, measurements)
- ✅ Temporal timeline extraction
- ✅ Tokenization strategies (sentence, paragraph, sliding window)
- ✅ Embedding generation with different models
- ✅ Batch document processing
- ✅ Document structure analysis
- ✅ Configuration options
- ✅ Error handling

#### PathologyProcessor (`test_pathology_processor.py`)
- ✅ Initialization with different models (UNI, UNI2, Virchow2, REMEDIS)
- ✅ WSI loading with parameters
- ✅ Tissue detection (Otsu, HSV, Otsu+HSV)
- ✅ Stain normalization (Reinhard, Macenko, Vahadane)
- ✅ Stain separation (H&E deconvolution)
- ✅ Patch extraction with tissue filtering
- ✅ Embedding generation (mocked)
- ✅ Embedding aggregation (mean, max, median, std, concat)
- ✅ Complete pipeline (`process_slide()`)
- ✅ Error handling

#### RadiologyProcessor (`test_radiology_processor.py`)
- ✅ Initialization with different models
- ✅ DICOM/NIfTI loading
- ✅ Preprocessing (denoising, normalization, windowing)
- ✅ Denoising methods (NLM, bilateral, median)
- ✅ Intensity normalization (z-score, min-max, percentile)
- ✅ Window/level adjustment with presets
- ✅ Hounsfield unit verification
- ✅ Spatial resampling
- ✅ Segmentation (lungs, brain, organs, tumors)
- ✅ Metal artifact reduction
- ✅ Embedding generation (mocked)
- ✅ Batch processing
- ✅ Image registration
- ✅ Feature extraction
- ✅ Error handling

### Integration Tests

#### Complete Workflows (`test_complete_workflows.py`)
- ✅ Clinical: PDF → processing → embeddings
- ✅ Clinical: Text pipeline with entity extraction
- ✅ Clinical: Batch processing workflow
- ✅ Pathology: WSI → patches → embeddings → aggregation
- ✅ Pathology: Preprocessing pipeline
- ✅ Radiology: DICOM → preprocessing → embeddings
- ✅ Radiology: Segmentation pipeline
- ✅ Multimodal: Complete integration workflow
- ✅ Multimodal: Fusion strategies
- ✅ Cancer patient analysis scenario
- ✅ Cohort analysis scenario
- ✅ Patient similarity search
- ✅ Error recovery and robustness

## Fixtures

The `conftest.py` file provides shared fixtures:

### Path Fixtures
- `project_root` - Path to HoneyBee project root
- `test_data_dir` - Path to test data directory
- `sample_wsi_path` - Path to sample WSI file
- `sample_clinical_pdf_path` - Path to sample clinical PDF

### Sample Data Fixtures
- `sample_clinical_text` - Sample clinical report text
- `sample_clinical_entities` - Expected entities from sample text
- `sample_image_2d` - Sample 2D medical image
- `sample_image_3d` - Sample 3D medical image (CT/MRI volume)
- `sample_wsi_patch` - Sample WSI patch (H&E stained)
- `sample_wsi_patches` - Batch of WSI patches
- `sample_embeddings` - Sample embeddings array

### Mock Model Fixtures
- `mock_embedder` - Mock embedding model
- `mock_tokenizer` - Mock tokenizer
- `mock_uni_model` - Mock UNI model
- `mock_radimagenet_model` - Mock RadImageNet model
- `mock_tissue_detector` - Mock tissue detector

### Configuration Fixtures
- `clinical_config` - Sample clinical processor configuration
- `honeybee_config` - Sample HoneyBee configuration
- `sample_dicom_metadata` - Sample DICOM metadata

## Writing New Tests

### Basic Test Structure

```python
import pytest
from honeybee.processors import ClinicalProcessor

class TestNewFeature:
    """Test new feature"""

    def test_basic_functionality(self):
        """Test basic functionality"""
        processor = ClinicalProcessor()
        result = processor.new_feature("input")

        assert result is not None
        assert expected_condition

    @pytest.mark.slow
    def test_slow_operation(self):
        """Test that takes > 5 seconds"""
        # Long-running test
        pass

    @pytest.mark.requires_models
    def test_with_real_model(self):
        """Test requiring actual model weights"""
        # Test with real model
        pass
```

### Using Fixtures

```python
def test_with_fixtures(sample_clinical_text, mock_embedder):
    """Test using fixtures"""
    processor = ClinicalProcessor()
    result = processor.process_text(sample_clinical_text)

    assert result is not None
```

### Using Mocks

```python
from unittest.mock import patch, MagicMock

@patch('honeybee.models.HuggingFaceEmbedder')
def test_with_mock(mock_embedder_class):
    """Test with mocked model"""
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.generate_embeddings.return_value = np.random.randn(1, 768)
    mock_embedder_class.return_value = mock_embedder_instance

    # Use the mock
    processor = ClinicalProcessor()
    embeddings = processor.generate_embeddings("text")

    assert embeddings.shape == (1, 768)
```

### Parametrized Tests

```python
@pytest.mark.parametrize("method", ["otsu", "hsv", "otsu_hsv"])
def test_all_methods(sample_image, method):
    """Test multiple methods"""
    processor = PathologyProcessor()
    result = processor.detect_tissue(sample_image, method=method)

    assert result is not None
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r tests/requirements-test.txt
      - name: Run tests
        run: pytest tests/ -v --cov=honeybee
```

## Best Practices

1. **Keep tests fast**: Use mocks for expensive operations (model loading, large file I/O)
2. **Test one thing**: Each test should verify a single behavior
3. **Use descriptive names**: Test names should clearly describe what they test
4. **Use fixtures**: Reuse common setup code via fixtures
5. **Mark appropriately**: Use markers to categorize tests
6. **Test edge cases**: Include tests for error conditions and edge cases
7. **Keep tests independent**: Tests should not depend on each other
8. **Update tests with code**: Keep tests in sync when changing implementation

## Troubleshooting

### Tests Fail to Import HoneyBee
```bash
# Make sure you're running from project root
cd /path/to/HoneyBee
pytest tests/
```

### Sample Data Tests Skip
```bash
# Tests marked with @pytest.mark.requires_sample_data will skip if
# sample files (sample.svs, sample.PDF) are not found in test directories

# To include these tests, make sure sample files exist:
ls testwsi/sample.svs
ls testclinical/sample.PDF
```

### Slow Tests Take Too Long
```bash
# Skip slow tests by default
pytest tests/

# Only run fast tests explicitly
pytest tests/ -m "not slow"
```

### Model Tests Fail
```bash
# Tests requiring model weights are marked with @pytest.mark.requires_models
# They skip by default unless you run with --runmodels

pytest tests/ --runmodels
```

## Contributing

When adding new features to HoneyBee:

1. Write tests for new functionality
2. Update existing tests if behavior changes
3. Ensure all tests pass before submitting PR
4. Add new fixtures to `conftest.py` if needed
5. Update this README with new test coverage

## Contact

For questions or issues with the test suite, please open an issue on the HoneyBee GitHub repository.
