"""
Shared pytest fixtures and configuration for HoneyBee tests

This module provides common fixtures, mock objects, and test data
that can be used across all test modules.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Add HoneyBee to path
honeybee_path = Path(__file__).parent.parent
sys.path.insert(0, str(honeybee_path))


# ============================================================================
# Path Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def project_root():
    """Path to HoneyBee project root"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Path to test data directory"""
    return project_root / "tests" / "fixtures"


@pytest.fixture(scope="session")
def sample_wsi_path(project_root):
    """Path to sample WSI file"""
    wsi_path = project_root / "testwsi" / "sample.svs"
    if wsi_path.exists():
        return wsi_path
    return None


@pytest.fixture(scope="session")
def sample_clinical_pdf_path(project_root):
    """Path to sample clinical PDF"""
    pdf_path = project_root / "testclinical" / "sample.PDF"
    if pdf_path.exists():
        return pdf_path
    return None


# ============================================================================
# Sample Data Fixtures
# ============================================================================


@pytest.fixture
def sample_clinical_text():
    """Sample clinical text for testing"""
    return """
    PATHOLOGY REPORT

    Patient Name: John Doe
    MRN: 12345678
    Date: January 15, 2024

    DIAGNOSIS: Invasive ductal carcinoma, Grade 2

    CLINICAL HISTORY:
    57-year-old female with palpable left breast mass.

    GROSS DESCRIPTION:
    The specimen consists of a 2.5 cm tumor in the upper outer quadrant.

    MICROSCOPIC DESCRIPTION:
    Sections show invasive ductal carcinoma, Grade 2 (Nottingham score 6).
    The tumor measures 2.3 cm in greatest dimension.

    IMMUNOHISTOCHEMISTRY:
    ER: Positive (95%, strong intensity)
    PR: Positive (80%, moderate to strong intensity)
    HER2: Negative (0)
    Ki-67: 20%

    STAGE: pT2 N0 M0 (Stage IIA)

    COMMENT:
    The findings are consistent with hormone receptor-positive,
    HER2-negative invasive ductal carcinoma.
    """


@pytest.fixture
def sample_clinical_entities():
    """Expected entities from sample clinical text"""
    return [
        {"type": "tumor", "text": "invasive ductal carcinoma"},
        {"type": "staging", "text": "Grade 2"},
        {"type": "measurement", "text": "2.5 cm"},
        {"type": "biomarker", "text": "ER: Positive"},
        {"type": "biomarker", "text": "HER2: Negative"},
        {"type": "staging", "text": "pT2 N0 M0"},
    ]


@pytest.fixture
def sample_image_2d():
    """Sample 2D medical image"""
    # 256x256 grayscale image with some structure
    image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    # Add some structure (circle in center)
    y, x = np.ogrid[:256, :256]
    mask = (x - 128) ** 2 + (y - 128) ** 2 <= 50**2
    image[mask] = 200
    return image


@pytest.fixture
def sample_image_3d():
    """Sample 3D medical image (CT/MRI volume)"""
    # 64x128x128 volume with some structure
    image = np.random.randint(-1000, 1000, (64, 128, 128), dtype=np.int16)
    # Add some anatomical-like structure
    z, y, x = np.ogrid[:64, :128, :128]
    mask = (x - 64) ** 2 + (y - 64) ** 2 + (z - 32) ** 2 <= 30**2
    image[mask] = 100  # Soft tissue density
    return image


@pytest.fixture
def sample_wsi_patch():
    """Sample WSI patch (H&E stained tissue)"""
    # 224x224x3 RGB patch
    patch = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    # Make it look more like H&E (purple-pink colors)
    patch[:, :, 0] = np.random.randint(150, 220, (224, 224))  # Red
    patch[:, :, 1] = np.random.randint(100, 180, (224, 224))  # Green
    patch[:, :, 2] = np.random.randint(180, 240, (224, 224))  # Blue
    return patch


@pytest.fixture
def sample_wsi_patches():
    """Sample batch of WSI patches"""
    return np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_embeddings():
    """Sample embeddings array"""
    return np.random.randn(10, 768).astype(np.float32)


@pytest.fixture
def sample_dicom_metadata():
    """Sample CT DICOM metadata"""
    from honeybee.processors.radiology.metadata import ImageMetadata

    return ImageMetadata(
        modality="CT",
        patient_id="TEST001",
        study_date="20240115",
        series_description="CHEST CT",
        pixel_spacing=(1.0, 1.0, 2.5),
        image_position=(0.0, 0.0, 0.0),
        image_orientation=[1, 0, 0, 0, 1, 0],
        window_center=-600.0,
        window_width=1500.0,
        rescale_intercept=-1024.0,
        rescale_slope=1.0,
        manufacturer="GE MEDICAL SYSTEMS",
        rows=512,
        columns=512,
        number_of_slices=64,
    )


@pytest.fixture
def sample_mri_metadata():
    """Sample MRI metadata"""
    from honeybee.processors.radiology.metadata import ImageMetadata

    return ImageMetadata(
        modality="MR",
        patient_id="TEST002",
        study_date="20240115",
        series_description="BRAIN MRI T1",
        pixel_spacing=(1.0, 1.0, 1.0),
        image_position=(0.0, 0.0, 0.0),
        image_orientation=[1, 0, 0, 0, 1, 0],
        manufacturer="SIEMENS",
        rows=256,
        columns=256,
        number_of_slices=176,
    )


@pytest.fixture
def sample_pet_metadata():
    """Sample PET metadata"""
    from honeybee.processors.radiology.metadata import ImageMetadata

    return ImageMetadata(
        modality="PT",
        patient_id="TEST003",
        study_date="20240115",
        series_description="FDG PET",
        pixel_spacing=(4.0, 4.0, 3.0),
        image_position=(0.0, 0.0, 0.0),
        image_orientation=[1, 0, 0, 0, 1, 0],
        rows=128,
        columns=128,
        number_of_slices=90,
    )


# ============================================================================
# Mock Model Fixtures
# ============================================================================


@pytest.fixture
def mock_embedder():
    """Mock embedding model that returns random embeddings"""
    embedder = MagicMock()
    embedder.generate_embeddings.return_value = np.random.randn(1, 768).astype(np.float32)
    return embedder


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer"""
    tokenizer = MagicMock()
    tokenizer.tokenize.return_value = ["patient", "with", "cancer"]
    tokenizer.encode.return_value = [101, 1234, 5678, 9999, 102]
    tokenizer.decode.return_value = "patient with cancer"
    tokenizer.cls_token_id = 101
    tokenizer.sep_token_id = 102
    tokenizer.pad_token_id = 0
    tokenizer.return_value = {
        "input_ids": [[101, 1234, 5678, 102]],
        "attention_mask": [[1, 1, 1, 1]],
    }
    return tokenizer


@pytest.fixture
def mock_uni_model():
    """Mock UNI model for pathology"""
    model = MagicMock()
    model.load_model_and_predict.return_value = MagicMock(
        cpu=lambda: MagicMock(numpy=lambda: np.random.randn(10, 1024).astype(np.float32))
    )
    return model


@pytest.fixture
def mock_tissue_detector():
    """Mock tissue detector"""
    detector = MagicMock()
    # Return mock tissue predictions
    detector.detect.return_value = np.random.rand(100, 100) > 0.5
    return detector


# ============================================================================
# Temporary File Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_pdf_file(temp_dir, sample_clinical_text):
    """Create a temporary text file (PDF simulation)"""
    pdf_path = temp_dir / "test_report.txt"
    pdf_path.write_text(sample_clinical_text)
    return pdf_path


@pytest.fixture
def temp_output_dir(temp_dir):
    """Temporary output directory"""
    output_dir = temp_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def clinical_config():
    """Sample clinical processor configuration"""
    return {
        "document_processor": {
            "use_ocr": False,  # Disable OCR for faster tests
            "confidence_threshold": 60,
        },
        "tokenization": {"model": "gatortron", "max_length": 512, "segment_strategy": "sentence"},
        "entity_recognition": {
            "use_rules": True,
            "use_patterns": True,
            "cancer_specific_extraction": True,
            "temporal_extraction": True,
        },
    }


@pytest.fixture
def honeybee_config():
    """Sample HoneyBee configuration"""
    return {
        "clinical": {
            "tokenization": {"model": "gatortron"},
            "entity_recognition": {"use_rules": True},
        }
    }


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Pytest configuration hook"""
    # Add custom markers
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")


def pytest_collection_modifyitems(config, items):
    """Modify test items during collection"""
    # Skip slow tests unless --runslow is passed
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_gpu = pytest.mark.skip(reason="need --rungpu option to run")
    skip_models = pytest.mark.skip(reason="need --runmodels option to run")

    for item in items:
        if "slow" in item.keywords and not config.getoption("--runslow", default=False):
            item.add_marker(skip_slow)
        if "gpu" in item.keywords and not config.getoption("--rungpu", default=False):
            item.add_marker(skip_gpu)
        if "requires_models" in item.keywords and not config.getoption(
            "--runmodels", default=False
        ):
            item.add_marker(skip_models)


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--rungpu", action="store_true", default=False, help="run GPU tests")
    parser.addoption(
        "--runmodels", action="store_true", default=False, help="run tests requiring model weights"
    )
