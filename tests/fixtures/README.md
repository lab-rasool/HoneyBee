# Test Fixtures Directory

This directory is reserved for additional test data files that may be needed for testing.

## Current Test Data

HoneyBee tests primarily use:
- **Sample WSI**: `testwsi/sample.svs` (in project root)
- **Sample Clinical PDF**: `testclinical/sample.PDF` (in project root)

## Adding New Test Data

If you need to add small test data files for testing:

1. Add files to this directory
2. Update fixtures in `conftest.py` to reference them
3. Add documentation here about the files

## Large Files

**Do not commit large files to git!**

For large test files (> 10MB):
- Add them to `.gitignore`
- Document how to obtain them (download links, generation scripts, etc.)
- Use mocks in tests when possible instead of real large files

## Example Files Structure

```
fixtures/
├── README.md                  # This file
├── sample_clinical_text.txt   # (Optional) Additional clinical text samples
├── sample_patches/            # (Optional) Small image patches for testing
│   ├── patch_001.png
│   └── patch_002.png
└── metadata/                  # (Optional) Sample metadata files
    └── sample_dicom_tags.json
```

## Generating Synthetic Test Data

For most tests, we generate synthetic data programmatically in `conftest.py`:

```python
@pytest.fixture
def sample_image_2d():
    """Sample 2D medical image"""
    return np.random.randint(0, 255, (256, 256), dtype=np.uint8)
```

This approach is preferred because:
- Tests run faster (no file I/O)
- No storage requirements
- Cross-platform compatibility
- Easy to modify for different test scenarios

## Using Fixtures

In your tests, reference fixtures via pytest parameters:

```python
def test_my_feature(sample_clinical_text):
    processor = ClinicalProcessor()
    result = processor.process_text(sample_clinical_text)
    assert result is not None
```

Available fixtures are defined in `conftest.py`. See the main test README for a complete list.
