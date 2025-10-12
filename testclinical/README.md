# HoneyBee Clinical Processing - Interactive Tutorial

This directory contains a comprehensive Jupyter notebook demonstrating all clinical processing capabilities of the HoneyBee framework.

## 📚 Contents

- **`HoneyBee_Clinical_Processing_Complete.ipynb`** - Complete interactive tutorial covering all features
- **`sample.PDF`** - Sample pathology report for testing

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# System dependencies
sudo apt-get install openslide-tools tesseract-ocr  # Ubuntu/Debian
# or
brew install openslide tesseract  # macOS

# Python dependencies
pip install torch transformers pytesseract pillow PyPDF2 pdf2image nltk opencv-python numpy

# NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 2. Launch the Notebook

```bash
cd /mnt/f/Projects/HoneyBee/clinicaltest
jupyter notebook HoneyBee_Clinical_Processing_Complete.ipynb
```

Or use JupyterLab:
```bash
jupyter lab HoneyBee_Clinical_Processing_Complete.ipynb
```

Or open in VS Code with the Jupyter extension.

## 📖 Notebook Contents

The comprehensive notebook includes:

### Section 1: Basic PDF Processing
- Load and process PDF documents
- OCR support for scanned documents
- Text extraction and document structure analysis
- Entity extraction from PDFs

### Section 2: Direct Text Processing
- Process clinical text without files
- Entity extraction from raw text
- Temporal timeline creation
- Entity relationship detection

### Section 3: Configuration Options
- Sentence segmentation with sliding window
- Paragraph segmentation with hierarchical processing
- Important segments strategy for long documents
- Custom entity recognition settings
- Output configuration options

### Section 4: Embedding Generation
- Single text embeddings
- Batch embedding generation
- Semantic similarity computation
- Different pooling methods (mean, cls, max)
- Multiple biomedical models (GatorTron, BioClinicalBERT, PubMedBERT)

### Section 5: Advanced Entity Extraction
- Cancer-specific entities (tumor, staging, biomarkers)
- Temporal entities and timeline
- Entity relationships
- Ontology mapping (SNOMED-CT, RxNorm)
- Measurement extraction

### Section 6: HoneyBee Main API
- High-level API usage
- `generate_embeddings()` examples
- `process_clinical()` convenience methods
- Multimodal integration

### Section 7: Batch Processing
- Process multiple documents efficiently
- Output generation and analysis
- Statistics and reporting

### Section 8: Complete Workflow Examples
- Cancer patient analysis pipeline
- Similarity search pipeline
- End-to-end examples
- Best practices

## 🎯 What You Can Do

✅ **Document Processing**
- Process clinical PDFs with OCR
- Extract text from scanned documents
- Analyze document structure
- Batch process multiple files

✅ **Entity Extraction**
- Cancer-specific entity recognition
- Biomarker status extraction (ER, PR, HER2, etc.)
- Tumor staging and grading
- Medication and dosage extraction
- Temporal information and timeline

✅ **Embeddings**
- Generate embeddings with 4 biomedical models
- Compute semantic similarity
- Find similar clinical cases
- Support for downstream ML tasks

✅ **Configuration**
- Flexible tokenization strategies
- Long document handling
- Custom entity recognition
- Output customization

## 📊 Example Usage

### Quick Processing

```python
from honeybee.processors import ClinicalProcessor

# Initialize processor
processor = ClinicalProcessor()

# Process a PDF
result = processor.process("sample.PDF")

# Or process text directly
result = processor.process_text("Patient diagnosed with breast cancer...")

# Extract entities
print(f"Found {len(result['entities'])} entities")
for entity in result['entities']:
    print(f"  {entity['text']} [{entity['type']}]")
```

### Generate Embeddings

```python
# Generate embeddings
embeddings = processor.generate_embeddings(
    text="Patient with stage III lung adenocarcinoma",
    model_name="gatortron"
)

print(f"Embedding shape: {embeddings.shape}")
```

### Using HoneyBee API

```python
from honeybee import HoneyBee

# Initialize
honeybee = HoneyBee()

# Process clinical data
result = honeybee.process_clinical(text="Clinical text here...")

# Generate embeddings
embeddings = honeybee.generate_embeddings(
    data="Text to embed",
    modality="clinical"
)
```

## 🔧 Troubleshooting

### Tesseract Not Found
```bash
# Install Tesseract
sudo apt-get install tesseract-ocr  # Ubuntu/Debian
brew install tesseract  # macOS
```

### NLTK Data Missing
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

### Model Download Issues
- Models are downloaded automatically on first use
- Requires internet connection
- May take several minutes for large models
- Some models may require HuggingFace authentication

### GPU/CUDA Issues
- All functionality works on CPU (slower)
- GPU is automatically detected and used if available
- Use `device="cpu"` to force CPU usage

## 📚 Resources

- **Documentation**: https://lab-rasool.github.io/HoneyBee/docs/clinical-processing/
- **GitHub**: https://github.com/lab-rasool/HoneyBee
- **Issues**: https://github.com/lab-rasool/HoneyBee/issues

## 🎓 Learning Path

1. **Start Here**: Run the notebook cells in order from top to bottom
2. **Experiment**: Modify the sample text and configurations
3. **Try Your Data**: Use your own clinical documents
4. **Explore Models**: Test different biomedical models
5. **Build Pipelines**: Create custom workflows for your use case

## 📝 Notes

- The notebook is fully self-contained and executable
- All code examples are tested and working
- Sample PDF (`sample.PDF`) is included for testing
- Output files are generated in the same directory
- The notebook can be run multiple times safely

## 🤝 Contributing

Found an issue or have a suggestion? Please open an issue on GitHub!

---

**Happy exploring! 🚀**
