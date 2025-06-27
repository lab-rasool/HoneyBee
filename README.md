<div align="center">
  <img src="website/public/images/logo.png" alt="HoneyBee Logo" width="200">
  
  # HoneyBee
  
  **A Scalable Modular Framework for Multimodal AI in Oncology**
  
  [![arXiv](https://img.shields.io/badge/arXiv-2405.07460-b31b1b.svg)](https://arxiv.org/abs/2405.07460)
  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![GitHub stars](https://img.shields.io/github/stars/lab-rasool/HoneyBee?style=social)](https://github.com/lab-rasool/HoneyBee/stargazers)
  [![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
  
  [Documentation](https://lab-rasool.github.io/HoneyBee/) | [Paper](https://arxiv.org/abs/2405.07460) | [Examples](examples/) | [Demo](app.py)
</div>

## 🚀 Overview

HoneyBee is a comprehensive multimodal AI framework designed specifically for oncology research and clinical applications. It seamlessly integrates and processes diverse medical data types—clinical text, radiology images, pathology slides, and molecular data—through a unified, modular architecture. Built with scalability and extensibility in mind, HoneyBee empowers researchers to develop sophisticated AI models for cancer diagnosis, prognosis, and treatment planning.

> [!WARNING]
> **Alpha Release**: This framework is currently in alpha. APIs may change, and some features are still under development.

## ✨ Key Features

### 🏗️ Modular Architecture
- **3-Layer Design**: Clean separation between data loaders, embedding models, and processors
- **Unified API**: Consistent interface across all modalities
- **Extensible**: Easy to add new models and data sources
- **Production-Ready**: Optimized for both research and clinical deployment

### 📊 Comprehensive Data Support

#### Medical Imaging
- **Pathology**: Whole Slide Images (WSI) - SVS, TIFF formats with tissue detection
- **Radiology**: DICOM, NIFTI processing with 3D support
- **Preprocessing**: Advanced augmentation and normalization pipelines

#### Clinical Text
- **Document Processing**: PDF support with OCR for scanned documents
- **NLP Pipeline**: Cancer entity extraction, temporal parsing, medical ontology integration
- **Database Integration**: Native [MINDS](https://github.com/lab-rasool/MINDS) format support
- **Long Document Handling**: Multiple tokenization strategies for clinical notes

#### Molecular Data
- **Genomics**: Support for expression data and mutation profiles
- **Integration**: Seamless combination with imaging and clinical data

### 🧠 State-of-the-Art Embedding Models

#### Clinical Text Embeddings
- **GatorTron**: Domain-specific clinical language model
- **BioBERT**: Biomedical text understanding
- **PubMedBERT**: Scientific literature embeddings
- **Clinical-T5**: Text-to-text clinical transformers

#### Medical Image Embeddings
- **REMEDIS**: Self-supervised medical image representations
- **RadImageNet**: Pre-trained radiological feature extractors
- **UNI**: Universal medical image encoder
- **Custom Models**: Easy integration of proprietary models

### 🛠️ Advanced Capabilities

#### Multimodal Integration
- **Cross-Modal Learning**: Unified representations across modalities
- **Attention Mechanisms**: Interpretable fusion strategies
- **Patient-Level Aggregation**: Comprehensive patient profiles

#### Analysis Tools
- **Survival Analysis**: Cox PH, Random Survival Forest, DeepSurv
- **Classification**: Multi-class cancer type prediction
- **Retrieval**: Similar patient identification
- **Visualization**: Interactive t-SNE dashboards

#### Clinical Applications
- **Risk Stratification**: Patient outcome prediction
- **Treatment Planning**: Personalized therapy recommendations
- **Biomarker Discovery**: Multi-omic pattern identification

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (optional, for GPU acceleration)

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y openslide-tools tesseract-ocr

# macOS
brew install openslide tesseract

# Windows
# Install from official websites:
# - OpenSlide: https://openslide.org/download/
# - Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
```

### Installation

```bash
# Clone the repository
git clone https://github.com/lab-rasool/HoneyBee.git
cd HoneyBee

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt')"

# Install HoneyBee in development mode
pip install -e .
```

### Environment Setup

Create a `.env` file in the project root:

```bash
# MINDS database credentials (if using MINDS format)
HOST=your_server
PORT=5433
DB_USER=postgres
PASSWORD=your_password
DATABASE=minds

# HuggingFace API (for some models)
HF_API_KEY=your_huggingface_api_key
```

## 🔬 Research Applications

HoneyBee has been successfully applied to:

- **Cancer Subtype Classification**: Automated identification of cancer subtypes from multimodal data
- **Survival Prediction**: Risk stratification and outcome prediction for treatment planning
- **Similar Patient Retrieval**: Finding patients with similar clinical profiles for precision medicine
- **Biomarker Discovery**: Identifying multimodal patterns associated with treatment response

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Fork and clone your fork
git clone https://github.com/YOUR_USERNAME/HoneyBee.git
cd HoneyBee

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -r requirements.txt
pip install -e .
```

## 🐛 Known Issues & Limitations

- **Alpha Status**: Some features are still under development
- **Memory Requirements**: WSI processing requires significant RAM (16GB+ recommended)
- **GPU Recommended**: While CPU fallback exists, GPU acceleration significantly improves performance
- **Limited Test Coverage**: Comprehensive test suite is planned for future releases

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📝 Citation

If you use HoneyBee in your research, please cite our paper:

```bibtex
@article{tripathi2024honeybee,
    title={HoneyBee: A Scalable Modular Framework for Creating Multimodal Oncology Datasets with Foundational Embedding Models},
    author={Aakash Tripathi and Asim Waqas and Yasin Yilmaz and Ghulam Rasool},
    journal={arXiv preprint arXiv:2405.07460},
    year={2024},
    eprint={2405.07460},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

---

<div align="center">
  Made with ❤️ by the <a href="https://github.com/lab-rasool">Lab Rasool</a> team
</div>
