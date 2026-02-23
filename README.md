<div align="center">
  <img src="website/public/images/logo.png" alt="HoneyBee Logo" width="200">

  # HoneyBee

  **A Scalable Modular Framework for Multimodal AI in Oncology**

  [![Nature Digital Medicine](https://img.shields.io/badge/Nature%20Digital%20Medicine-Published-success.svg)](https://www.nature.com/articles/s41746-025-02003-4)
  [![PyPI version](https://img.shields.io/pypi/v/honeybee-ml.svg)](https://pypi.org/project/honeybee-ml/)
  [![PyPI Downloads](https://static.pepy.tech/badge/honeybee-ml)](https://pepy.tech/projects/honeybee-ml)
  [![GitHub stars](https://img.shields.io/github/stars/lab-rasool/HoneyBee?style=social)](https://github.com/lab-rasool/HoneyBee/stargazers)
  [![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

  [Documentation  & Examples](https://lab-rasool.github.io/HoneyBee/) | [Paper](https://www.nature.com/articles/s41746-025-02003-4)
</div>

## Publication

**HoneyBee has been officially published in [Nature Digital Medicine](https://www.nature.com/articles/s41746-025-02003-4)!**

> Tripathi, A., Waqas, A., Schabath, M.B. et al. HONeYBEE: enabling scalable multimodal AI in oncology through foundation model-driven embeddings. *npj Digit. Med.* **8**, 622 (2025). https://doi.org/10.1038/s41746-025-02003-4

## Overview

HoneyBee is a comprehensive multimodal AI framework designed specifically for oncology research and clinical applications. It seamlessly integrates and processes diverse medical data types—clinical text, radiology images, pathology slides, and molecular data—through a unified, modular architecture. Built with scalability and extensibility in mind, HoneyBee empowers researchers to develop sophisticated AI models for cancer diagnosis, prognosis, and treatment planning.

> [!WARNING]
> **Alpha Release**: This framework is currently in alpha. APIs may change, and some features are still under development.

## Key Features

- **Multimodal data support**: clinical text, radiology (DICOM/NIFTI), pathology (WSI), and molecular data
- **3-layer modular architecture**: clean separation between loaders, processors, and embedding models
- **Clinical NLP pipeline**: OCR, cancer entity extraction, temporal parsing, and medical ontology mapping
- **Whole Slide Image processing**: tissue detection, patch extraction, stain normalization, and quality filtering
- **State-of-the-art embedding models**: GatorTron, BioBERT, PubMedBERT, UNI, REMEDIS, RadImageNet, and more
- **Cross-modal integration**: unified patient-level representations from multiple data modalities
- **Survival analysis**: Cox PH, Random Survival Forest, and DeepSurv
- **Similar patient retrieval**: find patients with matching clinical profiles
- **Interactive visualization**: t-SNE dashboards for embedding exploration
- **GPU-accelerated**: CuCIM backend for WSI processing with OpenSlide fallback

## Quick Start

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install -y openslide-tools tesseract-ocr

# macOS
brew install openslide tesseract
```

### Installation

```bash
pip install honeybee-ml
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Optional Extras

| Extra | Command | Includes |
|-------|---------|----------|
| Clinical | `pip install honeybee-ml[clinical]` | NLP, OCR, and text processing dependencies |
| Pathology | `pip install honeybee-ml[pathology]` | WSI loading and image processing |
| Molecular | `pip install honeybee-ml[molecular]` | Genomics and expression data support |
| All | `pip install honeybee-ml[all]` | Everything above |

## Research Applications

HoneyBee has been successfully applied to:

- **Cancer Subtype Classification**: Automated identification of cancer subtypes from multimodal data
- **Survival Prediction**: Risk stratification and outcome prediction for treatment planning
- **Similar Patient Retrieval**: Finding patients with similar clinical profiles for precision medicine
- **Biomarker Discovery**: Identifying multimodal patterns associated with treatment response

## License

See the [LICENSE](LICENSE) file for details.

## Citation

If you use HoneyBee in your research, please cite our paper:

```
Tripathi, A., Waqas, A., Schabath, M.B. et al. HONeYBEE: enabling scalable multimodal AI in
oncology through foundation model-driven embeddings. npj Digit. Med. 8, 622 (2025).
https://doi.org/10.1038/s41746-025-02003-4
```
