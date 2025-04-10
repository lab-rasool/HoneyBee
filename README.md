<div align="center" style="display: flex; align-items: center; justify-content: center;">
  <div style="flex: 1; text-align: left;">
    <img src="website/public/images/logo.png" alt="HoneyBee Logo" width="120px" height="140px">
  </div>
  <div style="flex: 2; text-align: left;">
    <h1>HoneyBee - A Scalable Modular Framework for Multimodal AI in Oncology</h1>
    <p><strong>An Open-Source Scalable Modular Framework for Multimodal AI in Oncology</strong></p>
    <p>
      <a href="https://arxiv.org/abs/2405.07460">
        <img src="https://img.shields.io/badge/arXiv-2405.07460-b31b1b.svg" alt="arXiv">
      </a>
      <a href="LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
      </a>
      <a href="https://github.com/lab-rasool/HoneyBee/stargazers">
        <img src="https://img.shields.io/github/stars/lab-rasool/HoneyBee?style=social" alt="GitHub stars">
      </a>
    </p>
  </div>
</div>

## 🚀 Overview

> [!NOTE]
> This is a work in progress, we are currently working on the aplha release. Please check back soon for updates.

HoneyBee is a comprehensive platform for developing AI models in oncology research. It provides tools for medical data processing, embedding generation, instruction tuning dataset creation, and advanced RAG (Retrieval-Augmented Generation) support.

## ✨ Key Features

### 📊 Medical Data Loaders

HoneyBee provides efficient loaders for various medical data formats:

- **Whole Slide Imaging:** SVS, TIFF
- **Radiology:** DICOM, NIFTI
- **Clinical Data:** PDF, [MINDS](https://github.com/lab-rasool/MINDS)
- **General:** Various image formats (PNG, JPG)
- And more formats coming soon!

### 🧠 Embedding Generation

Access to cutting-edge embedding models specifically tuned for medical data:

- 🔬 **Medical Text:** Support for specialized models like GatorTron, BioBERT, and more
- 📊 **Medical Imaging:** Integration with REMEDIS and RadImageNet
- 🔗 **Multimodal Analysis:** [SeNMo](https://github.com/lab-rasool/SeNMo) for cross-modal embeddings
- 🔄 Easy extensibility for custom embedding functions

### 🛠️ Advanced Capabilities

- **Dataset Creation:** Tools for generating instruction tuning datasets compatible with Hugging Face
- **RAG Support:** Implementations of modern retrieval-augmented generation pipelines
- **Scalable Processing:** Optimized for large medical datasets

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/lab-rasool/HoneyBee.git
cd HoneyBee

# Install dependencies
pip install -e .
```

## 📚 Documentation

For comprehensive documentation, visit our [website](https://lab-rasool.github.io/HoneyBee/).

## 🤝 Contributing

We welcome contributions from the community! Please see our contributing guidelines for more details on how to get involved.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📝 Citation

If you use HoneyBee in your research, please cite our paper:

```bibtex
@article{honeybee,
      title={HoneyBee: A Scalable Modular Framework for Creating Multimodal Oncology Datasets with Foundational Embedding Models},
      author={Aakash Tripathi and Asim Waqas and Yasin Yilmaz and Ghulam Rasool},
      year={2024},
      eprint={2405.07460},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
