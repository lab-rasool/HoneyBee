# HoneyBee

<img src="docs\assets\HoneyBee.png" width="280px" align="right"/>

Harmonized Oncology Network Enhancing Yield through Big Data Exploration and Evaluation (HONEYBEE) aims to provide a platform for the development of AI models for oncology. Including tools for medical data loading, embedding generation, huggingface instruction tuning dataset creation, and advanced RAG support. The current version includes the following dataloaders:

1. SVS
1. DICOM
1. NIFTI
1. TIFF
1. PDF
1. Images
1. MINDS-DB

Additionally, it includes the following Sentence Transformer style embeddings functions for Foundational medical models

1. HuggingFace text embeddings models (i.e. GatorTron, BioBERT, etc.)
1. REMEDIS Pathology
1. REMEDIS Radiology
1. [SeNMo]()

## Citation

If you use this code, please cite the following paper:

```
@article{honeybee,
      title={HoneyBee: A Scalable Modular Framework for Creating Multimodal Oncology Datasets with Foundational Embedding Models}, 
      author={Aakash Tripathi and Asim Waqas and Yasin Yilmaz and Ghulam Rasool},
      year={2024},
      eprint={2405.07460},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```