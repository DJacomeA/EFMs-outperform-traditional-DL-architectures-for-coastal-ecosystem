# Earth Foundation Models for Coastal Ecosystem Semantic Segmentation

> **Comparative analysis of Earth Foundation Models (EFMs) for high-fidelity coastal habitat mapping**

[Paper](link-to-paper) • [Dataset](link-to-dataset)

---
## Note: Configurations, Dataset and Checkpoints will be release after publication
---

---

## Overview

Coastal ecosystems—mangrove forests, tidal wetlands, and aquaculture ponds—face escalating threats from climate change and anthropogenic pressure, requiring precise monitoring for effective conservation. While traditional deep learning models (U-Net, DeepLabV3) provide foundational capabilities, they struggle with complex boundary delineation and spectral confusion in dynamic coastal environments.

This repository provides the official implementation for fine-tuning Earth Foundation Models (EFMs) on multi-temporal Sentinel-2 imagery to achieve state-of-the-art coastal habitat mapping. We systematically evaluate **Prithvi 100, 300 & 600**, **Satlas**, **DOFA**, and **TerraMind (Base & Large)** against traditional architectures, demonstrating that EFMs achieve superior accuracy with sharper boundaries and higher ground truth fidelity.

### 🔑 Key Innovations

- **Dynamic Adaptation**: The DOFA architecture, specifically, represents the current optimal solution for coastal semantic segmentation, balancing dynamic spectral adaptation with multi-scale spatial reasoning.
- **Efficient Transfer Learning**: Encoder-only fine-tuning preserves 98.9% of end-to-end performance while reducing computational overhead
- **Operational Scalability**: Regional-scale deployment validated (34.65s per 2,500 km²) across the Amazon Delta and Sundarbans

---

## 🎯 Performance Highlights

### Benchmark Results

| Model | Mean IoU | Mean Accuracy | Inference Time (256×256) | Parameters |
|-------|----------|---------------|--------------------------|------------|
| **DOFA** | **0.95 ± 0.0035** | **0.98 ± 0.0016** | 25.4 ms | 352.9M |
| TerraMind-Large | 0.94 ± 0.0038 | 0.97 ± 0.0014 | 25.1 ms | 319.5M |
| SegFormer | 0.91 ± 0.0003 | 0.97 ± 0.0003 | 11.6 ms | 24.9M |
| U-Net | 0.91 ± 0.0018 | 0.96 ± 0.0009 | **11.3 ms** | 32.6M |
| DeepLabV3 | 0.91 ± 0.0024 | 0.96 ± 0.0009 | 17.3 ms | 39.7M |
| Prithvi-600M | 0.89 ± 0.0036 | 0.95 ± 0.0014 | 26.8 ms | 650.9M |

**Key Achievements:**
- **+4.4% mIoU improvement** over best traditional model (U-Net)
- **Superior boundary delineation** in spectrally ambiguous transition zones (mangrove ↔ wetland)
- **Operational feasibility**: Enables regional-scale deployment

### Class-Specific Performance (DOFA)
| Class | IoU | Accuracy | Notes |
|-------|-----|----------|-------|
| Mangroves | 0.93 | 0.97 | Resolves spectral confusion with coastal wetlands |
| Coastal Wetlands | 0.94 | 0.98 | Sharp boundaries in tidal channels |
| Pond Aquaculture | 0.95 | 0.99 | Precise delineation of narrow inter-pond channels |
| Water | 0.97 | 0.99 | Robust to turbidity variations |
| Other Land | 0.95 | 0.97 | Minimal commission errors |

---

## 📋 Requirements

### Hardware
- **GPU**: NVIDIA GPU (Quadro RTX 8000/A6000 recommended)
- **Inference**: 8GB+ VRAM sufficient for inference on 256×256 tiles
- **Storage**: ~50GB for dataset and model checkpoints

### Software
- **OS**: Ubuntu 20.04 LTS (recommended) or Windows with WSL2
- **Python**: 3.10+
- **CUDA**: 12.1+
- **cuDNN**: 8.9.2+

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/DJacomeA/EFMs-outperform-traditional-DL-architectures-for-coastal-ecosystem.git
cd EFMs-outperform-traditional-DL-architectures-for-coastal-ecosystem

# Create virtual environment
conda create -n efm-coastal python=3.10
conda activate efm-coastal

# Install TerraTorch
pip install git+https://github.com/terrastackai/terratorch.git

# Fine-tuning
terratorch fit --config /clabs5_final_dofa_ignore.yaml 

# Predicct
terratorch predict -c /clabs5_final_dofa_ignore.yaml --ckpt_path /dofa_nod4/version_0/checkpoints/epoch=99.ckpt --predict_output_dir /CLABS_OUTPUT_FINAL/clabs_dofa_ignore --data.init_args.predict_data_root /dataset_stride/tiles/val
