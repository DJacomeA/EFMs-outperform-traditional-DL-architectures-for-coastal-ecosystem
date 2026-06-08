# Earth Foundation Models for Coastal Ecosystem Semantic Segmentation

> **Comparative analysis of Earth Foundation Models (EFMs) for high-fidelity coastal habitat mapping**

[Paper](link-to-paper) • [Dataset](link-to-dataset)

---
## Note: Configurations, Dataset and Checkpoints will be released after publication
---

## Overview

Coastal ecosystems—mangrove forests, tidal wetlands, and aquaculture ponds—face escalating threats from climate change and anthropogenic pressure, requiring precise monitoring for effective conservation. While traditional deep learning models (U-Net, DeepLabV3) provide foundational capabilities, they struggle with complex boundary delineation and spectral confusion in dynamic coastal environments.

This repository provides the official implementation for fine-tuning Earth Foundation Models (EFMs) on multi-temporal Sentinel-2 imagery to achieve state-of-the-art coastal habitat mapping. We systematically evaluate **Prithvi 100M, 300M & 600M**, **Satlas**, **DOFA**, and **TerraMind (Base & Large)** against traditional architectures, demonstrating that EFMs achieve superior accuracy with sharper boundaries and higher ground truth fidelity.

### 🔑 Key Innovations

- **Dynamic Adaptation**: The DOFA architecture represents the current optimal solution for coastal semantic segmentation, balancing dynamic spectral adaptation with multi-scale spatial reasoning.
- **Efficient Transfer Learning**: Encoder-only fine-tuning preserves 98.9% of end-to-end performance while reducing trainable parameters by 95.5%.
- **Operational Scalability**: Regional-scale deployment validated (34.65s per 2,500 km²) across the Amazon Delta and Sundarbans.

---

## 🎯 Performance Highlights

### Benchmark Results

| Model | Mean IoU | Mean Accuracy | Inference Time (256×256) | Parameters | Train VRAM |
|-------|----------|---------------|--------------------------|------------|------------|
| **DOFA** | **0.95 ± 0.0035** | **0.98 ± 0.0016** | 25.4 ms | 352.9M | 9.5 GB |
| TerraMind-Large | 0.94 ± 0.0038 | 0.97 ± 0.0014 | 25.1 ms | 319.5M | 8.6 GB |
| SegFormer | 0.91 ± 0.0003 | 0.97 ± 0.0003 | 11.6 ms | 24.9M | 0.6 GB |
| U-Net | 0.91 ± 0.0018 | 0.96 ± 0.0009 | **11.3 ms** | 32.6M | 0.7 GB |
| DeepLabV3 | 0.91 ± 0.0024 | 0.96 ± 0.0009 | 17.3 ms | 39.7M | 0.9 GB |
| UPerNet | 0.91 ± 0.0024 | 0.96 ± 0.0009 | 11.3 ms | 30.0M | 0.7 GB |
| MANet | 0.90 ± 0.0018 | 0.96 ± 0.0009 | 12.7 ms | 147.0M | 3.3 GB |
| Prithvi-300M | 0.93 ± 0.0018 | 0.97 ± 0.0009 | 25.3 ms | 319.8M | 8.6 GB |
| Prithvi-600M | 0.89 ± 0.0036 | 0.95 ± 0.0014 | 26.8 ms | 650.9M | 17.5 GB |
| Satlas | 0.92 ± 0.0024 | 0.97 ± 0.0009 | 58.5 ms | 98.9M | 2.2 GB |
| TerraMind-Base | 0.92 ± 0.0024 | 0.97 ± 0.0009 | 17.3 ms | 99.0M | 2.2 GB |

**Key Achievements:**
- **+4.4% mIoU improvement** over best traditional model (U-Net)
- **Superior boundary delineation** in spectrally ambiguous transition zones (mangrove ↔ wetland)
- **2.0–2.6× Bd-IoU improvement** over traditional architectures
- **Operational feasibility**: Enables regional-scale deployment with appropriate GPU infrastructure

### Class-Specific Performance (DOFA)

| Class | IoU | Accuracy | Boundary IoU | Edge F1 | Notes |
|-------|-----|----------|--------------|---------|-------|
| Mangroves | 0.93 | 0.97 | 0.50 | 0.67 | Resolves spectral confusion with coastal wetlands |
| Coastal Wetlands | 0.94 | 0.98 | 0.35 | 0.52 | Sharp boundaries in tidal channels |
| Pond Aquaculture | 0.95 | 0.99 | 0.51 | 0.67 | Precise delineation of narrow inter-pond channels |
| Water | 0.97 | 0.99 | 0.50 | 0.67 | Robust to turbidity variations |
| Other Land Cover | 0.95 | 0.97 | 0.41 | 0.58 | Minimal commission errors |

---

## 📋 Requirements

### Hardware

| Task | Minimum GPU | Recommended GPU | VRAM | Notes |
|------|-------------|-----------------|------|-------|
| **Training / Fine-tuning** | NVIDIA RTX 3090 (24 GB) | NVIDIA Quadro RTX 8000 / A6000 (48 GB) | 8.6–17.5 GB | Required for Prithvi-300M/600M, DOFA, TerraMind-Large |
| **Inference (256×256 tiles)** | NVIDIA RTX 2080 Ti (11 GB) | NVIDIA RTX 3080 / A4000 | 2.2–9.5 GB | 8 GB+ sufficient for single-tile inference |
| **Regional-scale inference** | Same as above | Multi-GPU or high-VRAM card | 9.5 GB | 2,500 km² processed in ~35s on Quadro RTX 8000 |

- **Storage**: ~50 GB for dataset and model checkpoints
- **RAM**: 32 GB system RAM recommended

### Software Environment

The codebase was developed and validated on the following stack:

| Component | Version | Purpose |
|-----------|---------|---------|
| OS | Ubuntu 20.04 LTS | Primary development platform |
| Python | 3.10.12 | Runtime environment |
| PyTorch | 2.1.0 | Deep learning framework |
| CUDA | 12.2 | GPU acceleration |
| cuDNN | 8.9.x | Deep learning GPU primitives |
| TerraTorch | 0.1.1 | Geospatial foundation model toolkit |
| GCC | 9.4.0 | Compiler for native extensions |

> **Note**: Windows users should use WSL2 with Ubuntu 20.04. macOS is not supported due to CUDA dependency.

---

## 🚀 Installation

### Step 1: System Prerequisites

Ensure your NVIDIA driver supports CUDA 12.2:

```bash
# Verify NVIDIA driver installation
nvidia-smi

# Expected output: Driver Version >= 535.54.03, CUDA Version >= 12.2
```

If `nvidia-smi` fails, install the latest NVIDIA drivers for your GPU from [https://www.nvidia.com/Download/index.aspx](https://www.nvidia.com/Download/index.aspx).

### Step 2: Install CUDA Toolkit 12.2 (if not present)

```bash
# Download and install CUDA 12.2 from NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.54.03_linux.run
sudo sh cuda_12.2.2_535.54.03_linux.run --silent --toolkit

# Add to PATH (append to ~/.bashrc)
echo 'export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA compiler
nvcc --version  # Expected: Cuda compilation tools, release 12.2
```

### Step 3: Create Conda Environment

We strongly recommend using **Miniconda** or **Anaconda**:

```bash
# Download Miniconda if not installed
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create environment with exact Python version
conda create -n efm-coastal python=3.10.12 -y
conda activate efm-coastal

# Verify Python version
python --version  # Should print: Python 3.10.12
```

### Step 4: Install PyTorch 2.1.0 with CUDA 12.1 Support

> **Note**: PyTorch 2.1.0 binaries are compiled against CUDA 12.1, which is forward-compatible with CUDA 12.2 runtime.

```bash
# Install PyTorch 2.1.0 + torchvision + torchaudio (CUDA 12.1 wheel)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch sees your GPU
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
PyTorch: 2.1.0+cu121
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA Quadro RTX 8000
```

### Step 5: Install TerraTorch 0.1.1

```bash
# Install TerraTorch from the official repository
pip install git+https://github.com/terrastackai/terratorch.git@0.1.1

# Or if you prefer to clone and install locally:
# git clone https://github.com/terrastackai/terratorch.git
# cd terratorch
# git checkout 0.1.1
# pip install -e .

# Verify installation
python -c "import terratorch; print(f'TerraTorch: {terratorch.__version__}')"
```

### Step 6: Clone Repository and Install Local Dependencies

```bash
# Clone the repository
git clone https://github.com/DJacomeA/EFMs-outperform-traditional-DL-architectures-for-coastal-ecosystem.git
cd EFMs-outperform-traditional-DL-architectures-for-coastal-ecosystem

# Install additional dependencies (if requirements.txt is provided)
pip install -r requirements.txt

# Or install common geospatial/ML packages manually:
pip install rasterio==1.3.8 geopandas==0.13.2 numpy==1.24.3 scipy==1.11.1     scikit-learn==1.3.0 matplotlib==3.7.2 seaborn==0.12.2 pandas==2.0.3     pillow==10.0.0 opencv-python==4.8.0.78 albumentations==1.3.1     tensorboard==2.13.0 pytorch-lightning==2.0.9 omegaconf==2.3.0
```

### Step 7: Verify Full Installation

Run the verification script to ensure all components are correctly linked:

```bash
python -c "
import torch
import terratorch
import rasterio
import cv2

print('=== Environment Verification ===')
print(f'Python: {torch.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'GPU name: {torch.cuda.get_device_name(0)}')
print(f'TerraTorch: {terratorch.__version__}')
print(f'Rasterio: {rasterio.__version__}')
print(f'OpenCV: {cv2.__version__}')
print('=== All checks passed ===')
"
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` during training | Reduce batch size to 2 or 4; use gradient accumulation; enable mixed precision (`precision=16` in TerraTorch config) |
| `nvcc not found` | Ensure `/usr/local/cuda-12.2/bin` is in your `PATH` |
| `TerraTorch import error` | Reinstall with `pip install --force-reinstall git+https://github.com/terrastackai/terratorch.git` |
| PyTorch CPU-only installed | Uninstall with `pip uninstall torch torchvision torchaudio` and reinstall with `--index-url https://download.pytorch.org/whl/cu121` |
| Driver/CUDA version mismatch | Upgrade NVIDIA driver to ≥ 535.54.03 or use CUDA 11.8 PyTorch wheels (`--index-url https://download.pytorch.org/whl/cu118`) |

---

## 📁 Dataset Setup

### Data Sources

1. **Sentinel-2 Imagery**: Annual cloud-free composites from Google Earth Engine
   - URL: [https://developers.google.com/earth-engine](https://developers.google.com/earth-engine)
   - Bands: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR1), B12 (SWIR2)
   - Resolution: 10 m (B2, B3, B4, B8), resampled from 20 m (B11, B12)
   - Time period: Full year 2022 (January 1 – December 31)

2. **Ground Truth**: ClarkLabs Aquaculture Dataset
   - URL: [https://www.clarku.edu/centers/geospatial-analytics/aquaculture-data-download/](https://www.clarku.edu/centers/geospatial-analytics/aquaculture-data-download/)
   - Resolution: 15 m (resampled to 10 m)
   - Coverage: 85 patches across 17 nations

### Preprocessing Pipeline

```bash
# Expected directory structure after preprocessing
dataset/
├── tiles/
│   ├── train/
│   │   ├── 256x256_patches/     # 8,245 tiles
│   │   └── labels/
│   └── val/
│       ├── 256x256_patches/     # 3,539 tiles
│       └── labels/
└── metadata/
    └── patch_index.csv
```

The preprocessing workflow:
1. Filter Sentinel-2 SR collection with 100% cloud cover threshold
2. Apply scaling factors to convert digital numbers to surface reflectance
3. Compute annual median composites to reduce residual cloud contamination
4. Clip to 85 ground truth patches and resample to 10 m
5. Generate 256×256 pixel tiles with 128-pixel stride
6. Stratified split: 70% train (8,245 tiles) / 30% validation (3,539 tiles)

---

## 🏋️ Training

### Configuration

All training configurations are stored as YAML files in the `configs/` directory. Key hyperparameters used in the paper:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max epochs | 100 | Early stopping with patience = 15 |
| Optimizer | AdamW | Weight decay = 0.05 |
| Initial learning rate | 1e-4 | Cosine annealing schedule |
| Batch size | 4 | Limited by 48 GB VRAM for large EFMs |
| Loss function | Cross-entropy | Standard multi-class segmentation |
| Input size | 256×256×6 | 6 Sentinel-2 bands |
| Random seeds | 0, 42, 123 | For full reproducibility |

### Fine-tuning Strategies

#### 1. End-to-End Fine-tuning (Highest Accuracy)
```bash
# Train all parameters (352.9M for DOFA)
terratorch fit --config configs/clabs5_final_dofa.yaml
```

#### 2. Encoder-Only Fine-tuning (Efficient, 98.9% Performance)
```bash
# Freeze pre-trained encoder, train only decoder (15.7M trainable params)
terratorch fit --config configs/clabs5_final_dofa_encoder_only.yaml
```

#### 3. Decoder-Only Fine-tuning (Not Recommended)
```bash
# Freeze decoder, fine-tune encoder only
# Note: This catastrophically degrades accuracy to 0.83 mIoU
terratorch fit --config configs/clabs5_final_dofa_decoder_only.yaml
```

### Monitoring Training

```bash
# Launch TensorBoard
tensorboard --logdir lightning_logs/

# View training curves (loss, mIoU, accuracy) at http://localhost:6006
```

---

## 🔮 Inference & Prediction

### Single-Patch Inference

```bash
# Predict on validation set using trained checkpoint
terratorch predict   --config configs/clabs5_final_dofa.yaml   --ckpt_path checkpoints/dofa/version_0/checkpoints/epoch=99.ckpt   --predict_output_dir outputs/clabs_dofa   --data.init_args.predict_data_root dataset/tiles/val
```

### Regional-Scale Deployment

For large-scale mapping (e.g., 50×50 km regions like Amazon Delta or Sundarbans):

```bash
# The fine-tuned DOFA model processes a 50×50 km area in ~34.65s
# Average rate: 2.89s per 14×14 km patch (1408×1408 pixels at 10m)

# Example: Process full regional mosaic
python scripts/regional_inference.py   --config configs/clabs5_final_dofa.yaml   --ckpt_path checkpoints/dofa_best.ckpt   --input_mosaic s3://your-bucket/region_50km.tif   --output_dir outputs/regional_maps/   --tile_size 1408   --overlap 128
```

---

## 📊 Evaluation Metrics

The paper employs a six-metric framework:

| Metric | Description | Best Model |
|--------|-------------|------------|
| **mAcc** | Mean pixel accuracy | DOFA: 0.98 |
| **mIoU** | Mean Intersection over Union | DOFA: 0.95 |
| **mF1** | Mean F1-score | DOFA: 0.98 |
| **Kappa** | Cohen's Kappa coefficient | DOFA: 0.97 |
| **mBd-IoU** | Mean Boundary IoU | DOFA: 0.46 |
| **mEdge-F1** | Mean Edge F1-score | DOFA: 0.62 |

Run evaluation:
```bash
python scripts/evaluate.py   --predictions outputs/clabs_dofa   --ground_truth dataset/tiles/val/labels   --metrics all
```

---

## 🗺️ Model Zoo

Pre-trained checkpoints will be released after publication:

| Model | Checkpoint | Config | mIoU | Size |
|-------|------------|--------|------|------|
| DOFA | `dofa_best.ckpt` | `clabs5_final_dofa.yaml` | 0.95 | ~1.4 GB |
| TerraMind-Large | `terramind_large_best.ckpt` | `clabs5_final_terramind_large.yaml` | 0.94 | ~1.3 GB |
| Prithvi-300M | `prithvi_300m_best.ckpt` | `clabs5_final_prithvi_300m.yaml` | 0.93 | ~1.3 GB |
| U-Net | `unet_best.ckpt` | `clabs5_final_unet.yaml` | 0.91 | ~130 MB |

---

## 🤝 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{jacome2026efm,
  title={Earth Foundation Models outperform traditional deep learning architectures for coastal ecosystem semantic segmentation and large-scale habitat mapping},
  author={Jacome, Daniel and Wang, Jianghao and Ge, Yong},
  journal={...},
  year={2026},
  publisher={...}
}
```

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

The TerraTorch library is maintained by the IBM Geospatial Team and licensed under the Apache License 2.0.

---

## 📧 Contact

- **Daniel Jacome** (First Author): dajarguello2022@igsnrr.ac.cn
- **Jianghao Wang** (Corresponding Author): wangjh@lreis.ac.cn
- **Yong Ge**: gey@lreis.ac.cn

**Affiliation**: State Key Laboratory of Resources and Environmental Information System, Institute of Geographic Sciences and Natural Resources Research, Chinese Academy of Sciences, Beijing, 100101, China.

---

## 🙏 Acknowledgments

This research was supported by the National Natural Science Foundation of China (No. 42571540, 42222110 & 42230110).
