# Flood Mamba Edge: Lightweight State Space Models for Real-Time Flood Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Edge AI](https://img.shields.io/badge/Platform-Edge%20%7C%20Jetson%20%7C%20MCU%20%7C%20NPU-brightgreen.svg)](#)

A high-performance, edge-optimized benchmarking framework featuring lightweight **Mamba / State Space Models (SSMs)** and real-time convolutional baselines for flood segmentation. Designed specifically for resource-constrained edge hardware (NVIDIA Jetson, Raspberry Pi, mobile SoCs, and embedded MCUs) operating at **128x128** resolution.

---

## 🌟 Supported Architectures & Provenance

All Mamba architectures in this repository are faithfully ported from their original official publications and source repositories, adapted with input resolution $128 \times 128$ and binary flood segmentation heads ($1$ output channel). Pure PyTorch native fallbacks are included so models run out-of-the-box without requiring precompiled CUDA C++ extensions.

| Model | Original Publication | Source Repository | Commit Hash | Key Architectural Focus |
| :--- | :--- | :--- | :--- | :--- |
| **MambaFloodLite** | ICCE MambaFloodLite | Local Verified Implementation | `main` | AxialDW + 4-directional SS2D selective scan + EESP bottleneck (285K params) |
| **LCMamNet** | Fan et al. (*arXiv:2607.24184*, 2026) | [Haoyu096/LCMamNet](https://github.com/Haoyu096/LCMamNet) | `8f3ba5bfce33f8133a4a63f357d27d171223d87c` | Cross-Directional Conv (CDConv), BiMamba multi-scale fusion, SBA skip connections |
| **UltraLight VM-UNet** | Wu et al. (*Patterns*, 2024; *arXiv:2403.19654*) | [wurenkai/UltraLight-VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet) | `27e44181b3cd5b0b2ab3ad1e3ddb8cad67368fbd` | Parallel Vision Mamba (PVM) layer (0.05M params), SC-Attention Bridge |
| **LightM-UNet** | Liao et al. (*arXiv:2403.05246*, 2024) | [MrBlankness/LightM-UNet](https://github.com/MrBlankness/LightM-UNet) | `38148ef8ac408014be0d668d9a98e2e97a68ed99` | Residual Mamba Blocks (ResMambaBlock) + DWConv skip upsampling |
| **SegFormer (B0)** | Xie et al. (*NeurIPS 2021*) | Official MiT-B0 | Standard Baseline | Lightweight hierarchical transformer |
| **STDC1** | Fan et al. (*CVPR 2021*) | Short-Term Dense Concatenate | Standard Baseline | Dual-path real-time detail preservation |
| **Fast-SCNN** | Poudel et al. (*BMVC 2019*) | Fast Segmentation ConvNet | Standard Baseline | Multi-branch real-time edge architecture |
| **UNet** | Ronneberger et al. (*MICCAI 2015*) | Standard U-Net | Standard Baseline | Encoder-decoder with direct skip connections |
| **ULite** | Lightweight ViT segmentation | Standard U-Lite | Standard Baseline | Ultra-compact axial attention baseline |
| **DABNet** | Depth-wise Asymmetric Bottleneck | Standard DABNet | Standard Baseline | Dilated asymmetric convolutional blocks |
| **DeepLabV3** | Chen et al. (*arXiv:1706.05587*) | ResNet50 ASPP | Standard Baseline | Heavy contextual baseline (pre-trained ImageNet) |
| **ENet** | Paszke et al. (*arXiv:1606.02147*) | Real-time ENet | Standard Baseline | Early downsampling for low latency |
| **ESPNetv2** | Mehta et al. (*CVPR 2019*) | ESPNetv2 | Standard Baseline | Extremely efficient spatial pyramid dilated convs |
| **EDANet** | Lo et al. (*arXiv:1809.06323*) | Efficient Dense Asymmetric Net | Standard Baseline | Dense connectivity with asymmetric convolutions |
| **SegNet** | Badrinarayanan et al. (*TPAMI 2017*) | SegNet | Standard Baseline | Max pooling indices decoder |

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/doantrongthai/flood_mamba_edge.git
cd flood_mamba_edge
pip install -r requirements.txt
```

### 1. Training a Mamba Model
Train `mambafloodlite` (or any other architecture) on flood segmentation at $128 \times 128$:
```bash
# Train MambaFloodLite
python main.py --model mambafloodlite --dataset floodkaggle --size 128 --batch_size 4 --epochs 50

# Train LCMamNet
python main.py --model lcmamnet --dataset floodkaggle --size 128 --batch_size 4 --epochs 50

# Train UltraLight VM-UNet
python main.py --model ultralight_vmunet --dataset floodkaggle --size 128 --batch_size 4 --epochs 50

# Train LightM-UNet
python main.py --model lightm_unet --dataset floodkaggle --size 128 --batch_size 4 --epochs 50
```

### 2. Multi-Seed Statistical Evaluation
Run rigorous 5-seed experiments ($[42, 123, 456, 789, 2024]$) to compute mean and standard deviations for research publications:
```bash
python benchmark.py --model mambafloodlite --dataset floodkaggle --size 128 --multiseed
```

### 3. Cross-Dataset Zero-Shot Domain Transfer
Evaluate domain generalization by training on a source dataset and testing zero-shot on other domains:
```bash
python cross_dataset_eval.py --model mambafloodlite --weights outputs/mambafloodlite_best.pth --source_dataset floodkaggle --target_datasets floodkaggle aifloodsense floodvn --size 128
```

### 4. ONNX Export for Edge Deployment
Export models directly to ONNX format for TensorRT, ONNX-Runtime, or edge accelerator deployment:
```bash
python export.py --model mambafloodlite --size 128 --output outputs/mambafloodlite.onnx
```

---

## 📁 Repository Structure

```
flood_mamba_edge/
├── models/
│   ├── __init__.py           # Model factory and deterministic initialization
│   ├── mamba_block.py        # Mamba S6 operator with pure PyTorch fallback
│   ├── mambafloodlite.py     # MambaFloodLite (AxialDW + 4-way selective scan)
│   ├── lcmamnet.py           # LCMamNet (CDConv + BiMamba multi-scale fusion)
│   ├── ultralight_vmunet.py  # UltraLight VM-UNet (Parallel Vision Mamba)
│   ├── lightm_unet.py        # LightM-UNet (Residual Mamba Blocks)
│   ├── segformer.py          # SegFormer (MiT-B0) baseline
│   ├── stdc1.py              # STDC1 baseline
│   ├── fast_scnn.py          # Fast-SCNN baseline
│   ├── unet.py               # UNet baseline
│   └── ...                   # Additional real-time baselines
├── losses/                   # BCE, Dice, Focal, Lovasz, Tversky loss functions
├── utils/
│   ├── dataloader.py         # Deterministic dataset loaders (default size=128)
│   ├── metrics.py            # mIoU, Dice, Boundary-IoU, FPS, latency, GMACs
│   └── trainer.py            # Unified training and validation engine
├── main.py                   # Main CLI entry point
├── benchmark.py              # Multi-seed and reproducibility test suite
├── export.py                 # ONNX edge deployment exporter
├── cross_dataset_eval.py     # Cross-domain generalization benchmark
├── requirements.txt
└── README.md
```

---

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
