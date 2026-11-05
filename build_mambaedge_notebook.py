"""
Script to build mambaedge.ipynb:
Comprehensive 5-Seed Training, INT8 QAT, ONNX Export, Cross-Dataset Evaluation,
and Jetson / Edge Deployment Artifact Bundler for MambaEdge (formerly m12_v16).
Follows exact edanet.ipynb structure with official Mamba CUDA kernel setup.
"""

import json
import os

def main():
    nb_filename = "mambaedge.ipynb"

    md_header = """# MambaEdge: Comprehensive 5-Seed Benchmark, INT8 QAT & Jetson Deployment
## Ultra-Lightweight (235K Params, >100 FPS) Multi-Dilated Spatial Boundary Mamba
### Full Pipeline: Multi-Seed Training (5 Seeds) | Quantization-Aware Training (INT8) | ONNX Export | Cross-Domain Generalization | Artifact Packaging

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/doantrongthai/flood_mamba_edge/blob/main/mambaedge.ipynb)

This production notebook executes the complete end-to-end evaluation and deployment suite for **MambaEdge**:
1. **Official Precompiled Mamba CUDA Kernel**: Instant Tesla T4 acceleration without compile overhead.
2. **5-Seed Statistical Reproducibility**: Trains across 5 standard random seeds (`[42, 123, 456, 789, 2024]`) on both Satellite (`floodkaggle`) and UAV (`aifloodsense`) datasets.
3. **INT8 Quantization-Aware Training (QAT)**: Fine-tunes weights with simulated integer quantization for deployment on NVIDIA Jetson / Edge devices.
4. **ONNX Export**: Generates clean, deployable single-tensor graph `.onnx` models for TensorRT / Triton / ONNX Runtime.
5. **Cross-Domain Generalization Benchmark**: Rigorous cross-dataset evaluation across `floodkaggle`, `aifloodsense`, and `floodvn`.
6. **Automatic Zip Packaging**: Packages all checkpoints (`.pth`), INT8 models, ONNX graphs, and evaluation logs for 1-click download.
"""

    cell_0 = """# 1. Clone Repository & Install Dependencies
!git clone https://github.com/doantrongthai/flood_mamba_edge.git
%cd flood_mamba_edge
!pip install -q -r requirements.txt
!pip install -q tabulate gdown pandas
print("[OK] flood_mamba_edge repository cloned and ready.")"""

    cell_1 = """# 2. Setup Official Hardware-Accelerated Mamba CUDA Kernel on Google Colab GPU (Tesla T4)
import zipfile, json, sys, sysconfig, pathlib, subprocess, torch

SP = pathlib.Path(sysconfig.get_paths()["purelib"])
FILE_ID = "1kuBAKlNtIc8G1KQFqO8xV7t9ZEM4XATB"

assert torch.cuda.get_device_capability(0) == (7, 5), f"Vui long chon Colab GPU la Tesla T4 (capability 7.5), hien tai: {torch.cuda.get_device_name(0)}"

# 1) Tai goi pre-compiled CUDA wheel
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"])
import gdown
gdown.download(id=FILE_ID, output="/content/pack.zip", quiet=False)

# 2) Giai nen vao site-packages
with zipfile.ZipFile("/content/pack.zip") as z:
    assert z.testzip() is None, "File tai ve bi hong, vui long tai lai"
    m = json.loads(z.read("__manifest__.json"))
    now = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda}
    bad = {k: (m[k], now[k]) for k in now if m[k] != now[k]}
    assert not bad, f"Moi truong Colab khong khop: {bad}"
    for name in z.namelist():
        if not name.startswith("__"):
            z.extract(name, SP)
print("Da giai nen vao:", SP)

# 3) Cai cac dependency bo tro
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "einops", "packaging", "ninja", "transformers", "triton"])

# 4) Kiem tra CUDA kernel Mamba tren GPU
import selective_scan_cuda, causal_conv1d_cuda
from mamba_ssm.modules.mamba_simple import causal_conv1d_fn
from mamba_ssm import Mamba
print("causal_conv1d_fn:", "OK (CUDA kernel)" if causal_conv1d_fn is not None else "FALLBACK")
y = Mamba(d_model=64).cuda()(torch.randn(2, 128, 64, device="cuda"))
y.mean().backward()
print("[OK] OFFICIAL MAMBA HARDWARE CUDA KERNEL ACTIVE:", y.shape)"""

    cell_2 = """# 3. Global Experiment Configuration
import os

MODEL_NAME = "mambaedge"

AUTO_PRETRAINED_MODELS = {'deeplabv3'}
IS_PRETRAINED = MODEL_NAME.lower() in AUTO_PRETRAINED_MODELS
PRETRAINED_FLAG = "--pretrained" if IS_PRETRAINED else ""

IMAGE_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 100
LR = 0.001
LOSS_NAME = "lovasz_focal"
NUM_CLASSES = 1

MULTISEED = True
SEEDS = [42, 123, 456, 789, 2024]
REP_SEED = 42

QAT_EPOCHS = 15
QAT_LR = 1e-4

def get_checkpoint_path(model, loss, dataset, seed, is_multiseed):
    clean_loss = loss.lower().replace('-', '_')
    candidates = [
        f"outputs/seed_{seed}/{model}_{clean_loss}_{dataset}_s{seed}.pth",
        f"outputs/{model}_{clean_loss}_{dataset}_s{seed}.pth",
        f"outputs/seed_{seed}/{model}_{loss}_{dataset}_s{seed}.pth",
        f"outputs/{model}_{loss}_{dataset}_s{seed}.pth"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0] if is_multiseed else candidates[1]

print(f"[OK] Configuration active for: {MODEL_NAME}")
print(f"     Image Size : {IMAGE_SIZE}x{IMAGE_SIZE} | Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE}")
print(f"     Seeds      : {SEEDS} (Representative seed: {REP_SEED})")"""

    cell_3 = """# 4. Multi-Seed Training on Satellite Dataset (floodkaggle)
if MULTISEED:
    seeds_str = " ".join(map(str, SEEDS))
    !python main.py --dataset floodkaggle --model {MODEL_NAME} --num_classes {NUM_CLASSES} --loss {LOSS_NAME} --size {IMAGE_SIZE} --epochs {EPOCHS} --batch_size {BATCH_SIZE} --lr {LR} --multiseed --seeds {seeds_str} {PRETRAINED_FLAG} --download
else:
    !python main.py --dataset floodkaggle --model {MODEL_NAME} --num_classes {NUM_CLASSES} --loss {LOSS_NAME} --size {IMAGE_SIZE} --epochs {EPOCHS} --batch_size {BATCH_SIZE} --lr {LR} --seed {REP_SEED} {PRETRAINED_FLAG} --download"""

    cell_4 = """# 5. INT8 Quantization-Aware Training (QAT) for Edge Deployment (floodkaggle)
weights_fk = get_checkpoint_path(MODEL_NAME, LOSS_NAME, "floodkaggle", REP_SEED, MULTISEED)
print(f"Running QAT on checkpoint: {weights_fk}")
!python qat.py --weights {weights_fk} --model {MODEL_NAME} --dataset floodkaggle --num_classes {NUM_CLASSES} --loss {LOSS_NAME} --size {IMAGE_SIZE} --epochs {QAT_EPOCHS} --lr {QAT_LR} --optimizer sgd"""

    cell_5 = """# 6. Export ONNX Computational Graph for Jetson / TensorRT (floodkaggle)
weights_fk = get_checkpoint_path(MODEL_NAME, LOSS_NAME, "floodkaggle", REP_SEED, MULTISEED)
print(f"Exporting ONNX from checkpoint: {weights_fk}")
!python export.py --model {MODEL_NAME} --weights {weights_fk} --size {IMAGE_SIZE} --num_classes {NUM_CLASSES} --output outputs/{MODEL_NAME}_floodkaggle.onnx"""

    cell_6 = """# 7. Multi-Seed Training on UAV Dataset (aifloodsense)
if MULTISEED:
    seeds_str = " ".join(map(str, SEEDS))
    !python main.py --dataset aifloodsense --model {MODEL_NAME} --num_classes {NUM_CLASSES} --loss {LOSS_NAME} --size {IMAGE_SIZE} --epochs {EPOCHS} --batch_size {BATCH_SIZE} --lr {LR} --multiseed --seeds {seeds_str} {PRETRAINED_FLAG} --download
else:
    !python main.py --dataset aifloodsense --model {MODEL_NAME} --num_classes {NUM_CLASSES} --loss {LOSS_NAME} --size {IMAGE_SIZE} --epochs {EPOCHS} --batch_size {BATCH_SIZE} --lr {LR} --seed {REP_SEED} {PRETRAINED_FLAG} --download"""

    cell_7 = """# 8. INT8 Quantization-Aware Training (QAT) for Edge Deployment (aifloodsense)
weights_afs = get_checkpoint_path(MODEL_NAME, LOSS_NAME, "aifloodsense", REP_SEED, MULTISEED)
print(f"Running QAT on checkpoint: {weights_afs}")
!python qat.py --weights {weights_afs} --model {MODEL_NAME} --dataset aifloodsense --num_classes {NUM_CLASSES} --loss {LOSS_NAME} --size {IMAGE_SIZE} --epochs {QAT_EPOCHS} --lr {QAT_LR} --optimizer sgd"""

    cell_8 = """# 9. Export ONNX Computational Graph for Jetson / TensorRT (aifloodsense)
weights_afs = get_checkpoint_path(MODEL_NAME, LOSS_NAME, "aifloodsense", REP_SEED, MULTISEED)
print(f"Exporting ONNX from checkpoint: {weights_afs}")
!python export.py --model {MODEL_NAME} --weights {weights_afs} --size {IMAGE_SIZE} --num_classes {NUM_CLASSES} --output outputs/{MODEL_NAME}_aifloodsense.onnx"""

    cell_9 = """# 10. Cross-Domain Generalization Benchmark (Trained on floodkaggle)
weights_fk = get_checkpoint_path(MODEL_NAME, LOSS_NAME, "floodkaggle", REP_SEED, MULTISEED)
print(f"Evaluating cross-dataset generalization for checkpoint: {weights_fk}")
!python cross_dataset_eval.py --model {MODEL_NAME} --weights {weights_fk} --source_dataset floodkaggle --target_datasets floodkaggle aifloodsense floodvn --size {IMAGE_SIZE} --num_classes {NUM_CLASSES} --download"""

    cell_10 = """# 11. Cross-Domain Generalization Benchmark (Trained on aifloodsense)
weights_afs = get_checkpoint_path(MODEL_NAME, LOSS_NAME, "aifloodsense", REP_SEED, MULTISEED)
print(f"Evaluating cross-dataset generalization for checkpoint: {weights_afs}")
!python cross_dataset_eval.py --model {MODEL_NAME} --weights {weights_afs} --source_dataset aifloodsense --target_datasets aifloodsense floodkaggle floodvn --size {IMAGE_SIZE} --num_classes {NUM_CLASSES} --download"""

    cell_11 = """# 12. Bundle All Deployable Weights, ONNX & Jetson Artifacts for Download
import os
import shutil
from google.colab import files

bundle_dir = f"{MODEL_NAME}_All_Artifacts"
os.makedirs(bundle_dir, exist_ok=True)
output_dir = "outputs"

if os.path.exists(output_dir):
    for item in sorted(os.listdir(output_dir)):
        src = os.path.join(output_dir, item)
        dst = os.path.join(bundle_dir, item)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        elif os.path.isfile(src):
            shutil.copy2(src, dst)

zip_name = f"{MODEL_NAME}_All_Results.zip"
shutil.make_archive(f"{MODEL_NAME}_All_Results", "zip", bundle_dir)
print(f"[OK] Successfully created deployment package: {zip_name} (Contains .pth, .onnx, QAT INT8, and eval reports)")

try:
    files.download(zip_name)
    print(f"[OK] Download triggered for {zip_name}")
except Exception as e:
    print(f"[NOTE] Please download {zip_name} directly from the Colab left file browser.")"""

    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [md_header]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_0]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_1]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_2]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_3]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_4]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_5]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_6]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_7]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_8]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_9]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_10]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_11]},
    ]

    nb_data = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), nb_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Generated deployment notebook: {nb_filename}")

if __name__ == "__main__":
    main()
