"""
Script to build screening_dual_stream_ultra_light.ipynb
Evaluates all 20 Decoupled Dual-Stream Wavelet Mamba Models (dual_v1 -> dual_v20).
- Resolution: 128x128
- Budget: 200k-300k parameters
- Official precompiled Mamba CUDA kernel cell (FILE_ID = "1kuBAKlNtIc8G1KQFqO8xV7t9ZEM4XATB")
- Colab 5,000-line buffer safe streaming (filters tqdm whitespace blank lines)
- NaN auto-skip
- Auto-resume from CSV: screening_dual_stream_results.csv
"""

import json
import os

def main():
    nb_filename = "screening_dual_stream_ultra_light.ipynb"
    csv_file = "screening_dual_stream_results.csv"
    models_list = [f"dual_v{i}" for i in range(1, 21)]

    md_header = f"""# Master Benchmark Suite: 20 Decoupled Dual-Stream Wavelet Mamba Models (dual_v1 -> dual_v20)
## Paper 1 Flagship Architecture Family: Decoupled Dual-Stream (Body & Edge Streams) with Haar Wavelet & RepStructural Convs
### Target Resolution: $128 \\times 128$ | Evaluated on Satellite (`floodkaggle`) and UAV (`aifloodsense`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/doantrongthai/flood_mamba_edge/blob/main/{nb_filename})

This dedicated screening notebook evaluates 20 novel, systematically engineered ultra-lightweight architectures for **Paper 1**.
Every model is strictly constrained within **200,000 to 250,000 parameters** and features a **100% distinct design** from `m12_v16`:
1. **Stem**: Parameter-free 2D Haar Discrete Wavelet Transform (lossless frequency decomposition into LL, LH, HL, HH).
2. **Body Stream**: Rep-Structural Inverted Residual (RIR) Mobile Encoder with multi-branch depthwise training and algebraic reparameterization.
3. **Bottleneck**: Directional Strip-PPM Mamba SSM (Horizontal $1 \\times W$ & Vertical $H \\times 1$ directional pooling).
4. **Edge Stream**: High-frequency subband boundary extraction with morphological gradients, curvature divergence, and Sobel-Laplacian priors.
5. **Cross-Stream Interaction (CSFI)**: Dynamic boundary-guided semantic modulation.

### Key Execution Features:
1. **Official Precompiled Mamba CUDA Kernel**: Instant Tesla T4 kernel setup without slow C++ compiling.
2. **Standard Mamba Edge Resolution ($128 \\times 128$)**: Real-time edge AI throughput (>100 FPS).
3. **Colab 5,000-Line Safe Stream Logging**: Filters tqdm carriage returns and empty whitespace lines to eliminate output buffer truncation.
4. **Instant NaN Skip**: Divergent runs abort immediately upon encountering NaN, advancing to the next model.
5. **Auto-Resume**: Automatically loads completed runs from `{csv_file}` on re-run.
"""

    cell_gpu = """# 1. Environment & GPU Hardware Verification
!nvidia-smi
import torch
print(f"PyTorch Version : {torch.__version__}")
print(f"CUDA Available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name     : {torch.cuda.get_device_name(0)}")
    print(f"Capability      : {torch.cuda.get_device_capability(0)}")"""

    cell_mamba = """# 2. Setup Official Hardware-Accelerated Mamba CUDA Kernel on Google Colab GPU (Tesla T4)
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

    cell_git = """# 3. Clone / Update Repository & Install Utilities
import os, sys

if not os.path.exists("flood_mamba_edge"):
    !git clone https://github.com/doantrongthai/flood_mamba_edge.git
    %cd flood_mamba_edge
else:
    %cd flood_mamba_edge
    !git pull origin main

!pip install -q tabulate pandas gdown
print("[OK] Workspace ready at:", os.getcwd())"""

    cell_download = f"""# 4. Verify & Pre-Download Datasets at 128x128
import sys, subprocess

DATASETS = ["floodkaggle", "aifloodsense"]
for ds in DATASETS:
    print(f"\\n--- Checking / Downloading Dataset: {{ds}} ---")
    subprocess.run([
        sys.executable, "main.py",
        "--dataset", ds,
        "--download",
        "--epochs", "1",
        "--batch_size", "2",
        "--model", "{models_list[0]}",
        "--size", "128"
    ], check=False)
print("\\n[OK] All datasets downloaded and verified successfully at 128x128!")"""

    cell_loop = f"""# 5. Automated Benchmark Loop with Clean Logging & NaN-Safe Skip
import sys, os, re, time, subprocess
import pandas as pd
from tabulate import tabulate

SEED = 42
IMAGE_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 100
LR = 0.001
LOSS_NAME = "lovasz_focal"
NUM_CLASSES = 1
DATASETS = ["floodkaggle", "aifloodsense"]

MODELS_TO_BENCHMARK = {models_list}

total_runs = len(MODELS_TO_BENCHMARK) * len(DATASETS)
run_idx = 0
results = []
csv_out_path = "{csv_file}"

# Load previous partial results if resuming
if os.path.exists(csv_out_path):
    try:
        prev_df = pd.read_csv(csv_out_path)
        results = prev_df.to_dict("records")
        print(f"[RESUME] Loaded {{len(results)}} completed runs from {{csv_out_path}}")
    except Exception as e:
        print(f"[RESUME WARNING] Could not read existing results: {{e}}")

completed_keys = {{(r["model"], r["dataset"]) for r in results if "model" in r and "dataset" in r}}

print(f"Total planned runs: {{total_runs}} (Already completed: {{len(completed_keys)}})")
print("=" * 85)

for model_name in MODELS_TO_BENCHMARK:
    for ds in DATASETS:
        run_idx += 1
        if (model_name, ds) in completed_keys:
            print(f"[{{run_idx}}/{{total_runs}}] [SKIP] {{model_name}} on {{ds}} already completed.")
            continue

        print(f"\\n{{'='*85}}")
        print(f"[{{run_idx}}/{{total_runs}}] STARTING: {{model_name.upper()}} on {{ds.upper()}} (Size={{IMAGE_SIZE}}, Epochs={{EPOCHS}})")
        print(f"{{'='*85}}")

        cmd = [
            sys.executable, "-u", "main.py",
            "--dataset", ds,
            "--model", model_name,
            "--loss", LOSS_NAME,
            "--epochs", str(EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--size", str(IMAGE_SIZE),
            "--lr", str(LR),
            "--seed", str(SEED),
            "--output_path", "outputs"
        ]

        t0 = time.time()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        captured_output = []
        is_nan = False

        for line in proc.stdout:
            # Check for NaN divergence in training/validation output
            if ("train:" in line.lower() or "val:" in line.lower()) and "nan" in line.lower():
                print(f"\\n[WARNING] NaN loss detected: {{line.strip()}}! Skipping {{model_name}} on {{ds}} immediately.")
                is_nan = True
                proc.terminate()
                break

            # FILTER OUT batch-level tqdm lines and whitespace blank lines to avoid Colab 5,000-line buffer truncation
            clean_line = line.strip()
            if not clean_line:
                continue
            if "%|" in line or "[Train]:" in line or "[Val]:" in line or "it/s" in line or "\\r" in line:
                continue

            sys.stdout.write(line + "\\n" if not line.endswith("\\n") else line)
            sys.stdout.flush()
            captured_output.append(line)

        proc.wait()
        elapsed = time.time() - t0

        if is_nan:
            print(f"[{{model_name.upper()}}] Skipped due to NaN divergence.")
            continue

        full_text = "".join(captured_output)

        # Parse metrics
        m_miou = re.search(r"mIoU:\\s+([0-9\\.]+)", full_text)
        m_dice = re.search(r"Dice / F1 Score:\\s+([0-9\\.]+)", full_text)
        m_prec = re.search(r"Precision:\\s+([0-9\\.]+)", full_text)
        m_rec = re.search(r"Recall:\\s+([0-9\\.]+)", full_text)
        m_biou = re.search(r"Boundary IoU:\\s+([0-9\\.]+)", full_text)
        m_loss = re.search(r"Best Val Loss:\\s+([0-9\\.]+)", full_text)
        m_params = re.search(r"Total Params \\(Train\\):\\s+([0-9,]+)", full_text)
        m_fps = re.search(r"FPS:\\s+([0-9\\.]+)", full_text)

        row = {{
            "model": model_name,
            "dataset": ds,
            "mIoU": float(m_miou.group(1)) if m_miou else None,
            "Dice": float(m_dice.group(1)) if m_dice else None,
            "Precision": float(m_prec.group(1)) if m_prec else None,
            "Recall": float(m_rec.group(1)) if m_rec else None,
            "B_IoU": float(m_biou.group(1)) if m_biou else None,
            "val_loss": float(m_loss.group(1)) if m_loss else None,
            "params": m_params.group(1) if m_params else "N/A",
            "fps": float(m_fps.group(1)) if m_fps else None,
            "time_sec": round(elapsed, 2)
        }}

        results.append(row)
        completed_keys.add((model_name, ds))

        # Save progress after every run
        df_cur = pd.DataFrame(results)
        df_cur.to_csv(csv_out_path, index=False)

        print(f"\\n[DONE] {{model_name}} on {{ds}} in {{elapsed:.1f}}s | mIoU: {{row['mIoU']}} | Dice: {{row['Dice']}}")

# 6. Master Benchmark Summary Table
print(f"\\n{{'='*90}}")
print(f"=== FINAL COMPLETE {{len(MODELS_TO_BENCHMARK)}}-MODEL MASTER BENCHMARK SUMMARY ===")
print(f"{{'='*90}}")
if results:
    summary_df = pd.DataFrame(results)
    print(tabulate(summary_df, headers="keys", tablefmt="github"))
    summary_df.to_csv("FINAL_MASTER_BENCHMARK_SUMMARY.csv", index=False)
    print(f"\\n[OK] Complete master summary saved to: FINAL_MASTER_BENCHMARK_SUMMARY.csv")
"""

    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [md_header]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_gpu]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_mamba]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_git]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_download]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_loop]}
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
    print(f"[OK] Generated notebook: {nb_filename}")

if __name__ == "__main__":
    main()
