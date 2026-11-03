import os
import sys
import copy
import time
import json
import argparse
import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import torch.ao.quantization as tq
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx

from models import get_model
from utils.dataloader import get_dataloaders
from utils.metrics import calculate_all_metrics, measure_inference_time
from losses import get_loss
from benchmark import set_seed


class DeployModel(nn.Module):
    """
    Wraps model to ensure purely single-tensor output for clean 
    graph tracing and Integer-only quantization.
    """
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        return out


def evaluate_model(model, dataloader, num_classes=1, device='cpu', desc="Evaluating", criterion=None):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc=desc, leave=False, ncols=100):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            if criterion is not None:
                loss = criterion(outputs, masks)
                total_loss += loss.item()

            if num_classes == 1:
                preds = torch.sigmoid(outputs)
            else:
                preds = torch.softmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(masks.cpu().numpy())

    metrics = calculate_all_metrics(all_preds, all_labels, num_classes)
    if criterion is not None:
        metrics['val_loss'] = total_loss / len(dataloader)
    return metrics


def estimate_peak_sram(model, input_size=(1, 3, 128, 128), is_int8=False):
    """
    Estimates peak activation memory (SRAM buffer / Arena) required on 
    resource-constrained microcontrollers (CMSIS-NN / TFLite Micro).
    On MCUs, operator execution requires simultaneous presence of both
    Input and Output activation tensors (ping-pong buffer), plus intermediate residuals.
    FP32 uses 4 bytes/element; INT8 uses 1 byte/element.
    """
    bytes_per_elem = 1 if is_int8 else 4
    max_buffer_bytes = 0

    def hook(module, inp, out):
        nonlocal max_buffer_bytes
        # Only count leaf operational layers (Conv2d, Linear, Pooling, etc.)
        if len(list(module.children())) == 0:
            in_bytes = 0
            if isinstance(inp, tuple) and len(inp) > 0 and isinstance(inp[0], torch.Tensor):
                in_bytes = inp[0].numel() * bytes_per_elem
            out_bytes = 0
            if isinstance(out, torch.Tensor):
                out_bytes = out.numel() * bytes_per_elem
            elif isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                out_bytes = sum(t.numel() * bytes_per_elem for t in out if isinstance(t, torch.Tensor))

            total_op_bytes = in_bytes + out_bytes
            if total_op_bytes > max_buffer_bytes:
                max_buffer_bytes = total_op_bytes

    hooks = []
    for m in model.modules():
        hooks.append(m.register_forward_hook(hook))

    device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
    dummy = torch.randn(input_size, device=device)
    with torch.no_grad():
        _ = model(dummy)

    for h in hooks:
        h.remove()

    return float(max_buffer_bytes / 1024.0)  # in KB


def run_qat(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 75)
    print("QUANTIZATION-AWARE TRAINING (QAT) PIPELINE FOR MCU DEPLOYMENT")
    print("=" * 75)
    print(f"Architecture         : {args.model}")
    print(f"Base Pretrained (.pth): {args.weights}")
    print(f"Dataset              : {args.dataset}")
    print(f"Input Dimension      : {args.size}x{args.size}")
    print(f"QAT Epochs           : {args.epochs} (Freeze BN/Observer at epoch {args.freeze_bn_epoch})")
    print(f"Optimizer            : {args.optimizer.upper()} (LR: {args.lr})")
    print(f"MCU Target Engine    : QNNPACK (Signed INT8, Per-Channel Weights, Affine Activations)")
    print(f"Device               : {device}")
    print("=" * 75 + "\n")

    set_seed(args.seed)

    if not args.weights or not os.path.exists(args.weights):
        raise FileNotFoundError(f"Base FP32 weights checkpoint not found: '{args.weights}'")

    # 1. Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=args.dataset, batch_size=args.batch_size, size=args.size,
        seed=args.seed, num_classes=args.num_classes, dataset_type=args.dataset
    )
    print(f"[OK] DataLoaders ready: {len(train_loader)} train batches | {len(test_loader)} test batches")

    # 2. Load Base FP32 Model
    base_model = get_model(args.model, num_classes=args.num_classes, seed=args.seed, input_size=args.size)
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    base_model.load_state_dict(state_dict)

    # 3. Evaluate Base FP32 Model Performance
    print("\n--- [Phase 1/4] Evaluating Base FP32 Performance on Test Set ---")
    fp32_deploy = DeployModel(base_model).to(device)
    fp32_metrics = evaluate_model(fp32_deploy, test_loader, args.num_classes, device, desc="FP32 Test")

    fp32_deploy.cpu()
    fp32_peak_sram = estimate_peak_sram(fp32_deploy, (1, 3, args.size, args.size), is_int8=False)
    fp32_latency = measure_inference_time(fp32_deploy, (1, 3, args.size, args.size), device='cpu', iterations=50)

    print(f"  FP32 mIoU:         {fp32_metrics['miou']:.4f}")
    print(f"  FP32 Dice / F1:    {fp32_metrics['dice']:.4f}")
    print(f"  FP32 Boundary IoU: {fp32_metrics['boundary_iou']:.4f}")
    print(f"  FP32 Peak SRAM:    {fp32_peak_sram:.2f} KB")
    print(f"  FP32 CPU Latency:  {fp32_latency['latency_ms']:.2f} ms ({fp32_latency['fps']:.2f} FPS)")

    # 4. Prepare Model for QAT (FX Graph Mode with QNNPACK Engine)
    print("\n--- [Phase 2/4] Tracing FX Graph & Inserting FakeQuantize Nodes ---")
    deploy_model = DeployModel(base_model)
    deploy_model.eval()

    dummy_input = torch.randn(2, 3, args.size, args.size)
    qconfig_mapping = tq.get_default_qat_qconfig_mapping('qnnpack')

    try:
        qat_model = prepare_qat_fx(deploy_model, qconfig_mapping, (dummy_input,))
        qat_model = qat_model.to(device)
        print("[OK] QAT Graph successfully prepared with FakeQuantize layers!")
    except Exception as e:
        print(f"[WARN] Architecture '{args.model}' is not supported by PyTorch FX Graph Mode ({e}).")
        print(f"       Skipping QAT fine-tuning for this architecture and preserving FP32 baseline.")
        os.makedirs(args.output_path, exist_ok=True)
        report_json = os.path.join(args.output_path, f"{args.model}_qat_benchmark_{args.dataset}.json")
        with open(report_json, 'w') as f:
            json.dump({
                'config': vars(args),
                'qat_supported': False,
                'fp32_metrics': fp32_metrics,
                'hardware': {
                    'fp32_flash_kb': os.path.getsize(args.weights) / 1024.0,
                    'fp32_peak_sram_kb': fp32_peak_sram,
                    'fp32_latency_ms': fp32_latency['latency_ms'],
                    'fp32_fps': fp32_latency['fps'],
                }
            }, f, indent=2)
        print(f"[OK] Benchmark report saved: {report_json}\n")
        return

    # 5. Setup QAT Optimizer & Scheduler
    criterion = get_loss(args.loss, num_classes=args.num_classes).to(device)
    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(qat_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(qat_model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 6. QAT Fine-tuning Loop (Tracking val_loss for best checkpoint consistency)
    print("\n--- [Phase 3/4] Quantization-Aware Fine-Tuning ---")
    best_val_loss = float('inf')
    best_qat_weights = None

    for epoch in range(args.epochs):
        qat_model.train()

        # Freezing strategy: Freeze Observer & BN stats in the last few epochs for scale stabilization
        if epoch >= args.freeze_bn_epoch:
            qat_model.apply(torch.ao.quantization.disable_observer)
            qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            freeze_status = "[FROZEN OBSERVER/BN]"
        else:
            freeze_status = "[CALIBRATING SCALES]"

        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"QAT Epoch {epoch+1}/{args.epochs} {freeze_status}", leave=False, ncols=100)

        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = qat_model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)

        # Validation at epoch end (Evaluating val_loss)
        val_metrics = evaluate_model(qat_model, val_loader, args.num_classes, device, desc="Val", criterion=criterion)
        val_loss = val_metrics['val_loss']
        val_miou = val_metrics['miou']

        print(f"Epoch {epoch+1:2d}/{args.epochs:2d} | Train: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f} {freeze_status}", end="")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_qat_weights = copy.deepcopy(qat_model.state_dict())
            print(" -> [BEST VAL]")
        else:
            print()

    # Restore best QAT weights
    if best_qat_weights is not None:
        qat_model.load_state_dict(best_qat_weights)

    # 7. Convert to True INT8 Quantized Module
    print("\n--- [Phase 4/4] Converting FakeQuantize to True INT8 Hardware Graph ---")
    qat_model.eval()
    qat_model = qat_model.cpu()

    try:
        int8_model = convert_fx(qat_model)
        print("[OK] Conversion complete! Model converted to true Integer-Only operations.")
    except Exception as e:
        print(f"[WARN] Failed to convert FX graph to native INT8 ({e}). Saving calibrated QAT weights.")
        os.makedirs(args.output_path, exist_ok=True)
        int8_pth_path = os.path.join(args.output_path, f"{args.model}_qat_weights_{args.dataset}.pth")
        torch.save(qat_model.state_dict(), int8_pth_path)
        return

    # 8. True INT8 Evaluation on Test Set
    int8_metrics = evaluate_model(int8_model, test_loader, args.num_classes, device='cpu', desc="INT8 Test")
    int8_peak_sram = estimate_peak_sram(int8_model, (1, 3, args.size, args.size), is_int8=True)
    int8_latency = measure_inference_time(int8_model, (1, 3, args.size, args.size), device='cpu', iterations=50)

    # 9. Compute Footprint & Savings
    os.makedirs(args.output_path, exist_ok=True)
    int8_pth_path = os.path.join(args.output_path, f"{args.model}_qat_int8_{args.dataset}.pth")
    torch.save(int8_model.state_dict(), int8_pth_path)

    int8_flash_kb = os.path.getsize(int8_pth_path) / 1024.0
    fp32_flash_kb = os.path.getsize(args.weights) / 1024.0

    delta_miou = (int8_metrics['miou'] - fp32_metrics['miou']) * 100.0
    delta_dice = (int8_metrics['dice'] - fp32_metrics['dice']) * 100.0
    sram_savings = ((fp32_peak_sram - int8_peak_sram) / fp32_peak_sram) * 100.0
    flash_savings = ((fp32_flash_kb - int8_flash_kb) / fp32_flash_kb) * 100.0

    # 10. Display Paper Results
    print("\n" + "=" * 80)
    print("QUANTIZATION RESULTS SUMMARY FOR PAPER (FP32 vs INT8 QAT)")
    print("=" * 80)
    print(f"{'Metric':<22} | {'FP32 Baseline':<16} | {'INT8 QAT (Ours)':<16} | {'Delta':<12}")
    print("-" * 72)
    print(f"{'mIoU (%)':<22} | {fp32_metrics['miou']*100:<16.2f} | {int8_metrics['miou']*100:<16.2f} | {delta_miou:+.2f}%")
    print(f"{'Dice / F1 (%)':<22} | {fp32_metrics['dice']*100:<16.2f} | {int8_metrics['dice']*100:<16.2f} | {delta_dice:+.2f}%")
    print(f"{'Precision (%)':<22} | {fp32_metrics['precision']*100:<16.2f} | {int8_metrics['precision']*100:<16.2f} | {(int8_metrics['precision']-fp32_metrics['precision'])*100:+.2f}%")
    print(f"{'Recall (%)':<22} | {fp32_metrics['recall']*100:<16.2f} | {int8_metrics['recall']*100:<16.2f} | {(int8_metrics['recall']-fp32_metrics['recall'])*100:+.2f}%")
    print(f"{'Boundary IoU (%)':<22} | {fp32_metrics['boundary_iou']*100:<16.2f} | {int8_metrics['boundary_iou']*100:<16.2f} | {(int8_metrics['boundary_iou']-fp32_metrics['boundary_iou'])*100:+.2f}%")
    print("-" * 72)
    print(f"{'Weight Flash (KB)':<22} | {fp32_flash_kb:<16.1f} | {int8_flash_kb:<16.1f} | -{flash_savings:.1f}%")
    print(f"{'Peak SRAM Buffer (KB)':<22} | {fp32_peak_sram:<16.1f} | {int8_peak_sram:<16.1f} | -{sram_savings:.1f}%")
    print(f"{'Latency (ms)':<22} | {fp32_latency['latency_ms']:<16.2f} | {int8_latency['latency_ms']:<16.2f} | {(int8_latency['latency_ms']-fp32_latency['latency_ms']):+.2f} ms")
    print(f"{'Throughput (FPS)':<22} | {fp32_latency['fps']:<16.2f} | {int8_latency['fps']:<16.2f} | {(int8_latency['fps']-fp32_latency['fps']):+.2f} FPS")
    print("=" * 80 + "\n")

    # 11. Print LaTeX Snippet
    print("LaTeX Table Snippet (Ready for Paper):")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Quantization-Aware Training Performance and MCU Footprint}")
    print(r"\label{tab:qat_results}")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\hline")
    print(r"\textbf{Format} & \textbf{mIoU (\%)} & \textbf{Dice (\%)} & \textbf{B-IoU (\%)} & \textbf{Flash (KB)} & \textbf{SRAM (KB)} & \textbf{FPS} \\ \hline")
    print(f"FP32 Baseline & {fp32_metrics['miou']*100:.2f} & {fp32_metrics['dice']*100:.2f} & {fp32_metrics['boundary_iou']*100:.2f} & {fp32_flash_kb:.1f} & {fp32_peak_sram:.1f} & {fp32_latency['fps']:.1f} \\\\")
    print(f"\\textbf{{INT8 (QAT Ours)}} & \\textbf{{{int8_metrics['miou']*100:.2f}}} & \\textbf{{{int8_metrics['dice']*100:.2f}}} & \\textbf{{{int8_metrics['boundary_iou']*100:.2f}}} & \\textbf{{{int8_flash_kb:.1f}}} & \\textbf{{{int8_peak_sram:.1f}}} & \\textbf{{{int8_latency['fps']:.1f}}} \\\\ \\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()

    # 12. Save JSON Report
    report_json = os.path.join(args.output_path, f"{args.model}_qat_benchmark_{args.dataset}.json")
    with open(report_json, 'w') as f:
        json.dump({
            'config': vars(args),
            'qat_supported': True,
            'best_val_loss': best_val_loss,
            'fp32_metrics': fp32_metrics,
            'int8_metrics': int8_metrics,
            'hardware': {
                'fp32_flash_kb': fp32_flash_kb,
                'int8_flash_kb': int8_flash_kb,
                'fp32_peak_sram_kb': fp32_peak_sram,
                'int8_peak_sram_kb': int8_peak_sram,
                'fp32_latency_ms': fp32_latency['latency_ms'],
                'int8_latency_ms': int8_latency['latency_ms'],
                'fp32_fps': fp32_latency['fps'],
                'int8_fps': int8_latency['fps'],
            }
        }, f, indent=2)

    print(f"[OK] INT8 checkpoint saved: {int8_pth_path}")
    print(f"[OK] Full benchmark metrics saved: {report_json}\n")


def main():
    parser = argparse.ArgumentParser(description="Quantization-Aware Training (QAT) Engine for Microcontroller Deployment")
    parser.add_argument('--weights', type=str, required=True,
                        help="Path to pretrained FP32 checkpoint (.pth)")
    parser.add_argument('--model', type=str, default='mambafloodlite',
                        help="Model architecture name (default: mambafloodlite)")
    parser.add_argument('--dataset', type=str, default='floodkaggle',
                        choices=['floodkaggle', 'aifloodsense', 'floodscene', 'floodvn'])
    parser.add_argument('--size', type=int, default=128, help="Image size (default: 128 for MCU deployment)")
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--loss', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=15, help="Number of QAT fine-tuning epochs (default: 15)")
    parser.add_argument('--freeze_bn_epoch', type=int, default=12,
                        help="Epoch to freeze BN stats & quantizer scales (default: 12)")
    parser.add_argument('--lr', type=float, default=1e-4, help="QAT learning rate (default: 1e-4)")
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'],
                        help="Optimizer (default: sgd with momentum 0.9)")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_path', type=str, default='outputs')
    args = parser.parse_args()

    if args.num_classes is None:
        args.num_classes = 1

    if args.loss is None:
        args.loss = 'bce' if args.num_classes == 1 else 'ce_dice'

    run_qat(args)


if __name__ == '__main__':
    main()
