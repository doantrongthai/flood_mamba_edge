import os
import sys
import argparse
import json
import torch
import numpy as np

from benchmark import set_seed, download_dataset, run_multiseed_experiments, verify_reproducibility
from utils.trainer import train_segmentation
from utils.dataloader import get_dataloaders
from utils.metrics import calculate_all_metrics, calculate_model_complexity, measure_inference_time
from models import get_model


def run_cross_dataset_evaluation(args):
    """
    Cross-Dataset Generalization Evaluation (Zero-Shot Domain Transfer):
    Evaluates a model trained on source_dataset directly on one or more target datasets.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "=" * 70)
    print("CROSS-DATASET GENERALIZATION EVALUATION (ZERO-SHOT DOMAIN TRANSFER)")
    print("=" * 70)
    print(f"Model Architecture : {args.model}")
    print(f"Weights Checkpoint : {args.weights}")
    print(f"Source Dataset     : {args.source_dataset}")
    print(f"Target Datasets    : {args.target_datasets}")
    print(f"Device             : {device}")
    print("=" * 70 + "\n")

    if not args.weights or not os.path.exists(args.weights):
        raise FileNotFoundError(f"Checkpoint weights not found: '{args.weights}'")

    # Load model
    model = get_model(args.model, num_classes=args.num_classes, seed=args.seed, input_size=args.size)
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    cross_results = {}

    for target_ds in args.target_datasets:
        print(f"\n--- Evaluating on Target Domain: {target_ds.upper()} ---")
        try:
            _, _, test_loader = get_dataloaders(
                dataset=target_ds, batch_size=args.batch_size, size=args.size,
                seed=args.seed, num_classes=args.num_classes, dataset_type=target_ds
            )
        except Exception as e:
            print(f"[SKIP] Could not load target dataset '{target_ds}': {e}")
            continue

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device)
                outputs = model(images)
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]

                if args.num_classes == 1:
                    preds = torch.sigmoid(outputs)
                else:
                    preds = torch.softmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(masks.cpu().numpy())

        metrics = calculate_all_metrics(all_preds, all_labels, args.num_classes)
        cross_results[target_ds] = metrics

        print(f"  mIoU:         {metrics['miou']:.4f}")
        print(f"  Dice / F1:    {metrics['dice']:.4f}")
        print(f"  Pixel Acc:    {metrics['pixel_accuracy']:.4f}")
        print(f"  Precision:    {metrics['precision']:.4f}")
        print(f"  Recall:       {metrics['recall']:.4f}")
        print(f"  FWIoU:        {metrics['fwiou']:.4f}")
        print(f"  Boundary IoU: {metrics['boundary_iou']:.4f}")

    print("\n" + "=" * 70)
    print("CROSS-DATASET SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Source -> Target':<30} | {'mIoU':<8} | {'Dice':<8} | {'Precision':<10} | {'Recall':<8} | {'Boundary IoU':<12}")
    print("-" * 88)
    for target_ds, m in cross_results.items():
        name_col = f"{args.source_dataset} -> {target_ds}"
        print(f"{name_col:<30} | {m['miou']:<8.4f} | {m['dice']:<8.4f} | {m['precision']:<10.4f} | {m['recall']:<8.4f} | {m['boundary_iou']:<12.4f}")
    print("=" * 70 + "\n")

    os.makedirs(args.output_path, exist_ok=True)
    report_file = os.path.join(args.output_path, f"{args.model}_cross_eval_{args.source_dataset}.json")
    with open(report_file, 'w') as f:
        json.dump({
            'source_dataset': args.source_dataset,
            'model': args.model,
            'weights': args.weights,
            'target_results': cross_results
        }, f, indent=2)
    print(f"Cross-dataset results saved to: {report_file}")
    return cross_results


def main():
    parser = argparse.ArgumentParser(description="Flood Mamba Edge: Unified Training & Evaluation Framework")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'cross_eval'],
                        help="Execution mode: 'train' or 'cross_eval'")
    parser.add_argument('--dataset', type=str, default='floodkaggle',
                        choices=['floodvn', 'floodkaggle', 'aifloodsense', 'floodscene'])
    parser.add_argument('--model', type=str, default='mambafloodlite',
                        help="Model architecture name (e.g. mambafloodlite, unet, segformer, etc.)")
    parser.add_argument('--size', type=int, default=128, help="Input image dimension (default: 128)")
    parser.add_argument('--num_classes', type=int, default=None,
                        help="Number of classes (default: 1 for binary flood segmentation)")
    parser.add_argument('--loss', type=str, default=None,
                        help="Loss function: bce, dice, focal, ce, ce_dice, lovasz, etc.")
    parser.add_argument('--aux_loss', type=str, default='boundary', choices=['boundary', 'bce', 'dice'])
    parser.add_argument('--aux_weight', type=float, default=0.4)
    parser.add_argument('--pretrained', action='store_true',
                        help="Load pretrained ImageNet weights for heavy baselines if supported")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_path', type=str, default='outputs')
    parser.add_argument('--quiet', action='store_true', help='Suppress tqdm progress bars for clean minimal logging')
    parser.add_argument('--download', action='store_true', help="Download dataset automatically if available")
    parser.add_argument('--verify_repro', action='store_true', help="Run strict reproducibility test")
    parser.add_argument('--multiseed', action='store_true', help="Run multi-seed experiment for paper statistics")
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 2024])

    # Cross-dataset evaluation parameters
    parser.add_argument('--weights', type=str, default=None, help="Checkpoint file path for cross-dataset evaluation")
    parser.add_argument('--source_dataset', type=str, default='floodkaggle')
    parser.add_argument('--target_datasets', type=str, nargs='+', default=['floodkaggle', 'aifloodsense'])

    args = parser.parse_args()

    if args.num_classes is None:
        args.num_classes = 1

    if args.loss is None:
        args.loss = 'bce' if args.num_classes == 1 else 'ce_dice'

    AUTO_PRETRAINED_MODELS = {'deeplabv3'}
    if args.model.lower() in AUTO_PRETRAINED_MODELS and not args.pretrained:
        args.pretrained = True
        print(f"[AUTO-PRETRAIN] Model '{args.model}' has a heavy ImageNet backbone. Automatically enabled --pretrained!")

    set_seed(args.seed)

    if args.download:
        download_dataset(args.dataset)

    if args.mode == 'cross_eval':
        run_cross_dataset_evaluation(args)
        return

    if args.verify_repro:
        verify_reproducibility(args, num_runs=2)
        return

    if args.multiseed:
        run_multiseed_experiments(args, seeds=args.seeds)
        return

    # Standard single-run training
    print("=" * 70)
    print(f"TRAINING PIPELINE | Mode: Single-Run | Dataset: {args.dataset}")
    print(f"Model: {args.model} (pretrained={args.pretrained}) | Size: {args.size} | Classes: {args.num_classes}")
    print(f"Loss: {args.loss} | Aux Loss: {args.aux_loss} (weight={args.aux_weight})")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr} | Seed: {args.seed}")
    print("=" * 70)

    train_segmentation(
        model_name=args.model, loss_name=args.loss, size=args.size,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        dataset=args.dataset, output_path=args.output_path,
        seed=args.seed, num_classes=args.num_classes, dataset_type=args.dataset,
        aux_loss_name=args.aux_loss, aux_weight=args.aux_weight, pretrained=args.pretrained,
        quiet=args.quiet
    )


if __name__ == '__main__':
    main()
