import os
import argparse
import json
import time
import torch
import numpy as np

from models import get_model
from utils.dataloader import get_dataloaders
from utils.metrics import calculate_all_metrics


def find_checkpoint(model_name, loss_name, source_dataset, seed, weights_dir='outputs'):
    clean_loss = loss_name.lower().replace('-', '_')
    names_to_try = [model_name]
    if model_name == 'bgfmamba':
        names_to_try.extend(['ablation_mamba_bidirectional', 'mambaedge'])

    candidates = []
    for m in names_to_try:
        candidates.extend([
            os.path.join(weights_dir, f"seed_{seed}", f"{m}_{clean_loss}_{source_dataset}_s{seed}.pth"),
            os.path.join(weights_dir, f"{m}_{clean_loss}_{source_dataset}_s{seed}.pth"),
            os.path.join(weights_dir, f"seed_{seed}", f"{m}_{loss_name}_{source_dataset}_s{seed}.pth"),
            os.path.join(weights_dir, f"{m}_{loss_name}_{source_dataset}_s{seed}.pth"),
        ])
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def evaluate_single_checkpoint(model, device, test_loader, num_classes=1, batch_size=4):
    all_preds = []
    all_labels = []
    inference_times = []

    model.eval()
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            start_t = time.perf_counter()
            outputs = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_times.append(time.perf_counter() - start_t)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            if num_classes == 1:
                preds = torch.sigmoid(outputs)
            else:
                preds = torch.softmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(masks.cpu().numpy())

    metrics = calculate_all_metrics(all_preds, all_labels, num_classes)
    avg_latency = float(np.mean(inference_times) / batch_size * 1000)
    metrics['latency_ms'] = avg_latency
    metrics['fps'] = float(1000.0 / avg_latency if avg_latency > 0 else 0)
    return metrics


def evaluate_cross_dataset(model_name, source_dataset, target_datasets,
                           loss_name='lovasz_focal', weights_path=None, seeds=None,
                           size=128, batch_size=4, num_classes=1, output_dir='outputs',
                           download=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if download:
        from benchmark import download_dataset
        for ds in target_datasets:
            if not os.path.exists(ds):
                print(f"[*] Target dataset '{ds}' not found on disk. Initiating download...")
                download_dataset(ds)

    is_multiseed = seeds is not None and len(seeds) > 0

    if not is_multiseed:
        seed_list = [42]
        weights_map = {42: weights_path}
    else:
        seed_list = seeds
        weights_map = {}
        for s in seed_list:
            if weights_path and '{seed}' in weights_path:
                w = weights_path.format(seed=s)
            else:
                w = find_checkpoint(model_name, loss_name, source_dataset, s, weights_dir=output_dir)
            weights_map[s] = w

    print("=" * 80)
    print("GENERALIZATION BENCHMARK: IN-DOMAIN, CROSS-DOMAIN & EXTERNAL TEST")
    print("=" * 80)
    print(f"Model Architecture    : {model_name}")
    print(f"Source Domain (Train) : {source_dataset.upper()}")
    print(f"Target Domains (Test) : {[d.upper() for d in target_datasets]}")
    print(f"Mode                  : {'MULTI-SEED (' + str(len(seed_list)) + ' seeds: ' + str(seed_list) + ')' if is_multiseed else 'SINGLE SEED'}")
    print(f"Inference Device      : {device}")
    print("=" * 80 + "\n")

    runs = {}
    domain_accumulators = {td: [] for td in target_datasets}

    for seed in seed_list:
        w_path = weights_map[seed]
        if not os.path.exists(w_path):
            print(f"[!] Warning: Checkpoint for seed {seed} not found at '{w_path}'. Skipping.")
            continue

        print(f"\n" + "-" * 70)
        print(f"--- EVALUATING SEED: {seed} (Checkpoint: {w_path}) ---")
        print("-" * 70)

        model = get_model(model_name, num_classes=num_classes, seed=seed, input_size=size)
        checkpoint = torch.load(w_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        model = model.to(device)

        runs[seed] = {}

        for target_ds in target_datasets:
            if target_ds.lower() == source_dataset.lower():
                eval_tag = "IN-DOMAIN"
            elif target_ds.lower() == 'floodvn':
                eval_tag = "EXTERNAL TEST"
            else:
                eval_tag = "CROSS-DOMAIN"

            try:
                _, _, test_loader = get_dataloaders(
                    dataset=target_ds, batch_size=batch_size, size=size,
                    seed=seed, num_classes=num_classes, dataset_type=target_ds
                )
            except Exception as e:
                print(f"[!] Could not load dataset '{target_ds}' for seed {seed}: {e}")
                continue

            metrics = evaluate_single_checkpoint(model, device, test_loader, num_classes=num_classes, batch_size=batch_size)
            metrics['eval_type'] = eval_tag
            runs[seed][target_ds] = metrics
            domain_accumulators[target_ds].append(metrics)

            print(f">>> [{eval_tag}] {source_dataset.upper()} -> {target_ds.upper()}:")
            print(f"    mIoU: {metrics['miou']:.4f} | Dice: {metrics['dice']:.4f} | Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | B-IoU: {metrics['boundary_iou']:.4f} | Latency: {metrics['latency_ms']:.2f} ms ({metrics['fps']:.1f} FPS)")

    if not is_multiseed:
        print("\n" + "=" * 80)
        print("CROSS-DATASET GENERALIZATION SUMMARY TABLE (SINGLE SEED)")
        print("=" * 80)
        print(f"{'Target Domain':<15} | {'Type':<15} | {'mIoU':<8} | {'Dice':<8} | {'Precision':<10} | {'Recall':<8} | {'B-IoU':<8}")
        print("-" * 80)
        single_res = runs.get(seed_list[0], {})
        for ds_name, m in single_res.items():
            print(f"{ds_name.upper():<15} | {m['eval_type']:<15} | {m['miou']:<8.4f} | {m['dice']:<8.4f} | {m['precision']:<10.4f} | {m['recall']:<8.4f} | {m['boundary_iou']:<8.4f}")
        print("=" * 80 + "\n")

        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f'{model_name}_cross_dataset_{source_dataset}.json')
        with open(out_file, 'w') as f:
            json.dump({
                'model': model_name,
                'source_dataset': source_dataset,
                'weights': weights_path,
                'input_size': size,
                'results': {ds: {'eval_type': m['eval_type'], 'metrics': m} for ds, m in single_res.items()}
            }, f, indent=2)
        print(f"[OK] Generalization benchmark saved: {out_file}")
        return single_res

    stats = {}
    for target_ds in target_datasets:
        m_list = domain_accumulators[target_ds]
        if not m_list:
            continue
        eval_tag = m_list[0]['eval_type']
        stat_dict = {'eval_type': eval_tag, 'num_seeds': len(m_list)}
        for k in ['miou', 'dice', 'precision', 'recall', 'pixel_accuracy', 'boundary_iou', 'fwiou', 'latency_ms', 'fps']:
            vals = [m[k] for m in m_list if k in m]
            if vals:
                stat_dict[k] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals))
                }
        stats[target_ds] = stat_dict

    print("\n" + "=" * 105)
    print(f"MULTI-SEED CROSS-DATASET GENERALIZATION SUMMARY TABLE ({len(seed_list)} Seeds: {seed_list})")
    print("=" * 105)
    print(f"{'Target Domain':<15} | {'Type':<14} | {'mIoU (Mean±Std)':<18} | {'Dice (Mean±Std)':<18} | {'B-IoU (Mean±Std)':<18} | {'FPS (Mean±Std)':<16}")
    print("-" * 105)
    for ds_name, s in stats.items():
        miou_str = f"{s['miou']['mean']:.4f} ± {s['miou']['std']:.4f}"
        dice_str = f"{s['dice']['mean']:.4f} ± {s['dice']['std']:.4f}"
        biou_str = f"{s['boundary_iou']['mean']:.4f} ± {s['boundary_iou']['std']:.4f}"
        fps_str  = f"{s['fps']['mean']:.1f} ± {s['fps']['std']:.1f}"
        print(f"{ds_name.upper():<15} | {s['eval_type']:<14} | {miou_str:<18} | {dice_str:<18} | {biou_str:<18} | {fps_str:<16}")
    print("=" * 105)

    print("\nLaTeX Multi-seed Format:")
    for ds_name, s in stats.items():
        print(f"% {s['eval_type']} ({ds_name.upper()}):")
        print(f"mIoU: ${s['miou']['mean']:.4f} \\pm {s['miou']['std']:.4f}$, Dice: ${s['dice']['mean']:.4f} \\pm {s['dice']['std']:.4f}$, B-IoU: ${s['boundary_iou']['mean']:.4f} \\pm {s['boundary_iou']['std']:.4f}$")

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f'{model_name}_cross_dataset_{source_dataset}_multiseed.json')
    with open(out_file, 'w') as f:
        json.dump({
            'model': model_name,
            'source_dataset': source_dataset,
            'seeds': seed_list,
            'input_size': size,
            'statistics': stats,
            'runs': runs
        }, f, indent=2)
    print(f"\n[OK] Multi-seed generalization benchmark saved: {out_file}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Cross-Dataset Generalization Evaluation Suite")
    parser.add_argument('--model',           type=str, default='mambafloodlite', help="Model name")
    parser.add_argument('--loss',            type=str, default='lovasz_focal',   help="Loss name used during training")
    parser.add_argument('--weights',         type=str, default=None,             help="Path to single trained .pth weights (optional)")
    parser.add_argument('--seeds',           type=int, nargs='+', default=None,  help="List of seeds for multi-seed evaluation (e.g. 42 123 456 789 2024)")
    parser.add_argument('--source_dataset',  type=str, default='floodkaggle',    help="Source domain used in training")
    parser.add_argument('--target_datasets', type=str, nargs='+', default=['floodkaggle', 'floodvn'],
                        help="List of target datasets to evaluate on")
    parser.add_argument('--size',            type=int, default=128,              help="Input dimension")
    parser.add_argument('--batch_size',      type=int, default=4)
    parser.add_argument('--num_classes',     type=int, default=1)
    parser.add_argument('--output_dir',      type=str, default='outputs')
    parser.add_argument('--download',        action='store_true',                help="Download missing datasets")
    args = parser.parse_args()

    evaluate_cross_dataset(
        model_name=args.model,
        source_dataset=args.source_dataset,
        target_datasets=args.target_datasets,
        loss_name=args.loss,
        weights_path=args.weights,
        seeds=args.seeds,
        size=args.size,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        output_dir=args.output_dir,
        download=args.download
    )


if __name__ == '__main__':
    main()
